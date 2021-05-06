import torch
from torch.fx import (
    GraphModule,
    Proxy,
    map_arg
)

from torch.fx.graph import (
    Graph,
    Node,
)

from torch.fx.node import Argument

from torch.quantization import (
    propagate_qconfig_,
    convert,
)

from ..quantization_mappings import (
    get_default_qat_module_mappings,
)

from ..quantize import (
    _remove_qconfig,
    is_activation_post_process
)

from ..utils import (
    get_combined_dict,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
    weight_is_quantized,
    activation_is_statically_quantized,
    activation_is_int8_quantized,
    activation_dtype,
    weight_dtype,
)

from .pattern_utils import (
    is_match,
    get_default_quant_patterns,
    get_default_output_activation_post_process_map,
    Pattern,
)

from .graph_module import (
    is_observed_module,
    is_observed_standalone_module,
    ObservedGraphModule,
    ObservedStandaloneGraphModule,
    QuantizedGraphModule,
)

from .quantization_patterns import (
    binary_op_supported_dtypes,
    binary_reference_op_supported_dtypes,
    BinaryOpQuantizeHandler,
    CatQuantizeHandler,
    CopyNodeQuantizeHandler,
    CustomModuleQuantizeHandler,
    DefaultQuantizeHandler,
    QuantizeHandler,
    StandaloneModuleQuantizeHandler,
)

from .utils import (
    _parent_name,
    all_node_args_have_no_tensors,
    quantize_node,
    get_custom_module_class_keys,
    get_new_attr_name_with_prefix,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    assert_and_get_unique_device,
    node_return_type_is_int,
    node_bool_tensor_arg_indexes,
)

from .qconfig_utils import (
    convert_dict_to_ordered_dict,
    get_flattened_qconfig_dict,
    get_object_type_qconfig,
    get_qconfig,
    QConfigAny,
)

import operator

from collections import defaultdict

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Define helper types
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
                    QConfigAny]

# ------------------------
# Helper Functions
# ------------------------

def insert_observer(
        node: Node, observer: torch.quantization.ObserverBase,
        model: torch.nn.Module,
        activation_post_process_map: Dict[str, List[str]],
        activation_post_process_indexes: Dict[str, int],
        env: Dict[Any, Any], observed_graph: Graph, load_arg: Callable,
        observed_node_names_set: Set[str]):
    """Insert observer for node by modifying the observed_graph and
       attach observer module to the model
       Args:
         node: Node
         observer: observer/fake_quantize module instance
    """
    # In eval mode fixed qparams node are the same as CopyNode and we
    # won't insert observer for them
    if not model.training:
        assert not isinstance(observer, torch.quantization.FixedQParamsFakeQuantize), \
            "Unexpected FixedQParamsFakeQuantize found when model.training is False."
    # respect device affinity when adding observers
    model_device = assert_and_get_unique_device(model)
    if model_device:
        observer.to(model_device)
    # add observer module as attribute
    prefix = node.name + '_activation_post_process_'
    get_new_observer_name = get_new_attr_name_with_prefix(prefix)
    observer_name = get_new_observer_name(model)
    setattr(model, observer_name, observer)
    # put observer instance activation_post_process map
    activation_post_process_map[node.name].append(observer_name)
    # initialize index map for activation_post_process
    if node.name not in activation_post_process_indexes:
        activation_post_process_indexes[node.name] = 0
    # insert observer call
    env[node.name] = observed_graph.create_node(
        'call_module', observer_name, (load_arg(node),), {})
    observed_node_names_set.add(node.name)

def maybe_insert_observer_for_special_module(
        quantize_handler: QuantizeHandler, modules: Dict[str, torch.nn.Module],
        prepare_custom_config_dict: Any, qconfig: Any, node: Node) -> Optional[List[int]]:
    """ Insert observer for custom module and standalone module
      Returns: standalone_module_input_idxs: the indexs for inputs that
      needs to be observed by parent module
    """
    assert modules is not None
    standalone_module_input_idxs = None
    if isinstance(quantize_handler, CustomModuleQuantizeHandler):
        custom_module = modules[node.target]  # type: ignore[index]
        custom_module_class_mapping = prepare_custom_config_dict.get(
            "float_to_observed_custom_module_class", {})
        observed_custom_module_class = \
            get_swapped_custom_module_class(
                custom_module, custom_module_class_mapping, qconfig)
        observed_custom_module = \
            observed_custom_module_class.from_float(custom_module)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, observed_custom_module)
    elif isinstance(quantize_handler, StandaloneModuleQuantizeHandler):
        # observe standalone module
        standalone_module = modules[node.target]  # type: ignore[index]
        standalone_module_name_configs = prepare_custom_config_dict.get("standalone_module_name", [])
        standalone_module_class_configs = prepare_custom_config_dict.get("standalone_module_class", [])
        class_config_map = {x[0]: (x[1], x[2]) for x in standalone_module_class_configs}
        name_config_map = {x[0]: (x[1], x[2]) for x in standalone_module_name_configs}
        config = class_config_map.get(type(standalone_module), (None, None))
        config = name_config_map.get(node.target, config)
        sm_qconfig_dict = {"": qconfig} if config[0] is None else config[0]
        sm_prepare_config_dict = {} if config[1] is None else config[1]
        prepare = \
            torch.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore[attr-defined]
        observed_standalone_module = \
            prepare(standalone_module, sm_qconfig_dict, sm_prepare_config_dict)
        standalone_module_input_idxs = \
            observed_standalone_module._standalone_module_input_quantized_idxs.int().tolist()
        preserved_attributes = set(sm_prepare_config_dict.get("preserved_attributes", []))
        observed_standalone_module = ObservedStandaloneGraphModule(
            observed_standalone_module, observed_standalone_module.graph, preserved_attributes)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name,
                observed_standalone_module)
        modules[node.target] = observed_standalone_module  # type: ignore[index]
    return standalone_module_input_idxs

def maybe_insert_observer_for_output_of_the_node(
        node: Node,
        quantize_handler: QuantizeHandler,
        qconfig: Any,
        modules: Dict[str, torch.nn.Module],
        model: torch.nn.Module,
        pattern: Any,
        activation_post_process_map: Dict[str, List[str]],
        activation_post_process_indexes: Dict[str, int],
        env: Dict[Any, Any],
        observed_graph: Graph,
        load_arg: Callable,
        observed_node_names_set: Set[str],
        matched_nodes: Optional[List[Node]],
        standalone_module_input_idxs: Optional[List[int]],
        quants: Dict[str, List[Tuple[DefaultQuantizeHandler, Callable]]]):
    """ Insert observer/fake_quantize module for output of the observed
    module if needed
    """
    # don't need to insert observer for output if activation does not
    # need to be statically quantized
    assert modules is not None
    # TODO: Add warnings in the quantize handlers that does not support fp16 quantization
    inserted_observer = False
    if not activation_is_statically_quantized(qconfig):
        return

    if isinstance(quantize_handler,
                  StandaloneModuleQuantizeHandler):
        assert node.op == "call_module"
        assert isinstance(node.target, str)
        sm_out_qidxs = modules[node.target]._standalone_module_output_quantized_idxs.tolist()  # type: ignore[operator]
        output_is_quantized = 0 in sm_out_qidxs

        if output_is_quantized:
            observed_node_names_set.add(node.name)
    else:
        should_insert_observer = quantize_handler.should_insert_observer_for_output(
            qconfig, model.training)
        should_mark_output_observed_from_input_observed_status = \
            quantize_handler.should_mark_output_observed_from_input_observed_status(
                observed_node_names_set)

        activation_post_process_ctr = quantize_handler.get_activation_ctr(
            qconfig, pattern)
        assert activation_post_process_ctr is not None, \
            "activation_post_process constructor not provided " + \
            "for pattern:" + str(pattern)

        if should_insert_observer:
            insert_observer(
                node, activation_post_process_ctr(),
                model, activation_post_process_map,
                activation_post_process_indexes,
                env, observed_graph,
                load_arg, observed_node_names_set)
            inserted_observer = True
        elif should_mark_output_observed_from_input_observed_status:
            observed_node_names_set.add(node.name)
            inserted_observer = True

    # insert observer for input of standalone module
    if standalone_module_input_idxs is not None:
        for idx in standalone_module_input_idxs:
            if node.args[idx].name not in observed_node_names_set:  # type: ignore[union-attr]
                new_observer = qconfig.activation()
                insert_observer(
                    node, new_observer, model,
                    activation_post_process_map,
                    activation_post_process_indexes,
                    env, observed_graph,
                    load_arg, observed_node_names_set)
                inserted_observer = True

    # we already inserted activation_post_process for the outputvalue
    # which is the same as the input value of the next op, so we
    # can skip inserting one activation_post_process for the input
    if node.name in quants and inserted_observer:
        quants[node.name].pop(0)

def maybe_insert_observer_for_input_arg_of_observed_node(
        node: Node, observed_node_names_set: Set[str],
        quants: Dict[str, List[Tuple[DefaultQuantizeHandler, Callable]]],
        model: torch.nn.Module,
        activation_post_process_map: Dict[str, List[str]],
        activation_post_process_indexes: Dict[str, int],
        env: Dict[str, str], observed_graph: Graph,
        load_arg: Callable):
    if node.name in quants:
        quant_act_ctrs = quants[node.name][:]
        for _, activation_post_process_ctr in quant_act_ctrs:
            if activation_post_process_ctr is not None:
                insert_observer(
                    node, activation_post_process_ctr(),
                    model, activation_post_process_map,
                    activation_post_process_indexes,
                    env, observed_graph, load_arg, observed_node_names_set)

def maybe_insert_observer_for_output_of_model(
        node: Node,
        model: torch.nn.Module,
        qconfig_map: Dict[str, QConfigAny],
        activation_post_process_map: Dict[str, List[str]],
        activation_post_process_indexes: Dict[str, int],
        env: Dict[Any, Any], observed_graph: Graph, load_arg: Callable,
        observed_node_names_set: Set[str]):
    if isinstance(node, Node):
        assert qconfig_map is not None
        local_qconfig = qconfig_map[node.name]
        assert local_qconfig is not None, \
            'qconfig of a node before a quantized output must exist'
        if node.name not in observed_node_names_set:
            insert_observer(
                node, local_qconfig.activation(),
                model,
                activation_post_process_map,
                activation_post_process_indexes,
                env, observed_graph, load_arg, observed_node_names_set)
    elif isinstance(node, list) or isinstance(node, tuple):
        for n in node:
            maybe_insert_observer_for_output_of_model(
                n,
                model,
                qconfig_map,
                activation_post_process_map,
                activation_post_process_indexes,
                env, observed_graph, load_arg, observed_node_names_set)
    elif isinstance(node, dict):
        for n in node.values():
            maybe_insert_observer_for_output_of_model(
                n,
                model,
                qconfig_map,
                activation_post_process_map,
                activation_post_process_indexes,
                env, observed_graph, load_arg, observed_node_names_set)
    else:
        raise Exception("hardcoding output to be quantized not supported: " + str(type(node)))

def insert_observers_for_model(
        model: GraphModule,
        modules: Dict[str, torch.nn.Module],
        matches: Dict[str, MatchResult],
        quants: Dict[str, List[Tuple[DefaultQuantizeHandler, Callable]]],
        observed_node_names_set: Set[str],
        qconfig_map: Dict[str, QConfigAny],
        activation_post_process_map: Dict[str, List[str]],
        activation_post_process_indexes: Dict[str, int],
        observed_graph: Graph,
        prepare_custom_config_dict: Dict[str, Any],
        input_quantized_idxs: List[int],
        output_quantized_idxs: List[int]) -> Optional[Node]:
    env: Dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    graph_inputs = [node.name for node in model.graph.nodes if node.op == "placeholder"]
    get_new_observer_name = get_new_attr_name_with_prefix(
        'activation_post_process_')
    placeholder_node_seen_cnt = 0
    output_node_seen_cnt = 0
    result_node: Optional[Node] = None
    for node in model.graph.nodes:
        if node.op == 'output':
            # If this output is hardcoded to be quantized, insert an
            # observer on the previous node if it does not already
            # exist.
            cur_output_node_idx = output_node_seen_cnt
            output_node_seen_cnt += 1
            if cur_output_node_idx in output_quantized_idxs:
                prev_node = node.args[0]
                maybe_insert_observer_for_output_of_model(
                    prev_node,
                    model,
                    qconfig_map,
                    activation_post_process_map,
                    activation_post_process_indexes,
                    env, observed_graph, load_arg, observed_node_names_set)

            observed_graph.output(load_arg(node.args[0]))
            result_node = node
            continue

        if node.name in observed_node_names_set:
            continue

        root_node, matched_nodes, pattern, obj, qconfig = matches.get(
            node.name, (None, None, None, None, None))
        env[node.name] = observed_graph.node_copy(node, load_arg)
        if root_node is node:
            # index for input of custom module that needs to be observed in
            # parent
            if qconfig is not None:
                assert obj is not None
                standalone_module_input_idxs = \
                    maybe_insert_observer_for_special_module(
                        obj, modules, prepare_custom_config_dict, qconfig,
                        node)
                maybe_insert_observer_for_output_of_the_node(
                    node, obj, qconfig, modules, model, pattern,
                    activation_post_process_map,
                    activation_post_process_indexes,
                    env,
                    observed_graph, load_arg, observed_node_names_set,
                    matched_nodes, standalone_module_input_idxs, quants)

        if node.op == 'placeholder':
            # skip adding observers at the graph input if the input is
            # overridden to be quantized
            cur_placeholder_node_idx = placeholder_node_seen_cnt
            placeholder_node_seen_cnt += 1
            if cur_placeholder_node_idx in input_quantized_idxs:
                observed_node_names_set.add(node.name)
                continue

        maybe_insert_observer_for_input_arg_of_observed_node(
            node, observed_node_names_set, quants,
            model, activation_post_process_map,
            activation_post_process_indexes,
            env,
            observed_graph, load_arg)

    return result_node


def in_nodes(a: Argument, nodes: Set[Node]) -> bool:
    """ Checks if argument `a` is in the nodes set
    if it is a list, check if all elements of a is in the nodes set
    recursively
    """
    if isinstance(a, Node):
        return a in nodes
    elif isinstance(a, list) or isinstance(a, tuple):
        return all([in_nodes(arg, nodes) for arg in a])
    return False

def is_activation_post_process_node(node: Node, modules: Dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_module" and \
        is_activation_post_process(modules[str(node.target)])

def handle_copy_nodes(
        observed_graph: Graph, matches: Dict[str, MatchResult],
        qconfig_map: Dict[str, QConfigAny],
        activation_post_process_map: Dict[str, List[str]],
        modules: Dict[str, torch.nn.Module]):
    observed_nodes: Set[Node] = set()
    copy_nodes: Set[Node] = set()
    non_tensor_input_nodes: Set[Node] = set()
    unmatched_nodes: Set[Node] = set()
    actpp_to_remove: Set[Node] = set()
    env: Dict[Any, Any] = {}

    def load_arg(a: Argument) -> Argument:
        return map_arg(a, lambda node: env[node.name])

    result_graph = Graph()
    cache_for_no_tensor_check: Dict[Node, bool] = dict()
    for node in observed_graph.nodes:
        root_node, matched_nodes, pattern, quantize_handler, qconfig = matches.get(
            node.name, (None, None, None, None, None))

        if is_activation_post_process_node(node, modules):
            # rule 1: if the input of a copy node is observed, we won't need to
            # insert observer for the output of copy node
            if in_nodes(node.args[0], copy_nodes) and in_nodes(node.args[0], observed_nodes):
                # we'll remove the activation_post_process if the previous node is
                # an observed copy node
                actpp_to_remove.add(node)

            # rule 2: if the previous node is an op without tensor input, we can remove the observer
            if in_nodes(node.args[0], non_tensor_input_nodes):
                actpp_to_remove.add(node)
            observed_nodes.add(node)

        if root_node is node and qconfig is not None:
            # recording copy nodes in the graph
            if isinstance(quantize_handler, CopyNodeQuantizeHandler):
                copy_nodes.add(node)
                # rule 3: if previous node is observed, the copy node will be observed as well
                if in_nodes(node.args[0], observed_nodes):
                    prev_node = node.args[0]
                    if (
                        isinstance(prev_node, Node) and
                        prev_node.op == "call_module" and
                        is_activation_post_process(modules[prev_node.target])  # type: ignore[index]
                    ):
                        prev_prev_node = prev_node.args[0]
                        # rule 3.1: If previous node is unmatched, the input to copy node should not
                        # be observed. For example, in the pattern of
                        #
                        # user_node_unmatched -> obs -> copy_node_matched -> next_node
                        #
                        # we delete `obs`, because user_node_unmatched is not quantizeable,
                        # and the input to copy_node_matched does not need observation.
                        if in_nodes(prev_prev_node, unmatched_nodes):
                            actpp_to_remove.add(prev_node)
                            observed_nodes.remove(prev_node)
                        else:
                            observed_nodes.add(node)
                    else:
                        observed_nodes.add(node)
        elif root_node is None and node.op != 'placeholder':
            # rule 4: remove observer for getitem if it is followed by an unmatched node
            if len(node.args) > 0:
                maybe_observer_node = node.args[0]
                if isinstance(maybe_observer_node, Node) and len(maybe_observer_node.args) > 0:
                    observed_node = maybe_observer_node.args[0]
                    if is_activation_post_process_node(maybe_observer_node, modules):
                        assert isinstance(observed_node, Node)
                        if (observed_node.op, observed_node.target) == ("call_function", operator.getitem):
                            actpp_to_remove.add(maybe_observer_node)
            unmatched_nodes.add(node)

        if all_node_args_have_no_tensors(node, modules, cache_for_no_tensor_check):
            non_tensor_input_nodes.add(node)

        # rule 5: for special node, we'll just remove observer for its input
        special_nodes = [
            # Note: observer placement for getitem is handled in multiple places
            # since we may need to remove the observer for input and output of getitem
            # the handling can be simplified if we support querying type of Proxy
            ("call_function", operator.getitem),
        ]
        if (node.op, node.target) in special_nodes:
            if in_nodes(node.args[0], observed_nodes):
                prev_node = node.args[0].args[0]
                if prev_node.name not in qconfig_map or qconfig_map[prev_node.name] is None:
                    actpp_to_remove.add(node.args[0])
                    # if the previous node is not quantized, remove node from copy nodes
                    if node in copy_nodes:
                        copy_nodes.remove(node)

        # rule 6: remove observers for inputs of masked_fill since
        # it is boolean Tensor
        # TODO: we need type info from proxy, what type of Tensor this is
        # similar to TorchScript
        bool_tensor_indexes = node_bool_tensor_arg_indexes(node)
        if bool_tensor_indexes:
            for idx in bool_tensor_indexes:
                maybe_observer_node = node.args[idx]
                while isinstance(maybe_observer_node, Node) and is_activation_post_process_node(maybe_observer_node, modules):
                    actpp_to_remove.add(maybe_observer_node)
                    # trace back from the current observer node
                    n = maybe_observer_node.args[0]
                    if isinstance(n, Node) and len(n.args) > 0:
                        maybe_observer_node = n.args[0]
                    else:
                        break

    for node in observed_graph.nodes:
        if node.op == "output":
            result_graph.output(map_arg(node.args[0], load_arg))
        elif node in actpp_to_remove:
            env[node.name] = env[node.args[0].name]
        else:
            env[node.name] = result_graph.node_copy(node, load_arg)

    return result_graph

def handle_cat_nodes(
        model: torch.nn.Module, observed_graph: Graph, matches: Dict[str, MatchResult],
        activation_post_process_map: Dict[str, List[str]],
        modules: Dict[str, torch.nn.Module]):
    observed_nodes: Set[Node] = set()
    # activation_post_process for cat nodes
    actpp_for_cat_nodes: Dict[Node, torch.nn.Module] = dict()

    # we'll set the activation_post_process for all tensor inputs of cat and output
    # of cat to be the same (the activation_post_process for the first Tensor
    # input in the list, for example, if we have:
    # x = torch.cat([x1, x2, x3], ...)
    # we'll set the activation_post_process for x1, x2, x3 and x to be the
    # activation_post_proceess of x1:
    # x1 -> obs1 -> cat -> obs1 -> ...
    #               /
    #       x2 -> obs1
    #
    # note that activation_post_process here needs to work for different
    # Tensor dimensions, e.g. MinMaxObserver, HistogramObserver, per tensor FakeQuantize
    for node in observed_graph.nodes:
        root_node, matched_nodes, pattern, quantize_handler, qconfig = matches.get(
            node.name, (None, None, None, None, None))

        if node.op == "call_module" and is_activation_post_process(modules[node.target]):
            observed_nodes.add(node)
            if node.args[0] in actpp_for_cat_nodes:
                parent_name, name = _parent_name(node.target)
                actpp_for_cat = actpp_for_cat_nodes[node.args[0]]
                setattr(modules[parent_name], name, actpp_for_cat)

        if root_node is node and qconfig is not None:
            if isinstance(quantize_handler, CatQuantizeHandler):
                if in_nodes(node.args[0], observed_nodes):
                    # set the activation post process to be the same
                    # input 0 for cat node is a list
                    assert isinstance(node.args[0], list) or \
                        isinstance(node.args[0], tuple), \
                        "Expecting first input of cat to be a list or tuple"
                    first_act_post_process = modules[node.args[0][0].target]
                    for arg in node.args[0]:
                        assert arg.op == "call_module" and is_activation_post_process(modules[arg.target])
                        parent_name, name = _parent_name(arg.target)
                        setattr(modules[parent_name], name, first_act_post_process)
                    actpp_for_cat_nodes[node] = first_act_post_process


    return observed_graph

# A dictionary for querying the weight index for a given op
WEIGHT_INDEX_DICT = {
    torch.nn.functional.conv1d : [1],
    torch.nn.functional.conv2d : [1],
    torch.nn.functional.conv3d : [1],
    torch.nn.functional.linear : [1],
}

def node_arg_is_weight(node: Node, arg: Any) -> bool:
    if isinstance(node, Node) and node.op == 'call_function' and \
            node.target in WEIGHT_INDEX_DICT:
        for i, node_arg in enumerate(node.args):
            if arg is node_arg and i in \
                    WEIGHT_INDEX_DICT[node.target]:  # type: ignore[index]
                return True
    return False

CONV_OPS_WITH_BIAS = {
    torch.nn.functional.conv1d,
    torch.nn.functional.conv2d,
    torch.nn.functional.conv3d,
}
CONV_BIAS_ARG_INDEX = 2

def node_arg_is_bias(node: Node, arg: Any) -> bool:
    if isinstance(node, Node) and node.op == 'call_function':
        if node.target in CONV_OPS_WITH_BIAS:
            for i, node_arg in enumerate(node.args):
                if arg is node_arg and i == CONV_BIAS_ARG_INDEX:
                    return True
        elif node.target is torch.nn.functional.linear:
            for kwarg_name, kwarg_value in node.kwargs.items():
                if kwarg_name == 'bias' and arg is kwarg_value:
                    return True
    return False


# weight prepacking ops
WEIGHT_PREPACK_OPS = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
}

class Quantizer:
    def __init__(self):
        # mapping from matched node to full qualified path of activation_post_process
        # must be filled before convert
        self.activation_post_process_map: Dict[str, List[str]] = {}

        # mapping from matched node to the index of activation_post_process that we are
        # using currently
        self.activation_post_process_indexes: Dict[str, int] = {}

        # mapping from node name to qconfig that should be used for that node
        # filled out for a model during _generate_qconfig_map
        self.qconfig_map: Dict[str, QConfigAny] = {}
        # mapping from fully qualified module name to module instance
        # for example,
        # {
        #   '': Model(...),
        #   'linear': Linear(...),
        #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
        # }
        self.modules: Dict[str, torch.nn.Module] = {}
        # mapping from a tuple of nodes in reverse order to uninitialized
        #   QuantizeHandler subclass. For example,
        # {
        #   # match a single node
        #   (<class 'torch.nn.modules.conv.Conv3d'>:
        #     <class 'torch.quantization.fx.quantize.ConvRelu'>),
        #   # match multiple nodes in reverse order
        #   ((<function relu at 0x7f766a7360d0>, <built-in function add>):
        #     <class 'torch.quantization.fx.quantize.Add'>),
        # }
        self.patterns: Dict[Pattern, QuantizeHandler] = {}
        self.prepare_custom_config_dict: Dict[str, Any] = {}

        # mapping from node name to the scope of the module which contains the node.
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}


    def _qat_swap_modules(
            self, root: torch.nn.Module,
            additional_qat_module_mapping: Dict[Callable, Callable]) -> None:
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        convert(root, mapping=all_mappings, inplace=True, remove_qconfig=False)

    def _generate_qconfig_map(
            self,
            root: torch.nn.Module,
            input_graph: Graph,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]]) -> None:
        global_qconfig = qconfig_dict.get("", None)
        self.node_name_to_scope = node_name_to_scope
        self.qconfig_map = dict()
        for node in input_graph.nodes:
            if node.op == "get_attr":
                module_name, _ = _parent_name(node.target)
                self.qconfig_map[node.name] = get_qconfig(
                    qconfig_dict, type(self.modules[module_name]), module_name, global_qconfig)
            elif node.op == "call_function":
                # precedence: [TODO] module_name_qconfig (need scope support
                # from fx)
                # > function_qconfig > global_qconfig
                # module_name takes precedence over function qconfig
                function_qconfig = get_object_type_qconfig(
                    qconfig_dict, node.target, global_qconfig)
                module_path, module_type = node_name_to_scope[node.name]
                qconfig = get_qconfig(
                    qconfig_dict, module_type, module_path, function_qconfig)
                self.qconfig_map[node.name] = qconfig
            elif node.op == "call_method":
                module_path, module_type = node_name_to_scope[node.name]
                # use the qconfig of the module that the node belongs to
                qconfig = get_qconfig(
                    qconfig_dict, module_type, module_path, global_qconfig)
                self.qconfig_map[node.name] = qconfig
            elif node.op == 'call_module':
                module_qconfig = get_qconfig(
                    qconfig_dict, type(self.modules[node.target]), node.target, global_qconfig)
                # regex is not supported eager mode propagate_qconfig_, we'll
                # need to set the qconfig explicitly here in case regex
                # is used
                self.modules[node.target].qconfig = module_qconfig
                self.qconfig_map[node.name] = module_qconfig

    def _match(self,
               model, graph, standalone_module_names, standalone_module_classes,
               custom_module_classes) -> Tuple[
                   Dict[str, MatchResult],
                   Dict[str, List[Tuple[DefaultQuantizeHandler, Callable]]]]:
        matches = self._find_matches(
            graph, self.modules, self.patterns, standalone_module_names,
            standalone_module_classes, custom_module_classes)

        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an DefaultQuantizeHandler object for each
        quants = self._find_quants(graph, self.modules, matches)
        return matches, quants

    def _prepare(
            self,
            model: GraphModule,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]],
            prepare_custom_config_dict: Optional[Dict[str, Any]],
            is_standalone_module: bool) -> ObservedGraphModule:
        """ standalone_module means it a submodule that is not inlined in
        parent module, and will be quantized separately as one unit.

        How the standalone module is observed is specified by `input_quantized_idxs` and
        `output_quantized_idxs` in the prepare_custom_config for the standalone module
        Returns:
            model(GraphModule): prepared standalone module
            attributes:
                _standalone_module_input_quantized_idxs(List[Int]): a list of
                    indexes for the graph input that is expected to be quantized,
                    same as input_quantized_idxs configuration provided
                    for the standalone module
                _standalone_module_output_quantized_idxs(List[Int]): a list of
                    indexs for the graph output that is quantized
                    same as input_quantized_idxs configuration provided
                    for the standalone module
        """
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        self.prepare_custom_config_dict = prepare_custom_config_dict

        additional_quant_patterns = \
            prepare_custom_config_dict.get("additional_quant_pattern", {})
        self.patterns = get_combined_dict(
            get_default_quant_patterns(), additional_quant_patterns)

        convert_dict_to_ordered_dict(qconfig_dict)
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)
        # TODO: support regex as well
        propagate_qconfig_(model, flattened_qconfig_dict)
        if model.training:
            additional_qat_module_mapping = prepare_custom_config_dict.get(
                "additional_qat_module_mapping", {})
            self._qat_swap_modules(model, additional_qat_module_mapping)

        self.modules = dict(model.named_modules())

        # fill self.qconfig_map, a map from node name to qconfig, used in _find_matches
        self._generate_qconfig_map(model, model.graph, qconfig_dict, node_name_to_scope)

        # match the patterns that will get quantized
        standalone_module_name_configs = prepare_custom_config_dict.get(
            "standalone_module_name", [])
        standalone_module_class_configs = prepare_custom_config_dict.get(
            "standalone_module_class", [])

        standalone_module_names = [config[0] for config in standalone_module_name_configs]
        standalone_module_classes = [config[0] for config in standalone_module_class_configs]
        custom_module_classes = get_custom_module_class_keys(
            prepare_custom_config_dict, "float_to_observed_custom_module_class")
        matches, quants = self._match(
            model, model.graph, standalone_module_names, standalone_module_classes,
            custom_module_classes)
        self.activation_post_process_map = defaultdict(list)
        observed_graph = Graph()
        observed_node_names_set: Set[str] = set()

        input_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "input_quantized_idxs", [])
        output_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "output_quantized_idxs", [])

        result_node = insert_observers_for_model(
            model, self.modules, matches, quants, observed_node_names_set,
            self.qconfig_map, self.activation_post_process_map, self.activation_post_process_indexes,
            observed_graph, prepare_custom_config_dict, input_quantized_idxs, output_quantized_idxs)

        self.modules = dict(model.named_modules())
        matches, quants = self._match(
            model, observed_graph, standalone_module_names, standalone_module_classes,
            custom_module_classes)
        observed_graph = handle_copy_nodes(
            observed_graph, matches, self.qconfig_map,
            self.activation_post_process_map, self.modules)

        self.modules = dict(model.named_modules())
        matches, quants = self._match(
            model, observed_graph, standalone_module_names, standalone_module_classes,
            custom_module_classes)
        observed_graph = handle_cat_nodes(
            model, observed_graph, matches, self.activation_post_process_map,
            self.modules)

        self.save_state(model)
        preserved_attributes = set(prepare_custom_config_dict.get("preserved_attributes", []))
        model = ObservedGraphModule(model, observed_graph, preserved_attributes)
        if is_standalone_module:
            assert result_node is not None
            assert isinstance(result_node.args[0], Node), \
                "standalone module only supports returning simple value currently"\
                "(not tuple, dict etc.)"
            # indicator for whether output is observed or not.
            # This used for correctly quantize standalone modules
            output_is_observed = \
                result_node.args[0].name in observed_node_names_set
            # these inputs are observed in parent
            # converting List[int] to Tensor since module attribute is
            # Union[Tensor, Module]
            model._standalone_module_input_quantized_idxs = \
                torch.tensor(input_quantized_idxs)
            model._standalone_module_output_quantized_idxs = torch.tensor(output_quantized_idxs)
        return model

    def save_state(self, observed: GraphModule) -> None:
        observed._activation_post_process_map = \
            self.activation_post_process_map  # type: ignore[assignment]
        observed._activation_post_process_indexes = \
            self.activation_post_process_indexes  # type: ignore[assignment]
        observed._patterns = self.patterns  # type: ignore[assignment]
        observed._qconfig_map = self.qconfig_map  # type: ignore[assignment]
        observed._prepare_custom_config_dict = \
            self.prepare_custom_config_dict  # type: ignore[assignment]
        observed._node_name_to_scope = self.node_name_to_scope  # type: ignore[assignment]

    def restore_state(self, observed: GraphModule) -> None:
        assert is_observed_module(observed), \
            'incoming model must be produced by prepare_fx'
        self.activation_post_process_map = \
            observed._activation_post_process_map  # type: ignore[assignment]
        self.activation_post_process_indexes = \
            observed._activation_post_process_indexes  # type: ignore[assignment]
        self.patterns = observed._patterns  # type: ignore[assignment]
        self.qconfig_map = observed._qconfig_map  # type: ignore[assignment]
        self.prepare_custom_config_dict = \
            observed._prepare_custom_config_dict  # type: ignore[assignment]
        self.node_name_to_scope = observed._node_name_to_scope  # type: ignore[assignment]

    def prepare(
            self,
            model: GraphModule,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]],
            prepare_custom_config_dict: Dict[str, Any] = None,
            is_standalone_module: bool = False) -> ObservedGraphModule:
        return self._prepare(
            model, qconfig_dict, node_name_to_scope, prepare_custom_config_dict,
            is_standalone_module)

    def _run_weight_observers(self, observed: GraphModule) -> None:
        r''' Extract the subgraph that produces the weight for dynamic quant
        or weight only quant node and run the subgraph to observe the weight.
        Note that the observers of dynamic quant or weight only quant ops are
        run during the convert step.
        '''
        for node in observed.graph.nodes:
            if node.op == 'call_function' and node.target in WEIGHT_INDEX_DICT:
                for i, node_arg in enumerate(node.args):
                    if i in WEIGHT_INDEX_DICT[node.target]:
                        # node_arg is weight
                        weight_observer_nodes = collect_producer_nodes(node_arg)
                        if weight_observer_nodes is not None:
                            weight_observer_module = \
                                graph_module_from_producer_nodes(
                                    observed, weight_observer_nodes)
                            # run the weight observer
                            weight_observer_module()
        return

    def _convert(self, model: GraphModule, is_reference: bool = False,
                 convert_custom_config_dict: Dict[str, Any] = None,
                 is_standalone_module: bool = False,
                 _remove_qconfig_flag: bool = True) -> QuantizedGraphModule:
        """ standalone_module means it a submodule that is not inlined in
        parent module, and will be quantized separately as one unit.

        Returns a quantized standalone module, whether input/output is quantized is
        specified by prepare_custom_config_dict, with
        input_quantized_idxs, output_quantized_idxs, please
        see docs for prepare_fx for details
        """
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        self.restore_state(model)
        # always run weight observers in the top level forward method
        # for dynamic quant ops or weight only quant ops
        self._run_weight_observers(model)

        # move to cpu since we only have quantized cpu kernels
        model.eval().cpu()
        self.modules = dict(model.named_modules(remove_duplicate=False))

        custom_module_classes = get_custom_module_class_keys(
            convert_custom_config_dict,
            "observed_to_quantized_custom_module_class")
        matches = self._find_matches(
            model.graph, self.modules, self.patterns,
            custom_module_classes=custom_module_classes)

        self.quantized_graph = Graph()
        env: Dict[str, Node] = {}
        # TODO: merge quant_env with env
        quant_env: Dict[str, Tuple[Node, torch.dtype]] = {}

        graph_inputs: List[str] = []
        for node in model.graph.nodes:
            if node.op == 'placeholder':
                graph_inputs.append(node.name)

        def load_non_quantized(n: Node) -> Node:
            if n.name not in env:
                assert n.name in quant_env, \
                    'trying to load float node but did not find ' + \
                    'node:' + n.name + \
                    ' in quantized or non quantized environment, env: ' + \
                    str(env) + ' quant_env:' + str(quant_env)
                quantized_node, _ = quant_env[n.name]
                env[n.name] = Proxy(quantized_node).dequantize().node
            return env[n.name]

        def load_quantized(n: Node) -> Node:
            assert n.name in quant_env, \
                'trying to load quantized node but did not find node:' + \
                n.name + ' in quant environment:' + str(quant_env)
            return quant_env[n.name][0]

        def load_x(n: Node) -> Node:
            assert n.name in env or n.name in quant_env, \
                'node ' + n.name + ' does not exist in either environment'
            if n.name in quant_env:
                return quant_env[n.name][0]
            else:
                return env[n.name]

        def load_arg(quantized: Optional[Union[List[int], bool, Tuple[int, ...]]]
                     ) -> Callable[[Node], Argument]:
            """
            Input: quantized, which can be None, list, boolean or tuple
              - if quantized is None, then we'll load the node as long as it
                exists
              - if quantized is a boolean, then all args will be
                quantized/not quantized
              - if quantized is an empty list or tuple, then it is the same as load_arg(quantized=False)
              - if quantized is a list or tuple, then arg should be a list and
                the args with corresponding indexes will be quantized


            Output: fn which takes arg_or_args, and loads them from the
                corresponding environment depending on the value of quantized.
            """
            assert quantized is None or \
                isinstance(quantized, (tuple, list, bool)), type(quantized)
            if isinstance(quantized, (tuple, list)) and len(quantized) == 0:
                # empty tuple or list means nothing is quantized
                quantized = False

            def load_arg_impl(arg_or_args):
                # we'll update the format of `quantized`
                # to better match arg_or_args
                updated_quantized: Optional[Union[List[int], bool, Tuple[int, ...]]] = quantized

                if isinstance(quantized, (tuple, list)) and \
                   len(quantized) == 1 and isinstance(arg_or_args, Node):
                    # when argument is one Node instead of tuple, we just need to check
                    # 0 is in the quantized list
                    updated_quantized = 0 in quantized

                if updated_quantized is None:
                    return map_arg(arg_or_args, load_x)
                if isinstance(updated_quantized, bool):
                    return map_arg(
                        arg_or_args,
                        load_quantized if updated_quantized else load_non_quantized)
                elif isinstance(updated_quantized, (tuple, list)):
                    assert isinstance(arg_or_args, (tuple, list)), arg_or_args
                    loaded_args = []
                    # for now, we only support quantizing positional arguments
                    for i, a in enumerate(arg_or_args):
                        if i in updated_quantized:
                            loaded_args.append(map_arg(a, load_quantized))
                        else:
                            loaded_args.append(map_arg(a, load_non_quantized))
                    return type(arg_or_args)(loaded_args)
            return load_arg_impl

        def node_arg_is_quantized(node_arg: Any) -> bool:
            if isinstance(node_arg, Node):
                assert node_arg.name in env or node_arg.name in quant_env, \
                    'Expecting node_arg to be in the environment'
                # there might be nodes appearing in both environemnts, but
                # quant_env will take precedence
                if node_arg.name in quant_env:
                    return True
                elif node_arg.name in env:
                    return False
                else:
                    return False
            elif isinstance(node_arg, list):
                quantized = map(node_arg_is_quantized, node_arg)
                if all(quantized):
                    return True
                elif not any(quantized):
                    return False
                else:
                    raise Exception(
                        "partially quantized inputs in list not handled yet")
            else:
                return False

        def is_output_quantized(node: Node, obj: QuantizeHandler) -> bool:
            """ Check if output node is quantized or not """
            assert self.modules is not None
            # by default the output for a quantizable node is expected to be quantized
            quantized = True

            # Need to get correct quantized/non-quantized state forn the output
            # of FixedQParamsQuantizeHandler
            # TODO: we may want to try to remove the special case here
            # as well
            if obj.should_mark_output_quantized_from_input_quantized_status():
                assert node.op in [
                    'call_module',
                    'call_function',
                    'call_method'], \
                    'FixedQParamsQuantizeHandler of type ' + node.op + ' is not handled'
                # TODO: need to extend this to consider all relevant args instead of just arg[0]
                quantized = node_arg_is_quantized(node.args[0])

            # the output is unquantized if the node is not a CopyNode
            # and activation is fp16 (since we will output fp32 currently for fp16
            # converter
            if not activation_is_int8_quantized(qconfig) or \
               not obj.input_output_observed():
                quantized = False
            if node_return_type_is_int(node):
                quantized = False

            return quantized

        def insert_quantize_node(node: Node) -> None:
            """ Given a activation_post_process module call node, insert a
            quantize node"""
            assert self.modules is not None
            assert isinstance(node.target, str)
            observer_module = self.modules[node.target]
            prev_node = node.args[0]
            if observer_module.dtype == torch.float32:
                # copy the observer for fp32 dtype
                env[node.name] = self.quantized_graph.node_copy(
                    node, load_non_quantized)
            elif isinstance(prev_node, Node) and prev_node.name in quant_env:
                # if previous node is already quantized, we'll just remove the
                # activation_post_process
                _, prev_dtype = quant_env[prev_node.name]
                current_dtype = observer_module.dtype
                if prev_dtype == current_dtype:
                    quant_env[node.name] = quant_env[prev_node.name]
                else:
                    root_module = self.modules[""]
                    assert isinstance(prev_node, Node)
                    observer_dtype: torch.dtype = observer_module.dtype  # type: ignore[assignment]
                    quant_env[node.name] = (
                        quantize_node(self, load_non_quantized(prev_node),
                                      observer_module, node, is_input=True),
                        observer_dtype)
            else:
                # replace activation post process with quantization ops
                root_module = self.modules[""]
                assert isinstance(node.args[0], Node)
                dtype: torch.dtype = observer_module.dtype  # type: ignore[assignment]
                quant_env[node.name] = (
                    quantize_node(self, load_non_quantized(node.args[0]),
                                  observer_module, node, is_input=True),
                    dtype)

        # additional state to override inputs to be quantized, if specified
        # by the user
        placeholder_node_seen_cnt = 0
        output_node_seen_cnt = 0
        input_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "input_quantized_idxs", [])
        output_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "output_quantized_idxs", [])

        for node in model.graph.nodes:
            if node.op == "output":
                cur_output_node_idx = output_node_seen_cnt
                output_node_seen_cnt += 1
                if cur_output_node_idx in output_quantized_idxs:
                    # Result are kept quantized if the user specified the
                    # output_quantized_idxs override.
                    graph_output = map_arg(node.args[0], load_x)
                else:
                    graph_output = map_arg(node.args[0], load_non_quantized)
                self.quantized_graph.output(graph_output)
                continue
            root_node, matched, matched_pattern, obj, qconfig = \
                matches.get(node.name, (None, None, None, None, None))
            if root_node is node:
                is_observed_standalone_module_node = (
                    node.op == 'call_module' and
                    is_observed_standalone_module(
                        self.modules[node.target])
                )
                if qconfig is None and not is_observed_standalone_module_node:
                    result = self.quantized_graph.node_copy(
                        node, load_non_quantized)
                    quantized = False
                else:
                    assert obj is not None
                    # We will get whether the output is quantized or not before
                    # convert for standalone module and after convert
                    # for non-standalone module, since _standalone_module_output_quantized_idxs
                    # is only available in observed standalone module
                    if is_observed_standalone_module_node:
                        out_quant_idxs = self.modules[node.target]._standalone_module_output_quantized_idxs.tolist()  # type: ignore[operator] # noqa: B950
                        assert len(out_quant_idxs) <= 1, "Currently standalone only support one output"
                        quantized = 0 in out_quant_idxs

                    result = obj.convert(
                        self, node, load_arg, is_reference=is_reference,
                        convert_custom_config_dict=convert_custom_config_dict)
                    if not is_observed_standalone_module_node:
                        quantized = is_output_quantized(node, obj)

                if quantized:
                    quant_env[node.name] = result, activation_dtype(qconfig)
                else:
                    env[node.name] = result
                continue
            elif root_node is not None:
                if qconfig is None:
                    # This branch is hit if all of these conditions are met:
                    # 1. we are in a fusion pattern of multiple nodes (i.e. add-relu)
                    # 2. the current node is not the "root_node" of the pattern
                    # 3. quantization for this pattern is disabled
                    #
                    # In this case, we need to make sure to populate the env with
                    # intermediate nodes manually, because the QuantizeHandler.convert
                    # function will not be called.
                    result = self.quantized_graph.node_copy(
                        node, load_non_quantized)
                    env[node.name] = result
                continue

            # handle activation post process calls
            if node.op == 'call_module' and \
                    is_activation_post_process(self.modules[node.target]):
                insert_quantize_node(node)
            elif node.op == 'placeholder':
                cur_placeholder_node_idx = placeholder_node_seen_cnt
                placeholder_node_seen_cnt += 1
                if cur_placeholder_node_idx in input_quantized_idxs:
                    quant_env[node.name] = \
                        self.quantized_graph.node_copy(node, load_non_quantized), activation_dtype(qconfig) if qconfig else None
                else:
                    env[node.name] = \
                        self.quantized_graph.node_copy(node, load_non_quantized)
            else:
                # copy quantized or non-quantized node
                env[node.name] = \
                    self.quantized_graph.node_copy(node, load_non_quantized)

        # remove activation post process
        act_post_process_removed_graph = Graph()
        env = {}

        def load_arg_simple(a: Argument) -> Argument:
            return map_arg(a, lambda node: env[node.name])
        for node in self.quantized_graph.nodes:
            if node.op == 'output':
                act_post_process_removed_graph.output(
                    map_arg(node.args[0], load_arg_simple))
                continue
            if node.op == 'call_module' and \
               is_activation_post_process(self.modules[node.target]):
                # remove activation post process node
                env[node.name] = env[node.args[0].name]
            else:
                env[node.name] = act_post_process_removed_graph.node_copy(
                    node, load_arg_simple)

        # removes qconfig and activation_post_process modules
        if _remove_qconfig_flag:
            _remove_qconfig(model)
        preserved_attributes = set(convert_custom_config_dict.get("preserved_attributes", []))
        model = QuantizedGraphModule(model, act_post_process_removed_graph, preserved_attributes)
        return model

    # Trace back from the weight node util we hit getattr, reconstruct the
    # graph module with the traced nodes and run the graph module to pack the
    # weight. then replace the original chain of ops with the packed weight.
    def _fold_weight(self, quantized: QuantizedGraphModule) -> QuantizedGraphModule:
        packed_weights = dict()
        # map from folded node name to the prepacked weight name
        folded_nodes = dict()
        # get packed weights
        for node in quantized.graph.nodes:
            if node.op == 'call_function' and node.target in WEIGHT_PREPACK_OPS:
                nodes_to_fold = collect_producer_nodes(node)
                if nodes_to_fold is not None:
                    for node_to_fold in nodes_to_fold:
                        folded_nodes[node_to_fold.name] = node

                    prepacking_module = graph_module_from_producer_nodes(
                        quantized, nodes_to_fold)
                    packed_weight = prepacking_module()
                    packed_weights[node.name] = packed_weight

        # remove folded nodes and replace the prepacking node with getattr
        folded_graph = Graph()
        env: Dict[Any, Any] = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])
        quantized_root = quantized
        quantized_graph = quantized.graph

        for node in quantized_graph.nodes:
            prepack_node = folded_nodes.get(node.name, None)
            if prepack_node is node:
                packed_weight = packed_weights[node.name]
                # add a prepacked attribute to root
                op_node = list(prepack_node.users)[0]
                module_path, _ = self.node_name_to_scope[op_node.name]
                get_new_packed_weight_name = \
                    get_new_attr_name_with_prefix(module_path + '_packed_weight_')
                packed_weight_name = get_new_packed_weight_name(quantized_root)
                setattr(quantized_root, packed_weight_name, packed_weight)
                # replace prepack node with a getattr node
                env[node.name] = folded_graph.create_node(
                    'get_attr', packed_weight_name, (), {})
            elif prepack_node is not None:
                # remove the foled node
                continue
            else:
                # copy other nodes
                env[node.name] = folded_graph.node_copy(node, load_arg)
        quantized = QuantizedGraphModule(quantized_root, folded_graph, quantized_root.preserved_attr_names)
        return quantized

    def _fold_quant_dequant(self, quantized: QuantizedGraphModule) -> QuantizedGraphModule:
        """ If quantize op is followed by a dequantize, we fold the ops together and remove the dequant.
            In the case where the only consumer of quantize_per_tensor is a dequant op, we erase both
            nodes from the graph, along with the qparams associated with quantize op.
        """

        for node in quantized.graph.nodes:
            if node.op == 'call_function' and node.target == torch.quantize_per_tensor:
                quant_uses = list(node.users)
                quant_args = node.args
                float_tensor = quant_args[0]
                for user in quant_uses:
                    is_dequant = user.op == 'call_method' and user.target == "dequantize"
                    if is_dequant:
                        user.replace_all_uses_with(float_tensor)
                        quantized.graph.erase_node(user)
                        # If dequant is the only user of quant node, we erase quant node
                        # and all it's inputs.
                        if len(quant_uses) == 1:
                            quantized.graph.erase_node(node)
                            for arg in quant_args[1:]:
                                if isinstance(arg, Node):
                                    quantized.graph.erase_node(arg)
        return quantized

    def convert(self, model: GraphModule, is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False,
                _remove_qconfig: bool = True) -> QuantizedGraphModule:
        quantized = self._convert(
            model, is_reference, convert_custom_config_dict, is_standalone_module, _remove_qconfig_flag=_remove_qconfig)
        if not is_reference:
            quantized = self._fold_weight(quantized)
            quantized = self._fold_quant_dequant(quantized)
        return quantized

    def _find_matches(
            self, graph: Graph, modules: Dict[str, torch.nn.Module],
            patterns: Dict[Pattern, QuantizeHandler],
            standalone_module_names: List[str] = None,
            standalone_module_classes: List[Callable] = None,
            custom_module_classes: List[Any] = None) -> Dict[str, MatchResult]:
        """
        Matches the nodes in the input graph to quantization patterns, and
        outputs the information needed to quantize them in future steps.

        Inputs:
          - graph: an fx.Graph object
          - modules: a mapping of fully qualified module name to instance,
              for example, {'foo': ModuleFoo, ...}
          - patterns: a mapping from a tuple of nodes in reverse order to
              uninitialized QuantizeHandler subclass.

        Outputs a map of
          node_name ->
            (node, matched_values, matched_pattern, QuantizeHandler instance,
             qconfig)

        For example, {
          'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                     <CopyNodeQuantizeHandler instance>, QConfig(...)),
          ...
        }
        """
        if custom_module_classes is None:
            custom_module_classes = []

        if standalone_module_classes is None:
            standalone_module_classes = []

        if standalone_module_names is None:
            standalone_module_names = []

        match_map: Dict[str, MatchResult] = {}
        all_matched : Set[str] = set()

        def record_match(pattern, node, matched):
            if isinstance(pattern, tuple):
                s, *args = pattern
                record_match(s, node, matched)
                if pattern[0] is not getattr:
                    for subpattern, arg in zip(args, node.args):
                        record_match(subpattern, arg, matched)
            else:
                matched.append(node)

        cache_for_no_tensor_check: Dict[Node, bool] = dict()
        for node in reversed(graph.nodes):
            if node.name not in match_map and node.name not in all_matched:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        skip_this_match = False
                        if value is BinaryOpQuantizeHandler:

                            # to properly check for dtype support, we need to
                            # navigate to the base node of an add-relu or mul-relu
                            # pattern
                            base_node = node
                            if (
                                (node.op == 'call_function' and
                                 node.target is torch.nn.functional.relu) or
                                (node.op == 'call_module' and
                                 isinstance(modules[node.target], torch.nn.ReLU))
                            ):
                                base_node = node.args[0]

                            this_node_qconfig = \
                                self.qconfig_map[base_node.name]
                            if this_node_qconfig:
                                dtypes = get_qconfig_dtypes(this_node_qconfig)
                                # TODO(future PR): update the pattern to quantize
                                # handler logic to take this into account.


                                # This needs to handle 3 cases
                                # 1) op and dtype is in either [is_ref or non-ref] list -> don't skip
                                # 2) op is not in either list (i.e. relu) -> don't skip
                                # 3) op is in non-ref list, but not for dtype, and op+dtype not in is_ref list -> skip

                                # note: the value of is_reference is unknown at prepare, so we have to cover both cases
                                # handle is_reference = False
                                skip_match_not_is_reference = (
                                    (base_node.target in binary_op_supported_dtypes) and
                                    (dtypes not in binary_op_supported_dtypes[base_node.target])
                                )

                                # handle is_reference = True
                                supported_is_reference = (
                                    (base_node.target in binary_reference_op_supported_dtypes) and
                                    (dtypes in binary_reference_op_supported_dtypes[base_node.target])
                                )

                                # only skip if not reference says skip and is_reference doesn't support
                                skip_this_match = skip_match_not_is_reference and not supported_is_reference

                        if not skip_this_match:
                            matched: List[Any] = []
                            record_match(pattern, node, matched)
                            for n in matched:
                                match_map[n.name] = (
                                    node, matched, pattern, value(self, node),  # type: ignore[operator]
                                    self.qconfig_map[n.name])
                                all_matched.add(n.name)
                            # break after finding the first match
                            break

        # add custom module instances to the match result
        assert self.modules is not None
        for node in graph.nodes:
            if node.op == 'call_module' and \
               type(self.modules[node.target]) in custom_module_classes:
                custom_module_qconfig = self.qconfig_map[node.name]
                match_map[node.name] = (
                    node, [node], None, CustomModuleQuantizeHandler(self, node),
                    custom_module_qconfig)

        def is_standalone_module(node_target):
            assert self.modules is not None
            return (
                node_target in standalone_module_names or  # type: ignore[operator]
                type(self.modules[node_target]) in standalone_module_classes  # type: ignore[operator]
            )

        # add standalone modules to the match
        for node in graph.nodes:
            if node.op == 'call_module' and \
               (is_standalone_module(node.target) or
                    is_observed_standalone_module(self.modules[node.target])):
                # add node to matched nodes
                custom_module_qconfig = self.qconfig_map[node.name]
                match_map[node.name] = (
                    node, [node], None,
                    StandaloneModuleQuantizeHandler(self, node),
                    custom_module_qconfig)

        return match_map

    def _find_quants(
            self, graph: Graph, modules: Dict[str, torch.nn.Module],
            matches: Dict[str, MatchResult]) -> Dict[str, List[Tuple[DefaultQuantizeHandler, Callable]]]:
        """
        Takes the nodes in the input graph and pending matches, and finds and
        returns the input and output nodes which need to be quantized.

        Inputs:
          - graph: an fx.Graph object
          - modules: a dictionary from module path to module
          - matches: output of self._find_matches function

        Outputs a map of
         node_name -> list of (QuantizeHandler instance (always DefaultQuantizeHandler),
         activation_post_process (observer/fake_quantize module) constructor)
         the reason why the value is a list is because each node can be configured with multiple
         qconfigs, for example in a subgraph of functional linear
         op followed by a sigmoid op, linear is configured with int8 static quantization
         and sigmoid is configured with float16 static quantization,
         then the output of linear (and input of sigmoid) needs first to be quantized to
         int8 and then float16
        """
        quants: Dict[str, List[Tuple[DefaultQuantizeHandler, Callable]]] = defaultdict(list)
        cache_for_no_tensor_check: Dict[Node, bool] = dict()

        def visit(node, matched_pattern, qconfig):
            def visit_arg(arg):
                is_weight = node_arg_is_weight(node, arg)
                is_bias = node_arg_is_bias(node, arg)
                is_activation = not (is_weight or is_bias)
                no_tensors = all_node_args_have_no_tensors(arg, modules, cache_for_no_tensor_check)
                # bias needs to be quantized if activation is fp16 and weight is fp16
                # this is the case for glow
                should_add_handler = qconfig is not None and (
                    (is_activation and
                     activation_is_statically_quantized(qconfig)) or
                    (is_weight and weight_is_quantized(qconfig)) or
                    (is_bias and activation_dtype(qconfig) == torch.float16)
                    and weight_dtype(qconfig) == torch.float16) and \
                    (not no_tensors)

                if should_add_handler:
                    act_post_process_ctr = qconfig.weight if is_weight else \
                        qconfig.activation
                    # overwrite the constructor from qconfig if it is int8 quantized
                    if activation_is_int8_quantized(qconfig):
                        act_post_process_ctr = \
                            get_default_output_activation_post_process_map().get(
                                matched_pattern,
                                act_post_process_ctr)
                    if len(quants[arg.name]) > 0:
                        _, last_act_post_process_ctr = quants[arg.name][-1]
                        if act_post_process_ctr == last_act_post_process_ctr:
                            # we won't add act_post_process_ctr if it is the same as the
                            # one
                            return visit_arg
                    quants[arg.name].append((
                        DefaultQuantizeHandler(self, arg), act_post_process_ctr))
            return visit_arg

        for node in graph.nodes:
            if node.name in matches:
                root_node, matched_nodes, matched_pattern, quantize_handler, \
                    qconfig = matches[node.name]
                if root_node is node and \
                        quantize_handler.input_output_observed():
                    # matched_nodes[-1] is the first op in the sequence and
                    # matched_nodes[0] is the last op in the sequence
                    # inputs
                    # matched_pattern is set to None for inputs because
                    # we only want to select QuantizeHandler object based
                    # on pattern for output, inputs will always use
                    # DefaultQuantizeHandler
                    map_arg(matched_nodes[-1].args, visit(matched_nodes[-1],
                            None, qconfig))
                    map_arg(matched_nodes[-1].kwargs, visit(matched_nodes[-1],
                            None, qconfig))

                    # output
                    # we don't insert observer for output of standalone module
                    if not isinstance(
                            quantize_handler, StandaloneModuleQuantizeHandler):
                        # passing in matched_pattern here so that we can
                        # customize activation_post_process constructor for
                        # output based on the pattern, e.g.
                        # for sigmoid op we'll use
                        # default_affine_fixed_qparam_fake_quant
                        map_arg(matched_nodes[0],
                                visit(None, matched_pattern, qconfig))
        return quants
