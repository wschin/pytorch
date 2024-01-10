import copy

import torch
import torch.utils._pytree as pytree
from torch._export.utils import _check_input_constraints_pre_hook
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from .exported_program import ExportedProgram


def _unlift(
    gm,
    inp_pos_to_param_buffer_name,
    in_spec,
    out_spec,
    state_dict,
    tensor_constants,
    buffers_to_mutate,
):
    count = 0
    buffer_name_to_node = {}
    # Step 1: make lifted params as get_attr
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if count in inp_pos_to_param_buffer_name:
                with gm.graph.inserting_after(node):
                    getattr_node = gm.graph.get_attr(
                        inp_pos_to_param_buffer_name[count]
                    )
                    node.replace_all_uses_with(getattr_node)
                    metadata = node.meta
                    gm.graph.erase_node(node)
                    getattr_node.meta = metadata
                    buffer_name_to_node[
                        inp_pos_to_param_buffer_name[count]
                    ] = getattr_node

            count += 1
        # Step 2: Find the all the buffers that were mutated and update them
        if node.op == "output":
            user_output_nodes = []
            # In the case that the same node is returned multiple times,
            # node.all_input_nodes will only iterate that node once
            for return_node in pytree.tree_flatten(node.args)[0]:
                return_node_name = return_node.name
                # we found a param/buffer mutation
                if return_node_name in buffers_to_mutate:
                    # TODO Fix situation here to replace dot with underscore...
                    buffer_node_name = buffers_to_mutate[return_node_name].replace(
                        ".", "_"
                    )
                    assert buffer_node_name in buffer_name_to_node
                    buffer_node = buffer_name_to_node[buffer_node_name]
                    with gm.graph.inserting_before(node):
                        buffer_update_node = gm.graph.call_function(
                            torch.ops.aten.copy_.default, (buffer_node, return_node)
                        )
                else:
                    user_output_nodes.append(return_node)
            with gm.graph.inserting_before(node):
                # Only return user outputs
                new_output = gm.graph.output(tuple(user_output_nodes))
                node.replace_all_uses_with(new_output)
                gm.graph.erase_node(node)

    # Step 3: Fix the input/output of the graph now that we deleted
    # some args.
    gm.graph.lint()

    if (
        in_spec.type == tuple
        and in_spec.num_children == 2
        and in_spec.children_specs[0].type == tuple
        and in_spec.children_specs[1].type == dict
    ):
        # if in_spec contains the args (tuple) and kwargs (dict)
        num_args = (
            in_spec.children_specs[0].num_children
            + in_spec.children_specs[1].num_children
        )
    else:
        num_args = in_spec.num_children

    names = [f"arg_{i}" for i in range(num_args)]

    gm.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )
    gm.recompile()

    # Step 4: Find state references in HigherOrderOps and recursively
    # fix them.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.cond:
            pred, true_graph, false_graph, operands = node.args
            true_gm = getattr(gm, true_graph.name)
            false_gm = getattr(gm, false_graph.name)
            inp_pos_to_param_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_param_buffer_name_for_submod[ix] = operand.target
                    if operand.target in state_dict:
                        value = state_dict[operand.target]
                    elif operand.target in tensor_constants:
                        value = tensor_constants[operand.target]
                    else:
                        raise RuntimeError("Unable to find value for ", operand.target)
                    true_gm.register_buffer(operand.target, value)
                    false_gm.register_buffer(operand.target, value)
                else:
                    real_operands.append(operand)
            node.args = (pred, true_graph, false_graph, real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(
                true_gm,
                inp_pos_to_param_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
                tensor_constants,
                buffers_to_mutate,
            )
            _unlift(
                false_gm,
                inp_pos_to_param_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
                tensor_constants,
                buffers_to_mutate,
            )
        if node.op == "call_function" and node.target.__name__ == "map_impl":
            body_graph, num_mapped, *operands = node.args
            body_gm = getattr(gm, body_graph.name)
            inp_pos_to_buffer_name_for_submod = {}
            real_operands = []
            # TODO Fix situation here to replace dot with underscore...
            state_dict_for_lookup = {
                key.replace(".", "_"): value for key, value in state_dict.items()
            }
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_buffer_name_for_submod[ix] = operand.target
                    if operand.target in state_dict_for_lookup:
                        value = state_dict_for_lookup[operand.target]
                    elif operand.target in tensor_constants:
                        value = tensor_constants[operand.target]
                    else:
                        raise RuntimeError(f"Unable to find value for {operand.target}")
                    body_gm.register_buffer(operand.target, value)
                else:
                    real_operands.append(operand)
            node.args = (body_graph, num_mapped, *real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(
                body_gm,
                inp_pos_to_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
                tensor_constants,
                buffers_to_mutate,
            )
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _construct_inp_pos_to_param_buffer_name(
    new_gm, graph_signature, state_dict, tensor_constants=None
):
    # TODO Fix the period in params/buffers names later
    # maybe a pass to replace graph signature with fixed names
    param_buffer_name_to_corrected_name = {}
    constant_name_to_corrected_name = {}

    for name, value in state_dict.items():
        if name in graph_signature.buffers:
            if "." in name:
                new_gm.register_buffer(name.replace(".", "_"), value)
                param_buffer_name_to_corrected_name[name] = name.replace(".", "_")
            else:
                new_gm.register_buffer(name, value)
        if name in graph_signature.parameters:
            if "." in name:
                new_gm.register_parameter(name.replace(".", "_"), value)
                param_buffer_name_to_corrected_name[name] = name.replace(".", "_")
            else:
                new_gm.register_parameter(name, value)

    if tensor_constants is not None and len(tensor_constants) > 0:
        assert hasattr(graph_signature, "lifted_tensor_constants")
        for name, value in tensor_constants.items():
            if name in graph_signature.lifted_tensor_constants:
                if isinstance(value, torch.Tensor):
                    new_gm.register_buffer(name.replace(".", "_"), value)
                else:
                    setattr(new_gm, name.replace(".", "_"), value)
                constant_name_to_corrected_name[name] = name.replace(".", "_")

    count = 0
    inp_pos_to_param_buffer_name = {}
    for node in new_gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in graph_signature.inputs_to_buffers:
                buffer_name = graph_signature.inputs_to_buffers[node.name]
                if buffer_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[
                        count
                    ] = param_buffer_name_to_corrected_name[buffer_name]
                else:
                    inp_pos_to_param_buffer_name[count] = buffer_name
            if node.name in graph_signature.inputs_to_parameters:
                param_name = graph_signature.inputs_to_parameters[node.name]
                if param_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[
                        count
                    ] = param_buffer_name_to_corrected_name[param_name]
                else:
                    inp_pos_to_param_buffer_name[count] = param_name
            if hasattr(graph_signature, "inputs_to_lifted_tensor_constants"):
                if node.name in graph_signature.inputs_to_lifted_tensor_constants:
                    constant_name = graph_signature.inputs_to_lifted_tensor_constants[
                        node.name
                    ]
                    if constant_name in constant_name_to_corrected_name:
                        inp_pos_to_param_buffer_name[
                            count
                        ] = constant_name_to_corrected_name[constant_name]
                    else:
                        inp_pos_to_param_buffer_name[count] = constant_name
            count += 1

    return inp_pos_to_param_buffer_name


class _StatefulGraphModuleFactory(type):
    """
    Metaclass that ensures a private constructor for _StatefulGraphModule
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
        )

    def _create(cls, root, graph, range_constraints=None):
        return super().__call__(
            root,
            graph,
            range_constraints=range_constraints,
        )


class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):
    def __init__(self, root, graph, range_constraints=None):
        super().__init__(root, graph)
        self.range_constraints = range_constraints or []


def _create_stateful_graph_module(
    plain_graph_module: torch.fx.GraphModule,
    range_constraints,
):
    stateful_gm = _StatefulGraphModule._create(
        plain_graph_module,
        plain_graph_module.graph,
        range_constraints=range_constraints,
    )
    stateful_gm.register_forward_pre_hook(
        _check_input_constraints_pre_hook, with_kwargs=True
    )
    return stateful_gm


def _unlift_exported_program_lifted_states(ep: ExportedProgram) -> torch.nn.Module:
    new_gm = copy.deepcopy(ep.graph_module)
    inp_pos_to_param_buffer_name = _construct_inp_pos_to_param_buffer_name(
        new_gm, ep.graph_signature, ep.state_dict, ep.tensor_constants
    )
    new_gm = _unlift(
        new_gm,
        inp_pos_to_param_buffer_name,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        ep.state_dict,
        ep.tensor_constants,
        ep.graph_signature.buffers_to_mutate,
    )
    unlift_gm = _create_stateful_graph_module(new_gm, ep.range_constraints)
    unlift_gm.meta.update(ep.graph_module.meta)
    return unlift_gm
