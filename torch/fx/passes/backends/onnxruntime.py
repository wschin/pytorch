import logging
from typing import Type
from typing import Dict, Tuple, List, Mapping

import torch
from torch._prims.nvfuser_executor import nvfuser_execute
from torch.autograd.grad_mode import F
from torch.nn import Module
from torch._ops import OpOverload

from torch.fx import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch._prims.executor import execute
from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table
import onnxruntime


def get_onnx_supported_table():
    from torch.onnx import _onnx_supported_ops
    from torch.onnx._globals import GLOBALS
    onnx_supported_ops = set()
    for aten_op_name, opsets_string in _onnx_supported_ops.onnx_supported_ops():
        # info[0] is aten symbol's name; e.g., "sum" for aten::sum.
        # info[1] is the ONNX opsets that can express info[0]; e.g., "15 16 17"
        # indicating opset 15, opset 16, and opset 17 can all express info[0].
        if str(GLOBALS.export_onnx_opset_version) in opsets_string.split(' '):
            onnx_supported_ops.add(aten_op_name)
    return onnx_supported_ops


onnx_supported_table = get_onnx_supported_table()


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def aten_to_dtype(self, dtype: torch.dtype, **kwargs):
    if len(kwargs) > 0 or not dtype:
        raise RuntimeError("No support for other to.dtype() formats other than to.dtype(self, dtype)")
    return torch._prims.convert_element_type(self, dtype)


# decomposition_table currently contains both aten2aten and aten2prim decomposition
# this is a hack to seperate them, as we only need aten2prim decomposition for nvfuser-supported aten graph lowering
aten2aten_decomp = {}
aten2prim_decomp = {}

aten2aten_decomp_skips = {
    "aten.native_layer_norm_backward.default",
    "aten.embedding_dense_backward.default",   # This is hurting nvfuser's perf
    "aten.addmm.default",
    # ONNX can directly export it. No need to decompose.
    "aten.embedding.default",
    "aten.unsqueeze.default",
}

for aten_op_name in onnx_supported_table:
    aten2aten_decomp_skips.add(f"aten.{aten_op_name}.default")

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        if str(op) not in aten2aten_decomp_skips:
            aten2aten_decomp[op] = decomp_fn

aten2prim_decomp[torch.ops.aten.to.dtype] = aten_to_dtype


class OrtOperatorSupport(OperatorSupport):
    """
    Operator support for nvFuser backend.

    Currently, partitioning is based on FX ATen graph. The fused subgraph will latter be decomposed into prims.
    To determine if an ATen ops is supported by nvFuser, we shall check the prim ops used in its ref decomposition.
    Only if all the prim ops in the ref has a nvfuser_impl, we say this Aten op is suppported by nvFuser.

    Note: When adding a rule, please add it to the corresponding section and follow the
    alphabetical order.
    """

    def __init__(self):

        # TODO: current list copied from torch/csrc/jit/codegen/cuda/parser.cpp is incorrect,
        # as that file is solely for TorchScript and doesn't represent the actual status
        # whether operation would be runnable by primTorch+nvFuser.
        # We will iterate on this list to reflect the the reality.
        support_dict = {
            # ===============================================================
            # Following supported aten ops is copied from torch/csrc/jit/codegen/cuda/parser.cpp
            # TODO: might need to update according to supported input types
            "torch.ops.aten.add": None,
            "torch.ops.aten.sub": None,
            # "torch.ops.aten.rsub": None,    # rsub decomp is supported at aten2aten level
            "torch.ops.aten.div": None,
            "torch.ops.aten.atan2": None,
            "torch.ops.aten.mul": None,
            "torch.ops.aten.max": None,
            "torch.ops.aten.min": None,
            "torch.ops.aten.pow": None,
            "torch.ops.aten.remainder": None,
            "torch.ops.aten.fmod": None,
            "torch.ops.aten.bitwise_and": None,
            "torch.ops.aten.__and__": None,
            "torch.ops.aten.bitwise_or": None,
            "torch.ops.aten.__or__": None,
            "torch.ops.aten.bitwise_xor": None,
            "torch.ops.aten.__xor__": None,
            "torch.ops.aten.bitwise_left_shift": None,
            "torch.ops.aten.__lshift__": None,
            "torch.ops.aten.bitwise_right_shift": None,
            "torch.ops.aten.__rshift__": None,
            "torch.ops.aten.eq": None,
            "torch.ops.aten.ne": None,
            "torch.ops.aten.ge": None,
            "torch.ops.aten.gt": None,
            "torch.ops.aten.le": None,
            "torch.ops.aten.lt": None,
            "torch.ops.aten.abs": None,
            "torch.ops.aten.bitwise_not": None,
            "torch.ops.aten.ceil": None,
            "torch.ops.aten.floor": None,
            "torch.ops.aten.frac": None,
            "torch.ops.aten.neg": None,
            "torch.ops.aten.relu": None,
            "torch.ops.aten.round": None,
            "torch.ops.aten.silu": None,
            "torch.ops.aten.trunc": None,
            "torch.ops.aten.log": None,
            "torch.ops.aten.log10": None,
            "torch.ops.aten.log1p": None,
            "torch.ops.aten.log2": None,
            "torch.ops.aten.lgamma": None,
            "torch.ops.aten.exp": None,
            "torch.ops.aten.expm1": None,
            "torch.ops.aten.erf": None,
            "torch.ops.aten.erfc": None,
            "torch.ops.aten.cos": None,
            "torch.ops.aten.acos": None,
            "torch.ops.aten.cosh": None,
            "torch.ops.aten.sin": None,
            "torch.ops.aten.asin": None,
            "torch.ops.aten.sinh": None,
            "torch.ops.aten.tan": None,
            "torch.ops.aten.atan": None,
            "torch.ops.aten.tanh": None,
            "torch.ops.aten.atanh": None,
            "torch.ops.aten.sqrt": None,
            "torch.ops.aten.rsqrt": None,
            "torch.ops.aten.reciprocal": None,
            "torch.ops.aten.sigmoid": None,
            "torch.ops.aten.isfinite": None,
            "torch.ops.aten.isinf": None,
            "torch.ops.aten.isnan": None,
            "torch.ops.aten.isneginf": None,
            "torch.ops.aten.isposinf": None,
            "torch.ops.aten.isreal": None,
            # "torch.ops.aten.rand_like": None,  # causing Node empty_like_default does not support nvfuser
            "torch.ops.aten.softplus": None,
            "torch.ops.aten.threshold": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.new_zero op
            # "torch.ops.aten.threshold_backward": None,
            "torch.ops.aten.clamp": None,
            # "torch.ops.aten.clone": None,
            # Failing with where(): incompatible function arguments: \
            # [<torch._C._nvfuser.TensorView, tensor, <torch._C._nvfuser.TensorView]
            # failing with BERT_pytorch_forward_0, which has aten.where.ScalarSelf in the decomps
            # "torch.ops.aten.where": None,
            # However, aten.where.self overload is fully supported
            "torch.ops.aten.where.self": None,
            "torch.ops.aten.lerp": None,
            "torch.ops.aten.addcmul": None,
            # "torch.ops.aten.native_dropout": None,    # missing refs for aten.rank_like
            "torch.ops.aten.dropout": None,
            # "torch.ops.aten.native_dropout_backward": None,   # missing refs for aten.type_as
            "torch.ops.aten.instance_norm": None,
            "torch.ops.aten._batch_norm_impl_index": None,
            # "torch.ops.aten.native_batch_norm": None,     # missing refs for aten.var
            "torch.ops.aten.batch_norm": None,
            "torch.ops.aten.cudnn_batch_norm": None,
            "torch.ops.aten._batch_norm_impl_index_backward": None,
            # "torch.ops.aten.native_batch_norm_backward": None,    # should have been handled at aten2aten decomp
            "torch.ops.aten.native_layer_norm": None,
            "torch.ops.aten.layer_norm": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.div
            # "torch.ops.aten.native_layer_norm_backward": None,
            "torch.ops.aten.softmax.int": None,
            "torch.ops.aten.log_softmax.int": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.amax
            # "torch.ops.aten._softmax": None,
            "torch.ops.aten._log_softmax_backward_data": None,
            # "torch.ops.aten._softmax_backward_data": None,  # Node _softmax_backward_data_default does not support nvfuser
            # "torch.ops.aten.var.dim": None,       # missing refs
            "torch.ops.aten.std.dim": None,
            "torch.ops.aten.sum": None,
            # "torch.ops.aten.mean.dim": None,      # missing refs
            "torch.ops.aten._grad_sum_to_size": None,
            "torch.ops.aten.sum_to_size": None,
            "torch.ops.aten._autocast_to_reduced_precision": None,
            "torch.ops.aten._autocast_to_full_precision": None,
            # "torch.ops.aten.to.dtype": None,      # causing segfault
            # "torch.ops.aten.type_as": None,       # missing refs
            "torch.ops.aten.linear": None,
            "torch.ops.aten.gelu": None,
            # "torch.ops.aten.gelu_backward": None,       # gelu_backward is handled at aten2aten decomp
            # "torch.ops.aten.hardtanh": None,        # has functional ref, using unsupported aten.clamp
            "torch.ops.aten.leaky_relu": None,
            "torch.ops.aten.square": None,
            # relying on aten->aten->prim decomp, aten2aten is using unsupported aten.conj_physical
            "torch.ops.aten.tanh_backward": None,
            # "torch.ops.aten.amax": None,      # missing prim decomp
            # "torch.ops.aten.amin": None,      # missing prim decomp
            # "torch.ops.aten.reshape": None,
            # "torch.ops.aten.view": None,      # missing prim decomp
            "torch.ops.aten.flatten.using_ints": None,

            # ===============================================================
            # call_function builtins and operator
            # ===============================================================
            "getattr": None,
            "_operator.getitem": None,

            # ===============================================================
            # Try more for BERT
            # ===============================================================
            "torch.ops.aten.embedding": None,
            "torch.ops.aten.unsqueeze": None,
        }
        for aten_op_name in onnx_supported_table:
            support_dict[f"torch.ops.aten.{aten_op_name}"] = None

        super().__init__(support_dict)

    def is_node_supported(
        self, submodules: Mapping[str, Module], node: Node
    ) -> bool:
        # nvFuser FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            print(f"1 Unsupported op={node.op}, target={node.target}")
            return False

        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)
            if target in self._support_dict:
                print(f"2 Supported op={node.op}, target={node.target}, target_name={target}")
                return True
            print(f"2 Unsupported op={node.op}, target={node.target}, target_name={target}")

        supported = super().is_node_supported(submodules, node)
        if not supported:
            print(f"3 Unsupported op={node.op}, target={node.target})")
            return False
        print(f"3 Supported op={node.op}, target={node.target}")
        return supported


def _jit_graph_to_onnx_model(graph, operator_export_type, opset_version):
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    This function is for testing purpose.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """

    # Shape inference is required because some ops' symbolic functions
    # generate sub-graphs based on inputs' types.
    torch.onnx.symbolic_helper._set_onnx_shape_inference(True)
    torch.onnx.symbolic_helper._set_opset_version(opset_version)

    graph = torch.onnx.utils._optimize_graph(
        graph, operator_export_type, params_dict={}
    )
    proto, _, _, _ = graph._export_onnx(
        {},
        opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
    return proto


def move_placeholder_to_front(graph_module: torch.fx.GraphModule):
    graph = graph_module.graph
    placeholders = list()
    first_not_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
        if first_not_placeholder is None and node.op != "placeholder":
            first_not_placeholder = node
    if first_not_placeholder is None:
        return
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)


def fx_to_torchscript(fx_module):
    for node in fx_module.graph.nodes:
        if (node.target == torch.ops.aten._to_copy and len(node.args) == 1 and len(node.kwargs) == 1 and 'dtype' in node.kwargs):
            node.target = torch.ops.aten.to
    for node in fx_module.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs
    for node in fx_module.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    fx_module.graph.lint()
    fx_module.recompile()
    return torch.jit.script(fx_module)


def decorate_script_module(script_module, expected_inputs, expected_outputs):
    for i, v in enumerate(script_module.graph.inputs()):
        if v.debugName() == "self":
            script_module.graph.eraseInput(i)
            break
    for i, input in enumerate(script_module.graph.inputs()):
        input.setType(torch._C.TensorType.create_from_tensor(expected_inputs[i]))
    for i, output in enumerate(script_module.graph.outputs()):
        output.setType(torch._C.TensorType.create_from_tensor(expected_outputs[i]))


def create_onnx_proto(script_module):
    onnx_proto = _jit_graph_to_onnx_model(script_module.graph, torch.onnx.OperatorExportTypes.ONNX, 14)
    return onnx_proto


def create_onnx_model(onnx_proto):
    import onnx
    return onnx.ModelProto.FromString(onnx_proto)


def create_onnx_session(onnx_proto):
    sess = onnxruntime.InferenceSession(onnx_proto, providers=["CUDAExecutionProvider"])
    return sess


def run_onnx_session(
        sess: Type[onnxruntime.InferenceSession],
        input_names: List[str],
        inputs: Tuple[torch.Tensor],
        output_names: List[str],
        outputs: Tuple[torch.Tensor]):
    torch.cuda.nvtx.range_push("ORT")
    import numpy as np
    _NP_DTYPE = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }
    binding = sess.io_binding()
    # Bind inputs.
    inputs = [a.contiguous() for a in inputs]
    for name, value in zip(input_names, inputs):
        dev = value.device
        binding.bind_input(
            name,
            dev.type,
            dev.index or 0,
            _NP_DTYPE[value.dtype],
            value.size(),
            value.data_ptr(),
        )
    # Pre-allocate outputs on the
    # same devices as PyTorch outputs.
    allocated_outputs = tuple(
        torch.empty(
            t.shape,
            dtype=t.dtype,
            layout=t.layout,
            device=t.device,
            requires_grad=t.requires_grad,
        )
        for t in outputs
    )
    # Bind pre-allocated outputs to ORT values
    # inside binding object.
    for name, value in zip(output_names, allocated_outputs):
        dev = value.device
        binding.bind_output(
            name,
            dev.type,
            dev.index or 0,
            _NP_DTYPE[value.dtype],
            value.size(),
            value.data_ptr(),
        )

    sess.run_with_iobinding(binding)
    torch.cuda.nvtx.range_pop()
    return allocated_outputs


def assert_allclose_with_detailed_error_message(x: torch.Tensor, y: torch.Tensor, rtol: float = 1e-03, atol: float = 1e-04):
    diff = x - y
    real_atol = torch.max(torch.abs(diff))
    max_value = torch.max(torch.abs(x), torch.abs(y))
    real_rtol = torch.max(diff / max_value)
    allclose = True if real_atol <= atol or real_rtol <= rtol else False
    if not allclose:
        raise RuntimeError(
            "ONNX output doesn't match baseline output with " +
            f"actual rtol={real_rtol} and actual atol={real_atol} " +
            f"but expected rtol={rtol} and expected atol={atol}.")


class OrtBackend:
    def __init__(self):
        self.supported_ops = OrtOperatorSupport()
        # TODO: this is a naive implementation of cache without proper guard
        self.partitioner_cache: Dict[GraphModule, GraphModule] = {}
        self.prim_outputs = {}
        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self.onnx_sessions = {}
        self.onnx_model = {}
        self.onnx_input_names = {}
        self.onnx_output_names = {}
        self.assert_allclose_to_baseline = False

    def lower_to_prims_and_execute(self, graph_module: GraphModule, *args, **kwargs):
        if graph_module in self.onnx_sessions:
            onnx_session = self.onnx_sessions[graph_module]
            input_names = self.onnx_input_names[graph_module]
            output_names = self.onnx_output_names[graph_module]
            prim_outputs = self.prim_outputs[graph_module]
        else:
            # Create a new graph by applying operator decomposition
            # onto the graph in "graph_module".
            prim_graph = torch.fx.Graph()
            # TODO(wechi): this is a workaround for #84311.
            move_placeholder_to_front(graph_module)
            # Sequentially go through each node in the graph and
            # decompose it into a set of primitive operators.
            # This decomposition triggers the actual computation of
            # "graph_module" and returns the result "prim_outputs".
            # "prim_outputs" is used as reference output so ORT knows how
            # # to generates the same types as PyTorch.
            #
            # TODO(wechi): replace this with symbolic_trace in meta_trace.py
            # to avoid actual computation.
            prim_outputs = DecompositionInterpreter(
                graph_module, prim_graph, decomposition_table=aten2aten_decomp).run(*args, **kwargs)
            # Store reference outputs. They are used to indicate output
            # tensors' types and devices when calling ORT.
            self.prim_outputs[graph_module] = prim_outputs
            # Wrap the new graph with primitive operators as a torch.fx.GraphModule
            # so that torch.jit.script can compile it into a torch.jit.ScriptModule.
            # This is necessary because most used ONNX exporter APIs only accepts
            # graph (type: torch._C.Graph) in torch.jit.ScriptModule.
            # TODO(wechi): We should have a new exporter to generate ONNX models
            # directly from torch.fx.Graph.
            prim_module = torch.fx.GraphModule(graph_module, prim_graph)
            # Compile the torch.fx.GraphModule into a torch.jit.ScriptModule.
            script_module = fx_to_torchscript(prim_module)
            # Post-processing step to add expected input and output type information
            # to the graph in torch.jit.ScriptModule. Expected inputs is "args" and "kwargs"
            # while expected outputs is "prim_outputs".
            if isinstance(prim_outputs, tuple):
                decorate_script_module(script_module, args, prim_outputs)
            else:
                decorate_script_module(script_module, args, (prim_outputs,))
            # Generate ONNX ModelProto from torch._C.Graph.
            onnx_proto = create_onnx_proto(script_module)
            # Initialize a ORT session to execute this ONNX model.
            onnx_session = create_onnx_session(onnx_proto)
            # Cache ORT session. It's reused for the same "graph_module".
            self.onnx_sessions[graph_module] = onnx_session
            # Generate ONNX model and extract its input and output names.
            # TODO(wechi): ORT session should provide a API to extract
            # input and output names from the underlying model.
            onnx_model = create_onnx_model(onnx_proto)
            input_names = [input.name for input in onnx_model.graph.input]
            output_names = [output.name for output in onnx_model.graph.output]
            self.onnx_input_names[graph_module] = input_names
            self.onnx_output_names[graph_module] = output_names

        if isinstance(prim_outputs, tuple):
            # ORT always returns a tuple of outputs. If the original is a tuple, just returning
            # ORT output is ok.
            onnx_outputs = run_onnx_session(onnx_session, input_names, args, output_names, prim_outputs)
            if self.assert_allclose_to_baseline:
                # Compute baseline.
                baeline_outputs = execute(graph_module, *args, executor="aten")
                # Ensure every output tensor is close to the corresponding baseline.
                for onnx_output, baseline_output in zip(onnx_outputs, baeline_outputs):
                    assert_allclose_with_detailed_error_message(onnx_output, baseline_output)
            return onnx_outputs
        else:
            # ORT always returns a tuple of outputs. If the original is a tensor, the first element
            # must be extracted and returned. Otherwise, type mismatch may happen in downstream
            # computation.
            onnx_outputs = run_onnx_session(onnx_session, input_names, args, output_names, (prim_outputs,))
            assert len(onnx_outputs) == 1
            if self.assert_allclose_to_baseline:
                # Compute baseline.
                baseline_outputs = execute(graph_module, *args, executor="aten")
                # Ensure output tensor is close to the corresponding baseline.
                assert_allclose_with_detailed_error_message(onnx_outputs[0], baseline_outputs)
            return onnx_outputs[0]

    def compile(self, graph_module: GraphModule, args) -> GraphModule:
        # entry function for nvFuser backend
        logging.debug("Compiling graph_module: ", graph_module.code)

        # FX graph based partitioning based on nvfuser supported ops
        if graph_module in self.partitioner_cache:
            logging.debug("partitioner_cache hit!")
            fused_graph_module = self.partitioner_cache[graph_module]
        else:
            partitioner = CapabilityBasedPartitioner(
                graph_module, self.supported_ops, allows_single_node_partition=False)
            fused_graph_module = partitioner.partition_and_fuse()
            self.partitioner_cache[graph_module] = fused_graph_module

        # Overriding fused_module's __call__() function with lower_to_prims_and_execute()
        for node in fused_graph_module.graph.nodes:
            # TODO: use a better way to identify fused submodule
            if node.op == "call_module" and "fused_" in node.name:
                fused_module = getattr(fused_graph_module, node.name)
                fused_module._wrapped_call = self.lower_to_prims_and_execute

        return fused_graph_module

    def __call__(self, graph_module: GraphModule, args) -> GraphModule:
        return self.compile(graph_module, args)
