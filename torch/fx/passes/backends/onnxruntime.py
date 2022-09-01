from typing import Type, Dict, Tuple, List, Mapping, Any

import torch
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
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


# The keys of this dictionary are OpOverload's which can be
# exported by ONNX exporter. Type of key is torch._ops.OpOverload.
support_dict: Dict[torch._ops.OpOverload, Any] = {}
for aten_op_name in onnx_supported_table:
    # Some assumptions for mapping ONNX exporter to "target"
    # in torch.fx.Node.
    # 1. ONNX exporters are not registered with overloads, so
    #    we assume the exporter function "def add(...)" supports all
    #    overloads of torch.ops.aten.add (type: torch._ops.OpOverload).
    #    This assumption is reasonable because in _run_symbolic_function,
    #    torch._C.Node.kind() is used as the key to lookup for exporting
    #    function and it doesn't include overload.
    # 2. Assume that exporting functions' names are in aten domain.
    op_overload_packet = getattr(torch.ops.aten, aten_op_name)
    for overload in op_overload_packet.overloads():
        op_overload = getattr(op_overload_packet, overload)
        print(f"op_overload:  {op_overload}, tyep of op_overload: {type(op_overload)}")
        support_dict[op_overload] = None

# decomposition_table currently contains both aten2aten and aten2prim decomposition
# this is a hack to seperate them, as we only need aten2prim decomposition for nvfuser-supported aten graph lowering
aten2aten_decomp = {}
aten2prim_decomp = {}

for op, decomp_fn in decomposition_table.items():
    if op in support_dict:
        # ONNX can express this op, no need to decompose.
        continue
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        # Assume ONNX can express ops after decomposition.
        # If no, exporter will fail and the user need to
        # remove this decomposition rule.
        aten2aten_decomp[op] = decomp_fn

# Added extra because they will be decomposed into supported
# primitives by DecompositionInterpreter.
# TODO(wechi): implement exporting functions for all
# primitive ops.
support_dict[torch.ops.aten._softmax.default] = None
# They are converted to ONNX-compatible ops by torch.jit.
# They don't have direct ONNX exporting functions.
extra_support_dict = {}
extra_support_dict["getattr"] = None
extra_support_dict["_operator.getitem"] = None


class OrtOperatorSupport(OperatorSupport):
    """
    Operator support for ONNXRuntime backend.
    """

    def __init__(self):
        super().__init__(extra_support_dict)

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return False
        if node.op == "call_function" and node.target in support_dict:
            print(f"[O] node.target: {node.target}, type of node.target: {type(node.target)}")
            return True
        print(f"[X] node.target: {node.target}, type of node.target: {type(node.target)}")
        return super().is_node_supported(submodules, node)


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
        self.partitioner_cache: Dict[torch.fx.GraphModule, torch.fx.GraphModule] = {}
        self.prim_outputs = {}
        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self.onnx_sessions = {}
        self.onnx_model = {}
        self.onnx_input_names = {}
        self.onnx_output_names = {}
        self.assert_allclose_to_baseline = False

    def lower_to_prims_and_execute(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
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
                baeline_outputs = torch._prims.executor.execute(graph_module, *args, executor="aten")
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
                baseline_outputs = torch._prims.executor.execute(graph_module, *args, executor="aten")
                # Ensure output tensor is close to the corresponding baseline.
                assert_allclose_with_detailed_error_message(onnx_outputs[0], baseline_outputs)
            return onnx_outputs[0]

    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        # FX graph based partitioning based on ONNX supported ops.
        if graph_module in self.partitioner_cache:
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

    def __call__(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        return self.compile(graph_module, args)
