import copy
from typing import Dict

import torch
from torch._prims.nvfuser_executor import nvfuser_execute
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

import typing as t

import logging

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

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        aten2aten_decomp[op] = decomp_fn

aten2aten_decomp_skips = {
    "aten.native_layer_norm_backward.default",
    "aten.embedding_dense_backward.default",   # This is hurting nvfuser's perf
    "aten.addmm.default"
}

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
            # call_function aten
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
        }

        super().__init__(support_dict)

    def is_node_supported(
        self, submodules: t.Mapping[str, Module], node: Node
    ) -> bool:

        # nvFuser FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            return False

        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)
            if target in self._support_dict:
                return True

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

def decoreate_script_module(script_module, expected_inputs, expected_outputs):
    for i, v in enumerate(script_module.graph.inputs()):
        if v.debugName() == "self":
            script_module.graph.eraseInput(i)
            break
    for i, input in enumerate(script_module.graph.inputs()):
        input.setType(torch._C.TensorType.create_from_tensor(expected_inputs[i]))
    #for i, output in enumerate(script_module.graph.outputs()):
    #    output.setType(torch._C.TensorType.create_from_tensor(expected_outputs[i]))

def create_onnx_proto(script_module):
    onnx_proto = _jit_graph_to_onnx_model(script_module.graph, torch.onnx.OperatorExportTypes.ONNX, 14)
    if False: # print ONNX model
        import onnx
        onnx.save(onnx_proto, '/tmp/onnx_model.onnx')
        m = onnx.load('/tmp/onnx_model.onnx')
        onnx.checker.check_model(m)
        print(m)
    return onnx_proto

def create_onnx_model(onnx_proto):
    import onnx
    onnx.save(onnx_proto, '/tmp/onnx_model.onnx')
    m = onnx.load('/tmp/onnx_model.onnx')
    onnx.checker.check_model(m)
    return m

def create_onnx_session(onnx_proto):
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_proto, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
    return sess

def run_onnx_session(sess, input_names, inputs, output_names, outputs):
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
    for name, value in zip(output_names, outputs):
        binding.bind_output(
            name
        )
    sess.run_with_iobinding(binding)
    return tuple(value.numpy() for value in binding.get_outputs())

class OrtBackend:
    def __init__(self):
        self.supported_ops = OrtOperatorSupport()

        # TODO: this is a naive implementation of cache without proper guard
        self.partitioner_cache: Dict[GraphModule, GraphModule] = {}

        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self.prim_decomp_cache: Dict[GraphModule, GraphModule] = {}
        self.ort_module_cache = {}
        self.prim_outputs = {}
        self.onnx_outputs = {}
        self.onnx_models = {}
        self.onnx_sessions = {}

    def lower_to_prims_and_execute(self, graph_module: GraphModule, *args, **kwargs):
        # `graph_module` is an Aten-Fx graph
        # "lowering to prims" and "trace execution" are grouped into this function, as they are both input dependent
        if graph_module in self.prim_decomp_cache:
            logging.debug("prim_decomp_cache hit!")
            prim_module = self.prim_decomp_cache[graph_module]
            onnx_sess = self.onnx_sessions[graph_module]
            onnx_model = self.onnx_models[graph_module]
            outputs = self.prim_outputs[graph_module]
        else:
            if False:
                prim_graph = torch.fx.Graph()
                DecompositionInterpreter(graph_module, prim_graph, decomposition_table=aten2aten_decomp).run(*args, **kwargs)
                prim_module = torch.fx.GraphModule(graph_module, prim_graph)
                self.prim_decomp_cache[graph_module] = prim_module
                logging.debug("Lower to prims graph: ", prim_module.code)

            prim_module = torch.fx.symbolic_trace(graph_module)
            self.prim_decomp_cache[graph_module] = prim_module

            duplicated_prim_graph = torch.fx.Graph()
            outputs = DecompositionInterpreter(graph_module, duplicated_prim_graph, decomposition_table=aten2aten_decomp).run(*args, **kwargs)
            print('outputs type:')
            print(type(outputs))
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            print('new outputs type:')
            print(type(outputs))
            print('duplicated_prim_graph.nodes after DecompositionInterpreter:')
            print([n for n in duplicated_prim_graph.nodes])
            self.prim_outputs[graph_module] = outputs
            # Make sure graph_module is not changed.
            if False:
                # It's not used because 'output' is the last node.
                # Clean nodes.
                for node in reversed(duplicated_prim_graph.nodes):
                    if node.op == 'output':
                        break
                    print('Removing node', node.op)
                    duplicated_prim_graph.erase_node(node)
            duplicated_prim_module = torch.fx.GraphModule(graph_module, duplicated_prim_graph)
            print('duplicated_prim_module.graph.nodes after torch.fx.GraphModule:')
            print([n for n in duplicated_prim_module.graph.nodes])
            duplicated_prim_module.graph.lint()
            print('duplicated_prim_module.graph.nodes after lint:')
            print([n for n in duplicated_prim_module.graph.nodes])
            duplicated_prim_module.recompile()
            print('duplicated_prim_module.graph.nodes after recompile:')
            print([n for n in duplicated_prim_module.graph.nodes])

            script_module = fx_to_torchscript(duplicated_prim_module)
            print('duplicated_prim_module.graph.nodes after fx_to_torchscript:')
            print([n for n in duplicated_prim_module.graph.nodes])
            decoreate_script_module(script_module, args, outputs)
            print('duplicated_prim_module.graph.nodes after decoreate_script_module:')
            print([n for n in duplicated_prim_module.graph.nodes])
            onnx_proto = create_onnx_proto(script_module)
            onnx_sess = create_onnx_session(onnx_proto)
            self.onnx_sessions[graph_module] = onnx_sess

            onnx_model = create_onnx_model(onnx_proto)
            print('duplicated_prim_module.graph.nodes after create_onnx_model:')
            print([n for n in duplicated_prim_module.graph.nodes])
            self.onnx_models[graph_module] = onnx_model
            
            if True:
                # Check shape mismatch.
                mismatch = False

                for ort_out, fx_out in zip(onnx_model.graph.output, outputs):
                    ort_shape = [i.dim_value for i in ort_out.type.tensor_type.shape.dim]
                    #jit_shape = jit_out.type().sizes()
                    fx_shape = [i for i in fx_out.shape]
                    if ort_shape != fx_shape:
                        print(f'{ort_shape} != {fx_shape}')
                        mismatch = True
                    else:
                        print(f'{ort_shape} == {fx_shape}')
                if mismatch:
                    import pdb; pdb.set_trace()
                    raise RuntimeError(f"Shape mismatch")

            # Assume ONNX exporter doesn't change the order of arguments and return values in ScriptModule.
            input_names = [input.name for input in onnx_model.graph.input]
            output_names = [output.name for output in onnx_model.graph.output]
            onnx_outputs = run_onnx_session(onnx_sess, input_names, args, output_names, outputs)
            self.onnx_outputs[graph_module] = onnx_outputs

        # invokes trace executor for running the prim graph
        nvfuser_outputs = execute(prim_module, *args, executor="nvfuser")
        if not isinstance(nvfuser_outputs, tuple):
            wrapped_nvfuser_outputs = (nvfuser_outputs,)
        else:
            wrapped_nvfuser_outputs = nvfuser_outputs
        for value in args:
            print(f'[input] module: {hash(prim_module)}, shape: {value.shape}, dtype: {value.dtype}')

        mismatch = False
        for prim_value, nv_value, onnx_value in zip(self.prim_outputs[graph_module], wrapped_nvfuser_outputs, self.onnx_outputs[graph_module]):
            print(f'[PRI output] module: {hash(prim_module)}, shape: {prim_value.shape}, dtype: {prim_value.dtype}')
            print(f'[NVF output] module: {hash(prim_module)}, shape: {nv_value.shape}, dtype: {nv_value.dtype}')
            print(f'[ORT output] module: {hash(prim_module)}, shape: {onnx_value.shape}, dtype: {onnx_value.dtype}')
            if prim_value.shape != nv_value.shape:
                mismatch = True
            if prim_value.shape != torch.Size(onnx_value.shape):
                mismatch = True
        if mismatch:
            raise RuntimeError(f"Shape mismatch")
        return nvfuser_outputs

        if graph_module in self.prim_decomp_cache:
            logging.debug("prim_decomp_cache hit!")
            prim_module = self.prim_decomp_cache[graph_module]
        else:
            prim_graph = torch.fx.Graph()
            expected_outputs = DecompositionInterpreter(graph_module, prim_graph, decomposition_table=aten2prim_decomp).run(*args, **kwargs)
            prim_module = torch.fx.GraphModule(graph_module, prim_graph)
            #print("prim_module begin")
            #prim_module.graph.print_tabular()
            #print("prim_module begin end")
            #duplicated_module = torch.fx.symbolic_trace(graph_module)
            self.prim_decomp_cache[graph_module] = prim_module
            self.expected_outputs[graph_module] = expected_outputs

            logging.debug("Lower to prims graph: ", prim_module.code)

            #for t in args:
            #    print("PTH in: ", t.device, t.shape, t.dtype, t.layout)
            #for t in expected_outputs:
            #    print("PTH out: ", t.device, t.shape, t.dtype, t.layout)

            ## Consider using prim module
            #modified_module = duplicated_module
            #self.ort_module_cache[graph_module] = graph_module
            #from torch.fx.passes.fake_tensor_prop import FakeTensorProp
            #from torch._subclasses.fake_tensor import (
            #    FakeTensor,
            #    FakeTensorMode,
            #)


            #print('JIT graph:')
            #f = fx_to_torchscript(modified_module)
            #print(f.graph)

            #def decoreate_torchscript(script_module, expected_outputs):
            #    for i, v in enumerate(script_module.graph.inputs()):
            #        if v.debugName() == "self":
            #            f.graph.eraseInput(i)
            #            break
            #    for i, input in enumerate(script_module.graph.inputs()):
            #        input.setType(torch._C.TensorType.create_from_tensor(args[i]))
            #    for i, output in enumerate(script_module.graph.outputs()):
            #        output.setType(torch._C.TensorType.create_from_tensor(expected_outputs[i]))

            #decoreate_torchscript(f, self.expected_outputs[graph_module])

            #import onnxruntime
            #onnx_proto = _jit_graph_to_onnx_model(f.graph, torch.onnx.OperatorExportTypes.ONNX, 14)
            #import onnx
            #onnx.save(onnx_proto, '/tmp/onnx_model.onnx')
            #m = onnx.load('/tmp/onnx_model.onnx')
            #print(m)
            #onnx.checker.check_model(m)
            ##import pdb; pdb.set_trace()
            #ort_sess = onnxruntime.InferenceSession(onnx_proto, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
            #import numpy as np
            #_NP_DTYPE = {
            #    torch.float16: np.float16,
            #    torch.float32: np.float32,
            #    torch.float64: np.float64,
            #    torch.uint8: np.uint8,
            #    torch.int8: np.int8,
            #    torch.int16: np.int16,
            #    torch.int32: np.int32,
            #    torch.int64: np.longlong,
            #    torch.bool: np.bool_,
            #}
            #binding = ort_sess.io_binding()
            #args = [a.contiguous() for a in args]
            #for jit_input, value in zip(f.graph.inputs(), args):
            #    dev = value.device
            #    binding.bind_input(
            #        jit_input.debugName(),
            #        dev.type,
            #        dev.index or 0,
            #        _NP_DTYPE[value.dtype],
            #        value.size(),
            #        value.data_ptr(),
            #    )

            #outputs = [
            #    torch.empty(
            #        t.shape,
            #        dtype=t.dtype,
            #        layout=t.layout,
            #        device=t.device,
            #        requires_grad=t.requires_grad,
            #    )
            #    for t in self.expected_outputs[graph_module]
            #]

            ##for o, value in zip(m.graph.output, outputs):
            ##    dev = value.device
            ##    binding.bind_output(
            ##        o.name,
            ##        dev.type,
            ##        dev.index or 0,
            ##        _NP_DTYPE[value.dtype],
            ##        value.size(),
            ##        value.data_ptr(),
            ##    )

            #for onnx_output in m.graph.output:
            #    binding.bind_output(
            #        onnx_output.name
            #    )
            #print('ORT run')
            ##import pdb; pdb.set_trace()
            #ort_sess.run_with_iobinding(binding)
            #for ort_output in binding.get_outputs():
            #  np_output = ort_output.numpy()
            #  print(f'[ORT output] shape: {np_output.shape}, dtype: {np_output.dtype}')
                

        # invokes trace executor for running the prim graph
        #print('My args: ', (type(i) for i in args))
        #return ort_module(*args)
        print("prim_module final")
        prim_module.graph.print_tabular()
        print("prim_module final end")
        return execute(prim_graph, *args, executor="aten")

    def compile(self, graph_module: GraphModule, args) -> GraphModule:
        # entry function for nvFuser backend
        logging.debug("Compiling graph_module: ", graph_module.code)
        print("Compiling graph_module: ", graph_module.code)


        # FX graph based partitioning based on nvfuser supported ops
        if graph_module in self.partitioner_cache:
            logging.debug("partitioner_cache hit!")
            fused_graph_module = self.partitioner_cache[graph_module]
        else:
            partitioner = CapabilityBasedPartitioner(
                graph_module, self.supported_ops, allows_single_node_partition=False)
            #import pdb; pdb.set_trace()
            fused_graph_module = partitioner.partition_and_fuse()
            from torch.fx.passes.fake_tensor_prop import FakeTensorProp
            FakeTensorProp(fused_graph_module).propagate(*args)

            #print('meta-------------------------------')
            #print(fused_graph_module.code)
            #for node in fused_graph_module.graph.nodes:
            #    if hasattr(node, 'meta'):
            #        print("Tensor: ", node.name, node.meta)
            #    else:
            #        print("Tensor: ", node.name, " no meta.")

            self.partitioner_cache[graph_module] = fused_graph_module

        print("Compiling fused_graph_module: ", fused_graph_module.code)
        # Overriding fused_module's __call__() function with lower_to_prims_and_execute()
        for node in fused_graph_module.graph.nodes:
            # TODO: use a better way to identify fused submodule
            if node.op == "call_module" and "fused_" in node.name:
                fused_module = getattr(fused_graph_module, node.name)
                fused_module._wrapped_call = self.lower_to_prims_and_execute

        return fused_graph_module

    def __call__(self, graph_module: GraphModule, args) -> GraphModule:
        # wrap self.compile as __call__ function to fit the interface for AOTAutograd's fw_compiler
        #from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        #from torch._subclasses.fake_tensor import (
        #    FakeTensor,
        #    FakeTensorMode,
        #)
        #FakeTensorProp(graph_module).propagate(*args)

        #print('Fake result')
        #print([n.meta['fake_result'] for n in graph_module.graph.nodes])
        #print('FakeProp')
        #print(graph_module.graph)
        #import pdb; pdb.set_trace()
        #import torch.utils._pytree as pytree
        #flat_tensor_args = pytree.tree_map(
        #    lambda x: x.detach().requires_grad_(x.requires_grad)
        #    if isinstance(x, torch.Tensor) else x, args
        #)
        #fake_mode = FakeTensorMode.push()
        #with fake_mode as mode:
        #    # Set input tensors that require grad to leaves
        #    fake_flat_tensor_args = pytree.tree_map(
        #        lambda x: mode.from_tensor(x) if mode else x
        #        if isinstance(x, torch.Tensor) else x, args
        #    )
        #    with torch.set_grad_enabled(True):
        #        out = graph_module(*fake_flat_tensor_args)
        #    out = pytree.tree_map(
        #        lambda x: x.detach().contiguous() if isinstance(x, torch.Tensor) else x, out
        #    )
        #    print("Fakeout")
        #    print(out)
        return self.compile(graph_module, args)
