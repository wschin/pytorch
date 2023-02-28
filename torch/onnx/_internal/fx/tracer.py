import inspect
import operator
import re
from contextlib import nullcontext
from typing import Any, Dict, Set, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map


class SymbolicTraceTorchOperatorMode(TorchDispatchMode):
    def __init__(self, root: torch.nn.Module, graph: "torch.fx.Graph"):
        self.graph: "torch.fx.Graph" = graph
        self.real_value_to_fx_value: Dict[Any, "torch.fx.Node"] = {}
        self.name_pool: Set[str] = set()
        self.root: "torch.nn.Module" = root

    def register_name(self, name: str):
        name_counter = 0
        while name in self.name_pool:
            name = re.sub("\d+$", "", name)
            name = f"{name}{name_counter}"
            name_counter += 1
        self.name_pool.add(name)
        return name

    def register_attr(self, name: str, value: Any):
        setattr(self.root, name, value)
        return self.graph.get_attr(name)

    def register_inputs(self, *real_values):
        for value in real_values:
            if value in self.real_value_to_fx_value:
                raise RuntimeError(
                    f"Cannot register graph input {value} multiple times."
                )
            name = self.register_name("input")
            fx_value = self.graph.placeholder(name)
            self.real_value_to_fx_value[value] = fx_value

    def register_outputs(self, *real_values):
        for value in real_values:
            if value not in self.real_value_to_fx_value:
                raise RuntimeError("Output value is not in traced variable pool.")
        fx_values = [
            self.real_value_to_fx_value[real_value] for real_value in real_values
        ]
        self.graph.create_node("output", "output", (tuple(fx_values),), {})

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **kwargs)

        def map_to_fx_value(value):
            if value in self.real_value_to_fx_value:
                return self.real_value_to_fx_value[value]
            else:
                # This is an input. Let's register a placeholder for it.
                if isinstance(value, torch.Tensor):
                    name = self.register_name("constant")
                    fx_value = self.register_attr(name, value)
                    self.real_value_to_fx_value[value] = fx_value
                    return fx_value
                else:
                    return value

        fx_args = tree_map(map_to_fx_value, args)
        fx_kwargs = tree_map(map_to_fx_value, kwargs)

        if isinstance(result, (tuple, list)):
            name = self.register_name("result")
            node = self.graph.create_node(
                "call_function", func, fx_args, fx_kwargs, name=name
            )
            if isinstance(result, torch.Tensor):
                self.real_value_to_fx_value[result] = node
            for i, v in enumerate(result):
                name_i = self.register_name("element")
                getitem_node_i = self.graph.create_node(
                    "call_function", operator.getitem, (node, i), {}, name=name_i
                )
                if isinstance(v, torch.Tensor):
                    self.real_value_to_fx_value[v] = getitem_node_i
        else:
            name = self.register_name("result")
            node = self.graph.create_node(
                "call_function", func, fx_args, fx_kwargs, name=name
            )
            if isinstance(result, torch.Tensor):
                self.real_value_to_fx_value[result] = node
        return result


def _trace_through_dispatcher(
    model: torch.nn.Module,
    *args: Tuple[Any, ...],
    enable_fake_tensor_mode: bool = False,
    **kwargs: Dict[str, Any],
):
    import pdb; pdb.set_trace()
    root = torch.nn.Module()
    graph = torch.fx.Graph()
    signature = inspect.signature(model.forward)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    assert len(bound.kwargs) == 0, bound.kwargs

    fake_tensor_context = nullcontext()
    if enable_fake_tensor_mode:
        fake_tensor_context = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_tensor_context, SymbolicTraceTorchOperatorMode(root, graph) as mode:
        mode.register_inputs(*bound.args)
        outputs = model(*bound.args)
        flat_outputs, _ = tree_flatten(outputs)
        mode.register_outputs(*flat_outputs)
    return (torch.fx.GraphModule(root, graph), bound.args)
