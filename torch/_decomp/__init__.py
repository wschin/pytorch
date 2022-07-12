import torch
import torch._ops
import torch.library
from functools import wraps
from itertools import chain
from typing import Callable, Union, Dict, Sequence, Tuple, NamedTuple
from torch.utils._pytree import tree_map
from collections import defaultdict
import inspect

__all__ = ["decomposition_table", "register_decomposition", "get_decompositions"]

# TODO: relax key type here; torch registrations should be possible to; but
# right now this type is accurate
decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}


meta_lib = torch.library.Library("aten", "IMPL", "Meta")


def register_decomposition(aten_op, registry=None, *, disable_meta: bool = False):
    """
    A decorator to register a function as a decomposition to the Python
    decomposition table.  Use it like this::

        @register_decomposition(torch.ops.aten.clamp_min)
        def clamp_min(x):
            return torch.clamp(self, min=min)

    If you are writing a new decomposition, consider contributing it
    directly to PyTorch in torch._decomp.decompositions.

    This API is experimental; we are almost certainly going to extend
    the API when we make decompositions eligible for use in transforms (e.g.,
    autograd) and not just backend tracing, where we then need to know if a
    decomposition can be used to simulate a transform.

    By default, if the decomposition is for an operator that doesn't have
    a Meta implementation, we will register it to the dispatcher.  Use
    `disable_meta` to disable this behavior.
    """

    def decomposition_decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        out_annotation = f.__annotations__.get("out")
        # Hack to detect when out is a Tuple. There seems to be no pretty way of doing this
        fn = f
        if out_annotation and getattr(out_annotation, "__origin__", None) is tuple:
            out_names = sig.return_annotation._fields
            # If out is a tuple, we need to register a function that unpacks all the out
            # elements as this is what native_functions.yaml expects

            @wraps(f)
            def _fn(*args, **kwargs):
                out_kwargs = tuple(kwargs.pop(o, None) for o in out_names)
                # Either all of the out kwargs are set or none of them
                is_none = out_kwargs[0] is None
                assert all((o is None) == is_none for o in out_kwargs)
                return f(*args, **kwargs, out=None if is_none else out_kwargs)

            out_params = [
                inspect.Parameter(
                    o,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=t,
                )
                for o, t in zip(out_names, out_annotation.__args__)
            ]
            # Drop the out parameter and concatenate the new kwargs in the signature
            params = chain(
                (v for k, v in sig.parameters.items() if k != "out"), out_params
            )
            _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                parameters=params, return_annotation=sig.return_annotation  # type: ignore[arg-type]
            )
            # Drop the out parameter and concatenate the new kwargs in the annotations
            _fn.__annotations__ = {
                k: v for k, v in f.__annotations__.items() if k != "out"
            }
            for o in out_params:
                _fn.__annotations__[o.name] = o.annotation

            fn = _fn

        nonlocal registry
        if registry is None:
            registry = decomposition_table

        def add_op_to_table(aten_op):
            overloads = []
            if isinstance(aten_op, torch._ops.OpOverload):
                overloads.append(aten_op)
            else:
                assert isinstance(aten_op, torch._ops.OpOverloadPacket)
                for ol in aten_op.overloads():
                    overloads.append(getattr(aten_op, ol))
            for op_overload in overloads:
                if op_overload in registry:
                    raise RuntimeError(f"duplicate registrations for {op_overload}")
                registry[op_overload] = fn
                # TODO: factor this logic into OpOverload or Library API
                name = op_overload._schema.name
                if op_overload._schema.overload_name:
                    name += "." + op_overload._schema.overload_name
                if (
                    not disable_meta
                    # TorchScript dumps a bunch of extra nonsense overloads
                    # which don't have corresponding dispatcher entries, we need
                    # to filter those out
                    and torch._C._dispatch_has_kernel(name)
                    # Don't register a meta kernel to any operator that has
                    # a CompositeImplicitAutograd kernel in core.
                    # Otherwise we won't be able to run autograd for that operator with the meta backend.
                    and "CompositeImplicitAutograd" not in torch._C._dispatch_dump(name)
                    and not torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta")
                ):
                    meta_lib.impl(op_overload, fn)

        # To handle allowing multiple aten_ops at once
        tree_map(add_op_to_table, aten_op)
        return fn

    return decomposition_decorator


def get_decompositions(
    aten_ops: Sequence[Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]]
) -> Dict[torch._ops.OpOverload, Callable]:
    """
    Retrieve a dictionary of decompositions corresponding to the list of
    operator overloads and overload packets passed as input.  Overload
    packets will include all decomposed overloads in the packet.  If there is
    no decomposition for a requested operator, it is silently ignored.

    This API is experimental; we are almost certainly going to give an alternate,
    more recommended formulation, where a user provides the set of operators
    they know how to implement, and we provide decompositions for everything
    not in this set.
    """
    packets_to_overloads = defaultdict(list)
    for opo in decomposition_table:
        packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions = {}
    for op in aten_ops:
        if isinstance(op, torch._ops.OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = decomposition_table[op_overload]
        elif isinstance(op, torch._ops.OpOverload) and op in decomposition_table:
            decompositions[op] = decomposition_table[op]
    return decompositions


# populate the table
import torch._decomp.decompositions
import torch._refs
