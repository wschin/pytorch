import torch

from torch import Tensor
from typing import Optional
import torch._prims as prims
import torch._prims.utils as utils
import torch._refs as refs
from torch._prims.utils import TensorLikeType, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims.wrappers import out_wrapper, elementwise_type_promotion_wrapper
from torch._refs import (
    _make_elementwise_unary_reference,
    _make_elementwise_binary_reference,
)
from torch._decomp import register_decomposition


__all__ = [
    "i0e",
    "i1",
    "i1e",
    "logit",
    "zeta",
]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i0e
)
def i0e(a):
    return prims.bessel_i0e(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i1
)
def i1(a):
    return prims.bessel_i1(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i1e
)
def i1e(a):
    return prims.bessel_i1e(a)


@register_decomposition(torch.ops.aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: TensorLikeType, eps: Optional[float] = None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.clamp(self, lo, hi)
    return torch.log(torch.true_divide(self, torch.sub(1, self)))


zeta = _make_elementwise_binary_reference(
    prims.zeta,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.special_zeta,
)
