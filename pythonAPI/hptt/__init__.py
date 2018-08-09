"""HPTT - Tensor Transposition Module based on the C++ High-Performance Tensor
Transposition library (HPTT)"""

from .hptt import (
    tensorTransposeAndUpdate,
    tensorTranspose,
    equal,
    transpose,
    ascontiguousarray,
    asfortranarray,
)

__all__ = [
    'tensorTransposeAndUpdate',
    'tensorTranspose',
    'equal',
    'transpose',
    'ascontiguousarray',
    'asfortranarray',
]
