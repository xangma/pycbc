"""Backward compatibility re-export for shift_sum."""
from pycbc.vetoes.chisq_jax import _compatible_shift_sum as shift_sum

__all__ = ["shift_sum"]
