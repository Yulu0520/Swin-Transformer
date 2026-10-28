"""Backward-compatible Stage 3 import with unchanged parameter names."""

from .memory_cross_attention import MemoryCrossAttention


class Stage3CrossAttention(MemoryCrossAttention):
    """Original Stage 3 module sharing attention with the Stage 4 path."""
