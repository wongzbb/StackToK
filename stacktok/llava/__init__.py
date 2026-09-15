"""LLaVA integration for StackTok."""

from .inject import inject_llava
from .patch import apply_llava_patches

__all__ = ["apply_llava_patches", "inject_llava"]
