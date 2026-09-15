# SPDX-License-Identifier: Apache-2.0
"""
StackTok LLaVA patch: utilities and monkey-patch of llava.mm_utils.process_images.

The adapter installs this patch before importing process_images from LLaVA.

When the patch is applied, each process_images call stores the image list in
_STACKTOK_LATEST_IMAGES. Padding patch indices are only computed when
_STACKTOK_USE_PADDING_INDICES is True.
"""

import math

from loguru import logger as eval_logger

_STACKTOK_LATEST_IMAGES = None
_STACKTOK_PADDING_PATCH_INDICES = None
_STACKTOK_USE_PADDING_INDICES = False


def set_use_padding_indices(use: bool):
    """Enable/disable padding index computation in the process_images wrapper (LLaVA-1.5 only)."""
    global _STACKTOK_USE_PADDING_INDICES
    _STACKTOK_USE_PADDING_INDICES = use


def calculate_padding_patch_indices(original_size,
                                    target_size=336,
                                    patch_size=14,
                                    include_overlap=False):
    """
    Compute patch indices that fall inside the padding region (after resize to target_size).

    Args:
        original_size: (width, height) of the original image
        target_size: Resize target (default 336)
        patch_size: Patch size (default 14)
        include_overlap: If False (default), only patches fully inside padding (//).
                        If True, any patch touching padding (ceil) is masked.

    Returns:
        List of patch indices in the padding region (no duplicates).
    """
    assert target_size % patch_size == 0, f"target_size ({target_size}) must be divisible by patch_size ({patch_size})"
    orig_width, orig_height = original_size
    if orig_width == orig_height:
        return []
    num_patches = target_size // patch_size  # 24 x 24 = 576

    if orig_width > orig_height:
        scale_factor = target_size / orig_width
        scaled_height = int(round(orig_height * scale_factor))
        padding_top_336 = (target_size - scaled_height) // 2
        padding_bottom_336 = target_size - scaled_height - padding_top_336
        if scaled_height == target_size:
            return []
        if include_overlap:
            pad_top_rows = math.ceil(padding_top_336 / patch_size)
            pad_bot_rows = math.ceil(padding_bottom_336 / patch_size)
        else:
            pad_top_rows = padding_top_336 // patch_size
            pad_bot_rows = padding_bottom_336 // patch_size
        padding_patch_indices = []
        for row in range(pad_top_rows):
            for col in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
        for row in range(num_patches - pad_bot_rows, num_patches):
            for col in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
    else:
        scale_factor = target_size / orig_height
        scaled_width = int(round(orig_width * scale_factor))
        padding_left_336 = (target_size - scaled_width) // 2
        padding_right_336 = target_size - scaled_width - padding_left_336
        if scaled_width == target_size:
            return []
        if include_overlap:
            pad_left_cols = math.ceil(padding_left_336 / patch_size)
            pad_right_cols = math.ceil(padding_right_336 / patch_size)
        else:
            pad_left_cols = padding_left_336 // patch_size
            pad_right_cols = padding_right_336 // patch_size
        padding_patch_indices = []
        for col in range(pad_left_cols):
            for row in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
        for col in range(num_patches - pad_right_cols, num_patches):
            for row in range(num_patches):
                padding_patch_indices.append(row * num_patches + col)
    return padding_patch_indices


def get_latest_images(clear=True):
    global _STACKTOK_LATEST_IMAGES
    imgs = _STACKTOK_LATEST_IMAGES
    if clear:
        _STACKTOK_LATEST_IMAGES = None
    return imgs or []  # avoid returning None


def get_padding_patch_indices(clear=True):
    """
    Return padding patch indices (List[List[int]]) set by the last process_images call.
    """
    global _STACKTOK_PADDING_PATCH_INDICES
    indices = _STACKTOK_PADDING_PATCH_INDICES
    if clear:
        _STACKTOK_PADDING_PATCH_INDICES = None
    return indices


def apply_llava_patches():
    """
    Apply monkey-patch to llava.mm_utils.process_images so that latest images and
    padding patch indices are stored for StackTok. Idempotent; safe to call multiple times.
    """
    try:
        import llava.mm_utils as mm_utils
    except ImportError:
        eval_logger.warning("[StackTok] LLaVA not installed. Patch skipped.")
        return

    if getattr(mm_utils, "_stacktok_patched", False):
        return

    original_process_images = mm_utils.process_images

    def _new_process_images(flattened_visuals, *args, **kwargs):
        global _STACKTOK_LATEST_IMAGES, _STACKTOK_PADDING_PATCH_INDICES, _STACKTOK_USE_PADDING_INDICES

        if flattened_visuals:
            _STACKTOK_LATEST_IMAGES = flattened_visuals
            if _STACKTOK_USE_PADDING_INDICES:
                try:
                    padding_patch_indices = []
                    for image in flattened_visuals:
                        patch_indices = calculate_padding_patch_indices(image.size,
                                                                        target_size=336,
                                                                        patch_size=14,
                                                                        include_overlap=False)
                        padding_patch_indices.append(patch_indices)
                    _STACKTOK_PADDING_PATCH_INDICES = padding_patch_indices
                except Exception as exc:
                    eval_logger.warning(f"[StackTok] Error calculating padding indices: {exc}")

        return original_process_images(flattened_visuals, *args, **kwargs)

    mm_utils.process_images = _new_process_images
    mm_utils._stacktok_patched = True

    eval_logger.info("[StackTok] llava.mm_utils.process_images successfully patched.")
