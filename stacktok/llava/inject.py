# SPDX-License-Identifier: Apache-2.0
"""
StackTok LLaVA injection: coverage-based vision token subset selection for LLaVA models.

Injects StackTok into a LLaVA model so that a subset of vision tokens is selected
under the maximum coverage criterion (cover text tokens + vision token set).
"""

import types

from loguru import logger as eval_logger

from ..core import StackTokCore
from .multimodal import (
    encode_images_stacktok,
    encode_images_stacktok_multi,
    prepare_inputs_labels_for_multimodal_stacktok,
    restore_image_features_sorted,
)
from .patch import apply_llava_patches, set_use_padding_indices
from .vision_tower import CLIPVisionTowerStackTok


def inject_llava(model, language_tokenizer=None, target_vision_tokens=64, **stacktok_kwargs):
    """
    Inject StackTok into a LLaVA model: coverage-based subset selection of vision tokens.

    A subset of vision tokens is selected under the maximum coverage criterion
    to cover text tokens (question/keywords) and the vision token set. All
    parameters are passed to StackTokCore.

    Args:
        model: LLaVA model to inject StackTok into
        language_tokenizer: language tokenizer (for text token embedding)
        target_vision_tokens: Target subset size (default: 64)
        **stacktok_kwargs: StackTok configuration overrides.
    Returns:
        model: Model with StackTok injection applied

    Example:
        >>> from stacktok import inject_llava
        >>> model = inject_llava(model, target_vision_tokens=64)
        >>> model = inject_llava(model, target_vision_tokens=128)
    """
    stacktok_config = {
        "target_vision_tokens": target_vision_tokens,
        "softmax_tv_temperature": 0.02,
        "softmax_vv_temperature": 0.2,
        "remove_padding_indices": None,  # None = auto (resolved below); True only for LLaVA 1.5
        "stacktok_beta_min": 0.3,
        "stacktok_beta_max": 0.9,
        "stacktok_swap_mode": "auto",
        "stacktok_epsilon_swap": 1e-6,
        "stacktok_swap_passes": 1,
        "stacktok_swap_auto_max_k": 16,
        "stacktok_global_multicrop": True,
        **stacktok_kwargs,
    }

    if language_tokenizer is None:
        raise ValueError("language_tokenizer is required for StackTok selection.")
    apply_llava_patches()
    stacktok_config["device"] = model.device

    # Resolve remove_padding_indices: only LLaVA 1.5 supports True; LLaVA 1.6 must be False
    _model_path = getattr(model.config, "_name_or_path", None) or getattr(
        model.config, "name_or_path", "") or ""
    _model_path = str(_model_path).lower()
    _is_llava_15 = "llava-v1.5" in _model_path
    if stacktok_config.get("remove_padding_indices") is None:
        stacktok_config["remove_padding_indices"] = _is_llava_15
    elif stacktok_config.get("remove_padding_indices") is True and not _is_llava_15:
        eval_logger.warning(
            "[StackTok] remove_padding_indices=True is only supported for LLaVA 1.5; forcing False for this model."
        )
        stacktok_config["remove_padding_indices"] = False

    # Patch image preprocessing before injection; here we only enable padding-index computation for LLaVA-1.5.
    set_use_padding_indices(_is_llava_15)

    eval_logger.info(
        f"[StackTok] Injecting StackTok A/B/C selection: "
        f"target_tokens={stacktok_config['target_vision_tokens']}, "
        f"tv_temp={stacktok_config['softmax_tv_temperature']}, "
        f"vv_temp={stacktok_config['softmax_vv_temperature']}, "
        f"beta=[{stacktok_config['stacktok_beta_min']}, {stacktok_config['stacktok_beta_max']}], "
        "force_full_k=True, "
        f"swap={stacktok_config['stacktok_swap_mode']}, "
        f"global_multicrop={stacktok_config['stacktok_global_multicrop']}, "
        f"device={stacktok_config['device']}")

    stacktok_core = StackTokCore(**stacktok_config)

    vision_tower = model.get_vision_tower()
    model.encode_images_stacktok = types.MethodType(encode_images_stacktok, model)
    model.restore_image_features_sorted = types.MethodType(restore_image_features_sorted, model)
    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        prepare_inputs_labels_for_multimodal_stacktok, model)
    model.encode_images_stacktok_multi = types.MethodType(encode_images_stacktok_multi, model)
    stacktok_core._main_model_embed_tokens = model.get_model().embed_tokens
    stacktok_core._language_tokenizer = language_tokenizer

    vision_tower._stacktok_core = stacktok_core
    vision_tower.remove_padding_indices = stacktok_core.remove_padding_indices
    vision_tower.forward = types.MethodType(CLIPVisionTowerStackTok.forward, vision_tower)
    vision_tower.mm_projector = model.get_model().mm_projector
    vision_tower._question_for_vision = None
    vision_tower.set_question = types.MethodType(_set_question, vision_tower)
    vision_tower.get_question = types.MethodType(_get_question, vision_tower)

    patch_conv_copy_for_hook("vicuna_v1", vision_tower)

    eval_logger.info("[StackTok] LLaVA components patched; StackTok A/B/C injection complete.")

    return model


def _set_question(self, question: str):
    self._question_for_vision = question


def _get_question(self):
    return self._question_for_vision


def patch_conv_copy_for_hook(conv_name, vision_tower):
    """
    Patch the 'copy' method of a conversation template to capture the prompt for StackTok.
    Imports llava.conversation only inside this function so LLaVA remains optional.
    """
    try:
        from llava.conversation import conv_templates
    except ImportError:
        eval_logger.warning(
            "[StackTok] LLaVA conversation not found. Question hooking disabled.")
        return

    if conv_name not in conv_templates:
        return

    conv = conv_templates[conv_name]

    if getattr(conv, "_stacktok_patched", False):
        return

    orig_copy = conv.copy

    def patched_copy(self):
        new_conv = orig_copy()
        original_append = new_conv.append_message

        def patched_append_message(self, role, message):
            if role == self.roles[0] and message:
                vision_tower.set_question(message)
            return original_append(role, message)

        new_conv.append_message = types.MethodType(patched_append_message, new_conv)
        return new_conv

    conv.copy = types.MethodType(patched_copy, conv)
    conv._stacktok_patched = True
