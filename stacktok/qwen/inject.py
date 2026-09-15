# SPDX-License-Identifier: Apache-2.0
"""
StackTok for Qwen2.5-VL: injection and monkey-patching for token selection.
"""

import types

from loguru import logger as eval_logger

from ..core import StackTokCore
from .model import Qwen2_5_VL_StackTok
from .vision import Qwen2_5_VisionTransformerStackTok


def inject_qwen2_5_vl(qwen_model,
                      language_tokenizer=None,
                      processor=None,
                      retain_ratio=0.2,
                      **stacktok_kwargs):
    """
    Inject StackTok token selection into Qwen2.5-VL.

    Qwen2.5-VL uses dynamic resolution, so the number of vision tokens per image
    varies. StackTok uses a relative ``retain_ratio`` (fraction of tokens to keep)
    instead of an absolute count.

    Args:
        qwen_model: Qwen2.5-VL model instance
        language_tokenizer: Language tokenizer
        processor: Qwen2.5-VL processor (used to patch apply_chat_template for question hook)
        retain_ratio: Fraction of vision tokens to retain (default: 0.2, i.e. keep 20%).
        **stacktok_kwargs: Additional StackTok configuration overrides.

    Returns:
        (qwen_model, processor) with StackTok applied.

    Example:
        >>> from stacktok.qwen import inject_qwen2_5_vl
        >>> model, processor = inject_qwen2_5_vl(model, processor=processor, retain_ratio=0.2)
    """
    if language_tokenizer is None:
        raise ValueError("language_tokenizer is required for StackTok selection.")
    if not 0.0 < float(retain_ratio) <= 1.0:
        raise ValueError("retain_ratio must be in the interval (0, 1].")

    stacktok_config = {
        "softmax_tv_temperature": 0.01,
        "softmax_vv_temperature": 0.2,
        "device": qwen_model.device,
        "remove_padding_indices": False,  # only LLaVA 1.5 supports True; Qwen must be False
        "stacktok_beta_min": 0.3,
        "stacktok_beta_max": 0.9,
        "stacktok_swap_mode": "auto",
        "stacktok_epsilon_swap": 1e-6,
        "stacktok_swap_passes": 1,
        "stacktok_swap_auto_max_k": 16,
        **stacktok_kwargs,
    }

    eval_logger.info(
        f"[StackTok-Qwen2.5] Injecting StackTok A/B/C: "
        f"retain_ratio={retain_ratio}, force_full_k=True, device={stacktok_config['device']}")
    stacktok_core = StackTokCore(**stacktok_config)
    stacktok_core.retain_ratio = retain_ratio
    eval_logger.info("[StackTok-Qwen2.5] core initialized")
    stacktok_core._main_model_embed_tokens = qwen_model.get_input_embeddings()
    stacktok_core._language_tokenizer = language_tokenizer
    qwen_model.model._stacktok_core = stacktok_core
    qwen_model.model._question_for_vision = None
    qwen_model.set_question = types.MethodType(_set_question, qwen_model)
    qwen_model.model.get_question = types.MethodType(_get_question, qwen_model.model)
    qwen_model.model.forward = types.MethodType(Qwen2_5_VL_StackTok.forward, qwen_model.model)
    qwen_model.model.get_video_features = types.MethodType(
        Qwen2_5_VL_StackTok.get_video_features, qwen_model.model)
    qwen_model.model.get_image_features = types.MethodType(
        Qwen2_5_VL_StackTok.get_image_features, qwen_model.model)
    qwen_model.model.visual.forward = types.MethodType(Qwen2_5_VisionTransformerStackTok.forward,
                                                       qwen_model.model.visual)
    eval_logger.info("[StackTok] Qwen2_5_VLModel.forward patched")
    if processor is not None:
        patch_qwen2_5_vl_processor_for_question_hook(processor, qwen_model)
        eval_logger.info(
            "[StackTok] Qwen2.5-VL processor.apply_chat_template patched for question hook")
    else:
        eval_logger.warning(
            "[StackTok] No processor provided, skipping apply_chat_template patch")
    eval_logger.info("[StackTok-Qwen2.5] injection done")

    return qwen_model, processor


def _set_question(self, question: str):
    """Set question on qwen_model; stored on model."""
    self.model._question_for_vision = question


def _get_question(self):
    """Get question from qwen_model.model."""
    return self._question_for_vision


def patch_qwen2_5_vl_processor_for_question_hook(processor, stacktok_model_instance):
    """
    Patch processor.apply_chat_template to capture question text and set it on stacktok_model_instance.
    """
    original_apply_chat_template = processor.apply_chat_template

    def patched_apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        question_text = extract_question_from_messages(messages)
        if question_text:
            stacktok_model_instance.set_question(question_text)
        return original_apply_chat_template(messages,
                                            tokenize=tokenize,
                                            add_generation_prompt=add_generation_prompt,
                                            **kwargs)

    processor.apply_chat_template = patched_apply_chat_template


def extract_question_from_messages(messages):
    """
    Extract question text from Qwen2.5-VL message format.
    User messages may have content as str or list of {"type": "text", "text": "..."} / {"type": "image", ...}.
    """
    question_parts = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, str):
                question_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        if text_content:
                            question_parts.append(text_content)
    full_question = " ".join(question_parts).strip().replace("<image>", "").strip()
    return full_question if full_question else None


__all__ = ["inject_qwen2_5_vl", "extract_question_from_messages"]
