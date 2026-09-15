# SPDX-License-Identifier: Apache-2.0
"""
Runtime integration for StackTok vision-token selection.
"""

import torch
from loguru import logger as eval_logger

from .selector import StackTokSelector
from .text import QuestionTextProcessor


class StackTokCore:
    """
    Connect text encoding and model features to the StackTok selector.
    """

    def __init__(
        self,
        target_vision_tokens=64,
        softmax_tv_temperature=0.02,
        softmax_vv_temperature=0.2,
        device="cuda",
        remove_padding_indices=False,
        stacktok_beta_min=0.3,
        stacktok_beta_max=0.9,
        stacktok_swap_mode="auto",
        stacktok_epsilon_swap=1e-6,
        stacktok_swap_passes=1,
        stacktok_swap_auto_max_k=16,
        stacktok_global_multicrop=True,
        **kwargs,
    ):
        self.device = device
        self.target_vision_tokens = target_vision_tokens
        self.softmax_tv_temperature = softmax_tv_temperature
        self.softmax_vv_temperature = softmax_vv_temperature
        self.remove_padding_indices = remove_padding_indices
        self.stacktok_beta_min = stacktok_beta_min
        self.stacktok_beta_max = stacktok_beta_max
        self.stacktok_swap_mode = stacktok_swap_mode
        self.stacktok_epsilon_swap = stacktok_epsilon_swap
        self.stacktok_swap_passes = stacktok_swap_passes
        self.stacktok_swap_auto_max_k = stacktok_swap_auto_max_k
        self.stacktok_global_multicrop = stacktok_global_multicrop
        self.extra_kwargs = kwargs
        self._logged_single_path = False
        self._logged_multicrop_path = False
        self._init_processors()
        eval_logger.info(f"[StackTok] target_vision_tokens={self.target_vision_tokens}, "
                         f"beta=[{self.stacktok_beta_min}, {self.stacktok_beta_max}], "
                         "force_full_k=True, "
                         f"swap={self.stacktok_swap_mode}")

    def _init_processors(self):
        """Initialize token selector and text processor."""
        self.token_selector = StackTokSelector(
            target_vision_tokens=self.target_vision_tokens,
            beta_min=self.stacktok_beta_min,
            beta_max=self.stacktok_beta_max,
            swap_mode=self.stacktok_swap_mode,
            epsilon_swap=self.stacktok_epsilon_swap,
            swap_passes=self.stacktok_swap_passes,
            swap_auto_max_k=self.stacktok_swap_auto_max_k,
            enable_global_multicrop=self.stacktok_global_multicrop,
        )
        self.text_processor = QuestionTextProcessor()

    def _encode_text_with_token_pooling(self, text: str):
        """
        Encode text (question/keywords) into text token embeddings for coverage.
        The model language embedding table keeps text and projected vision
        features in the same representation space.
        """
        enc = self._language_tokenizer(text.split(),
                                       is_split_into_words=True,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True)
        input_ids = enc["input_ids"].to(self.device)  # [1, T]
        with torch.no_grad():
            tok_emb = self._main_model_embed_tokens(input_ids)[0]  # [T, D]
        start_idx = 0
        if input_ids.shape[1] > 1 and input_ids[0,
                                                0].item() == self._language_tokenizer.bos_token_id:
            start_idx = 1
        return tok_emb[start_idx:]  # [num_words, hidden_dim]

    def apply_selection(
        self,
        projected_features,
        encoder_features,
        question_text,
        padding_patch_indices=None,
    ):
        """
        Apply coverage-based subset selection: select vision tokens that cover
        text tokens (question) and the vision token set (multimodal coverage).

        Args:
            projected_features: Features in the language-model embedding space.
            encoder_features: Features from the vision encoder.
            question_text: Question string used to obtain text embeddings.
            padding_patch_indices: Optional per-image padding indices.

        Returns:
            selected_features: Selected subset of mm_projector features
            selected_indices: Indices of selected vision tokens
        """
        vision_feat = projected_features
        vision_feat_clip = encoder_features
        if vision_feat_clip is not None and vision_feat_clip.shape[1] - vision_feat.shape[1] == 1:
            vision_feat_clip = vision_feat_clip[:, 1:, :]

        text_for_coverage = self.text_processor.extract_keywords_simple(question_text)
        text_token_embedding = self._encode_text_with_token_pooling(text_for_coverage)

        selected_features, selected_indices = self.select_vision_tokens(
            vision_features=vision_feat,
            vision_features_clip=vision_feat_clip,
            text_token_embedding=text_token_embedding,
            padding_patch_indices_list=padding_patch_indices,
        )
        return selected_features, selected_indices

    def select_vision_tokens(
        self,
        vision_features: torch.Tensor,
        vision_features_clip: torch.Tensor,
        text_token_embedding: torch.Tensor,
        padding_patch_indices_list: list = None,
    ):
        """
        Subset selection under maximum coverage: greedy selection of vision tokens
        to cover text tokens (question) and the vision token set.

        Args:
            vision_features: [batch_size, num_tokens, hidden_dim] (mm_projector output, used as selected features)
            vision_features_clip: [batch_size, num_tokens, hidden_dim] (CLIP space for vision-vision coverage/diversity)
            text_token_embedding: Text token embedding (question/keywords) for text-vision coverage
            padding_patch_indices_list: Optional per-image padding patch indices to exclude from selection

        Returns:
            selected_tokens_batch: [batch_size, target_vision_tokens, hidden_dim] selected subset
            selected_indices_list: Per-batch selected indices
        """
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)
        if vision_features_clip.dim() == 2:
            vision_features_clip = vision_features_clip.unsqueeze(0)
        batch_size, num_tokens, _ = vision_features.shape

        if num_tokens <= self.target_vision_tokens and not (
                batch_size > 1 and self.stacktok_global_multicrop
                and batch_size * num_tokens > self.target_vision_tokens):
            return vision_features, [list(range(num_tokens))] * batch_size

        selected_tokens_list = []
        selected_indices_list = []

        if batch_size > 1 and self.stacktok_global_multicrop:
            if not self._logged_multicrop_path:
                eval_logger.info(
                    "[StackTok] Algorithm B active: multi-crop global allocation; "
                    f"crops={batch_size}, total_budget(target_vision_tokens)={self.target_vision_tokens}, "
                    f"Algorithm C swap_mode={self.stacktok_swap_mode}, swap_auto_max_k={self.stacktok_swap_auto_max_k}"
                )
                self._logged_multicrop_path = True
            selected_indices_list, selected_tokens_list, _ = self.token_selector.stacktok_multicrop(
                text_token_embedding=text_token_embedding,
                vision_tokens_by_crop=[vision_features[i] for i in range(batch_size)],
                vision_tokens_clip_by_crop=[vision_features_clip[i] for i in range(batch_size)],
                tv_temp=self.softmax_tv_temperature,
                vv_temp=self.softmax_vv_temperature,
                padding_patch_indices_by_crop=padding_patch_indices_list,
            )
        else:
            if not self._logged_single_path:
                eval_logger.info(
                    "[StackTok] Algorithm A active: single-crop selection; "
                    f"budget(target_vision_tokens)={self.target_vision_tokens}, "
                    f"Algorithm C swap_mode={self.stacktok_swap_mode}, swap_auto_max_k={self.stacktok_swap_auto_max_k}"
                )
                self._logged_single_path = True
            for batch_idx in range(batch_size):
                vision_tokens = vision_features[batch_idx]
                vision_tokens_clip = vision_features_clip[batch_idx]
                if padding_patch_indices_list is not None and len(padding_patch_indices_list) > 0:
                    padding_patch_indices = padding_patch_indices_list[batch_idx]
                else:
                    padding_patch_indices = None

                selected_indices, selected_tokens, _ = self.token_selector.stacktok_single(
                    text_token_embedding=text_token_embedding,
                    vision_tokens=vision_tokens,
                    vision_tokens_clip=vision_tokens_clip,
                    tv_temp=self.softmax_tv_temperature,
                    vv_temp=self.softmax_vv_temperature,
                    padding_patch_indices=padding_patch_indices,
                )
                selected_tokens_list.append(selected_tokens)
                selected_indices_list.append(selected_indices)

        if selected_tokens_list and len({tokens.shape[0] for tokens in selected_tokens_list}) == 1:
            selected_tokens_batch = torch.stack(selected_tokens_list, dim=0)
        else:
            selected_tokens_batch = selected_tokens_list

        return selected_tokens_batch, selected_indices_list

    def apply_selection_preprocess_qwen(self,
                                        image_embeds,
                                        image_features,
                                        question_text,
                                        target_vision_tokens=None):
        """
        Coverage-based subset selection for Qwen2.5-VL: select vision tokens
        to cover text (question only, or question+answer if provided) and vision set. Single-sample path.

        Args:
            image_embeds: [num_image_tokens, hidden_dim] mm_projector output
            image_features: CLIP features for vision-vision coverage
            question_text: Question string
            answer_text: Optional answer text (for compatibility; when no scout, use question only)
            target_vision_tokens: Target subset size (uses self.token_selector.target_vision_tokens if None)

        Returns:
            selected_indices: Indices of selected vision tokens
            selected_features: Selected subset of image features (single sample)
        """
        if target_vision_tokens is not None:
            self.token_selector.target_vision_tokens = target_vision_tokens

        text_for_embedding = f"Question: {question_text}"
        if getattr(self, "clean_text", False):
            text_for_embedding = self.text_processor.extract_keywords_simple(text_for_embedding)

        text_token_embedding = self._encode_text_with_token_pooling(text_for_embedding)
        selected_features, selected_indices = self.select_vision_tokens(
            vision_features=image_embeds,
            vision_features_clip=image_features,
            text_token_embedding=text_token_embedding,
        )
        return selected_indices[0], selected_features[0]
