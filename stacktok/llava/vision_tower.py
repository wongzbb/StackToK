# SPDX-License-Identifier: Apache-2.0
"""
StackTok CLIP encoder: forward runs CLIP -> mm_projector -> coverage-based subset selection.

Subset of vision tokens is selected to cover text tokens (question) and the
vision token set (maximum coverage formulation).
"""

import torch
import torch.nn as nn
from loguru import logger as eval_logger

from .patch import get_padding_patch_indices


def _indices_to_output(selected_indices, device):
    if not selected_indices:
        return torch.empty(0, dtype=torch.long, device=device)
    lengths = {len(idx) for idx in selected_indices}
    if len(lengths) == 1:
        return torch.tensor(selected_indices, dtype=torch.long, device=device)
    return [torch.tensor(idx, dtype=torch.long, device=device) for idx in selected_indices]


class CLIPVisionTowerStackTok(nn.Module):
    """
    StackTok vision tower: CLIP -> mm_projector -> subset selection (coverage criterion).
    """

    @torch.no_grad()
    def forward(self, images):
        """
        Forward with coverage-based subset selection: CLIP -> mm_projector -> select
        vision tokens that cover text (question) and vision set; return selected subset.

        Args:
            images: Input image tensor(s).
        Returns:
            Selected vision features (mm_projector space); second return is selected_indices.
        """
        question = self.get_question()

        if question:
            # Strip leading <image>\n
            if question.startswith("<image>\n"):
                question = question[8:]
            # Strip leading "<image> <image> ...\\n" and keep the rest as question text
            elif question.startswith("<image> "):
                parts = question.split("\n", 1)
                if len(parts) == 2:
                    prefix, remaining = parts[0], parts[1]
                    tokens = prefix.split()
                    if all(t == "<image>" for t in tokens):
                        question = remaining
                    else:
                        raise ValueError(f"Invalid question: {question}")
        else:
            question = "What do you see in this image?"
            eval_logger.error(f"No question found for StackTok, using default: {question}")

        if self.remove_padding_indices:
            padding_patch_indices = get_padding_patch_indices(clear=True)
        else:
            padding_patch_indices = None

        if isinstance(images, list):
            image_features = []
            image_indices = []
            for image_index, image in enumerate(images):
                image_feature_before_mm_projection = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True).hidden_states[-2]
                image_feature_after_mm_projection = self.mm_projector(
                    image_feature_before_mm_projection[:, 1:].to(dtype=self.dtype))

                current_padding = None
                if padding_patch_indices and image_index < len(padding_patch_indices):
                    current_padding = [padding_patch_indices[image_index]]
                selected_feature, selected_indices = self._stacktok_core.apply_selection(
                    projected_features=image_feature_after_mm_projection,
                    encoder_features=image_feature_before_mm_projection,
                    question_text=question,
                    padding_patch_indices=current_padding,
                )
                image_features.append(selected_feature)
                image_indices.append(selected_indices[0] if isinstance(selected_indices, list)
                                     and selected_indices else selected_indices)
            return image_features, _indices_to_output(image_indices, self.device)
        else:
            image_feature_before_mm_projection = self.vision_tower(
                images.to(device=self.device,
                          dtype=self.dtype), output_hidden_states=True).hidden_states[-2]
            image_feature_after_mm_projection = self.mm_projector(
                image_feature_before_mm_projection[:, 1:].to(dtype=self.dtype))

            selected_image_feature_after_mm_projection, selected_indices = self._stacktok_core.apply_selection(
                projected_features=image_feature_after_mm_projection,
                encoder_features=image_feature_before_mm_projection,
                question_text=question,
                padding_patch_indices=padding_patch_indices,
            )
            return selected_image_feature_after_mm_projection, _indices_to_output(
                selected_indices, self.device)
