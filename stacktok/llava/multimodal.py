# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from llava.constants import (
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import get_anyres_image_grid_shape
except ImportError:
    pass  # LLaVA not installed; functions here will raise at call time if LLaVA is missing


def encode_images_stacktok(self, images):
    image_features, _ = self.get_model().get_vision_tower()(images)
    return image_features


def encode_images_stacktok_multi(self, images):
    image_features, keep_idx = self.get_model().get_vision_tower()(images)
    return image_features, keep_idx


def _split_features_or_indices(value, split_sizes):
    if torch.is_tensor(value):
        return list(torch.split(value, split_sizes, dim=0))
    out = []
    start = 0
    for size in split_sizes:
        out.append(value[start:start + size])
        start += size
    return out


def _as_crop_list(value):
    if isinstance(value, list):
        return value
    return [value[i] for i in range(value.shape[0])]


# Restore selected spatial crops to their original grid.
def restore_image_features_sorted(self, image_feature, cur_keep_idx, width, height):
    image_feature_list = _as_crop_list(image_feature)
    cur_keep_idx_list = _as_crop_list(cur_keep_idx)
    num_img = len(image_feature_list)
    feature_dim = image_feature_list[0].shape[-1] if num_img else self.model.image_newline.shape[-1]

    restored_features = torch.zeros((num_img, 576, feature_dim),
                                    device=image_feature_list[0].device,
                                    dtype=image_feature_list[0].dtype)
    mask = torch.zeros(num_img, 576, dtype=torch.bool, device=image_feature_list[0].device)
    for crop_idx, (crop_feature,
                   crop_keep_idx) in enumerate(zip(image_feature_list, cur_keep_idx_list)):
        crop_keep_idx = crop_keep_idx.to(device=image_feature_list[0].device, dtype=torch.long)
        if crop_keep_idx.numel() == 0:
            continue
        restored_features[crop_idx, crop_keep_idx] = crop_feature
        mask[crop_idx, crop_keep_idx] = True

    assert width * height == restored_features.shape[0], "width * height must equal num_img"
    restored_features = restored_features.view(height, width, 24, 24,
                                               feature_dim)  # [height, width, 24, 24, feature_dim]
    restored_features = restored_features.permute(
        0, 2, 1, 3, 4).contiguous()  # [height, 24, width, 24, feature_dim]
    restored_features = restored_features.view(height, 24, width * 24,
                                               feature_dim)  # [height, 24, width*24, feature_dim]
    restored_features = restored_features.view(height * 24, width * 24,
                                               feature_dim)  # [height*24, width*24, feature_dim]
    grid_with_newline = restored_features

    mask = mask.view(height, width, 24, 24)  # [height, width, 24, 24]
    mask = mask.permute(0, 2, 1, 3).contiguous()  # [height, 24, width, 24]
    mask = mask.view(height * 24, width * 24)  # [height*24, width*24]

    image_feature_select = grid_with_newline[mask]
    return image_feature_select


# Restore selected spatial crops to their original grid.
def prepare_inputs_labels_for_multimodal_stacktok(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    modalities=None,
    image_sizes=None,
):
    # image_sizes = kwargs.get("image_sizes", None)
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(images, list) or images.ndim == 5:
        if isinstance(images, list):
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features, keep_idxs = self.encode_images_stacktok_multi(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = _split_features_or_indices(image_features, split_sizes)
        keep_idxs = _split_features_or_indices(keep_idxs, split_sizes)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        if mm_patch_merge_type == "flat":
            image_features = [torch.cat(_as_crop_list(x), dim=0) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                image_feature_list = _as_crop_list(image_feature)
                keep_idx_list = _as_crop_list(keep_idxs[image_idx])
                if len(image_feature_list) > 1:

                    base_image_feature = image_feature_list[0]
                    image_feature = image_feature_list[1:]
                    cur_keep_idx = keep_idx_list[1:]
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                        image_sizes[image_idx], self.config.image_grid_pinpoints,
                        self.get_vision_tower().config.image_size)

                    if "unpad" in mm_patch_merge_type:
                        image_feature = self.restore_image_features_sorted(
                            image_feature, cur_keep_idx, num_patch_width, num_patch_height)

                    else:
                        image_feature = torch.cat(image_feature, dim=0)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature_list[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device),
                            ),
                            dim=0,
                        )
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:

        image_features = self.encode_images_stacktok(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config,
                                                                      "mm_use_im_start_end", False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0,
                                    input_ids.shape[1],
                                    dtype=torch.long,
                                    device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            start = image_token_indices[i] + 1
            end = image_token_indices[i + 1]
            cur_input_ids_noim.append(cur_input_ids[start:end])
            cur_labels_noim.append(cur_labels[start:end])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full((cur_image_features.shape[0], ),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(
                torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                       dtype=cur_new_embed.dtype,
                                       device=cur_new_embed.device), cur_new_embed),
                          dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0,
                                                          cur_len,
                                                          dtype=position_ids.dtype,
                                                          device=position_ids.device)
        else:
            new_input_embeds_padded.append(
                torch.cat((cur_new_embed,
                           torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                       dtype=cur_new_embed.dtype,
                                       device=cur_new_embed.device)),
                          dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0,
                                                         cur_len,
                                                         dtype=position_ids.dtype,
                                                         device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
