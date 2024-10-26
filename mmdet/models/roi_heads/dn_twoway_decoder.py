# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from mmdet.core import auto_fp16
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from ..utils.common import LayerNorm2d
from ..builder import HEADS
from ..builder import build_head

from mmdet.core import (build_bbox_coder, multi_apply, force_fp32)
from mmdet.models.builder import HEADS, build_loss

@HEADS.register_module()
class DN_OffsetDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        offset_head_depth: int = 3,
        offset_coordinate='rectangle',
        loss_offset=dict(type='SmoothL1Loss', loss_weight=8*2.0),
        offset_coder=dict(
            type='DeltaXYOffsetCoder_Transformer',
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]),
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.offset_coordinate = offset_coordinate
        self.transformer_dim = transformer_dim
        self.offset_query = nn.Embedding(1, transformer_dim)
        if isinstance(transformer, dict): 
            self.transformer = build_head(transformer)
        else:
            self.transformer = transformer
        # self.offset_prediction_head = nn.ModuleList()
        # self.offset_prediction_head.append(MLP(transformer_dim, offset_head_hidden_dim, transformer_dim, offset_head_depth))
        # self.offset_prediction_head.append(nn.Linear(transformer_dim, 2))
        self.fcs = nn.ModuleList()
        num_fcs = offset_head_depth
        for i in range(num_fcs):
            in_channels = (
                transformer_dim if i == 0 else transformer_dim)
            self.fcs.append(nn.Linear(in_channels, transformer_dim))
        self.fc_offset = nn.Linear(transformer_dim, 2)
        self.offset_coder = build_bbox_coder(offset_coder)
        self.loss_offset = build_loss(loss_offset)
    def forward_test(self,        
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,):
        x, _ = self.predict_offset(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )# [1,256], [1,256,64,64]
        x = x[-1]
        for fc in self.fcs:
            x = self.relu(fc(x))
        offset = self.fc_offset(x)
        offset = self.offset_coder.decode(offset,)
        return offset, None
    
    @force_fp32(apply_to=('offset_pred', 'offset_targets'))
    def loss(self, offset_pred, offset_targets):
        if offset_pred.size(0) == 0:
            loss_offset = offset_pred.sum() * 0
        else:
            loss_offset = self.loss_offset(offset_pred,
                                        offset_targets)
        return dict(loss_offset=loss_offset)
    @auto_fp16(apply_to=('image_embeddings', 'image_pe', 'sparse_prompt_embeddings', 'dense_prompt_embeddings'))
    def forward_train(self,
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                gt_offsets,
                **kwargs):
        # print(f'length of offset: {gt_offsets[0].shape[0]}')
        xx, _ = self.predict_offset(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,)
        losses = dict()
        for i, x in enumerate(xx):
            for fc in self.fcs:
                x = self.relu(fc(x))
            offset = self.fc_offset(x)
            offset_targets = self.offset_coder.encode(gt_offsets)   
            loss = self.loss_offset(offset, offset_targets)
            losses[f's{i}.loss']=loss
        return losses
        
    def predict_offset(self,
        image_embeddings: torch.Tensor, # Bx(embed_dim=256 in vit-h)x(embed_H)x(embed_W)
        image_pe: torch.Tensor, # Bx(embed_dim=256 in vit-h)x(embed_H)x(embed_W)
        sparse_prompt_embeddings: torch.Tensor, # BxNx(embed_dim)
        dense_prompt_embeddings: torch.Tensor, # Bx(embed_dim)x(embed_H)x(embed_W)
        ):
        output_tokens = torch.cat([self.offset_query.weight], dim=0) #[1,256] <- [1,256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # (n, 3, 256) <- (n, 1, 256) + (n, 2, 256)
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # (n,256,64,64) <- (1,256,64,64) 
        src = src + dense_prompt_embeddings  # (n,256,64,64) = (n,256,64,64) + (n,256,64,64) 
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # (n,256,64,64) <- (1,256,64,64) 
        b, c, h, w = src.shape
    
        # Run the transformer
        hss, _ = self.transformer(src, pos_src, tokens) # queries:(1,3,256) ,  keys:(1,4096,256)  <-  (图片特征， 位置信息，（output，prompt）)
        offset_token_outs = []
        for hs in hss:
            offset_token_outs.append(hs[:, 0, :]) # [1,256]
        # src = src.transpose(1, 2).view(b, c, h, w) # [1,256,64,64] <- [1,4096,256]
        return offset_token_outs, None