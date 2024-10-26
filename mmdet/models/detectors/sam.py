# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from mmcv.runner import auto_fp16
import torch
from torch import nn
from torch.nn import functional as F

from mmdet.models.backbones import ImageEncoderViT
from mmdet.models.dense_heads import PromptEncoder
from mmdet.models.roi_heads import MaskDecoder

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from typing import Any, Dict, List, Tuple
from ..builder import DETECTORS
from ..backbones import ImageEncoderViT
from ..dense_heads import PromptEncoder
from ..roi_heads import MaskDecoder
from ..utils.transformer import TwoWayTransformer
from functools import partial
# 这个 Sam 类实现了根据图像和输入提示预测对象掩码的模型。
@DETECTORS.register_module()
class Sam_core(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    # __init__ 方法:
    # 1. 输入参数:
    #     - image_encoder: 用于将图像编码为图像 embedding 的骨干网络,用于有效地预测掩码。
    #     - prompt_encoder: 用于编码各种类型的输入提示。
    #     - mask_decoder: 根据图像 embedding 和编码的提示预测掩码。
    #     - pixel_mean: 输入图像像素的归一化均值。
    #     - pixel_std: 输入图像像素的归一化标准差。
    # 2. 调用父类初始化。
    # 3. 记录 image_encoder、prompt_encoder 和 mask_decoder。
    # 4. 使用 register_buffer 注册 pixel_mean 和 pixel_std。
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) :
        """-> None
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    # 这个 device 属性返回 pixel_mean 的设备类型。
    # pixel_mean 在 __init__ 方法中使用 register_buffer 注册为 buffer, 它的设备类型由输入图像的设备类型决定。
    # 所以,这个 device 属性返回模型中用于图像处理的设备类型,通常为CPU或GPU。
    @property
    def device(self) :#-> Any
        return self.pixel_mean.device

    # 这个 forward 方法实现了根据提供的图像和提示端到端预测掩码。它包含:
    # 1. 输入参数:
    #     - batched_input: 输入图像列表,每个图像是一个字典,包含:
    #         - 'image': 输入图像,形状为 3xHxW, 已经转换为模型输入。
    #         - 'original_size': 转换前的原始图像大小, 形状为 (H, W)。
    #         - 'point_coords': 批量点提示,形状为 BxNx2, 已经转换为模型输入空间。
    #         - 'point_labels': 批量点提示标签,形状为 BxN。
    #         - 'boxes': 批量框提示,形状为 Bx4, 已经转换为模型输入空间。
    #         - 'mask_inputs': 批量掩码输入, 形状为 Bx1xHxW。
    #     - multimask_output: 模型是否应该预测多个遮挡掩码, 或者返回一个掩码。
    # 2. 使用 torch.stack 将所有图像堆叠在一起,并使用 image_encoder 获得其 image_embeddings。
    # 3. 对每个图像记录和其对应的 image_embedding:
    #     - 如果有 'point_coords' ,则 points 为其值，否则 points 为 None。
    #     - 使用 prompt_encoder 获得 sparse_embeddings 和 dense_embeddings。
    #     - 使用 mask_decoder 获得 low_res_masks 和 iou_predictions。
    #     - 使用 postprocess_masks 将 low_res_masks 处理为 masks, masks的值超过 mask_threshold 则为 1,否则为 0。
    # 4. 将每个图像的预测结果组织为字典, 组成列表输出。
    # 5. 返回输出列表。
    # 所以,这个 forward 方法实现了根据输入图像和各种类型的提示(点提示、框提示和掩码提示)预测掩码的完整流程。
    # 它先使用 image_encoder 提取图像特征, 再使用 prompt_encoder 对不同类型的提示进行编码,
    # 最后使用 mask_decoder 预测掩码和掩码质量, 并使用 postprocess_masks 进行后处理,输出最终掩码预测结果。
    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) :
        """-> List[Dict[str, torch.Tensor]]
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs
    @torch.no_grad()
    def sam_hq_forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        # image_embeddings, interm_embeddings = self.image_encoder.sam_hq_forward(input_images)
        image_embeddings, interm_embeddings = self.image_encoder.sam_hq_forward(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                }
            )

        return outputs, interm_embeddings
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
    # 这个 postprocess_masks 方法实现了掩码的后处理, 包含去除填充和将掩码放大到原始图像大小两步。
    # 1. 输入参数:
    #     - masks: 从 mask_decoder 输出的批量掩码,形状为 BxCxHxW。
    #     - input_size: 输入图像大小,形状为 (H, W)。用于去除填充。
    #     - original_size: 调整大小前的原始图像大小,形状为 (H, W)。
    # 2. 使用 F.interpolate 将 masks 放大到 image_encoder.img_size x image_encoder.img_size。
    # 3. 使用 masks[..., : input_size[0], : input_size[1]] 去除 masks 中的填充。
    # 4. 使用 F.interpolate 再将 masks 放大到 original_size 大小。
    # 5. 返回放大后的 masks。
    # 所以, 这个 postprocess_masks 方法实现了掩码的后处理,包含去除填充和恢复原始大小两步。
    # 它使模型最终输出与原始输入图像大小相匹配的掩码, 为掩码的实际应用提供方便。
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) :
        """-> torch.Tensor
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    # 这个 preprocess 方法实现了图像预处理,包含归一化像素值和填充为方形输入两步。
    # 1. 输入参数 x 为输入图像, 形状为 CxHxW。
    # 2. 使用记录在 __init__ 中的 pixel_mean 和 pixel_std 对图像进行归一化。
    # 3. 计算输入图像的高 h和宽 w。
    # 4. 计算需要填充的高 padh 和宽 padw, 使图像变为 image_encoder.img_size x image_encoder.img_size。
    # 5. 使用 F.pad 对图像进行填充。
    # 6. 返回预处理后的图像 x。
    # 所以,这个 preprocess 方法实现了图像归一化和填充两步预处理。
    # 它使输入图像适合 image_encoder 的输入要求, 为后续处理提供标准化的输入。
    def preprocess(self, x: torch.Tensor) :#-> torch.Tensor
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam_core(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )
    
@DETECTORS.register_module()
class Sam(Sam_core):
      #default setting is vit_base
    def __init__(self, 
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
            checkpoint=None,
            train_cfg=None,
            test_cfg=None,
            pixel_mean = [0, 0, 0], 
            pixel_std = [1, 1, 1]):
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
    def forward_train(self,img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        pass
# prompt boxes        
    def forward_test(self, 
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert gt_masks or gt_bboxes
        x = self.image_encoder(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=gt_bboxes,
                masks=gt_masks,
            )
        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=x.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        masks = self.postprocess_masks(
                low_res_masks,
                input_size=img_metas['img_shape'][:2],
                original_size=img_metas["ori_shape"][:2],
            )
        masks = masks > self.mask_threshold
        return masks
    @torch.no_grad()
    def sam_hq_forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        # image_embeddings, interm_embeddings = self.image_encoder.sam_hq_forward(input_images)
        image_embeddings, interm_embeddings = self.image_encoder.sam_hq_forward(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                }
            )

        return outputs, interm_embeddings
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

