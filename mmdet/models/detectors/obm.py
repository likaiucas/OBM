from mmcv.runner import auto_fp16
import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from torch import nn
from torch.nn import functional as F
import random
from mmdet.models.backbones import ImageEncoderViT
from mmdet.models.dense_heads import PromptEncoder
from mmdet.utils import get_root_logger
from typing import Any, Dict, List, Tuple
from ..builder import DETECTORS
from ..backbones import ImageEncoderViT
from ..dense_heads import PromptEncoder
from ..roi_heads import OffsetDecoder
from ..utils.transformer import TwoWayTransformer
from functools import partial
from .base import BaseDetector



@DETECTORS.register_module()
class obm_core(BaseDetector):
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
        offset_decoder: OffsetDecoder,
        pretrained=None,
        train_cfg=None,
        test_cfg=None,
        pixel_mean = [0, 0, 0], 
        pixel_std = [1, 1, 1]
    ) :
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
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
        if isinstance(image_encoder, nn.Module):
            self.image_encoder = image_encoder
        else:
            self.image_encoder = build_backbone(image_encoder)
        if isinstance(prompt_encoder, nn.Module):
            self.prompt_encoder = prompt_encoder
        else:
            self.prompt_encoder =build_head(prompt_encoder)
        if isinstance(offset_decoder, nn.Module):
            self.offset_decoder = offset_decoder
        else:
            self.offset_decoder = build_head(offset_decoder)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    # 这个 device 属性返回 pixel_mean 的设备类型。
    # pixel_mean 在 __init__ 方法中使用 register_buffer 注册为 buffer, 它的设备类型由输入图像的设备类型决定。
    # 所以,这个 device 属性返回模型中用于图像处理的设备类型,通常为CPU或GPU。
    @property
    def device(self) :#-> Any
        return self.pixel_mean.device
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_offsets=None,
                      **kwargs):
        assert gt_masks or gt_bboxes
        prompt_len= len(gt_labels[0])
        if prompt_len>self.train_cfg['max_num']:
            loc = random.sample(range(0, prompt_len), self.train_cfg['max_num'])
            gt_bboxes[0] = gt_bboxes[0][loc,:]
            gt_labels[0] = gt_labels[0][loc]
            gt_masks[0] = gt_masks[0][loc,:]
            gt_offsets[0] = gt_offsets[0][loc,:]
        features = self.image_encoder(img)
        losses = dict()
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=gt_bboxes[0],
                masks=None,
            )
        loss = self.offset_decoder.forward_train(
                image_embeddings=features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                gt_offsets=gt_offsets,
                **kwargs
            )
        losses.update(loss)
        return losses
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
        x = self.image_encoder(img[0])
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=gt_bboxes[0][0],
                masks=None,
            )
        offset, _ = self.offset_decoder.forward_test(
                image_embeddings=x,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
        
        return gt_bboxes[0][0], None, offset
        
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
        
    def aug_test(self,):
        pass
    def extract_feat(self):
        pass
    def simple_test(self):
        pass


@DETECTORS.register_module()
class OBM(obm_core):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
      #default setting is vit_base
    def __init__(self, 
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
            checkpoint=None,
            pretrained=None,
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
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
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
        offset_decoder=OffsetDecoder(
            # num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            # iou_head_depth=3,
            # iou_head_hidden_dim=256,
        )
        super().__init__(image_encoder, prompt_encoder, offset_decoder,train_cfg,test_cfg, pixel_mean, pixel_std)
        
