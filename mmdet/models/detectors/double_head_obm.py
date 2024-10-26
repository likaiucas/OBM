from mmcv.runner import auto_fp16
import torch
from ..builder import DETECTORS, build_backbone, build_head
from mmdet.core import build_sampler
from torch import nn
from torch.nn import functional as F
import random
from typing import Tuple
from ..builder import DETECTORS
from .base import BaseDetector



@DETECTORS.register_module()
class DoubleHead_OBM(BaseDetector):
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
        image_encoder,
        prompt_encoder,
        mask_decoder1,
        mask_decoder2,
        pretrained=None,
        train_cfg=None,
        test_cfg=None,
        pixel_mean = [0, 0, 0], 
        pixel_std = [1, 1, 1],
        sampler = None
    ) :
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.sampler = build_sampler(sampler, context=self) if sampler is not None else None
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
            
        if isinstance(mask_decoder1, nn.Module):
            self.mask_decoder1 = mask_decoder1
        else:
            self.mask_decoder1 = build_head(mask_decoder1)
            
        if isinstance(mask_decoder2, nn.Module):
                self.mask_decoder2 = mask_decoder2
        else:
            self.mask_decoder2 = build_head(mask_decoder2)
            
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
                      gt_building_masks=None,
                      gt_masks=None,
                      gt_offsets=None,
                      **kwargs):
        assert gt_masks or gt_bboxes
        
        if self.sampler is not None:
            prompt_len= len(gt_offsets[0])
            if prompt_len>self.train_cfg['max_num']:
                loc = self.sampler.custom_sampler(gt_offsets[0]).cpu().numpy()
                gt_bboxes[0] = gt_bboxes[0][loc]
                gt_labels[0] = gt_labels[0][loc]
                gt_masks[0] = gt_masks[0][loc]
                if gt_building_masks is not None:
                    gt_building_masks[0] =  gt_building_masks[0][loc]#gt_seg[0][loc,:]
                else:
                    gt_building_masks = gt_masks
                gt_offsets[0] = gt_offsets[0][loc]
        prompt_len= len(gt_offsets[0])
        if prompt_len>self.train_cfg['max_num']:
            loc = random.sample(range(0, prompt_len), self.train_cfg['max_num'])
            gt_bboxes[0] = gt_bboxes[0][loc,:]
            gt_labels[0] = gt_labels[0][loc]
            gt_masks[0] = gt_masks[0][loc,:]
            if gt_building_masks is not None:
                gt_building_masks[0] =  gt_building_masks[0][loc,:]#gt_seg[0][loc,:]
            else:
                gt_building_masks = gt_masks
            gt_offsets[0] = gt_offsets[0][loc,:]
                
                
        features = self.image_encoder(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=gt_bboxes[0],
                masks=None,
            )
        loss1 = self.mask_decoder1.forward_train(
                image_embeddings=features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                gt_offsets=gt_offsets,
                gt_masks=gt_masks,
                **kwargs
            )
        loss2 = self.mask_decoder2.forward_train(
                image_embeddings=features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                gt_offsets=gt_offsets,
                gt_masks=gt_building_masks,
                **kwargs
            )
        losses = dict(loss_offset1=loss1['loss_offset'], loss_mask1=loss1['loss_mask'],
                      loss_offset2=loss2['loss_offset'], loss_mask2=loss2['loss_mask'],)
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
        x = self.image_encoder(img[0])
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=gt_bboxes[0][0],
                masks=None,
            )
        offset1, prob, masks1 = self.mask_decoder1.forward_test(
                image_embeddings=x,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        offset2, prob, masks2 = self.mask_decoder2.forward_test(
                image_embeddings=x,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        
        masks1 = self.postprocess_masks(masks1, img_metas[0][0]['img_shape'][:2], img_metas[0][0]['ori_shape'][:2])
        masks2 = self.postprocess_masks(masks2, img_metas[0][0]['img_shape'][:2], img_metas[0][0]['ori_shape'][:2])
        masks1 = masks1>0
        masks2 = masks2>0
        return gt_bboxes[0][0], torch.cat([masks1,masks2], dim=1), (offset1+offset2)/2
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