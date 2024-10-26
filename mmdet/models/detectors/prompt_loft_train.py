import numpy as np
import mmcv
import torch
import cv2
import math

from ..builder import DETECTORS
from .loft import LOFT


@DETECTORS.register_module()
class prompt_LOFT_train(LOFT):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(prompt_LOFT_train, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.anchor_bbox_vis = [[287, 433, 441, 541]]
        self.with_vis_feat = True
        
    def simple_test(self, img, img_metas, gt_bboxes, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, gt_bboxes, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        x = x[0]
        gt_bboxes=gt_bboxes[0]
        gt_bboxes = [b.squeeze(0) for b in gt_bboxes]
        # proposal_list: list(tensor(N,5))
        gt_bboxes = [torch.cat((b, torch.ones(b.shape[0]).unsqueeze(1).cuda()), 1) for b in gt_bboxes]
        # p = p
        # gt_bboxes = torch.cat((gt_bboxes,p),1)
        proposal_list = gt_bboxes
        # return self.roi_head.aug_test(
        #     x, proposal_list, img_metas, rescale=rescale)
        # img_metas = [i[0] for i in img_metas]
        img_metas = img_metas[0]
        return self.roi_head.aug_test2(x, proposal_list, img_metas, rescale=rescale)
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            # proposal_list = self.rpn_head.forward(gt_bboxes)
            # proposal_list = proposal_list[0]
            bbox_pred = [self.duplicate_tensor(gt_bbox, num=self.rpn_head.N)for gt_bbox in gt_bboxes]
            N = [bbox_p.size(0) for bbox_p in bbox_pred]
            ones_column = [torch.ones(NN, 1).cuda() for NN in N]
            proposal_list = [torch.cat((bbox_pre, ones_col), dim=1) for bbox_pre, ones_col in zip(bbox_pred, ones_column)]
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train_prompt(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def generate_gaussian_offsets(self, num_boxes, mean, std_dev, size):
        # 生成高斯分布偏移量
        offsets = torch.randn(num_boxes, size) * std_dev + mean
        return offsets

    def duplicate_tensor(self, tensor, num=5):
        N = tensor.size(0)
        duplicated_tensor = tensor.repeat(num, 1)
        offset = self.generate_gaussian_offsets(N*num, 0, 4, 4).cuda()
        duplicated_tensor[:, :2]-=abs(offset[:, :2])
        duplicated_tensor[:, 2:4]+=abs(offset[:, 2:4])
        duplicated_tensor = torch.cat((tensor, duplicated_tensor), dim=0)   
        # duplicated_tensor[:, :4]+=offset[:, :4]
        return duplicated_tensor

