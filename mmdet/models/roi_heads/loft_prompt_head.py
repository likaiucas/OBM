# -*- encoding: utf-8 -*-
'''
@File    :   offset_roi_head.py
@Time    :   2021/01/17 21:10:35
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   RoI head for offset model training
'''

import numpy as np
import torch
from abc import abstractmethod
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2roi, bbox2result, roi2bbox
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import OffsetTestMixin

def xyxy2xywh(rois):
    rois[:,2] = rois[:,2]-rois[:,0]
    rois[:,3] = rois[:,3]-rois[:,1]
    return rois
    
@HEADS.register_module()
class LoftPromptHead(StandardRoIHead, OffsetTestMixin):
    def __init__(self,
                offset_roi_extractor=None,
                offset_head=None,
                inference_repeat_tensor=100,
                inference_aug=False,
                noise_box = None,
                **kwargs):
        assert offset_head is not None
        super(LoftPromptHead, self).__init__(**kwargs)
        self.inference_repeat_tensor=inference_repeat_tensor
        self.inference_aug = inference_aug
        self.noise_box = noise_box
        if offset_head is not None:
            self.init_offset_head(offset_roi_extractor, offset_head)

        self.with_vis_feat = False

    def init_offset_head(self, offset_roi_extractor, offset_head):
        self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        self.offset_head = build_head(offset_head)

    def init_weights(self, pretrained):
        super(LoftPromptHead, self).init_weights(pretrained)
        self.offset_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_offsets=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        if self.with_offset:
            # print("mask_results['mask_pred']: ", mask_results['mask_pred'].shape)
            # print("mask_results['mask_targets']: ", mask_results['mask_targets'].shape)
            # print("bbox_results['bbox_feats']: ", bbox_results['bbox_feats'].shape)
            offset_results = self._offset_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_offsets, img_metas)
            # TODO: Support empty tensor input. #2280
            if offset_results['loss_offset'] is not None:
                losses.update(offset_results['loss_offset'])

        return losses

    def _offset_forward_train(self, 
                              x, 
                              sampling_results, 
                              bbox_feats, 
                              gt_offsets,
                              img_metas):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # if pos_rois.shape[0] == 0:
        #     return dict(loss_offset=None)
        offset_results = self._offset_forward(x, pos_rois)

        offset_targets = self.offset_head.get_targets(sampling_results, gt_offsets,
                                                  self.train_cfg)

        loss_offset = self.offset_head.loss(offset_results['offset_pred'], offset_targets)

        offset_results.update(loss_offset=loss_offset, offset_targets=offset_targets)
        return offset_results

    def _offset_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            offset_feats = self.offset_roi_extractor(
                x[:self.offset_roi_extractor.num_inputs], rois)
        else:
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]

        # self._show_offset_feat(rois, offset_feats)

        offset_pred = self.offset_head(offset_feats)
        offset_results = dict(offset_pred=offset_pred, offset_feats=offset_feats)
        return offset_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)

            if self.with_vis_feat:
                offset_results = self.simple_test_offset_rotate_feature(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return bbox_results, segm_results, offset_results, self.vis_featuremap
            else:
                offset_results = self.simple_test_offset(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                
                return bbox_results, segm_results, offset_results
        else:
            offset_results = self.simple_test_offset(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            
            return bbox_results, None, offset_results

    def aug_test2(self, x, proposal_list, img_metas, rescale=False):
        return_box=xyxy2xywh(roi2bbox(bbox2roi(proposal_list))[0])
        if self.noise_box is not None:
            proposal_list = [noise(proposal_list[0], self.noise_box)]
        # if len(proposal_list[0])==0:
        #     return None, None, None, None
        if self.inference_aug:
            num = 100
            proposal_list2 = [duplicate_tensor(proposal_list[0],num)]
            rois = bbox2roi(proposal_list2)
            # bbox_results = self._bbox_forward(x, rois)
            # mask_results = self._mask_forward(x, rois)
            offset_results = self._offset_forward(x, rois)
            offset_results = self.offset_head.offset_fusion(offset_results['offset_pred'], model='max')
            offset_results = self.post_fusion(offset_results, num+1, model='mean')
            rois = bbox2roi(proposal_list)
            bbox = roi2bbox(rois)
            offset_results = self.offset_head.offset_coder.decode(bbox[0], offset_results)            
            bbox = xyxy2xywh(bbox[0])

        else:
            rois = bbox2roi(proposal_list)
            bbox = roi2bbox(rois)
            # bbox = xyxy2xywh(bbox[0])
            # bbox_results = self._bbox_forward(x, rois)
            # mask_results = self._mask_forward(x, rois)
            offset_results = self._offset_forward(x, rois)
            offset_results = self.offset_head.offset_fusion(offset_results['offset_pred'], model='max')
            offset_results = self.offset_head.offset_coder.decode(bbox[0], offset_results)
            bbox = xyxy2xywh(bbox[0])
            
        # mask_results = merge_aug_masks(mask_results['mask_pred'].cpu().numpy(),[img_metas],self.test_cfg)
        # mask_results = self.simple_test_mask(x, img_metas,
        #                  bbox,
        #                  torch.zeros(len(bbox)).to(bbox.device).long(), rescale=True)
        # # mask_results = self._mask_forward(x, rois)
        # mask_results = torch.from_numpy(np.array(mask_results)).to(bbox.device).transpose(1,0).contiguous()
        # # mask_results = torch.cat((mask_results, mask_results), dim=1)
        
        mask_results = self._mask_forward(x, rois)['mask_pred']
        mask_results = self.get_out(rois, mask_results, img_metas)
        
        return return_box.cpu().numpy(), mask_results, offset_results.cpu().numpy(), None
    
    def get_out(self, rois, masks, img_metas):
        meta = img_metas[0]
        masks_out = torch.zeros((len(rois), 1, meta['ori_shape'][0], meta['ori_shape'][1]))>0.5
        for i, (roi, mask) in enumerate(zip(rois,masks)):
            roi2 = roi.tolist()
            w,h = int(roi2[3]-roi2[1]+0.5), int(0.5+roi2[4]-roi2[2])
            # resize masks
            mask = F.interpolate(mask.unsqueeze(0), size=(h,w), mode = 'bilinear')>0.5
            # w, h  = mask[0,0,:,:].shape
            # w, h  = masks_out[i, 0, int(roi2[2]):int(roi2[2])+w , int(roi2[1]):int(roi2[1])+h].shape
            masks_out[i, 0, int(roi2[2]):int(roi2[2])+h , int(roi2[1]):int(roi2[1])+w] = mask[0,0,:,:]
            
        return masks_out
    
    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result
    def post_fusion(self, offset_pred, num, model='mean'):
        if model=='mean':
            split_offsets = offset_pred.split(int(offset_pred.size(0)/num), dim=0)
            s = 0
            for n in range(num):
                s+=split_offsets[n]
            offset_pred = s/n
        return offset_pred
    
    def forward_train_prompt(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_offsets=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    None,
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        if self.with_offset:
            # print("mask_results['mask_pred']: ", mask_results['mask_pred'].shape)
            # print("mask_results['mask_targets']: ", mask_results['mask_targets'].shape)
            # print("bbox_results['bbox_feats']: ", bbox_results['bbox_feats'].shape)
            offset_results = self._offset_forward_train(x, sampling_results,
                                                    None,
                                                    gt_offsets, img_metas)
            # TODO: Support empty tensor input. #2280
            if offset_results['loss_offset'] is not None:
                losses.update(offset_results['loss_offset'])

        return losses

def generate_gaussian_offsets(num_boxes, mean, std_dev, size):
    # 生成高斯分布偏移量
    offsets = torch.randn(num_boxes, size) * std_dev + mean
    return offsets

def duplicate_tensor(tensor, num=5):
    N = tensor.size(0)
    duplicated_tensor = tensor.repeat(num, 1)
    offset = generate_gaussian_offsets(N*num, 0, 1, 4).cuda()
    duplicated_tensor[:, :2]-=abs(offset[:, :2])
    duplicated_tensor[:, 2:4]+=abs(offset[:, 2:4])
    duplicated_tensor = torch.cat((tensor, duplicated_tensor), dim=0)
    # duplicated_tensor[:, :4]+=offset[:, :4]
    return duplicated_tensor

def noise(x, mean=0):
    offset = torch.rand(x.shape).cuda()*mean
    x[:, :2]-=offset[:, :2]
    x[:, 2:4]+=offset[:, 2:4]
    x[x<0]=0
    x[x>1024]=1024.
    return x