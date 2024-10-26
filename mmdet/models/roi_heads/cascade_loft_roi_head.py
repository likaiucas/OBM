import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks, merge_aug_offsets,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead
from .test_mixins import OffsetTestMixin

@HEADS.register_module()
class CascadeLOFTRoIHead(CascadeRoIHead, OffsetTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 offset_roi_extractor=None,
                 offset_head=None,
                inference_repeat_tensor=100,
                inference_aug=True,
                **kwargs):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.inference_repeat_tensor=inference_repeat_tensor
        self.inference_aug = inference_aug
        super(CascadeLOFTRoIHead, self).__init__(
            num_stages = num_stages,
            stage_loss_weights = stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,**kwargs)
        if offset_head is not None:
            self.init_offset_head(offset_roi_extractor, offset_head)

        self.with_vis_feat = False
    
    def init_offset_head(self, offset_roi_extractor, offset_head):
        # self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        # self.offset_head = build_head(offset_head)
        self.offset_head = nn.ModuleList()
        if not isinstance(offset_head, list):
            offset_head = [offset_head for _ in range(self.num_stages)]
        assert len(offset_head) == self.num_stages
        for head in offset_head:
            self.offset_head.append(build_head(head))
        if offset_roi_extractor is not None:
            self.share_roi_extractor_offset = False
            self.offset_roi_extractor = nn.ModuleList()
            if not isinstance(offset_roi_extractor, list):
                offset_roi_extractor = [
                    offset_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(offset_roi_extractor) == self.num_stages
            for roi_extractor in offset_roi_extractor:
                self.offset_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor_offset = True
            self.offset_roi_extractor = self.bbox_roi_extractor
            
    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()
            if self.with_offset:
                if not self.share_roi_extractor_offset:
                    self.offset_roi_extractor[i].init_weights()
                self.offset_head[i].init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
                
        # offset heads
        if self.with_offset:
            offset_rois = rois[:100]
            for i in range(self.num_stages):
                offset_results = self._offset_forward(i, x, offset_rois)
                outs = outs + (offset_results['offset_pred'],)
        return outs

    def _offset_forward_train(self, 
                              stage,
                              x, 
                              sampling_results, 
                              bbox_feats, 
                              gt_offsets,
                              img_metas):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # if pos_rois.shape[0] == 0:
        #     return dict(loss_offset=None)
        offset_results = self._offset_forward(stage, x, pos_rois)

        offset_targets = self.offset_head[stage].get_targets(sampling_results, gt_offsets,
                                                  self.train_cfg)

        loss_offset = self.offset_head[stage].loss(offset_results['offset_pred'], offset_targets)

        offset_results.update(loss_offset=loss_offset, offset_targets=offset_targets)
        return offset_results

    def _offset_forward(self, stage, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            offset_feats = self.offset_roi_extractor[stage](
                x[:self.offset_roi_extractor[stage].num_inputs], rois)
        else:
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]

        # self._show_offset_feat(rois, offset_feats)

        offset_pred = self.offset_head[stage](offset_feats)
        # num = torch.Tensor(len(rois)).cuda()/256
        offset_results = dict(offset_pred=offset_pred, offset_feats=offset_feats)
        return offset_results

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
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                # TODO: Support empty tensor input. #2280
                if mask_results['loss_mask'] is not None:
                    for name, value in mask_results['loss_mask'].items():
                        losses[f's{i}.{name}'] = (
                            value * lw if 'loss' in name else value)
            if self.with_offset:
                offset_results = self._offset_forward_train(i, x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_offsets, img_metas)
                if offset_results['loss_offset'] is not None:
                    for name, value in offset_results['loss_offset'].items():
                        losses[f's{i}.{name}'] = (
                            value * lw if 'loss' in name else value)
                # losses[f's{i}.num_pos']=offset_results['loss_num']
                        # losses[f's{i}.{name}'] = (
                        #     value * lw if 'num' in name else value)
            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses
    def simple_test_offset(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        # ori_shape = img_metas[0]['ori_shape']
        # scale_factor = img_metas[0]['scale_factor']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            offset_result = [[] for _ in range(2)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            offset_rois = bbox2roi([_bboxes])
            offset_feats = self.offset_roi_extractor(
                x[:len(self.offset_roi_extractor.featmap_strides)], offset_rois)

            offset_pred = self.offset_head(offset_feats)
            offset_result = self.offset_head.get_offsets(offset_pred, 
                                                        _bboxes,
                                                       scale_factor,
                                                       rescale)
        return offset_result
    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            ms_scores.append(bbox_results['cls_score'])

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'][:, :-1].argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results['bbox_pred'], img_metas[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    aug_masks.append(
                        mask_results['mask_pred'].sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_metas] * self.num_stages,
                                               self.test_cfg)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result
        if self.with_offset:
            _bboxes = (
                    det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
                    if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            aug_offsets = []
            for i in range(self.num_stages):
                offset_results = self._offset_forward(i, x, mask_rois)
                aug_offsets.append(
                    offset_results['offset_pred'].cpu().numpy())
            merged_offsets = merge_aug_offsets(aug_offsets,
                                            [img_metas] * self.num_stages,
                                            self.test_cfg)
            offset_result = self.offset_head[-1].get_offsets(
                merged_offsets, _bboxes, scale_factor, rescale)
        if self.with_mask:
            # if self.with_offset:
            return ms_bbox_result['ensemble'], ms_segm_result['ensemble'], offset_result
            # results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            return ms_bbox_result['ensemble'], None, offset_result

            
        # if self.with_mask:
        #     if self.with_offset:
        #         results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'], offset_result)
        #     results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        # else:
        #     results = ms_bbox_result['ensemble']

        # return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
