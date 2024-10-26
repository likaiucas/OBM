from ..builder import DETECTORS
import torch
from .cascade_loft import CascadeLOFT
@DETECTORS.register_module()
class CascadeLOFTprompt(CascadeLOFT):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CascadeLOFT, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.anchor_bbox_vis = [[287, 433, 441, 541]]
        self.with_vis_feat = False
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