import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks, merge_aug_offsets,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead
from .test_mixins import OffsetTestMixin
from . cascade_loft_roi_head import CascadeLOFTRoIHead
import torch.nn.functional as F
from mmdet.core import bbox2roi, bbox2result, roi2bbox


@HEADS.register_module()
class CascadePromptHead(CascadeLOFTRoIHead):
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
                 noise_box = None,
                **kwargs):
        super(CascadePromptHead, self).__init__(
                 num_stages=num_stages,
                 stage_loss_weights=stage_loss_weights,
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 offset_roi_extractor=offset_roi_extractor,
                 offset_head=offset_head,
                **kwargs)
        self.noise_box = noise_box
        
    def aug_test2(self, x, proposal_list, img_metas, rescale=False):
        return_box=xyxy2xywh(roi2bbox(bbox2roi(proposal_list))[0])
        if self.noise_box is not None:
            proposal_list = [noise(proposal_list[0], self.noise_box)]
        if self.inference_aug:
            num = 100
            proposal_list2 = [duplicate_tensor(proposal_list[0],num)]
            rois = bbox2roi(proposal_list2)
            # bbox = roi2bbox(proposal_list)
        else:
            rois = bbox2roi(proposal_list)
            bbox = roi2bbox(rois)

        # offset_results = self._offset_forward(x, rois)
        # offset_results = self.offset_head.offset_fusion(offset_results['offset_pred'], model='max')
        aug_offsets = []
        for i in range(self.num_stages):
            offset_results = self._offset_forward(i, x, rois)
            offset_results = self.offset_head[i].offset_fusion(offset_results['offset_pred'], model='max')
            aug_offsets.append(offset_results.cpu().numpy())
        merged_offsets = merge_aug_offsets(aug_offsets,
                                        [img_metas] * self.num_stages,
                                        self.test_cfg)
        offset_results = merged_offsets
        
        # offset_results = self._offset_forward(0, x, rois)
        # offset_results = self.offset_head[0].offset_fusion(offset_results['offset_pred'], model='max')
        
        # offset_results = self.offset_head[0].offset_fusion(merged_offsets, model='max')
        # mask_results = merge_aug_masks(mask_results['mask_pred'].cpu().numpy(),[img_metas],self.test_cfg)
        if self.inference_aug:
            offset_results = self.post_fusion(offset_results, num+1, model='mean')
            rois = bbox2roi(proposal_list)
            bbox = roi2bbox(rois)
            offset_results = self.offset_head[-1].offset_coder.decode(bbox[0], offset_results)
            bbox = xyxy2xywh(bbox[0])
        else:
            offset_results = self.offset_head[-1].offset_coder.decode(bbox[0], offset_results)
            bbox = xyxy2xywh(bbox[0])
        
        mask_results=[]
        for i in range(self.num_stages):
            mask_result = self._mask_forward(i, x, rois)['mask_pred']
            mask_results.append(mask_result)
        mask_results = sum(mask_results) 
        mask_results = self.get_out(rois, mask_results, img_metas)
        return return_box.cpu().numpy(), mask_results, offset_results.cpu().numpy(), None
    
    def get_out(self, rois, masks, img_metas):
        meta = img_metas[0]
        masks_out = torch.zeros((len(rois), 1, meta['ori_shape'][0], meta['ori_shape'][1]))>0.5*self.num_stages
        for i, (roi, mask) in enumerate(zip(rois,masks)):
            roi2 = roi.tolist()
            w,h = int(roi2[3]-roi2[1]+0.5), int(0.5+roi2[4]-roi2[2])
            # resize masks
            mask = F.interpolate(mask.unsqueeze(0), size=(h,w), mode = 'bilinear')>0.5*self.num_stages
            # w, h  = mask[0,0,:,:].shape
            # w, h  = masks_out[i, 0, int(roi2[2]):int(roi2[2])+w , int(roi2[1]):int(roi2[1])+h].shape
            masks_out[i, 0, int(roi2[2]):int(roi2[2])+h , int(roi2[1]):int(roi2[1])+w] = mask[0,0,:,:]
            
        return masks_out
    
    def post_fusion(self, offset_pred, num, model='mean'):
        if not isinstance(offset_pred, torch.Tensor):
            offset_pred = torch.Tensor(offset_pred).cuda()
        if model=='mean':
            split_offsets = offset_pred.split(int(offset_pred.size(0)/num), dim=0)
            s = 0
            for n in range(num):
                s+=split_offsets[n]
            offset_pred = s/n
        return offset_pred
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
def xyxy2xywh(rois):
    rois[:,2] = rois[:,2]-rois[:,0]
    rois[:,3] = rois[:,3]-rois[:,1]
    return rois

def noise(x, mean=0):
    offset = torch.rand(x.shape).cuda()*mean
    x[:, :2]-=offset[:, :2]
    x[:, 2:4]+=offset[:, 2:4]
    x[x<0]=0
    x[x>1024]=1024.
    return x