import torch
import random
from ..builder import BBOX_SAMPLERS
from ..transforms import bbox2roi
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class OffsetSampler(BaseSampler):
    def __init__(self, max_num=40, t_len=20, **kwargs):
        self.max_num = max_num
        self.t_len = t_len
        
    def sample(self, gt_offsets, gt_bboxes, gt_labels=None, 
               gt_building_masks=None, gt_masks=None, **kwargs):
        index = self.custom_sampler(gt_offsets[0]).cpu().numpy()
        gt_offsets[0] = gt_offsets[0][index]
        gt_bboxes[0] = gt_bboxes[0][index]
        if gt_labels is not None:
            gt_labels[0] = gt_labels[0][index]
        if gt_building_masks is not None:
            gt_building_masks[0] = gt_building_masks[0][index]
        if gt_masks is not None:
            gt_masks[0] = gt_masks[0][index]
        
        prompt_len= len(gt_labels[0])
        if prompt_len>self.max_num:
            loc = random.sample(range(0, prompt_len), self.max_num)
            gt_bboxes[0] = gt_bboxes[0][loc,:]
            gt_labels[0] = gt_labels[0][loc]
            gt_masks[0] = gt_masks[0][loc,:]
            if gt_building_masks is not None:
                gt_building_masks[0] =  gt_building_masks[0][loc,:]#gt_seg[0][loc,:]
            else:
                gt_building_masks = gt_masks
            gt_offsets[0] = gt_offsets[0][loc,:]
        return gt_offsets, gt_bboxes, gt_labels, gt_building_masks, gt_masks
    
    def custom_sampler(self, vector_list):
        length = torch.norm(vector_list, dim=1)
        p = length / self.t_len
        return torch.rand(len(vector_list)).to(p.device)<p
    
    def _sample_pos(self,):
        pass
    def _sample_neg(self,):
        pass
    
@BBOX_SAMPLERS.register_module()
class OffsetSamplerUnit(BaseSampler):
    def __init__(self, max_num=40, t_len=20, **kwargs):
        self.max_num = max_num
        self.t_len = t_len
        
    def sample(self, gt_offsets, gt_bboxes, gt_labels=None, 
               gt_building_masks=None, gt_masks=None, **kwargs):
        index = self.custom_sampler(gt_offsets[0]).cpu().numpy()
        gt_offsets[0] = gt_offsets[0][index]
        gt_bboxes[0] = gt_bboxes[0][index]
        if gt_labels is not None:
            gt_labels[0] = gt_labels[0][index]
        if gt_building_masks is not None:
            gt_building_masks[0] = gt_building_masks[0][index]
        if gt_masks is not None:
            gt_masks[0] = gt_masks[0][index]
        
        prompt_len= len(gt_labels[0])
        if prompt_len>self.max_num:
            loc = random.sample(range(0, prompt_len), self.max_num)
            gt_bboxes[0] = gt_bboxes[0][loc,:]
            gt_labels[0] = gt_labels[0][loc]
            gt_masks[0] = gt_masks[0][loc,:]
            if gt_building_masks is not None:
                gt_building_masks[0] =  gt_building_masks[0][loc,:]#gt_seg[0][loc,:]
            else:
                gt_building_masks = gt_masks
            gt_offsets[0] = gt_offsets[0][loc,:]
        return gt_offsets, gt_bboxes, gt_labels, gt_building_masks, gt_masks
    
    def custom_sampler(self, vector_list):
        length = torch.norm(vector_list, dim=1)
        p = torch.rand(len(vector_list)).to(length.device)
        p[length>self.t_len] = 1
        
        return p>0.5
    
    def _sample_pos(self,):
        pass
    def _sample_neg(self,):
        pass