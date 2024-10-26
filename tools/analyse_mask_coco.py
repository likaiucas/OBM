import json
import sys
sys.path.append("./")
import os
from pycocotools.coco import COCO
import numpy as np
import pickle as pkl
from tabulate import tabulate
from mmdet.utils.boundary_iou import mask_to_boundary
import cv2
from pycocotools import mask as maskUtils

def roof_offset2building(mask, offset):
    lenth = np.sqrt(np.sum(offset**2))
    offset_unit = offset/lenth
    n = int(lenth+0.5)
    seg2=  mask
    # seg single building roof
    for nn in range(n):
        
        offset_ = -offset_unit*(nn+1)
        translation_matrix = np.float32([[1,0,offset_[0]],[0,1,offset_[1]]])
        seg_ = cv2.warpAffine(seg2, translation_matrix, (1024,1024)) # building layer
        mask[seg_>0]=True
    return mask

def roof_offset2foot_print(mask, offset):
    lenth = np.sqrt(np.sum(offset**2))
    offset_unit = offset/lenth
    n = int(lenth+0.5)
    seg2=  mask
    # seg single building roof

    offset_ = -offset_unit
    translation_matrix = np.float32([[1,0,offset_[0]],[0,1,offset_[1]]])
    seg_ = cv2.warpAffine(seg2, translation_matrix, (1024,1024)) # building layer
        
    return seg_

def fix_json(js_path='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json'):
    js = json.load(open(js_path))
    anns = js['annotations']
    keys = ['roof_bbox', 'roof_mask', 'building_bbox', 'building_mask', 'segmentation', 'bbox', 'footprint_mask', 'footprint_bbox']
    for ann in anns:
        for keys_ in keys:
            if keys_ not in ann:
                continue            
            item = ann[keys_]
            if isinstance(item, list):
                item = np.array(item)
                item[item<0] = 0
                item[item>1024] = 1024
                item = item.tolist()
                ann[keys_] = item
    js['annotations'] = anns
    json.dump(js, open('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/obm_huizhou_test.json','w'))


def mask_iou(mask1, mask2):
    # mask1 = mask_to_boundary(mask1)
    # mask2 = mask_to_boundary(mask2)
    # area1 = mask1.sum()
    # area2 = mask2.sum()
    inter = (mask1 & mask2).sum()
    uni = (mask1 | mask2).sum()
    mask_iou = inter / uni
    if np.isnan(mask_iou):
        mask_iou = 1
    return mask_iou

def decode_mask(mask):
    if not isinstance(mask, list):
        mask = [mask]
    elif not isinstance(mask[0], list):
        mask = [mask]
        
    if not isinstance(mask[0], dict):
        
        mask = maskUtils.merge(maskUtils.frPyObjects(mask, 1024,1024))
        mask = maskUtils.decode(mask)
    else:
        mask = maskUtils.merge(mask)
        mask = maskUtils.decode(mask)
    return mask

def compare_ordered_json_AL(coco:COCO,
                         pred_list:list,
                         start_len = None,
                         end_len = None,
                         ):
    def _crop_mask(mask, bbox):
        x1, y1, x2, y2 = [int(a) for a in bbox]
        return mask[y1:y2+y1, x1:x1+x2]
    
    def _detect_in_range(ann_len, start_len=start_len, end_len=end_len):
        assert start_len is not None or end_len is not None
        if start_len is None:
            return ann_len <end_len
        elif end_len is None:
            return start_len<ann_len
        else:
            return start_len<ann_len and ann_len <end_len
    mean_roof = []
    mean_roofb = []
    mean_building = []
    mean_footprint = []
    # mean = []
    for pred in pred_list:
        
        ann = coco.loadAnns(pred['id'])[0]
        ann_offset = np.array(ann["offset"])
        ann_len = np.sqrt(np.sum(ann_offset**2))
        if not _detect_in_range(ann_len):
            continue
        ann_roof_mask = decode_mask(ann["roof_mask"])
        # ann_roof_box = ann["roof_bbox"]
        pred_roof_mask = decode_mask(pred['roof_mask'])
        roof_box = ann['building_bbox']
        
        ann_roof_mask = _crop_mask(ann_roof_mask, roof_box)
        pred_roof_mask = _crop_mask(pred_roof_mask, roof_box)
        mean_roof.append(mask_iou(ann_roof_mask, pred_roof_mask))
        mean_roofb.append(mask_iou(mask_to_boundary(ann_roof_mask, 0.05), mask_to_boundary(pred_roof_mask, 0.05)))
        
        
        
    return [start_len, end_len, np.mean(mean_roof), np.mean(mean_roofb)]

ann = COCO('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test_fixed.json')


def measure_detail(pred_path, anns=ann, ):
    preds = json.load(open(pred_path))
    txt=os.path.join('eval_mask_log',os.path.basename(pred_path).replace('.json', '.txt'))
    headers = ['start_len', 'end_len', 'roof_iou', 'roof_boundary_iou']#, 'building_iou', 'building_boundary_iou'
    range_len = [compare_ordered_json_AL(anns, preds,n, n+10) for n in range(0,100,10)]
    range_len.append(compare_ordered_json_AL(anns, preds,100, None))
    averg=np.average(np.array(range_len)[:,2:], 0)
    range_len.append(['-','-','-','-'])
    range_len.append([None, None, 'average_roof_iou', 'average_roof_boundary_iou', ])#'average_building_iou', 'average_building_boundary_iou'
    range_len.append([None, None, averg[0], averg[1]])
    _,_, mr, mrb = compare_ordered_json_AL(anns, preds,0, None)
    range_len.append(['-','-','-','-'])
    range_len.append(['In total',None])
    range_len.append([None, None, mr, mrb])
    table = tabulate(range_len, headers, tablefmt="pipe")
    f = open(txt+'\n','a')
    print(txt,file = f)
    print(table, file = f)
    print(table)
    f.close()
    


def measure_detail_single(anns, preds, txt):
    headers = ['start_len', 'end_len', 'roof_iou', 'roof_boundary_iou']#, 'building_iou', 'building_boundary_iou'
    range_len = [compare_ordered_json_AL(anns, preds,n, n+10) for n in range(0,100,10)]
    range_len.append(compare_ordered_json_AL(anns, preds,100, None))
    averg=np.average(np.array(range_len)[:,2:], 0)
    range_len.append(['-','-','-','-'])
    range_len.append([None, None, 'average_roof_iou', 'average_roof_boundary_iou', ])#'average_building_iou', 'average_building_boundary_iou'
    range_len.append([None, None, averg[0], averg[1]])
    _,_, mr, mrb = compare_ordered_json_AL(anns, preds,0, None)
    range_len.append(['-','-','-','-'])
    range_len.append(['In total',None])
    range_len.append([None, None, mr, mrb])
    table = tabulate(range_len, headers, tablefmt="pipe")
    f = open(txt+'\n','a')
    print(txt,file = f)
    print(table, file = f)
    print(table)
    f.close()

def main():
    jss = os.listdir('eval_ouput_log')
    
    for js in jss:
        
        if js.endswith('.json'):
            pred_path = os.path.join('eval_ouput_log', js) 
        else:
            continue
        # pred_path = 'obm_seg_b_roofguide.json'
        # os.system(f'cp {pred_path} eval_ouput_log')
        pred = json.load(open(pred_path))# off_bbox_mask cas_guassian
        # ann = json.load(open('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json'))
        ann = COCO('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test_fix.json')
        measure_detail(ann,pred, 
                    os.path.join(
                        'eval_ouput_log',
                        os.path.basename(pred_path).replace('.json', '.txt')
                    )
                    )
        
if __name__=='__main__':
    # existing number under 0, please run fix_json() to fix the json file
    # fix_json('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/obm_huizhou_test.json')
    
    # pred_path = 'smlcdr_obm_pretrain_boxnoise2.json'
    # os.system(f'cp {pred_path} eval_mask_log')
    # pred = json.load(open(pred_path))# off_bbox_mask cas_guassian
    # # ann = json.load(open('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json'))
    # # ann = COCO('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test_fixed.json')
    # measure_detail(pred_path)
    
    
    # pred_path = [
    #     'smlcdr_obm_pretrain.json',
    #     'smlcdr_obm_pretrain_boxnoise1.json',
    #     'smlcdr_obm_pretrain_boxnoise2.json',
    #     'smlcdr_obm_pretrain_boxnoise3.json',
    #     'smlcdr_obm_pretrain_boxnoise4.json',
    #     'smlcdr_obm_pretrain_boxnoise5.json',
    #     'smlcdr_obm_pretrain_boxnoise6.json',
    #     'smlcdr_obm_pretrain_boxnoise7.json',
    #     'smlcdr_obm_pretrain_boxnoise8.json',
    #     'smlcdr_obm_pretrain_boxnoise9.json',
    #     'smlcdr_obm_pretrain_boxnoise10.json',
    #     'smlcdr_obm_pretrain_boxnoise20.json',
    #     'smlcdr_obm_pretrain_boxnoise30.json',
    #     'smlcdr_obm_pretrain_boxnoise40.json',
    #     'smlcdr_obm_pretrain_boxnoise50.json',
    #     ]
    pred_path = os.listdir('./')
    pred_path = [cfg for cfg in pred_path if (('cas' in cfg) or ('loft' in cfg)) and ('noise' in cfg) ]
    print(pred_path)
    [os.system(f'cp {i} eval_mask_log')for i in pred_path]
    from multiprocessing import Pool
    pool = Pool(15)
    pool.map(measure_detail, pred_path)
    
    
    
    
    
