import json
import os
from pycocotools.coco import COCO
import numpy as np
import pickle as pkl
from tabulate import tabulate
def bboxiou(pred_box, ann_box, ann_offset, pred_offset=None):
    ann_box[:, 2:] += ann_box[:, :2]
    matched_id = []
    offset = []
    for i, bbox in enumerate(pred_box):
        x1, y1, x2, y2, score = bbox
        x1min = np.maximum(ann_box[:,0], x1)
        y1min = np.maximum(ann_box[:,1], y1)
        x2max = np.minimum(ann_box[:,2], x2)
        y2max = np.minimum(ann_box[:,3], y2)
        
        w = np.maximum(0, x2max - x1min)
        h = np.maximum(0, y2max - y1min)
        inter = w * h
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (ann_box[:, 2] - ann_box[:, 0]) * (ann_box[:, 3] - ann_box[:, 1])
        iou = inter / (area1 + area2 - inter)
        if max(iou)>0:
            idx = np.argmax(iou)
            matched_id.append(idx)
            offset.append(ann_offset[idx])
        else:
            matched_id.append(-1)
            if pred_offset is not None:
                offset.append(pred_offset[i])
            else:
                offset.append([0, 0])
    return matched_id, offset 
        
    
def compare_json_pkl(ann_list:list,
                    pred_list:list,
                    using_bbox_key='building_bbox',):
    def annwise2imagewise(anns):
        imagewise=[]
        bbox =[]
        offset = []
        now_image = None
        for ann in anns:
            if ann['image_id']!=now_image:
                if now_image is not None:
                    imagewise.append({'image_id':now_image,
                                      using_bbox_key:bbox,
                                      'offset':offset})
                now_image = ann['image_id']
                bbox = []
                offset = []
            bbox.append(ann[using_bbox_key])
            offset.append(ann['offset'])
        return imagewise
    imagewise_ann = annwise2imagewise(ann_list)
    distances = []
    for ann, pred in zip(imagewise_ann, pred_list):
        ann_bbox = np.array(ann[using_bbox_key])
        pred_bbox = np.array(pred[0][0])
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred[2])
        
        matched_id, matched_offset = bboxiou(pred_bbox, ann_bbox, ann_offset, )
        distance = np.sqrt(np.sum((matched_offset - pred_offset)**2, 1))
        distances.extend(distance.tolist())
    print('average offset distance: {}'.format(np.mean(distances)))
    return distances 

def compare_ordered_json(ann_list:COCO,
                         pred_list:list,
                         check=False,
                         ):
    mean_dis = []
    mean_len = []
    mean_ang = []
    for pred in pred_list:
        ann = ann_list.loadAnns(pred['id'])[0]
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred['offset'][:2])
        distance = np.sqrt(np.sum((ann_offset - pred_offset)**2))
        ann_len = np.sqrt(np.sum(ann_offset**2))
        pred_len = np.sqrt(np.sum(pred_offset**2))
        a = sum(ann_offset*pred_offset)/(np.linalg.norm(ann_offset) * np.linalg.norm(pred_offset))
        # if pred_len<10:
        #     continue
        if check:
            if sum(ann['building_bbox'])!=sum(pred['building_bbox']):
                # print(pred["file_name"])
                continue
        # print('yes')
        if not np.isnan(a):
            mean_ang.append(np.arccos(a))
            # mean_dis.append(distance)
            # mean_len.append(abs(ann_len-pred_len))
        else:
            mean_ang.append(0)
            # mean_dis.append(0)
            # mean_len.append(0)
        mean_dis.append(distance)
        mean_len.append(abs(ann_len-pred_len))
        # mean_len.append(ann_len-pred_len)
        
    return mean_dis, mean_len, mean_ang

def compare_ordered_json_AL(coco:COCO,
                         pred_list:list,
                         start_len = None,
                         end_len = None,
                         ):
    def _detect_in_range(ann_len, start_len=start_len, end_len=end_len):
        assert start_len is not None or end_len is not None
        if start_len is None:
            return ann_len <end_len
        elif end_len is None:
            return start_len<ann_len
        else:
            return start_len<ann_len and ann_len <end_len
    mean_dis = []
    mean_len = []
    mean_ang = []
    for pred in pred_list:
        ann = coco.loadAnns(pred['id'])[0]
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred['offset'][:2])
        distance = np.sqrt(np.sum((ann_offset - pred_offset)**2))
        ann_len = np.sqrt(np.sum(ann_offset**2))
        pred_len = np.sqrt(np.sum(pred_offset**2))
        a = sum(ann_offset*pred_offset)/(np.linalg.norm(ann_offset) * np.linalg.norm(pred_offset))
        # if pred_len>10:
        #     continue
        if not _detect_in_range(ann_len):
            continue
        if not np.isnan(a):
            mean_ang.append(np.arccos(a))
            # mean_dis.append(distance)
            # mean_len.append(abs(ann_len-pred_len))
        else:
            mean_ang.append(0)
            # mean_dis.append(0)
            # mean_len.append(0)
        mean_dis.append(distance)
        mean_len.append(abs(ann_len-pred_len))
                # mean_len.append(ann_len-pred_len)
        
    return [start_len, end_len, np.mean(mean_dis), np.mean(mean_len), np.mean(mean_ang)]

def measure_detail(anns, preds, txt):
    headers = ['start_len', 'end_len', 'mean_dis', 'mean_len', 'mean_ang']
    range_len = [compare_ordered_json_AL(anns, preds,n, n+10) for n in range(0,100,10)]
    range_len.append(compare_ordered_json_AL(anns, preds,100, None) )
    averg=np.average(np.array(range_len)[:,2:5], 0)
    range_len.append(['-','-','-','-','-','-'])
    range_len.append([None, None, 'averge_mdis', 'average_mlen', 'average_mang'])
    range_len.append([None, None, averg[0], averg[1], averg[2]])
    mean_dis, mean_len, mean_ang = compare_ordered_json(anns, preds)
    range_len.append(['-','-','-','-','-','-'])
    range_len.append(['In total',None])
    range_len.append([None, None, np.mean(mean_dis), np.mean(mean_len), np.mean(mean_ang)])
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
        ann = COCO('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json')
        measure_detail(ann,pred, 
                    os.path.join(
                        'eval_ouput_log',
                        os.path.basename(pred_path).replace('.json', '.txt')
                    )
                    )
if __name__=='__main__':
    # main()
    
    pred_path = 'cascade_loft_r50_fpn_prompt_nomask.json'
    os.system(f'cp {pred_path} eval_ouput_log')
    pred = json.load(open(pred_path))# off_bbox_mask cas_guassian
    # ann = json.load(open('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json'))
    ann = COCO('../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json')
    measure_detail(ann,pred, 
                os.path.join(
                    'eval_ouput_log',
                    os.path.basename(pred_path).replace('.json', '.txt')
                )
                )
    
    
    
