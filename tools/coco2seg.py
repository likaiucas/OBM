import json 
import numpy as np
from pycocotools import mask as maskUtils
import cv2
import os

from multiprocessing.pool import Pool

if not os.path.exists('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/labels/'):
    os.mkdir('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/labels/')

def numpy_mask_to_coco_rle(binary_mask):
    rle = maskUtils.encode(np.array(binary_mask, order='F', dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def coco_polygon_to_mask(coco_polygon, image_height, image_width):
    rle = maskUtils.frPyObjects(coco_polygon, image_height, image_width)
    rle = maskUtils.merge(rle)
    binary_mask = maskUtils.decode(rle)
    return binary_mask

def search_anns(img_id, anns):
    return [ann for ann in anns if ann['image_id']==img_id]

def coco2mask(ann):
    seg1 = [ann] if not isinstance(ann, list) else ann
    if not isinstance(seg1[0], dict):
        seg1 = maskUtils.merge(maskUtils.frPyObjects(seg1, 1024,1024))
        seg1 = maskUtils.decode(seg1)
    else:
        seg1 = maskUtils.merge(seg1)
        seg1 = maskUtils.decode(seg1)
    return seg1
def process_json(js):
    js = json.load(open(js,'r'))
    imgs = js['images']
    anns = js['annotations']
    for img in imgs:
        ann = search_anns(img['id'], anns)
        roofs=[]
        buildings=[]
        for a in ann:
            roofs.append(coco2mask(a['segmentation']))
            buildings.append(coco2mask(a['building_seg']))
        roofs = np.array(roofs).sum(0)>0
        buildings = np.array(buildings).sum(0)>0
        buildings = buildings.astype('uint8')
        buildings[roofs]=2
        cv2.imwrite(os.path.join('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/labels/',img['file_name']), buildings)
        
            
if __name__=="__main__":
    js_list=[
        '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_shanghai_xian_test.json',
        '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_beijing_trainval.json',
        '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_chengdu_trainval.json',
        '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_haerbin_trainval.json',
        '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_jinan_trainval.json',
        '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_shanghai_trainval.json',
    ]
    p = Pool(6)
    p.map(process_json, js_list)