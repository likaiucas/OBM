import json
import numpy as np
import cv2
import sys
sys.path.append("./")
from os.path import join as ospj
import os
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from multiprocessing import Pool
from mmdet.utils.boundary_iou import mask_to_boundary
from tqdm import tqdm
import multiprocessing


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def visual_pred(ann_list:list,img_dir:str,save_dir:str):
    if not os.path.exists(save_dir):
        os.system('mkdir {}'.format(save_dir))
    ann = json.load(open('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json')) 
    ann_coco = COCO('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json')
    # ann_list = list(zip(ann['annotations'],ann_list))
    name_str = ''
    while ann_list:
    # for pred in ann_list:
        pred = ann_list.pop()
        if name_str==pred['file_name']:
            pred_offset = np.array(pred['offset'])
            ann = ann_coco.loadAnns(pred['id'])[0]
            # bbox = pred['building_bbox']
            bbox = ann['roof_bbox']

            # if calculate_iou(ann['building_bbox'], pred['building_bbox'])==0:
            #     print('no')
            
            start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
            endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
            cv2.arrowedLine(img,start, endpoint, (0,0,255),2,0,0,0.2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0), 2)
            
            pred_offset = np.array(ann['offset'])
            start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
            endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
            cv2.arrowedLine(img,start, endpoint, (0,255,0),2,0,0,0.2)
            
            img = cv2.putText(img, str(np.sqrt(sum(pred_offset*pred_offset)))[:5], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (254, 67, 101), 1)
        else:
            if name_str:
                cv2.imwrite(ospj(save_dir,name_str),img)
            name_str=pred['file_name']    
            path = ospj(img_dir,name_str)
            img = cv2.imread(path)

def decode_rle(ann):
    rle = [ann] if not isinstance(ann, list) else ann
    rle = maskUtils.merge(rle)
    binary_mask = maskUtils.decode(rle)     
    return binary_mask

def decode_polygon(ann):
    rle = maskUtils.frPyObjects(ann, 1024, 1024)
    rle = maskUtils.merge(rle)
    binary_mask = maskUtils.decode(rle)
    return binary_mask

def visual_anns(ann_path:str,img_dir:str,save_dir='/config_data/BONAI_data/trainval-001/vis3d'):
    if not os.path.exists(save_dir):
        os.system('mkdir {}'.format(save_dir))
    ann_coco = COCO(ann_path)
    ann_js = json.load(open(ann_path))
    imgs = ann_js['images']
    for img in imgs:
        img_id = img['id']
        anns_id = ann_coco.getAnnIds(img_id)
        anns = ann_coco.loadAnns(anns_id)
        image = cv2.imread(os.path.join(img_dir, img['file_name']))
        roof_segs = np.zeros_like(image[:,:,0])
        building_segs = np.zeros_like(image[:,:,0])
        for ann in anns:
            building_seg = decode_rle(ann['building_seg'])
            roof_seg = decode_polygon(ann['segmentation'])
            # image[building_seg]=[0,0,255]
            # image[roof_seg]=[0,255,0]
            roof_segs[roof_seg>0] = 1
            building_segs[building_seg>0] = 1
            
            pred_offset = np.array(ann['offset'])
            bbox = ann['roof_bbox']
            start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
            endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
            cv2.arrowedLine(image,start, endpoint, (255,255,255),2,0,0,0.2)
        building_segs[roof_seg]=0
        seg = np.zeros_like(image)
        seg[:,:,1]=roof_segs*255
        seg[:,:,2]=building_segs*255
        im = image*0.5+seg*0.5
        cv2.imwrite(ospj(save_dir,img['file_name'] ),im.astype('uint8'))  
        
def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file)
        
SAVE_PATH = '/irsa/lk/BONAI_data/trainval-001/vis3d_loft_selfmasks'
ANN_COCO = COCO('/irsa/lk/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_shanghai_xian_test.json')


# def _draw_boundary_offset(id, coco:COCO, save_dir=SAVE_PATH):
#     if not os.path.exists(save_dir):
#         os.system('mkdir {}'.format(save_dir))
#     img = coco.loadImgs(id)[0]
#     annIds = coco.getAnnIds(id)
#     anns = coco.loadAnns(annIds)
#     offsets = [ann['offset'] for ann in anns]   
#     offsets = np.array(offsets)
#     offsets_len = np.sqrt(np.sum(offsets*offsets,axis=1))
#     max_offset = np.max(offsets_len)
#     offsets_gray = offsets_len/max_offset*230
#     segs = np.zeros((1024,1024))
#     buildings = []
#     image = cv2.imread(os.path.join('/irsa/lk/BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test', img['file_name']))
#     # for each building
#     for i, ann in enumerate(anns):
#         roof_seg = ann['roof_mask']
#         # roof_seg = ann['roof_mask']
#         # roof_seg = ANN_COCO.loadAnns(ann['id'])[0]['segmentation']
#         seg = decode_rle(roof_seg) if isinstance(roof_seg, dict) else decode_polygon(roof_seg)
#         # seg = mask_to_boundary(seg, 0.002)
#         # draw prediction offset
#         pred_offset = np.array(ann['offset'])
#         bbox = ANN_COCO.loadAnns(ann['id'])[0]['roof_bbox']
#         start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
#         endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
#         cv2.arrowedLine(image,start, endpoint, (0,255,0),2,0,0,0.2)
#         # # draw ann offset
#         # pred_offset = np.array(ANN_COCO.loadAnns(ann['id'])[0]['offset'])
#         # start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
#         # endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
#         # cv2.arrowedLine(image,start, endpoint, (0,255,0),2,0,0,0.2)
        
#         # # draw building
#         segs[seg>0]=seg[seg>0]
#     # segs2 = np.zeros_like(image)
#     # segs2[:,:,2] = segs*255
#     # segs2[segs>0,1]=0
#     # segs2[segs>0,0]=0
    
#     # im = image*0.5+segs2*0.5
#     segs = mask_to_boundary(segs, 0.002)
#     image[segs>0,1]=0
#     image[segs>0,0]=0
#     image[segs>0,2]=255
#     cv2.imwrite(ospj(save_dir,img['file_name'] ),image.astype('uint8'))  
    
    
def _draw(id, coco:COCO, save_dir=SAVE_PATH):
    if not os.path.exists(save_dir):
        os.system('mkdir {}'.format(save_dir))
    img = coco.loadImgs(id)[0]
    # if img['file_name'] not in ['shanghai_arg__L18_107176_219488__0_1024.png','shanghai_arg__L18_107176_219440__0_0.png','shanghai_arg__L18_107176_219488__1024_1024.png']:
    #     return 
    annIds = coco.getAnnIds(id)
    anns = coco.loadAnns(annIds)
    offsets = [ann['offset'] for ann in anns]   
    offsets = np.array(offsets)
    offsets_len = np.sqrt(np.sum(offsets*offsets,axis=1))
    max_offset = np.max(offsets_len)
    offsets_gray = offsets_len/max_offset*220
    segs = np.zeros((1024,1024))
    buildings = []
    # for each building
    for i, ann in enumerate(anns):
        roof_seg = ann['roof_mask']
        # roof_seg = ann['roof_mask']
        # roof_seg = ANN_COCO.loadAnns(ann['id'])[0]['segmentation']
        seg = decode_rle(roof_seg) if isinstance(roof_seg, dict) else decode_polygon([roof_seg])
        # roof_seg = np.array(roof_seg).reshape(-1,2)
        offset = np.array(ann['offset'])
        if offsets_len[i]==0:
            continue
        offset_unit = offset/offsets_len[i]
        n = int(offsets_len[i]+0.5)
        gray = int(offsets_gray[i])
        seg[seg>0]=gray
        # seg single building roof
        for nn in range(n):
            seg2=  decode_rle(roof_seg) if isinstance(roof_seg, dict) else decode_polygon([roof_seg])
            offset_ = -offset_unit*(nn+1)
            translation_matrix = np.float32([[1,0,offset_[0]],[0,1,offset_[1]]])
            seg_ = cv2.warpAffine(seg2, translation_matrix, (1024,1024)) # building layer
            gray_ = int(offsets_gray[i]*(1-nn/n))
            seg_[seg>0] = 0
            seg[seg_>0]=gray_
            
        # segs[seg>0]=seg[seg>0]
        segs = np.maximum(seg, segs)
        # segs+=seg
            # buildings.append(seg)
        # segs = sum(buildings)
        # segs = segs/segs.max()*255
    cv2.imwrite(ospj(save_dir,img['file_name']),segs,)
    return 

def vis_3d(coco_path, save_path):
    coco = COCO(coco_path) if isinstance(json.load(open(coco_path)), dict) else None
    if not coco:
        js_anns = json.load(open(coco_path))
        imgs = []
        categories=[{'id': 1, 'name': 'building', 'supercategory': 'building'}]
        for ann in js_anns:
            img=dict(file_name = ann['file_name'],
                     id = ann['image_id'],
                     height = 1024,
                     width = 1024)
            imgs.append(img)
        coco = dict(images=imgs, annotations=js_anns, categories=categories)
        save_json('tmp.json',coco)
        coco = COCO('tmp.json')
    
    imgids = coco.getImgIds()
    # [ _draw(id, coco, save_path) for id in tqdm(imgids)]
    # pool = Pool(50)
    # pool.map(lambda x: _draw(x, coco), imgids)
    tasks = [(_draw, id, coco, save_path) for id in imgids]
    
    # [ _draw_boundary_offset(id, coco, save_path) for id in tqdm(imgids)]
    pool = multiprocessing.Pool(processes=10)
    pool.map(wrapper, tasks)
    pool.close()


def wrapper(args):
    func, a, b, c =args
    return func(a,b,c)
       
def vis_boundary_offset(coco_path, save_path):
    coco = COCO(coco_path) if isinstance(json.load(open(coco_path)), dict) else None
    if not coco:
        js_anns = json.load(open(coco_path))
        imgs = []
        categories=[{'id': 1, 'name': 'building', 'supercategory': 'building'}]
        for ann in js_anns:
            img=dict(file_name = ann['file_name'],
                     id = ann['image_id'],
                     height = 1024,
                     width = 1024)
            imgs.append(img)
        coco = dict(images=imgs, annotations=js_anns, categories=categories)
        save_json('tmp.json',coco)
        coco = COCO('tmp.json')
    
    imgids = coco.getImgIds()
    
    tasks = [(_draw_boundary_offset, id, coco, save_path) for id in imgids]
    
    # [ _draw_boundary_offset(id, coco, save_path) for id in tqdm(imgids)]
    pool = multiprocessing.Pool(processes=40)
    pool.map(wrapper, tasks)
    pool.close()

def _draw_boundary_offset(id, coco:COCO, save_dir=SAVE_PATH):
    if not os.path.exists(save_dir):
        os.system('mkdir {}'.format(save_dir))
    img = coco.loadImgs(id)[0]
    annIds = coco.getAnnIds(id)
    # if img['file_name'] != 'L18_104536_210376__0_1024.png':
    #     return 
    anns = coco.loadAnns(annIds)
    offsets = [ann['offset'] for ann in anns]   
    offsets = np.array(offsets)
    offsets_len = np.sqrt(np.sum(offsets*offsets,axis=1))
    max_offset = np.max(offsets_len)
    offsets_gray = offsets_len/max_offset*230
    segs = np.zeros((1024,1024))
    buildings = []
    foot_segs = np.zeros((1024,1024))
    image = cv2.imread(os.path.join('/irsa/lk/BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test', img['file_name']))
    # for each building
    for i, ann in enumerate(anns):
        roof_seg = ann['roof_mask']
        # roof_seg = ann['roof_mask']
        # roof_seg = ANN_COCO.loadAnns(ann['id'])[0]['segmentation']
        seg = decode_rle(roof_seg) if isinstance(roof_seg, dict) else decode_polygon([roof_seg])
        seg = mask_to_boundary(seg, 0.002)
        # seg = mask_to_boundary(seg, 0.002)
        # draw prediction offset
        pred_offset = np.array(ann['offset'])
        bbox = ANN_COCO.loadAnns(ann['id'])[0]['roof_bbox']
        start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
        endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
        # cv2.arrowedLine(image,start, endpoint, (0,255,0),2,0,0,0.2)
        # # draw ann offset
        # pred_offset = np.array(ANN_COCO.loadAnns(ann['id'])[0]['offset'])
        # start = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
        # endpoint = (int(start[0]-pred_offset[0]),int(start[1]-pred_offset[1]))
        # cv2.arrowedLine(image,start, endpoint, (0,255,0),2,0,0,0.2)

        # # draw building
        segs[seg>0]=seg[seg>0]
        
        translation_matrix = np.float32([[1,0,-pred_offset[0]],[0,1,-pred_offset[1]]])
        seg_ = cv2.warpAffine(seg, translation_matrix, (1024,1024))
        # seg_[seg>0] = 0
        foot_segs[seg_>0]=seg_[seg_>0]        
        

    # foot_segs = mask_to_boundary(foot_segs, 0.002)
    image[foot_segs>0,1]=255
    image[foot_segs>0,0]=0
    image[foot_segs>0,2]=0
    # im = image*0.5+segs2*0.5
    # segs = mask_to_boundary(segs, 0.002)
    image[segs>0,1]=0
    image[segs>0,0]=0
    image[segs>0,2]=255
    cv2.imwrite(ospj(save_dir,img['file_name'] ),image.astype('uint8'))  

if __name__=='__main__':
    # p = 'double_obm_seg_b_all.json'
    # pred = json.load(open(p))
    
    ## compare_ordered_json(pred,'/config_data/BONAI_data/trainval-001/trainval/images/')
    # visual_pred(pred,'/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test', os.path.join('../vis',os.path.basename(p).replace('.json','')))
    # visual_pred(ann,'visual','visual')
    
    ## plot masks 
    """
    data_root = '/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/'
    citys=[ 'chengdu','shanghai', 'beijing', 'jinan', 'haerbin',]
    for city in citys:
        jsp = data_root + 'coco_building_seg/building_seg_bonai_{}_trainval.json'.format(city)
        
        visual_anns(jsp,
                '/config_data/BONAI_data/trainval-001/trainval/images',
                f'/config_data/BONAI_data/trainval-001/visual_train/{city}/')
    """
        
    ##  vis 3d
    # vis_3d('cascade_loft_r50_fpn_prompt.json', '/irsa/lk/BONAI_data/trainval-001/vis3d_cas_selfmasks')
    # vis_3d('loft_foa_r50_fpn_2x_bonai_prompt.json', '/irsa/lk/BONAI_data/trainval-001/vis3d_loft_selfmasks')
    # vis_3d('smlcdr_obm_pretrain.json', '/irsa/lk/BONAI_data/trainval-001/vis3d_obm_selfmasks')
    vis_boundary_offset('smlcdr_obm_pretrain.json', '/irsa/lk/BONAI_data/trainval-001/boundary_obm2')
    vis_boundary_offset('cascade_loft_r50_fpn_prompt.json', '/irsa/lk/BONAI_data/trainval-001/boundary_cas2')
    vis_boundary_offset('loft_foa_r50_fpn_2x_bonai_prompt.json', '/irsa/lk/BONAI_data/trainval-001/boundary_loft2')
    # vis_3d('/irsa/lk/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json', '/irsa/lk/BONAI_data/trainval-001/v3d_bon_gt')
    # vis_boundary_offset('/irsa/lk/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json', '/irsa/lk/BONAI_data/trainval-001/boundary_bon_gt')
    
 