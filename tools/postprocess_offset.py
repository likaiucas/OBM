import json
import numpy as np
import math
import os

def convert2cocostyle(js, w, h):
    images = []
    annotations = []
    check = []
    for i in js:   
        if i['image_id'] not in check:
            images.append({'id':i['image_id'],
                       'file_name':i['file_name'],
                       'height':w,
                       'width':h})
            check.append(i['image_id'])
        annotations.append({'id':i['id'],
                            'image_id':i['image_id'],
                            'category_id':1,
                            'segmentation':[],
                            'area':0,
                            'building_bbox':i['building_bbox'],
                            'offset':i['offset'],
                            'bbox':[]})
    coco_dict = {'images':images,
                'annotations':annotations,
                'categories':[{'id':1,'name':'building'}]}
    return coco_dict

def systh_json(js:list, ann_js:list):
    offsets = []
    lengths = []
    angles = []
    offset = None
    ims = set()
    for i, j in zip(js, ann_js):
        if i['image_id'] not in ims:
            if offset is not None:
                offsets.append(np.array(offset))
                lengths.append(np.array(length))
                angles.append(np.array(angle))
            ims.add(i['image_id'])
            offset = []
            length = []
            angle = []
        
        offset_x, offset_y = i['offset'][0], i['offset'][1]
        agl1 = math.atan2(offset_y, offset_x)
        offset_x, offset_y = j['offset'][0], j['offset'][1]
        agl2 = math.atan2(offset_y, offset_x)
        angle.append([agl1, agl2])
        i['offset'].extend(j['offset'])
        offset.append(i['offset'])
        
        length.append([np.sqrt(i['offset'][1]*i['offset'][1]+i['offset'][0]*i['offset'][0]), 
                    np.sqrt(j['offset'][1]*j['offset'][1]+j['offset'][0]*j['offset'][0])])
    return offsets, lengths, angles

def fixangle(angle=None, length=None, offset=None, model='guassia_std'):
    assert model in ['max', 'linear_mean', 'guassia_std', 'average_mean']
    def gaussian_kernel(x, y):
        y = y[None,:].repeat(loc_pred.shape[0],0)
        # x向量的长度
        x_length = np.linalg.norm(x, axis=1)
        y_length = np.linalg.norm(y, axis=1)
        # 计算样本之间的欧氏距离
        distance = x_length - y_length
        sigma = np.std(distance)
        # 计算高斯核函数的值
        kernel_value = np.exp(-distance**2 / (2 * sigma*sigma))
        return kernel_value
    
    mean = []
    
    if model =='guassia_std':
        fixed = []
        for loc in offset:
            loc_pred, loc_true = loc[:,:2], loc[:,2:]
            loc_pred_mean = np.mean(loc_pred, axis=0)
            len_of_pred = (loc_pred_mean[0]*loc_pred_mean[0]+loc_pred_mean[1]*loc_pred_mean[1])**0.5  
            loc_pred_std = np.std(loc_pred, axis=0)
            ad_mean = loc_pred_mean+loc_pred_std*0.5
            kernel_value = gaussian_kernel(loc_pred, ad_mean)
            weight = kernel_value[:,None].repeat(2,1)
            loc_pred_mean = loc_pred_mean/len_of_pred
            loc_pred_mean = loc_pred_mean[None,:].repeat(loc_pred.shape[0],0)
            len_pred = np.sqrt(np.sum(loc_pred*loc_pred,axis=1)[:,None]).repeat(2,1)
            loc_pred = loc_pred*(1-weight)+weight*loc_pred_mean*len_pred
            fixed.append(loc_pred)
        return fixed
    
    if model =='average_mean':
        fixed = []
        for loc in offset:
            loc_pred, loc_true = loc[:,:2], loc[:,2:]
            loc_pred_mean = np.mean(loc_pred, axis=0)
            len_of_pred = (loc_pred_mean[0]*loc_pred_mean[0]+loc_pred_mean[1]*loc_pred_mean[1])**0.5  

            loc_pred_mean = loc_pred_mean/len_of_pred
            loc_pred_mean = loc_pred_mean[None,:].repeat(loc_pred.shape[0],0)
            len_pred = np.sqrt(np.sum(loc_pred*loc_pred,axis=1)[:,None]).repeat(2,1)
            loc_pred = loc_pred*0.5+0.5*loc_pred_mean*len_pred
            fixed.append(loc_pred)
        return fixed  
            
    if model=='linear_mean':
        fixed=[]
        for loc in offset:
            loc_pred, loc_true = loc[:,:2], loc[:,2:]
            loc_pred_mean = np.mean(loc_pred, axis=0)
            len_of_pred = (loc_pred_mean[0]*loc_pred_mean[0]+loc_pred_mean[1]*loc_pred_mean[1])**0.5  
            loc_pred_mean = loc_pred_mean/len_of_pred 
            loc_pred_mean = loc_pred_mean[None,:].repeat(loc_pred.shape[0],0)
            loc_pred= loc_pred_mean*np.sqrt(np.sum(loc_pred*loc_pred,axis=1)[:,None]).repeat(2,1)
            fixed.append(loc_pred)
            loc_true[np.isnan(loc_true)] = 0
            a = np.arccos(sum(loc_pred*loc_true)/(np.linalg.norm(loc_pred) * np.linalg.norm(loc_true)))
            # 计算两个向量的夹角
            # a = sum(loc_pred*loc_true)/(np.linalg.norm(loc_true) * np.linalg.norm(loc_pred))
            mean.extend(a.tolist())
        return fixed
            
    if model=='max':
        fixed = []
        for offset in offsets:
            pred, true = offset[:,:2], offset[:,2:]
            len_offset = np.sqrt(np.sum(pred*pred,axis=1))
            loc = np.where(len_offset==len_offset.max())
            loc = loc if not isinstance(loc, tuple) else loc[0][0]
            std_offset = pred[loc]/len_offset[loc]
            offset = np.array([n*std_offset for n in len_offset])
            fixed.append(offset)
        return fixed
    
        # for ag, lenth in zip(angle, length):
        #     ag0 = ag[:,0]
        #     le0 = lenth[:,0]
        #     loc = np.where(le0==le0.max())
        #     ag_std = ag0[loc][0]
        #     ag0 = ag0/ag0*ag_std
        #     ag[:,0] = ag0
        #     mean.extend((ag0-ag[:,1]).tolist())

    for ag, lenth in zip(angle, length):
        # 极坐标系下坐标转为笛卡尔坐标系下坐标
        x1 = lenth[:,0]*np.cos(ag[:,0])
        y1 = lenth[:,0]*np.sin(ag[:,0])
        x2 = lenth[:,1]*np.cos(ag[:,1])
        y2 = lenth[:,1]*np.sin(ag[:,1])
        if np.isnan(x1).any():
            print('no')
        # 计算两个点的夹角
        agl = np.arccos((x1*x2+y1*y2)/(lenth[:,0]*lenth[:,1]))
        # 计算两个点的距离
        dis = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        ag[:,0] = x1
        ag[:,1] = y1    
        lenth[:,0] = x2
        lenth[:,1] = y2
        # mean.extend(agl.tolist())
            
    mean = np.array(mean)
    mean = mean[~np.isnan(np.array(mean))].mean()
    return angle

def fixoffsets_js(js_list, fixed):
    fixed2 = []
    for i in fixed:
        fixed2.extend(i.tolist())
    for i, j in zip(js_list, fixed2):
        i['offset'] = j[:2]
    return js_list
    
if __name__=="__main__":
    json_path = 'loft_foa_r50_fpn_2x_bonai_prompt_huizhou_test.json'
    js = json.load(open(json_path))
    # json_path2 = '../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json' 
    # ann_js = json.load(open(json_path2))
    offsets, lengths, angles = systh_json(js, js)
    offset = fixangle(offset = offsets, model='max')
    js_fixed = fixoffsets_js(js, offset)
    
    json.dump(js_fixed, open('mx_'+os.path.basename(json_path),'w'))
    print('yes')
    
    # coco_dict = convert2cocostyle(js, 1024, 1024)
    # json.dump(coco_dict, open('polar_max_out_coco.json','w'))
    