import argparse
import os
import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from copy import copy
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

class OffsetTest(object):
    def __init__(self, model):
        self.model=model
        self.feature_map=None
        
    def set_image(self, image):
        if isinstance(image, str):
            self.img_meta = {'filename': image, 
                        'ori_filename': os.path.basename(image),  
                        'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28, 103.53]), 
                        'std': np.array([58.395, 57.12, 57.375]), 'to_rgb': True}}
            image = mmcv.imread(image)
            image= mmcv.imnormalize(image, self.img_meta['img_norm_cfg']['mean'], self.img_meta['img_norm_cfg']['std'],
                                            self.img_meta['img_norm_cfg']['to_rgb'])
            self.img_meta['ori_shape']=image.shape
            self.img_meta['img_shape']=image.shape
            self.img_meta['pad_shape']=image.shape
            self.image = torch.from_numpy(image)
            
        if isinstance(image, np.ndarray):
            self.img_meta = {'filename': './demo/tmp.png', 
                        'ori_filename': 'tmp.png',  
                        'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28, 103.53]), 
                        'std': np.array([58.395, 57.12, 57.375]), 'to_rgb': True}}
            image= mmcv.imnormalize(image, self.img_meta['img_norm_cfg']['mean'], self.img_meta['img_norm_cfg']['std'],
                                            self.img_meta['img_norm_cfg']['to_rgb'])
            self.image=torch.from_numpy(image)            
            
        self.image=self.image.unsqueeze(0)
        self.image=self.image.float()
        self.image=self.image.cuda()
        self.image=self.image.permute(0,3,1,2)
        with torch.no_grad():
            self.feature_map=self.model.extract_feat(self.image)
        print('successfully set image')
        
    def inference(self, prompt_bbox):
        assert self.feature_map is not None # run set_image first
        with torch.no_grad():
            proposal_list = prompt_bbox
            feature_map = copy(self.feature_map)
            feature_map = [f.contiguous() for f in feature_map]
            results = self.model.roi_head.aug_test2(feature_map, proposal_list, [self.img_meta], rescale=True)
        bbox, _, offset = results
        return bbox, offset 
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', 
                        default='configs/cascade_loft_prompt/cascade_loft_r50_fpn_prompt.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='work_dirs/cascade_loft_r50_fpn_noscale/epoch_24.pth',
                        help='checkpoint file')
    parser.add_argument('--image',
                        default='/config_data/BONAI/tests/data/buildings.png',
                        help='inference image')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    # if not distributed:
    #     model = MMDataParallel(model, device_ids=[0])
    model.eval()
    model = OffsetTest(model.cuda())
    model.set_image(args.image)
    #[606, 510, 678, 578]559, 607, 639, 704
    p = [
         [644, 386, 80, 84],
         [556, 366, 72, 70],
        
         ]
    pp2 = [[pp[0], pp[1], pp[0]+pp[2], pp[1]+pp[3], 1.] for pp in p]
    bbox, offset = model.inference([torch.Tensor(pp2).cuda()])
    print(f'bbox:\n{bbox}')
    print(f'offset:\n{offset}')
    
if __name__ == '__main__':
    main()