import argparse
import os
import warnings
import sys
sys.path.append("./")
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', 
                        default='configs/loft_foa/loft_foa_r50_fpn_2x_bonai_huizhou_nms.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='../work_dirs/loft_foa_r50_fpn_2x_bonai/epoch_24.pth',
                        help='checkpoint file')
    parser.add_argument('--out', 
                        default='./loft_huizhou_nms.pkl',
                        help='output result file in pickle format')
    parser.add_argument('--merged-out', help='output merged result file in pickle format')
    parser.add_argument('--merge-iou-threshold', 
        type=float,
        default=0.1, 
        help='threshold of iou')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        default=True,
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default=False,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--city',
        type=str,
        default='huizhou', 
        help='dataset for evaluation')
    parser.add_argument('--show', 
                        default=True, 
                        action='store_true', help='show results')
    parser.add_argument(
        '--show-dir',
        default='./show_dir_soft_nms',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.7,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', 
            default=dict(jsonfile_prefix='./loft_huizhou'),
            nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--nms-score',
        type=float,
        default=0.5,
        help='nms threshold (default: 0.5)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.city == 'shanghai_xian':
        data_root = '../BONAI_data/BONAI-20230403T091731Z-002/BONAI/'
        cfg.data.test.ann_file = data_root + 'coco/bonai_shanghai_xian_test.json'
        cfg.data.test.img_prefix = data_root + 'test/test/'
        
    elif args.city == 'huizhou':
        data_root = '../BONAI_data/BONAI-20230403T091731Z-002/BONAI/'
        cfg.data.test.ann_file = data_root + 'coco/obm_huizhou_test.json'
        cfg.data.test.img_prefix = data_root + 'test/huizhou_test/'
        
    else:
        raise(RuntimeError("do not support the input city: ", len(args.city)))

    if cfg.test_cfg.get('rcnn', False):
        cfg.test_cfg.rcnn.nms.iou_threshold = args.nms_score
        print("NMS config for testing: {}".format(cfg.test_cfg.rcnn.nms))


    print("Dataset for evaluation: ", cfg.data.test.ann_file)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    main()
