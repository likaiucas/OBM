import os 
from tqdm import tqdm

config_list = os.listdir('/irsa/lk/BONAI2/configs/sensitive_analysis')
config_list = [cfg for cfg in config_list if ('cas' in cfg) or ('loft' in cfg) ]


ckpt_cas = '../work_dirs/cascade_loft_r50_fpn_3x/epoch_24.pth'
ckpt_loft = '../work_dirs/loft_foa_r50_fpn_2x_bonai/epoch_24.pth'
for cfg in tqdm(config_list):
    cfg = os.path.join('configs/sensitive_analysis', cfg)
    if 'cas' in cfg:
        os.system(f'python tools/test_offset.py --config={cfg} --checkpoint={ckpt_cas}')
    elif 'loft' in cfg:
        os.system(f'python tools/test_offset.py --config={cfg} --checkpoint={ckpt_loft}')