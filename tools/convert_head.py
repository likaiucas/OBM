import torch
from mmcv.runner.checkpoint import _load_checkpoint
checkpoint = _load_checkpoint('/config_data/segment-anything/sam_vit_b_01ec64.pth')
ckpt = dict()
for k in checkpoint.keys():
    
    if 'mask_decoder' in k:
        n_k = k.replace('mask_decoder', 'mask_decoder1')
        n_k2 = k.replace('mask_decoder', 'mask_decoder2')
        
        ckpt[n_k]=checkpoint[k]
        ckpt[n_k2]=checkpoint[k]
    else:
        ckpt[k] = checkpoint[k]
torch.save('./pretrained/double_obm.pth', ckpt)
print('yes')