from mmcv.runner import auto_fp16
import torch
from ..builder import DETECTORS, build_head
from mmdet.core import build_sampler
from torch import nn
from torch.nn import functional as F
import random
from mmdet.models.roi_heads.mask_decoder_hq import MaskDecoderHQ
from mmdet.utils import get_root_logger
from typing import Any, Dict, List, Tuple
from ..builder import DETECTORS
from .base import BaseDetector
from .sam import sam_model_registry, _build_sam

@DETECTORS.register_module()
class obm_hq(BaseDetector):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(self, 
                model_type, 
                ckpt, 
                hq_decoder:MaskDecoderHQ,
                offset_sampler = dict(
                    type='OffsetSampler',
                    max_num = 40,
                    t_len = 24, # lower the prompt whose length is shorter than t_len)
                    ),
                train_cfg=None,
                test_cfg=None,
                pixel_mean = [0, 0, 0], 
                pixel_std = [1, 1, 1],
                input_keys = ['box']#,'point'
        ): 
        super().__init__()
        assert model_type in ["vit_b","vit_l","vit_h"]
        self.input_keys=input_keys
        self.sampler = build_sampler(offset_sampler)
        # activation=nn.GELU
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.obm_hq_decoder = hq_decoder if isinstance(hq_decoder, MaskDecoderHQ) else build_head(hq_decoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) :#-> Any
        return self.pixel_mean.device
    def aug_test(self,):
        pass
    def extract_feat(self):
        pass
    def simple_test(self):
        pass
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
    def forward_test(self,                       
                     img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert gt_masks or gt_bboxes
        assert len(gt_bboxes)==1
        x = self.image_encoder.sam_hq_forward(img[0])
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=gt_bboxes[0][0],
                masks=None,
            )
        masks_hq, offsets = self.obm_hq_decoder(
                image_embeddings=x,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
        masks = masks>self.mask_threshold
        return gt_bboxes[0][0], None, offsets
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_building_masks=None,
                      gt_masks=None,
                      gt_offsets=None,
                      **kwargs):
        assert gt_masks or gt_bboxes
        assert len(gt_bboxes)==1
        gt_offsets, gt_bboxes, gt_labels, gt_building_masks, gt_masks = self.sampler.sample(
            gt_offsets, gt_bboxes, gt_labels, gt_building_masks, gt_masks
        )
        roof_box = masks_to_boxes(self.to_tensor(gt_masks[0], img.device))
        try:
            roof_points = masks_sample_points(self.to_tensor(gt_masks[0], img.device),1)
            input_keys = self.input_keys
        except:
            input_keys = ['box']
        
        roof_masks_256= self.get_masks(gt_masks[0], img.device, (256,256))

        noise_roof_masks = masks_noise(roof_masks_256.unsqueeze(0).float())
        
        batched_input = []
        for b_i in range(len(img)):
            dict_input = dict()
            input_image = img[b_i].contiguous()
            dict_input['image'] = input_image 
            input_type = random.choice(input_keys)
            if input_type == 'box':
                dict_input['boxes'] = roof_box#[b_i:b_i+1]
            elif input_type == 'point':
                point_coords = roof_points
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[0], device=point_coords.device)[:,None]
            elif input_type == 'noise_mask':
                dict_input['mask_inputs'] = noise_roof_masks[b_i:b_i+1]
            elif input_type == 'point_box':
                point_coords = roof_points
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[0], device=point_coords.device)[:,None]
                dict_input['boxes'] = roof_box
            else:
                raise NotImplementedError 
            dict_input['original_size'] = img[b_i].shape[:2]
            batched_input.append(dict_input)
        
        with torch.no_grad():
            # batched_output, interm_embeddings = self.sam.image_encoder(batched_input, multimask_output=False)
            image_embeddings, interm_embeddings = self.image_encoder.sam_hq_forward(img)
            if "point_coords" in dict_input:
                points = (dict_input["point_coords"], dict_input["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=dict_input.get("boxes", None),
                masks=dict_input.get("mask_inputs", None),
            )
            
        encoder_embedding = image_embeddings
        image_pe = self.prompt_encoder.get_dense_pe()
        
        masks_hq, offsets = self.obm_hq_decoder(image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,)
        loss_mask, loss_dice = loss_masks(masks_hq, roof_masks_256.unsqueeze(1).float(), len(masks_hq)) # (17, 1, 256, 256) (17, 256, 256)  17
        loss_offset = self.obm_hq_decoder.loss_offset(offsets, gt_offsets[0])
        
        return {"loss_mask": loss_mask, "loss_dice":loss_dice, 'loss_offset':loss_offset}
                
    def to_tensor(self, masks, device):
        return torch.from_numpy(masks.masks).to(device)

    def get_masks(self, masks, device, size):
        mask = torch.from_numpy(masks.masks).to(device)
        mask = F.interpolate(mask[:,None,:,:], size=size, mode='nearest')
        return mask.squeeze(1)>0



def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    # No need to upsample predictions as we are using normalized coordinates :)

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice





