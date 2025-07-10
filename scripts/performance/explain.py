from pathlib import Path
from functools import partial

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import torch as pt
import torch.nn.functional as F

from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr import LimeBase

import hydra
from omegaconf import DictConfig

from importables.pytorch.dataset import SegmentationDataset
from importables.project.cloud_classes import CloudClass

from pysnic.algorithms.snic import snic

import cv2

import pandas as pd

from tqdm import tqdm

# Paths 
# -----
BASE_PATH = Path('./')

DATA_PATH = BASE_PATH / '_data'
DS_PATH = DATA_PATH / 'dataset'
SEEDS_PATH = DATA_PATH / 'seeds'
XAI_PATH = DATA_PATH / 'auxiliary' / 'xai'
XAI_PATH.mkdir(exist_ok=True)

PLOTS_PATH = BASE_PATH / 'plots'
XAI_PLOTS_PATH = PLOTS_PATH / 'xai'
XAI_PLOTS_PATH.mkdir(exist_ok=True)

# Pytorch
# -------
IMG_NAME = 'LC81010142014189LGN00_26_13'
IMG_NAME = 'LC81930452013126LGN01_21_14'
IMG_NAME = 'LC81750512013208LGN00_10_8'

SEG_CT = 49
COMPACT = 8

MODELS = [
    ('prithvi', 'finetuning', 'Prithvi', 8),
    ('satlas', 'finetuning', 'SatlasNet', 16),
    ('lsknet', 'finetuning', 'LSKNet', 32)
]

# Helper functions
# ----------------
def binarize_mask(lbl: pt.Tensor, cat: int) -> pt.Tensor:
    mask = pt.zeros_like(lbl)
    mask[lbl == cat] = 1
    
    return mask

def class_logit_agg(model, cat: int, mask: pt.Tensor, orig_logits: pt.Tensor, 
                    input: pt.Tensor) -> pt.Tensor:
    logits: pt.Tensor = model(input)
    logits = F.softmax(logits, dim=1) - F.softmax(orig_logits, dim=1)
    logits = logits[:, cat, :, :]
    
    return logits[:, mask].mean(dim=1) - logits[:, ~mask].mean(dim=1)

# Plotting
# --------
def boost_hsv(img: np.ndarray, sat_gamma: float = 1, val_gamma: float = 1) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Must be RGB, [0, 1] each band.
        sat_gamma (float): Gamma for saturation band.
        val_gamma (float): Gamma for value band.

    Returns:
        np.ndarray: RGB. np.float32. [0,1].
    """
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # H: [0, 360], S: [0, 1], V: [0, 1]
    
    s, v = s**sat_gamma, v**val_gamma

    hsv = cv2.merge([h, s, v])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb

# LIME
# ----
def sp_binomial_sampler(_, *, M, B, sigma):
    """
    returns z′  of shape 1×M   where each super-pixel is kept (1) or dropped (0)
    """
    p = pt.sigmoid(pt.tensor(1.0 / sigma**2, device='cuda'))
    output = pt.rand(B, M, device='cuda') < p
    
    return output

def sp_from_interp(z_prime: pt.Tensor, original_img, *, baseline, seg_map, **kwargs):
    """
    z_prime_vec : 1×M   binary mask
    original_img: C×H×W original
    returns      : C×H×W perturbed image for the forward pass
    """
    
    batch_idx = pt.arange(kwargs['B'], device='cuda')[:, None, None, None]
    batch_idx = batch_idx.expand_as(seg_map)
    z_p_batch = z_prime[batch_idx, seg_map]
    
    return pt.where(z_p_batch, original_img, baseline)

def combine_segmentation_masks(mask1: pt.Tensor,
                               mask2: pt.Tensor) -> pt.LongTensor:
    """
    Given two segmentation masks of identical shape, returns a new mask
    whose labels are the Cartesian product of the inputs:
      new_label = f(old1, old2)
    such that every distinct (old1, old2) pair maps to a distinct 0..N-1 ID.

    Args:
      mask1, mask2: LongTensors of shape (H, W) or (B, H, W)
                    with non-negative integer class IDs.

    Returns:
      new_mask: LongTensor of the same shape, with values in [0 .. C1*C2-1],
                but only as many distinct labels as actually appear.
    """
    # flatten spatial (and batch) dims
    flat1 = mask1.reshape(-1)
    flat2 = mask2.reshape(-1)

    # choose an offset larger than any label in mask2
    offset = int(mask2.max().item()) + 1

    # encode each pair as a single integer
    pair_idx = flat1 * offset + flat2   # unique if offset > max(mask2)

    # find all unique pair-indices and remap to [0..K-1]
    unique_pairs, new_flat = pt.unique(pair_idx, sorted=True, return_inverse=True)

    # reshape back to original
    new_mask = new_flat.reshape(mask1.shape)
    
    return new_mask

# Main
# ----
@hydra.main(config_path=str(BASE_PATH / '..' / '..' / 'conf'), config_name="main_config", version_base="1.3")
def main(cfg: DictConfig):
    df = pd.read_csv(SEEDS_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv', index_col=0)
    
    df = df[(df['split'] == 'test') & df['subset'] == True]
    df = df[df['biome'] != 'Snow/Ice']
    df = df[df['clear'] >= .2]
    df = df[df['shadow'] >= .1]
    df = df[(df['cloud'] + df['thin_cloud'] - df['shadow']).abs() <= .2]
    
    def uniform_similarity(*args, **kwargs): return 1.0
    
    B = 1
    TRIALS = 2**11
    
    surrogate = SkLearnLinearModel("linear_model.Ridge", alpha=1.0)
    
    print('Total images: ', len(df))
    
    for _, row in tqdm(list(df.iterrows()),
                         desc='Calculating GLIME attribution', unit='image'):
        results_df = pd.DataFrame(columns=['Prithvi', 'SatlasNet', 'LSKNet'])
        
        IMG_NAME = row['image_name'].split('.')[0]
        
        # --- Initialize ---
        def get_model_file(model_name: str, version: str):
            return BASE_PATH / 'models' / model_name / 'logs' / f'seed-{cfg.seed}' / version / 'model.pt'
        
        ds = SegmentationDataset(cfg.seed, 'test', str(DATA_PATH), subset_ratio=cfg.subset_ratio)
        
        # --- Image ---
        img = Image.open(DS_PATH / 'img' / f'{IMG_NAME}.png')
        # img_arr = np.array(img)
        img_tensor: pt.Tensor = ds.img_transform(img).cuda().unsqueeze(0).detach()
        
        rgb = boost_hsv(np.array(img).astype(np.float32) / 255,
                        sat_gamma=.8, val_gamma=.8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        
        # --- Label ---
        lbl = Image.open(DS_PATH / 'label' / f'{IMG_NAME}.png')
        lbl_tensor: pt.Tensor = ds.mask_transform(lbl).cuda().detach()
        # lbl_arr = np.array(lbl)
        # lbl_arr_pretty = CLASS_REG.class_recolor_map(lbl_arr)
        
        # --- Per-class labels ---
        land_mask = binarize_mask(lbl_tensor, CloudClass.CLEAR.value)
        land_mask = land_mask.squeeze(0).bool()
        
        cloud_mask = binarize_mask(lbl_tensor, CloudClass.CLOUD.value)
        cloud_mask = cloud_mask.squeeze(0).bool()
        
        thin_cloud_mask = binarize_mask(lbl_tensor, CloudClass.THIN_CLOUD.value)
        thin_cloud_mask = thin_cloud_mask.squeeze(0).bool()
        
        cloud_shadow_mask = binarize_mask(lbl_tensor, CloudClass.CLOUD_SHADOW.value)
        cloud_shadow_mask = cloud_shadow_mask.squeeze(0).bool()
        
        cloud_full_mask = cloud_mask | thin_cloud_mask
        
        # --- Baseline ---
        snic_seg_map = pt.tensor(snic(lab, SEG_CT, COMPACT)[0]).cuda()
        seg_map = combine_segmentation_masks(snic_seg_map, lbl_tensor)
        seg_map = pt.stack(3 * [seg_map], dim=0).unsqueeze(0)
        
        seg_ids = seg_map.unique()            
        M = len(seg_ids)
        
        # baseline = pt.zeros_like(img_tensor)
        baseline = img_tensor[:, :, land_mask].mean(dim=2)
        baseline = baseline.reshape(1, 3, 1, 1).expand(1, 3, 224, 224)
        
        sampler = partial(sp_binomial_sampler, M=M, B=B)
        from_interp = partial(sp_from_interp, baseline=baseline, seg_map=seg_map)

        for name, ver, title, sample_size in MODELS:
            pt.manual_seed(cfg.seed)
            
            model_path = get_model_file(name, ver)
            model: pt.jit.ScriptModule = pt.jit.load(model_path, map_location='cuda').eval()
            logits = model(img_tensor)
            forward_pass = partial(class_logit_agg, model, CloudClass.CLOUD_SHADOW.value, 
                                cloud_shadow_mask, logits)
            
            explainer = LimeBase(forward_func=forward_pass,
                                interpretable_model=surrogate,
                                similarity_func=uniform_similarity,
                                perturb_func=sampler,
                                perturb_interpretable_space=True,
                                from_interp_rep_transform=from_interp,
                                to_interp_rep_transform=None)
            
            attr = explainer.attribute(
                inputs=img_tensor,
                n_samples=TRIALS,
                perturbations_per_eval=sample_size,
                # show_progress=True,
                sigma=.5, B=B,
            ).cuda()
            
            attr = (attr - (attr_min:=attr.min()))/(attr.max() - attr_min)
            
            batch_idx = pt.arange(B, device='cuda')[:, None, None, None]
            batch_idx = batch_idx.expand_as(seg_map)
            attr = attr[batch_idx, seg_map].mean(dim=1).squeeze(0)
            
            col = [
                land_score:=attr[land_mask].mean().cpu().item(),
                cloud_score:=attr[cloud_full_mask].mean().cpu().item(),
                cs_score:=attr[cloud_shadow_mask].mean().cpu().item(),
            ]
            
            if land_score >= cs_score or cloud_score >= cs_score:
                print(f'Aborting GLIME on image {IMG_NAME}')
                break
                
            results_df[title] = col
            
            attr = attr.cpu().numpy()
            np.save(XAI_PATH / 'images' / f'{IMG_NAME}_{name}', attr)
        
        if results_df.count().sum() == 9:
            results_df.index = ['Land', 'Cloud', 'Cloud-shadow']
            results_df.to_csv(XAI_PATH / 'tables' / f'{IMG_NAME}.csv')

if __name__ == '__main__':
    main()