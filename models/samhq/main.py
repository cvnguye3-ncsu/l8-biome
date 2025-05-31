from functools import partial
from pathlib import Path
from typing import Callable

import torch as pt
import torchvision.transforms.v2 as transform
import pytorch_lightning as L

import pandas as pd

from sam import Sam
from image_encoder import ImageEncoderViT
from prompt_encoder import PromptEncoder
from mask_decoder_hq import MaskDecoderHQ
from transformer import TwoWayTransformer

import hydra
from omegaconf import DictConfig

from segment_anything_hq import SamPredictor

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator
from randomgen import ChaCha

from PIL import Image
import cv2

from importables.project.cloud_classes import CloudClass
from importables.pytorch.metrics import SemanticSegmentationMetrics

from tqdm import tqdm

# Paths 
# -----
DATA_PATH = Path('../../_data/')
SEED_PATH = DATA_PATH / 'seeds'
DS_PATH = DATA_PATH / 'dataset'
IMG_PATH = DS_PATH / 'img'
LBL_PATH = DS_PATH / 'label'

# Constants
# ---------
CLASSES = [CloudClass.CLEAR,
           CloudClass.CLOUD,
           CloudClass.THIN_CLOUD,
           CloudClass.CLOUD_SHADOW]

IMG_SIZE = 224
upsize: Callable[[NDArray], NDArray] = partial(cv2.resize, dsize=(256, 256), interpolation=cv2.INTER_NEAREST_EXACT)

# CUBIC is pretty 'glith grainy'.
# Linear is pretty reasonable.
# Nearest is close to Linear, with a touch of grain.
downsize: Callable[[NDArray], NDArray] = partial(cv2.resize, dsize=(224, 224), interpolation=cv2.INTER_LINEAR_EXACT)

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    # checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(pt.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    
    for n, p in sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False

    return sam

def SAM_predictor(weight_path: str, seed: int) -> tuple[SamPredictor, Generator]:
    L.seed_everything(seed)
    weights = pt.load(weight_path, map_location='cuda')
    
    sam =_build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    ).cuda()
    sam.load_state_dict(weights)
    
    sam_predictor = SamPredictor(sam)
    rng = Generator(ChaCha(seed))
    
    return sam_predictor, rng
    
def SAM_predict(img: NDArray, lbl: NDArray, num_points_sample: int, 
                sam_predictor: SamPredictor, rng: Generator) -> NDArray:
    img = upsize(img)
    lbl = upsize(lbl)
    
    sam_predictor.set_image(img)
    
    categories, counts = np.unique(lbl, return_counts=True)
    
    if categories.size == 1:
        return np.full((224, 224), categories.item(), dtype=np.long)
    
    logits = np.full((4, 224, 224), -np.inf)
    
    for i in range(categories.size):
        cat = categories[i]
        fore_count = counts[i]
        back_count = np.sum(counts[categories != cat])
        
        # Get foreground pixels
        class_positions = np.argwhere(lbl == cat)[:, [1,0]]
        class_pixels = rng.choice(class_positions, min(fore_count, num_points_sample), replace=False)
        class_pixel_labels = np.ones((class_pixels.shape[0]), dtype=np.long)
        
        # Get background pixels
        notclass_positions = np.argwhere(lbl != cat)[:, [1,0]]
        notclass_pixels = rng.choice(notclass_positions, min(back_count, num_points_sample), replace=False)
        notclass_pixel_labels = np.zeros((notclass_pixels.shape[0]), dtype=np.long)
        
        pixels = np.concat((class_pixels, notclass_pixels), axis=0)
        pixel_labels = np.concat((class_pixel_labels, notclass_pixel_labels), axis=0)

        # Predictions
        _, _, class_logit = sam_predictor.predict(pixels, pixel_labels, hq_token_only=True)
        class_logit = downsize(class_logit.transpose(1, 2, 0))
        
        logits[cat, :, :] = class_logit
        
    return logits.argmax(axis=0)

@hydra.main(config_path="../../conf", config_name="samhq", version_base="1.3")
def main(cfg: DictConfig):
    predictor, rng = SAM_predictor(cfg.weight_path, cfg.seed)
    sam_predict = partial(SAM_predict, num_points_sample=cfg.num_sample_points,
                          sam_predictor=predictor, rng=rng)
    
    seed_df = pd.read_csv(SEED_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv', index_col=0)
    seed_df = seed_df[seed_df['split'] == 'test']
    seed_df = seed_df[seed_df['subset'] == True]
    
    trans = transform.Compose([
        transform.ToImage(),
        transform.ToDtype(pt.long)
    ])
    
    metrics = SemanticSegmentationMetrics()
    
    for _, row in tqdm(list(seed_df.iterrows()),
                       desc='Applying SAM . . .', unit='image'):
        label = Image.open(LBL_PATH / row['image_name'])
        img = Image.open(IMG_PATH / row['image_name'])
        
        pred = sam_predict(np.array(img), np.array(label))
        metrics.update(trans(pred), trans(label))
        
    perf_df = metrics.compute()
    
    print(perf_df.to_latex(index=True, 
                           caption="Class Metrics", label="tab:class_metrics", float_format="%.2f"))

if __name__ == '__main__':
    main()