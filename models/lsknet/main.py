import os
import re
from pathlib import Path

import torch as pt

import hydra
from omegaconf import DictConfig

from importables.pytorch.train import train_model
from UNetFormer_lsk import UNetFormer_LSK_s

import UNetFormer_lsk

def change_tensor_names(weights: dict) -> None:
    pattern = re.compile(r"(patch_embed|block|norm)(\d)(.+)")
    
    for key in list(weights.keys()):
        key: str
        
        searched = pattern.search(key)
        if searched == None:
            continue
        
        module = searched.group(1)
        idx = int(searched.group(2)) - 1
        leftover = searched.group(3)
        
        match module:
            case 'patch_embed':
                new_name = f'patch_embeds.{idx}{leftover}'
            case 'block':
                new_name = f'stages.{idx}{leftover}'
            case 'norm':
                new_name = f'norms.{idx}{leftover}'
        
        tensor = weights.pop(key)
        weights[new_name] = tensor

@hydra.main(config_path="../../conf", config_name="lsknet", version_base="1.3")
def main(cfg: DictConfig):
    ext = cfg.weight_path.split('.')[-1]
    
    model = UNetFormer_LSK_s(resize_embedding=cfg.resize_embedding,
                             freeze_encoder_layers=cfg.frozen_layers)
    
    weights = pt.load(cfg.weight_path, map_location='cuda', weights_only=False)
    weights: dict = weights['state_dict']
    
    match ext:
        case 'pth' | 'pt':
            change_tensor_names(weights)
            
            weights.pop('head.weight')
            weights.pop('head.bias')
    
            model.backbone.load_state_dict(weights)
            
        case 'ckpt':
            model.load_state_dict(weights)
    
    model_path = Path(os.path.abspath(UNetFormer_lsk.__file__))
    train_model(cfg, model, model_path)

if __name__ == '__main__':
    main()