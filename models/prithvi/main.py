import os

from collections import OrderedDict
from omegaconf import DictConfig
import torch as pt

import hydra

import Prithvi
from Prithvi import MaskedAutoencoderViT
from importables.pytorch.train import train_model

def load_custom_weights(model: pt.nn.Module, state_dict):
    model_dict = model.state_dict()
    filtered_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                print(f"Skipped: {k} due to shape mismatch {v.shape} != {model_dict[k].shape}")
        else:
            print(f"Skipped: {k} not found in model")

    model.load_state_dict(filtered_dict, strict=False)


@hydra.main(config_path="../../conf", config_name="prithvi", version_base="1.3")
def main(cfg: DictConfig):
    # --- MODEL ---
    model = MaskedAutoencoderViT(
        depth=12,
        freeze_encoder_layers=cfg.frozen_layers,
        resize_embedding=cfg.resize_embedding)

    ext = cfg.weight_path.split('.')[-1]
    
    if ext == 'pt':
        weights: OrderedDict = pt.load(
            cfg.weight_path, weights_only=True,
        )
        full_keys = [(key, weights[key])
                    for key in weights.keys() if 'decoder' not in key]
        full_keys = OrderedDict(full_keys)
        full_keys.pop('mask_token')
        
        load_custom_weights(model, full_keys)
    elif ext == 'ckpt':
        ckpt = pt.load(cfg.weight_path, map_location='cuda', weights_only=False)
        weights = ckpt['state_dict']
        model.load_state_dict(weights)

    # --- FITMENT ---
    model_path = os.path.abspath(Prithvi.__file__)
    train_model(cfg, model, model_path)

if __name__ == '__main__':
    main()