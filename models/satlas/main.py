import os
from pathlib import Path
from enum import Enum, auto

import torch as pt

import hydra
from omegaconf import DictConfig

from importables.pytorch.train import train_model

from SatlasNet import Model
import models.satlas.SatlasNet as SatlasNet

class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()
    
@hydra.main(config_path="../../conf", config_name="satlas", version_base="1.3")
def main(cfg: DictConfig):
    ext = cfg.weight_path.split('.')[-1]
    
    if ext == 'pt' or ext == 'pth':
        weights = pt.load(cfg.weight_path, weights_only=False, map_location='cuda')
    elif ext == 'ckpt':
        ckpt = pt.load(cfg.weight_path, weights_only=False, map_location='cuda')
        weights = ckpt['state_dict']

    # Initialize a pretrained model using the Model() class.
    model = Model(num_categories=4, fpn=cfg.fpn,
                  weights=weights,
                  freeze_encoder_layers=cfg.frozen_layers,
                  resize_embedding=cfg.resize_embedding)

    # FITMENT
    # -------
    model_path = Path(os.path.abspath(SatlasNet.__file__))
    train_model(cfg, model, model_path)

if __name__ == '__main__':
    main()
