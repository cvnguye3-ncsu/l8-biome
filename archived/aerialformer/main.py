import os

import hydra
from omegaconf import DictConfig
import torch as pt

import AerialFormer
from AerialFormer import AerialFormer
from importables.pytorch.train import train_model

@hydra.main(config_path="../../conf", config_name="aerialformer", version_base="1.3")
def main(cfg: DictConfig):
    # --- MODEL ---
    model = AerialFormer(False, class_count=4)
    
    ext = cfg.weight_path.split('.')[-1]
    
    if ext == 'pth':
        weights = pt.load(cfg.weight_path)
        
        for key in weights.keys():
            print(key)
        
        model.encoder.load_state_dict(weights)
        model.encoder.load_state_dict()
        model.decoder.load_state_dict(weights) 
    
    # --- FITMENT ---
    model_path = os.path.abspath(AerialFormer.__file__)
    train_model(cfg, model, model_path)

if __name__ == '__main__':
    
    main()