import os
from pathlib import Path
from enum import Enum, auto
from itertools import chain

import torch as pt
from torch import Tensor

import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate as hydra_instant

from importables.general.instantiators import instantiate_callbacks
from importables.general.evaluation import plot_curves, tabulate_metrics, plot_confusion_matrix, tabulate_per_image_metrics
from importables.general.cloud_classes import ClassRegistry 

IMG_SIZE = 224
    
@hydra.main(version_base="1.3", config_path="./conf", config_name="main")
def main(cfg: DictConfig):
    device = cfg.training.trainer.accelerator
    
    ext = cfg.weight_path.split('.')[-1]
    
    if ext == 'pt' or ext == 'pth':
        weights = pt.load(cfg.weight_path, weights_only=False, map_location=device)
    elif ext == 'ckpt':
        ckpt = pt.load(cfg.weight_path, weights_only=False, map_location=device)
        weights = ckpt['state_dict']

    model = Model(num_categories=4, fpn=cfg.fpn,
                  weights=weights,
                  freeze_encoder_layers=cfg.frozen_layers,
                  resize_embedding=cfg.resize_embedding)
    model_path = Path(os.path.abspath(SatlasNet.__file__))
    
    # --- setup ---
    L.seed_everything(cfg.seed, workers=True)
    if cfg.gpu_compile: model.compile()
    pt.set_float32_matmul_precision(cfg.matmul_precision)
    
    model.opt_params = OmegaConf.to_container(cfg.training.optimizer)
    model.sched_params = OmegaConf.to_container(cfg.training.scheduler)
    model.log_params = {
        'batch_size': cfg.data.loader.train.batch_size * cfg.training.trainer.accumulate_grad_batches,
        'sync_dist': ...
    }
    
    class_weights = pt.load(Path(cfg.data_folder) / 'seeds' / f'{cfg.seed}_{cfg.subset_ratio}_class_balance.pt', map_location=device)
    model.class_weights = pt.sqrt(class_weights).float()
    
    class_reg: ClassRegistry = hydra_instant(cfg.training.classes)
    model.reduce_func = class_reg.reduce_classes
    model.reduce_matrix = class_reg.REDUCE_MATRIX_PYTORCH.to(device)
    class_names = list(class_reg.PRETTY_NAMES.values())
    
    # --- data ---
    train_dataloader = hydra_instant(cfg.data.loader.train)
    val_dataloader = hydra_instant(cfg.data.loader.val)
    test_dataloader = hydra_instant(cfg.data.loader.test)
    
    # --- training ---
    callbacks = instantiate_callbacks(cfg.callbacks)
    logger: CSVLogger = hydra_instant(cfg.logger.csv_logger)

    trainer: L.Trainer = hydra_instant(cfg.training.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    if cfg.test_run: return
    
    pt.jit.script(model).save(Path(logger.log_dir) / "model.pt")

    # --- evaluation ---
    exp_path = Path(cfg.log_path) / model.logger.name / f'version_{model.logger.version}'
    ckpt = list((exp_path / 'checkpoints').glob('*.ckpt'))[0]

    if (csv_path:=exp_path / 'metrics.csv').exists():
        plot_curves(csv_path, exp_path)

    if cfg.perf_eval:
        predictions = trainer.predict(model, test_dataloader, ckpt_path=ckpt)

        logits, lbls, img_names = zip(*predictions)
        
        logits = [class_reg.reduce_classes(logit) for logit in logits]
        lbls = [class_reg.reduce_classes(lbl) for lbl in lbls]
        logits, lbls = pt.cat(logits, dim=0).to(device), pt.cat(lbls, dim=0).to(device)
        preds = logits.argmax(dim=1)
        
        img_names = list(chain.from_iterable(img_names))
        
        tabulate_per_image_metrics(logits, lbls, img_names)
        tabulate_metrics(preds, lbls, class_reg.class_ct, class_names, exp_path)
        plot_confusion_matrix(preds, lbls, class_names, exp_path)

if __name__ == '__main__':
    main()
