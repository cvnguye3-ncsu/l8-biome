from pathlib import Path
from itertools import chain

import torch as pt
from torch.utils.data import DataLoader
from torch import Tensor

import pytorch_lightning as L
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

from omegaconf import DictConfig

from importables.pytorch.my_callbacks import MyCallback
from importables.pytorch.dataset import SegmentationDataset
from importables.general.plotting import plot_curves, tabulate_metrics, plot_confusion_matrix

def eval_model(cfg: DictConfig, model: L.LightningModule, 
               val_dataloader: DataLoader, trainer: L.Trainer):
    exp_name = model.logger.name
    exp_ver = model.logger.version

    log_path = Path(cfg.log_path)
    exp_path = log_path / exp_name / f'version_{exp_ver}'
    csv_path = exp_path / 'metrics.csv'

    ckpt = (exp_path / 'checkpoints').glob('*.ckpt')
    ckpt = list(ckpt)[0]

    if csv_path.exists():
        plot_curves(csv_path, exp_path)

    if cfg.perf_eval:
        model.eval()
        
        predictions = trainer.predict(model, val_dataloader, ckpt_path=ckpt)

        preds, labels, imgnames = zip(*predictions)
        preds: Tensor = pt.cat(preds, dim=0).cuda()
        labels: Tensor = pt.cat(labels, dim=0).cuda()
        
        imgnames = list(chain.from_iterable(imgnames))
        
        # Performance metrics: Clouds, thin clouds, cloud shadows
        tabulate_metrics(preds, labels, exp_path)
        
        # Confusion matrix
        plot_confusion_matrix(preds, labels, exp_path)
     
def set_training_params(cfg: DictConfig, model: L.LightningModule):
    L.seed_everything(cfg.seed)
    if cfg.gpu_compile: model.compile()
    pt.set_float32_matmul_precision(cfg.matmul_precision)
        
def set_model_hyperparameters(cfg: DictConfig, model: L.LightningModule):
    model.learning_rate = cfg.learning_rate
    model.weight_decay = cfg.l2_reg
    model.betas = cfg.betas

    model.warmup_epochs = cfg.warmup_epochs
    model.max_epochs = cfg.max_epochs
    model.warmup_starting_lr_ratio = cfg.warmup_starting_lr_ratio
    model.warmup_ending_lr_ratio = cfg.warmup_ending_lr_ratio

    model.batch_grad_acc = cfg.batch_grad_acc
    
    SEEDS_PATH = Path(cfg.data_folder) / 'seeds'
    
    class_weights = pt.load(SEEDS_PATH / f'{cfg.seed}_{cfg.subset_ratio}_class_balance.pt', 
                            map_location='cuda')
    class_weights = 1/pt.sqrt(class_weights)
    model.class_weights = class_weights.float()
    
    model.sync_dist = cfg.sync_logging

def train_model(cfg: DictConfig, model: LightningModule, model_path: Path):
    # --- hyperparameters ---
    set_training_params(cfg, model)
    set_model_hyperparameters(cfg, model)
    
    # --- data ---
    train_dataset = SegmentationDataset(seed=cfg.seed, split='train', 
                                        data_location=cfg.data_folder, data_aug=True,
                                        subset_ratio=cfg.subset_ratio)
    val_dataset = SegmentationDataset(seed=cfg.seed, split='val', 
                                      data_location=cfg.data_folder, subset_ratio=cfg.subset_ratio)
    test_dataset = SegmentationDataset(seed=cfg.seed, split='test', 
                                       data_location=cfg.data_folder, subset_ratio=cfg.subset_ratio)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=4, persistent_workers=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.batch_size,
                                num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=cfg.batch_size,
                                num_workers=4, persistent_workers=True, pin_memory=True)

    # --- callbacks ---
    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        patience=cfg.patience,
        min_delta=cfg.patience_tol
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1', mode='max', save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_f1:.4f}"
    )

    input_shape = (cfg.batch_size,) + model.input_shape
    my_callback = MyCallback(model_path, input_shape)

    # --- logging ---
    exp_name = f"seed-{cfg.seed}"
    csv_logger = CSVLogger(cfg.log_path, name=exp_name)

    progress_bar = TQDMProgressBar(refresh_rate=10)

    # --- training ---
    trainer = L.Trainer(logger=csv_logger,
                        callbacks=[
                            early_stop_callback,
                            checkpoint_callback,
                            my_callback,
                            progress_bar],
                        max_epochs=cfg.max_epochs,
                        fast_dev_run=cfg.test_run,
                        strategy=cfg.gpu_strategy,
                        devices=cfg.gpu_devices,
                        enable_model_summary=False,
                        log_every_n_steps=1,
                        sync_batchnorm=cfg.sync_batchnorm,
                        accumulate_grad_batches=cfg.batch_grad_acc,
                        detect_anomaly=cfg.debug_grad)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
    if cfg.test_run: return
    
    log_dir = Path(csv_logger.log_dir)

    scripted_model: pt.jit.ScriptModule = pt.jit.script(model)
    scripted_model.save(log_dir / "model.pt")

    # --- evaluation ---
    eval_model(cfg, model, test_dataloader, trainer)