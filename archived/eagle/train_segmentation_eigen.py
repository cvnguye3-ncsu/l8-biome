import sys
import pytz
from datetime import datetime
from pathlib import Path
import warnings
from itertools import chain

import os
from os.path import join

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing
import torchvision.transforms as T 

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from importables.pytorch.dataset import SegmentationDataset

import torch as tf
from torch import Tensor

from data import ContrastiveSegDataset
from utils import *
from modules import *
from eigen_modules import  *

import hydra
from omegaconf import DictConfig, OmegaConf

from importables.general.plotting import plot_curves, tabulate_metrics, plot_confusion_matrix
    
warnings.filterwarnings(action='ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

def scheduler(step_schedulers, step):
    return 0 if step > step_schedulers else 0

class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, 
                 continuous, dim, pretrained_weights,
                 dino_patch_size, dino_feat_type, 
                 model_type, projection_type, dropout,
                 eigen_cluster, eigen_cluster_out, extra_clusters, centroid_mode,
                 global_loss_weight, contrastive_temp, neg_samples, pointwise, zero_clamp, shift_bias, shift_value, stabalize, feature_samples,
                 use_head, n_images, 
                 lr, lr_cluster, lr_linear, lr_cluster_eigen, rec_weight, pos_inter_weight, momentum_limit, neg_inter_weight, correspondence_weight, local_pos_weight, local_pos_aug_weight, step_schedulers):
        super().__init__()
        
        self.n_classes = n_classes

        dim = n_classes if not continuous else dim
        self.dim = dim

        self.net = DinoFeaturizer(dim, 
                pretrained_weights, 
                dino_patch_size, dino_feat_type, model_type, 
                projection_type, dropout)
        
        self.lr = lr
        self.lr_cluster = lr_cluster
        self.lr_linear = lr_linear
        self.lr_cluster_eigen = lr_cluster_eigen
        self.rec_weight = rec_weight
        
        self.pos_inter_weight = pos_inter_weight
        
        self.neg_inter_weight = neg_inter_weight
        self.correspondence_weight = correspondence_weight
        self.local_pos_weight = local_pos_weight
        self.local_pos_aug_weight = local_pos_aug_weight
        
        self.step_schedulers = step_schedulers
        
        self.train_cluster_probe_eigen = ClusterLookup(eigen_cluster-1, eigen_cluster_out)
        self.train_cluster_probe_eigen_aug = ClusterLookup(eigen_cluster-1, eigen_cluster_out)
        
        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        
        self.centroid_mode = centroid_mode
        self.use_head = use_head
        self.n_images = n_images
        self.neg_samples = neg_samples
        self.momentum_limit = momentum_limit
        
        if self.use_head:
            self.project_head = nn.Linear(dim, dim)

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)
        
        self.CELoss = newLocalGlobalInfoNCE(n_classes, dim, extra_clusters, centroid_mode, global_loss_weight, contrastive_temp)
        self.eigen_loss_fn = EigenLoss(eigen_cluster)
        # self.eigen_new_loss_fn = new_EigenLoss(cfg)
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()

        self.contrastive_corr_loss_fn = CorrespondenceLoss(neg_samples, pointwise, zero_clamp, shift_bias, shift_value, stabalize, feature_samples)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        self.val_steps = 0
        
        self.update_prams = 0.0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, _):
        if self.centroid_mode == 'learned' or self.centroid_mode == 'prototype':
            net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, centroid_optim, cluster_eigen_optim, cluster_eigen_optim_aug = self.optimizers()
        else:
            net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, cluster_eigen_optim, cluster_eigen_optim_aug = self.optimizers()
        
        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()
        project_head_optim.zero_grad()
        cluster_eigen_optim.zero_grad()
        cluster_eigen_optim_aug.zero_grad()
        
        scheduler_cluster = optim.lr_scheduler.StepLR(cluster_probe_optim, step_size=50, gamma=0.1)
        
        if self.centroid_mode == 'learned' or self.centroid_mode == 'prototype':
            centroid_optim.zero_grad()

        with torch.no_grad():
            # ind = batch["ind"]
            img = batch["img"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            # label_pos = batch["label_pos"]
            img_pos_aug = batch["img_pos_aug"]

        feats, feats_kk, code, code_kk = self.net(img)
        feats_pos, feats_pos_kk, code_pos, code_pos_kk = self.net(img_pos)
        feats_pos_aug, feats_pos_aug_kk, code_pos_aug, code_pos_aug_kk = self.net(img_pos_aug)
        log_args = dict(sync_dist=False, rank_zero_only=True)
        
        code_pos_z, code_pos_aug_z = code_pos_kk.permute(0,2,3,1).reshape(-1, self.dim), code_pos_aug_kk.permute(0,2,3,1).reshape(-1, self.dim)

        if self.use_head:
            code_pos_z = self.project_head(code_pos_z)
            code_pos_aug_z = self.project_head(code_pos_aug_z) #[25088, 70]
            code_pos_aug_z = F.normalize(code_pos_aug_z, dim=1)
            code_pos_z = F.normalize(code_pos_z, dim=1)
        
        feats_pos_reshaped = feats_pos_kk.view(feats_pos.shape[0], feats_pos.shape[1], -1)
        corr_feats_pos = torch.matmul(feats_pos_reshaped.transpose(2, 1), feats_pos_reshaped)
        corr_feats_pos = F.normalize(corr_feats_pos, dim=1)
        
        feats_pos_aug_reshaped = feats_pos_aug_kk.view(feats_pos_aug.shape[0], feats_pos_aug.shape[1], -1)
        corr_feats_pos_aug = torch.matmul(feats_pos_aug_reshaped.transpose(2, 1), feats_pos_aug_reshaped)
        corr_feats_pos_aug = F.normalize(corr_feats_pos_aug, dim=1)

        loss = 0    

        if self.neg_samples == 0:
            (
            pos_inter_loss, pos_inter_cd, _, _
            ) = self.contrastive_corr_loss_fn(
                        feats, feats_pos,
                        code, code_pos
                    )

            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            loss += (self.pos_inter_weight * pos_inter_loss) * self.correspondence_weight
        
        elif self.neg_samples > 0:
            update_params = scheduler(self.step_schedulers, self.global_step)
            update_params = min(update_params, self.momentum_limit)
            
            (
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd
            ) = self.contrastive_corr_loss_fn(
                feats_kk, feats_pos_kk, feats_pos_aug_kk,
                code_kk, code_pos_kk, code_pos_aug_kk
            )
            neg_inter_loss = neg_inter_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()

            # 2. Eigenloss
            # pos 
            feats_pos_re = feats_pos_kk.reshape(feats_pos.shape[0], feats_pos.shape[1], -1).permute(0,2,1)
            code_pos_re = code_pos_kk.reshape(code_pos.shape[0], code_pos.shape[1], -1).permute(0,2,1)        
            eigenvectors =  self.eigen_loss_fn(img, feats_pos_re, code_pos_re)

            eigenvectors = eigenvectors[:, :, 1:].reshape(eigenvectors.shape[0], feats_pos.shape[-1], feats_pos.shape[-1], -1).permute(0,3,1,2)
            cluster_eigen_loss, cluster_eigen_probs = self.train_cluster_probe_eigen(eigenvectors, 1, log_probs = True)
            cluster_eigen_probs = cluster_eigen_probs.argmax(1)

            # # pos_aug
            feats_pos_aug_re = feats_pos_aug_kk.reshape(feats_pos_aug.shape[0], feats_pos_aug.shape[1], -1).permute(0,2,1)
            code_pos_aug_re = code_pos_aug_kk.reshape(code_pos_aug.shape[0], code_pos_aug.shape[1], -1).permute(0,2,1)
            eigenvectors_aug = self.eigen_loss_fn(img, feats_pos_aug_re, code_pos_aug_re)

            eigenvectors_aug = eigenvectors_aug[:, :, 1:].reshape(eigenvectors_aug.shape[0], feats_pos.shape[-1], feats_pos.shape[-1], -1).permute(0,3,1,2)
            cluster_eigen_aug_loss, cluster_eigen_aug_probs = self.train_cluster_probe_eigen_aug(eigenvectors_aug, 1, log_probs = True)
            cluster_eigen_aug_probs = cluster_eigen_aug_probs.argmax(1)

            local_pos_mid_loss = self.CELoss(code_pos_z, code_pos_aug_z, cluster_eigen_probs, corr_feats_pos)
            local_pos_loss = local_pos_mid_loss 
            
            local_pos_aug_mid_loss = self.CELoss(code_pos_aug_z, code_pos_z, cluster_eigen_aug_probs, corr_feats_pos_aug)
            local_pos_aug_loss = local_pos_aug_mid_loss 
            
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('loss/local_pos_loss', local_pos_loss, **log_args)
            self.log('loss/local_pos_aug_loss', local_pos_aug_loss, **log_args)
            self.log('loss/cluster_eigen_loss', cluster_eigen_loss, **log_args)

            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)
            
            loss += (cluster_eigen_aug_loss + cluster_eigen_loss)/2

            loss += (self.pos_inter_weight * pos_inter_loss +
                     self.neg_inter_weight * neg_inter_loss 
                     ) * self.correspondence_weight * (1.0 - update_params)

            loss += (self.local_pos_weight * local_pos_loss + self.local_pos_aug_weight * local_pos_aug_loss
                    ) * (update_params)

            self.log('cd/update_params', update_params, **log_args)
            
        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        # detached_code = torch.clone(code.detach())
        detached_code_kk = torch.clone(code_kk.detach())

        linear_logits = self.linear_probe(detached_code_kk)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code_kk, None)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)
        scheduler_cluster.step()

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()
        cluster_eigen_optim.step()
        cluster_eigen_optim_aug.step()
        
        if self.use_head:
            project_head_optim.step()
            
        if self.centroid_mode == 'learned' or self.centroid_mode == 'prototype':
            centroid_optim.step()
            
        return loss

    def on_train_start(self):
        # tb_metrics = {
        #     **self.linear_metrics.compute(training=True),
        #     **self.cluster_metrics.compute(training=True)
        # }
        
        self.logger.log_hyperparams(None)

    def validation_step(self, batch, _):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, feats_kk, code, code_kk = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
            code_kk = F.interpolate(code_kk, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code_kk)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code_kk, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            return {
                'img': img[:self.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.n_images].detach().cpu(),
                "label": label[:self.n_images].detach().cpu()}

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(training=True),
                **self.cluster_metrics.compute(training=True),
            }

            if self.global_step > 2:
                self.log_dict(tb_metrics)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

    def predict_step(self, batch, _):
        _, masks, img_names = batch
        masks = masks.squeeze(1)

        logit = self._logit_step(batch)
        preds = logit.argmax(dim=1)

        return preds, masks, img_names

    def configure_optimizers(self): # project_head_cluster_optim
        main_params = list(self.net.parameters())

        if self.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=self.lr_linear)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=self.lr_cluster)
        cluster_eigen_optim = torch.optim.Adam(list(self.train_cluster_probe_eigen.parameters()), lr=self.lr_cluster_eigen)
        cluster_eigen_optim_aug = torch.optim.Adam(list(self.train_cluster_probe_eigen_aug.parameters()), lr=self.lr_cluster_eigen)

        if self.use_head == True and (self.centroid_mode == 'learned' or self.centroid_mode == 'prototype'):
            project_head_optim = torch.optim.Adam(self.project_head.parameters(), lr=self.lr)
            centroid_optim = torch.optim.Adam(self.CELoss.parameters(), lr=self.lr)
            return net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, centroid_optim, cluster_eigen_optim, cluster_eigen_optim_aug
        
        elif self.use_head == True and (self.centroid_mode == 'mean' or self.centroid_mode == 'medoid'):
            project_head_optim = torch.optim.Adam(self.project_head.parameters(), lr=self.lr)
            return net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, cluster_eigen_optim, cluster_eigen_optim_aug

        else:
            return net_optim, linear_probe_optim, cluster_probe_optim, cluster_eigen_optim, cluster_eigen_optim_aug

EAGLE_PATH = Path("E:/l8-biome/models/eagle/")

@hydra.main(config_path=str(EAGLE_PATH), config_name="train_config_cocostuff.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = Path(cfg.pytorch_data_dir)
    
    data_dir =  join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")
    
    exp_name = f"{cfg.experiment_name}"

    tz = pytz.timezone('Asia/Seoul')
    prefix = "{}/{}_{}_{}_{}".format(cfg.dataset_name, cfg.log_dir, datetime.now(tz).strftime('%b%d_%H-%M-%S'),cfg.model_type, exp_name)

    cfg.full_name = prefix
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(cfg.seed)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.RandomApply([T.GaussianBlur((3, 3))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=str(pytorch_data_dir  / 'train'),
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        mask=True,
    )
    
    val_loader_crop = "center"
    
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=str(pytorch_data_dir / 'val'),
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(224, False, val_loader_crop),
        target_transform=get_transform(224, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    val_batch_size = cfg.batch_size // 2

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, 
                                     cfg.continuous, cfg.dim, cfg.pretrained_weights,
                 cfg.dino_patch_size, cfg.dino_feat_type, 
                 cfg.model_type, cfg.projection_type, cfg.dropout,
                 cfg.eigen_cluster, cfg.eigen_cluster_out, cfg.extra_clusters, cfg.centroid_mode,
                 cfg.global_loss_weight, cfg.contrastive_temp, cfg.neg_samples, cfg.pointwise, cfg.zero_clamp, cfg.shift_bias, cfg.shift_value, cfg.stabalize, cfg.feature_samples,
                 cfg.use_head, cfg.n_images, 
                 cfg.lr, cfg.lr_cluster, cfg.lr_linear, cfg.lr_cluster_eigen, cfg.rec_weight, 
                 cfg.pos_inter_weight, cfg.momentum_limit, cfg.neg_inter_weight, cfg.correspondence_weight, cfg.local_pos_weight, cfg.local_pos_aug_weight, cfg.step_schedulers)
    
    tb_logger = CSVLogger(save_dir=EAGLE_PATH / 'logs', name=cfg.log_dir+"_"+exp_name)

    gpu_args = dict(devices=[0], val_check_interval=cfg.val_freq)

    if gpu_args["val_check_interval"] > len(train_loader) // 4:
        gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, prefix),
                every_n_train_steps=100,
                save_top_k=1,
                monitor="test/linear/mIoU",
                mode="max",
                filename='{epoch:02d}-{step:08d}-{test/linear/mIoU:.2f}'
            ),
            EarlyStopping(monitor='test/linear/mIoU',
                          patience=4,
                          mode='max',
                          min_delta=1e-3)
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)
    
    scripted_model = tf.jit.script(model)
    model_path = EAGLE_PATH / 'logs' / f'{cfg.log_dir}_{exp_name}' / f'version_{tb_logger.version}' / "model.pt"
    print(model_path.absolute())
    scripted_model.save(model_path)
    
    # === EVALUATION ===
    dataset = SegmentationDataset(pytorch_data_dir  / 'test' / 'img',
                                  pytorch_data_dir  / 'test' / 'label')
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            num_workers=4, persistent_workers=True)
    
    eval_model(model, dataloader, trainer, tb_logger.log_dir)
    
def eval_model(model: pl.LightningModule, 
               val_dataloader: DataLoader, trainer: pl.Trainer,
               log_dir: str, ):
    exp_name = model.logger.name
    exp_ver = model.logger.version

    log_path = Path(log_dir)
    exp_path = log_path / exp_name / f'version_{exp_ver}'
    csv_path = exp_path / 'metrics.csv'

    ckpt = (exp_path / 'checkpoints').glob('*.ckpt')
    ckpt = list(ckpt)[0]

    if csv_path.exists():
        plot_curves(csv_path, exp_path)

    model.eval()
    
    predictions = trainer.predict(model, val_dataloader,
                                    ckpt_path=ckpt)

    preds, labels, imgnames = zip(*predictions)
    preds: Tensor = tf.cat(preds, dim=0)
    labels: Tensor = tf.cat(labels, dim=0)
    imgnames = list(chain.from_iterable(imgnames))
    
    # Performance metrics: Clouds, thin clouds, cloud shadows
    tabulate_metrics(preds, labels, exp_path)
    
    # Confusion matrix
    plot_confusion_matrix(preds, labels, exp_path)

if __name__ == "__main__":
    prep_args()
    my_app()