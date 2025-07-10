import torch as pt
import pytorch_lightning as L
import torch.optim.lr_scheduler as LR

from importables.pytorch_lightning.metrics import get_metrics
import torch.nn.functional as F

import abc

class ModelSkeleton(abc.ABC, L.LightningModule):
    class_weights: pt.Tensor
    reduce_matrix: pt.Tensor
    reduce_func: callable
    
    log_params: dict
    opt_params: dict
    sched_params: dict
    
    def __init__(self, cls_ct: int):
        super().__init__()
        
        self.cls_ct = cls_ct
        
        self.train_metrics = get_metrics(cls_ct, 'train_')
        self.val_metrics = get_metrics(cls_ct, 'val_')
        
    @abc.abstractmethod
    def _image_preprocessing(img: pt.Tensor) -> pt.Tensor:
        ...

    def training_step(self, batch):
        imgs, lbls, _ = batch
        lbls = lbls.squeeze(1)
        logits = self.forward(imgs)
        
        loss = F.cross_entropy(logits, lbls, weight=self.class_weights)
        self.train_metrics.update(self.reduce_matrix * logits, lbls)

        self.log(f'train_loss', loss, **self.log_params)
        self.log_dict(self.train_metrics, **self.log_params)
        
        return loss

    def validation_step(self, batch):
        imgs, lbls, _ = batch
        lbls = lbls.squeeze(1)
        logits = self.forward(imgs)

        loss = F.cross_entropy(logits, lbls, weight=self.class_weights)
        self.val_metrics.update(self.reduce_matrix * logits, lbls)

        self.log(f'val_loss', loss, **self.log_params)
        self.log_dict(self.val_metrics, **self.log_params)

    def predict_step(self, batch, _):
        imgs, lbls, img_names = batch
        lbls = lbls.squeeze(1)

        logits = self.forward(imgs)

        return logits, lbls, img_names

    def configure_optimizers(self):
        optimizer = pt.optim.AdamW(list(self.decoder_params) + list(self.encoder_params), 
                                   **self.opt_params)
        scheduler = LR.LinearLR(optimizer, **self.sched_params)

        return [optimizer], [scheduler]