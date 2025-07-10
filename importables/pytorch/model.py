import torch as pt
import pytorch_lightning as L
import torch.optim.lr_scheduler as LR

from torchmetrics import Accuracy, JaccardIndex, F1Score

from torch.nn.functional import cross_entropy

import abc

class ModelSkeleton(abc.ABC, L.LightningModule):
    learning_rate: float = 1
    betas: tuple[float, float] = (.9, .999)
    weight_decay: float = 1e-2
    
    warmup_epochs: int = 1
    max_epochs: int = 50
    warmup_starting_lr_ratio: float = 1
    warmup_ending_lr_ratio: float = 1

    batch_grad_acc: int = 1
    
    class_weights: pt.Tensor
    sync_dist: bool
    
    def __init__(self, resize_embedding: bool, class_count: int):
        super().__init__()
        
        self.resize_embedding = resize_embedding
        self.class_count = class_count

        self.acc_func = Accuracy(task='multiclass', num_classes=class_count)
        self.miou_func = JaccardIndex(task='multiclass', num_classes=class_count, average='macro')
        self.f1_func = F1Score(task='multiclass', num_classes=class_count, average='macro')
        
    @abc.abstractmethod
    def _image_preprocessing(img: pt.Tensor) -> pt.Tensor:
        ...
    
    def _logit_step(self, batch):
        imgs, _, _ = batch

        logit = self.forward(imgs)

        return logit

    def _general_step(self, batch):
        _, masks, _ = batch
        masks = masks.squeeze(1)

        logit = self._logit_step(batch)

        # Logging.
        loss = cross_entropy(logit, masks, weight=self.class_weights)

        acc = self.acc_func(logit, masks)
        miou = self.miou_func(logit, masks)
        f1 = self.f1_func(logit, masks)

        return loss, acc, miou, f1

    def predict_step(self, batch, _):
        _, masks, imgnames = batch
        masks = masks.squeeze(1)

        logit = self._logit_step(batch)

        preds = logit.argmax(dim=1)

        return preds, masks, imgnames

    # --- OPTIMIZATION ---
    def _opt_step(self, batch, split: str):
        B, *_ = batch[0].shape
        loss, acc, miou, f1 = self._general_step(batch)
        
        self.log(f'{split}_loss', loss, sync_dist=self.sync_dist,batch_size=B*self.batch_grad_acc)
        
        self.log(f'{split}_miou', miou, sync_dist=self.sync_dist, batch_size=B*self.batch_grad_acc)
        self.log(f'{split}_f1', f1, sync_dist=self.sync_dist, batch_size=B*self.batch_grad_acc)
        self.log(f'{split}_acc', acc, sync_dist=self.sync_dist, batch_size=B*self.batch_grad_acc)

        return loss
    
    def training_step(self, batch, _):
        return self._opt_step(batch, 'train')

    def validation_step(self, batch, _):
        return self._opt_step(batch, 'val')
    
    def configure_optimizers(self):
        # TODO: fused does not work for AdamW.
        warm_up_params = {
            'start_factor': self.warmup_starting_lr_ratio,
            'end_factor': self.warmup_ending_lr_ratio,
            'total_iters': self.warmup_epochs
        }
        
        optim_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay, 
            'betas': self.betas,
            'foreach': True
        }

        optimizer = pt.optim.AdamW(list(self.decoder_params) + list(self.encoder_params), **optim_params)
        scheduler = LR.LinearLR(optimizer, **warm_up_params)

        return [optimizer], [scheduler]