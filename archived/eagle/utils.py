import torch.multiprocessing
import torchvision.transforms as T
from torchmetrics import Metric

import numpy as np
from scipy.optimize import linear_sum_assignment

from PIL import Image

def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args

class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)

normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])

class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self, training=False):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        # prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)
            
        metric_dict = {
            self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
            self.prefix + "Accuracy": opc.item()
        }
        
        if not training:
            metric_dict["assignments"] = (self.assignments[1]).tolist()
        
        return {k: 100 * v for k, v in metric_dict.items()}