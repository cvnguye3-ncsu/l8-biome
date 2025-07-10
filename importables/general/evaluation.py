from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sb

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from torch import Tensor
from torchmetrics.classification import ConfusionMatrix

from importables.pytorch_lightning.metrics import get_metrics, tabulate_compute

# HELPER
# ------
def _clean_metrics(metrics_path, outpath, train_metrics, val_metrics):
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df.groupby('epoch').agg(
        {**{metric: 'mean' for metric in train_metrics}, 
         **{metric: 'max' for metric in val_metrics}}
    ).reset_index()
    
    metrics_df.to_csv(outpath / "figure_data" / "metrics_cleaned.csv")
    return metrics_df

# PLOTTING 
# --------
def _plot_confusion_matrix(cm: NDArray, class_names: list[str], outpath: str):
    _, ax = plt.subplots()
    sb.heatmap(100*cm, annot=True, fmt=".2f", annot_kws={"size": 10}, cbar=True, cmap='Blues', linewidth=.5, ax=ax)

    for text in ax.texts:
        text.set_text(f"{float(text.get_text()):.2f}%")

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    ax.set_title('Confusion Matrix of Performance', fontsize=14)

    plt.savefig(outpath, dpi=600, bbox_inches='tight')

def _plot_curves(metrics: list[pd.Series], metric_names: list[str],
                 title: str, outpath: Path):
    plt.figure(figsize=(10, 5))
    for metric, label in zip(metrics, metric_names):
        plt.plot(metric.index, metric.values, label=label, marker='o')

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath, dpi=600, bbox_inches='tight')

# CALLABLE
# --------
def tabulate_metrics(preds: Tensor, lbls: Tensor, 
                     cls_ct: int, class_names: list[str],
                     outpath: Path, extension: str = ''):
    metrics = get_metrics(cls_ct).cuda()
    
    for preds, lbls in zip(preds.split(4), lbls.split(4, dim=0)):
        metrics.update(preds, lbls)
    
    df = tabulate_compute(cls_ct, class_names, metrics.compute())
    df.to_csv(outpath / 'figure_data' / 'perf_metrics.csv')

    latex_table = df.to_latex(index=True, float_format="%.3f",
                              caption="Class Metrics", label="tab:class_metrics")
    with open(outpath / 'figures' / f"perf_metrics{extension}.tex", "w") as file:
        file.write(latex_table)
        
def tabulate_per_image_metrics(logits: Tensor, lbls: Tensor, img_names: list[str],
                         cls_ct: int, class_names: list[str], 
                         outpath: Path):
    metrics = get_metrics(cls_ct).cuda()
    df_list = []
    
    for pred, lbl, img_name in zip(logits.split(1), lbls.split(1), img_names):
        met_vals = metrics(pred, lbl)
        
        row_df = tabulate_compute(cls_ct, class_names, met_vals)
        row_df = row_df.stack()
        row_df.index = [f'{row}_{col}' for row, col in row_df.index]
        row_df = row_df.to_frame().T
        
        row_df['img_name'] = [img_name]

        df_list.append(row_df)
        
    df = pd.concat(df_list)
    df.to_csv(outpath / 'figure_data' / 'per_image_stats.csv')

def plot_confusion_matrix(preds: Tensor, lbls: Tensor, 
                          class_names: list[str], outpath: Path):
    cm_func = ConfusionMatrix(task='multiclass', num_classes=len(class_names),
                              normalize='all').cuda()
    cm = cm_func(preds, lbls).cpu().numpy()

    np.save(outpath / 'figure_data' / 'confusion_matrix.npy', cm)
    _plot_confusion_matrix(cm, class_names, outpath / 'figures' / 'confusion_matrix.png')

def plot_curves(metrics_path: Path, outpath: Path):
    metrics = ['loss', 'acc', 'iou']
    train_metrics = [f"train_{met}" for met in metrics]
    val_metrics = [f"val_{met}" for met in metrics]

    metrics_df = _clean_metrics(metrics_path, outpath, train_metrics, val_metrics) 
    
    train_loss = metrics_df['train_loss']
    val_loss = metrics_df['val_loss']

    _plot_curves([train_loss, val_loss],
                 ['Train loss', 'Validation loss'],
                 "Loss Curves",
                 outpath / 'figures' / 'loss_curves.png')