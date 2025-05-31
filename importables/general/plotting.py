from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sb

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from torch import Tensor
from torchmetrics.classification import ConfusionMatrix

from importables.pytorch.metrics import SemanticSegmentationMetrics
from importables.project.cloud_classes import CLASS_REG

# HELPER
# ------
def _clean_metrics(metrics_path, outpath, train_metrics, val_metrics):
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df.groupby('epoch').agg(
        {**{metric: 'mean' for metric in train_metrics}, 
         **{metric: 'max' for metric in val_metrics}}
    ).reset_index()
    
    metrics_df.to_csv(outpath / "plot_data" / "metrics_cleaned.csv")
    return metrics_df

# PLOTTING 
# --------
def _plot_confusion_matrix(cm: NDArray, outpath: str):
    _, ax = plt.subplots()
    sb.heatmap(100*cm, annot=True, fmt=".2f", annot_kws={"size": 10}, cbar=True, cmap='Blues', linewidth=.5, ax=ax)

    # Add percentage symbol to annotations
    for text in ax.texts:
        text.set_text(f"{float(text.get_text()):.2f}%")

    # Set custom axis labels
    ax.set_xticklabels(CLASS_REG.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_REG.CLASSES, rotation=0)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    ax.set_title('Confusion Matrix of Performance', fontsize=14)

    plt.savefig(outpath, dpi=600, bbox_inches='tight')

def _plot_curves(metrics: list[pd.Series],
                 metric_names: list[str],
                 title: str,
                 outpath: Path):
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
def tabulate_metrics(preds: Tensor, labels: Tensor,
                     outpath: Path, extension: str = ''):
    metrics = SemanticSegmentationMetrics()
    
    for preds, labels in zip(preds.split(4), labels.split(4, dim=0)):
        metrics.update(preds, labels)
    
    df = metrics.compute()
    df.to_csv(outpath / 'plot_data' / 'perf_metrics.csv')

    latex_table = df.to_latex(
        index=True, caption="Class Metrics", label="tab:class_metrics", float_format="%.2f")
    with open(outpath / 'plot' / f"perf_metrics{extension}.tex", "w") as file:
        file.write(latex_table)

def plot_confusion_matrix(preds: Tensor, labels: Tensor,
                          outpath: Path):
    cm_func = ConfusionMatrix(task='multiclass', num_classes=len(CLASS_REG.CLASSES),
                              normalize='all').cuda()
    cm = cm_func(preds, labels).cpu().numpy()

    np.save(outpath / 'plot_data' / 'confusion_matrix.npy', cm)
    _plot_confusion_matrix(cm, outpath / 'plot' / 'confusion_matrix.png')

def plot_curves(metrics_path: Path, outpath: Path):
    metrics = ['acc', 'f1', 'loss', 'miou']
    train_metrics = [f"train_{met}" for met in metrics]
    val_metrics = [f"val_{met}" for met in metrics]
    
    metrics_df = _clean_metrics(metrics_path, outpath, train_metrics, val_metrics) 
    
    train_loss = metrics_df['train_loss']
    val_loss = metrics_df['val_loss']
    
    train_f1 = metrics_df['train_f1']
    
    val_f1 = metrics_df['val_f1']
    val_acc = metrics_df['val_acc']
    val_miou = metrics_df['val_miou']

    # Overfitting relative to the loss.
    _plot_curves([train_loss, val_loss],
                 ['Train loss', 'Validation loss'],
                 "Loss Curves",
                 outpath / 'plot' / 'loss_curves.png')

    # Whether loss and performance metrics cohere for training.
    _plot_curves([train_loss, train_f1],
                 ["Loss", "F1-measure"],
                 "Training Loss vs Metric",
                 outpath / 'plot' / 'train_curves.png')

    # Whether loss and performance metrics cohere for validation.
    _plot_curves([val_loss, val_f1],
                 ["Loss", "F1-measure"],
                 "Validation Loss vs Metric",
                 outpath / 'plot' / 'val_curves.png')

    _plot_curves([val_acc, val_f1, val_miou],
                 ["Accuracy", "F1-measure", "mIOU"],
                 "Validation Performance Metrics",
                 outpath / 'plot' / 'val_metrics_curves.png')