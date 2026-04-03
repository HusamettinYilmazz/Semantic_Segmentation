from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def plot_confusion_matrix(cm, class_names, save_path=None):
    cm_normalized = cm.numpy().astype(float) / (cm.numpy().sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_normalized * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=100
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (%)")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)

def compute_confusion_matrix(y_preds, y_true, class_names, ignore_index=255):
    """
    y_preds: [B, H, W] long
    y_true:  [B, H, W] long
    """
    num_classes = len(class_names)

    y_preds = y_preds.cpu().numpy().flatten()
    y_true  = y_true.cpu().numpy().flatten()

    valid   = (y_true != ignore_index) & (y_true >= 0) & (y_true < num_classes)
    y_preds = y_preds[valid]
    y_true  = y_true[valid]

    cm = confusion_matrix(y_true, y_preds, labels=list(range(num_classes)))

    return torch.tensor(cm, dtype=torch.long)

def compute_iou_per_class(cm):
    return (cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-10)).cpu().numpy()

def compute_per_class_accuracy(cm):
    return (cm.diag() / (cm.sum(dim=1) + 1e-10)).cpu().numpy()
