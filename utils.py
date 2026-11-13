import torch
import matplotlib.pyplot as plt
from pathlib import Path


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)  # (B,1,H,W)
    targets = targets.float()
    dims = (1,2,3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    dims = (1,2,3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def plot_loss_curves(training_losses, validation_losses, output_dir):
    epochs = range(1, len(training_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, training_losses, label='Training loss')
    ax.plot(epochs, validation_losses, label='Validation loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_validation_dice(validation_dice_scores, output_dir):
    epochs = range(1, len(validation_dice_scores) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, validation_dice_scores, label='Validation Dice Score', marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    ax.set_title('Validation Dice Score')
    best_score = max(validation_dice_scores)
    best_epoch = validation_dice_scores.index(best_score) + 1
    ax.scatter(
        best_epoch,
        best_score,
        color="red",
        zorder=3,
        label=f"Best: epoch {best_epoch} ({best_score:.4f})"
    )
    ax.legend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "val_dice_score.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
