import torch
from utils import dice_score_from_logits
from pathlib import Path
from torchvision.transforms import functional as TF
import random
from PIL import Image
import matplotlib.pyplot as plt
from args import get_args
import pandas as pd
import os
from dataset import KneeDataset, KneeTestDataset
from torch.utils.data import DataLoader
from model import UNetLext


def evaluate_model(model, validation_loader, device, best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    total_dice = 0
    total_samples = 0

    with torch.no_grad():
        for xrays, masks in validation_loader:
            xrays = xrays.to(device)
            masks = masks.to(device)
            logits = model(xrays)
            dice_scores = dice_score_from_logits(logits, masks)
            total_dice += dice_scores.sum().item()
            total_samples += dice_scores.size(0)

    validation_dice_score = total_dice / total_samples

    return validation_dice_score


def save_test_predictions(model, test_loader, device, best_model_path, output_dir):
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    predictions_dir = Path(output_dir) / "test_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for xrays, filenames in test_loader:
            xrays = xrays.to(device)
            logits = model(xrays)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for prediction, filename in zip(preds, filenames):
                predicted_mask_image = TF.to_pil_image(prediction.cpu())
                prediction_path = predictions_dir / filename
                predicted_mask_image.save(prediction_path)

    generator = random.Random(42)
    selected_indices = generator.sample(range(len(test_loader.dataset)), k=3)
    examples = []

    for index in selected_indices:
        xray, filename = test_loader.dataset[index]
        predicted_mask = Image.open(predictions_dir / filename).convert("L")
        examples.append((xray, filename, predicted_mask))

    fig, axes = plt.subplots(nrows=len(examples), ncols=2, figsize=(8, 3 * len(examples)))
    for row, (xray, filename, predicted_mask) in enumerate(examples):
        axes[row, 0].imshow(xray.squeeze(0).numpy(), cmap="gray")
        axes[row, 0].set_title(f"Original: {filename}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(predicted_mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Predicted mask")
        axes[row, 1].axis("off")

    fig.suptitle("Test Set Predictions")
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / "test_prediction_examples.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'validation.csv'))
    test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))
    validation_dataset = KneeDataset(val_set)
    test_dataset = KneeTestDataset(test_set)
    validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    model = UNetLext(basic_width=32, depth=4,input_channels=1, output_channels=1).to(device)
    best_model_path = Path(args.out_dir) / "best_model.pth"
    validation_dice_score = evaluate_model(model, validation_loader, device, best_model_path)
    print(f'Validation Dice score: {validation_dice_score:.4f}')
    save_test_predictions(model, test_loader, device, best_model_path, args.out_dir)
