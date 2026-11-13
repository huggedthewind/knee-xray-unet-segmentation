from args import get_args
from dataset import KneeDataset, KneeTestDataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from model import UNetLext
import torch
from trainer import train_model
from evaluate import evaluate_model, save_test_predictions
from utils import plot_loss_curves, plot_validation_dice

args = get_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'validation.csv'))
test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))

train_dataset = KneeDataset(train_set, augment=True)
validation_dataset = KneeDataset(val_set)
test_dataset = KneeTestDataset(test_set)

train_generator = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, generator=train_generator)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

model = UNetLext(basic_width=32, depth=4,input_channels=1, output_channels=1).to(device)

training_losses, validation_losses, validation_dice_scores = train_model(model, train_loader, validation_loader, device, args)

best_model_validation_dice = evaluate_model(model, validation_loader, device, os.path.join(args.out_dir, "best_model.pth"))
print(f'Validation Dice score: {best_model_validation_dice:.4f}')

plot_loss_curves(training_losses, validation_losses, args.out_dir)
plot_validation_dice(validation_dice_scores, args.out_dir)
save_test_predictions(model, test_loader, device, os.path.join(args.out_dir, "best_model.pth"), args.out_dir)
