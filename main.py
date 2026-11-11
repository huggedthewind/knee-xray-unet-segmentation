from args import get_args
from dataset import KneeDataset, KneeTestDataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from model import UNetLext
import torch
from trainer import train_model
from evaluate import evaluate_model, save_test_predictions
from utils import plot_loss_curves

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'validation.csv'))
test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))

train_dataset = KneeDataset(train_set)
validation_dataset = KneeDataset(val_set)
test_dataset = KneeTestDataset(test_set)

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

model = UNetLext(basic_width=32, depth=4,input_channels=1, output_channels=1).to(device)

training_losses, validation_losses = train_model(model, train_loader, validation_loader, device, args)

validation_dice_score = evaluate_model(model, validation_loader, device, os.path.join(args.out_dir, "best_model.pth"))
print(f'Validation Dice score: {validation_dice_score:.4f}')

plot_loss_curves(training_losses, validation_losses, args.out_dir)
save_test_predictions(model, test_loader, device, os.path.join(args.out_dir, "best_model.pth"), args.out_dir)