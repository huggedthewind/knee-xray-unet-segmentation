from args import get_args
from dataset import KneeDataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from model import UNetLext
import torch

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'validation.csv'))

train_dataset = KneeDataset(train_set)
validation_dataset = KneeDataset(val_set)

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False)

model = UNetLext(
    basic_width=32,
    depth=4,
    input_channels=1,
    output_channels=1
).to(device)
