from args import get_args
from dataset import KneeDataset
import pandas as pd
import os
from torch.utils.data import DataLoader

args = get_args()

train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'validation.csv'))

train_dataset = KneeDataset(train_set)
validation_dataset = KneeDataset(val_set)

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False)
