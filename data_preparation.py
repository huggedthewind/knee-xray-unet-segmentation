from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Complete master CSV
xrays_dir = Path("data/xrays")
masks_dir = Path("data/masks")
csv_dir = Path("data/CSVs")
outputs_dir = Path("outputs")

csv_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

rows = []

xray_paths = sorted(
    xrays_dir.glob("*.png"),
    key=lambda path: int(path.stem)
)

for xray_path in xray_paths:
    mask_path = masks_dir / xray_path.name

    if mask_path.exists():
        mask_value = str(mask_path)
    else:
        mask_value = "None"

    rows.append({
        "xrays": str(xray_path),
        "masks": mask_value
    })

dataset_df = pd.DataFrame(rows)
dataset_df.to_csv(csv_dir / "dataset.csv", index=False)

print(dataset_df)
print(f"Total samples: {len(dataset_df)}")
print(f"Labeled samples: {(dataset_df['masks'] != 'None').sum()}")
print(f"Test samples: {(dataset_df['masks'] == 'None').sum()}")

# Test dataset CSV
test_df = dataset_df[dataset_df["masks"] == "None"]
test_df.to_csv(csv_dir / "test.csv", index=False)

print("\nTest dataset:")
print(test_df)
print(f"Test samples: {len(test_df)}")

# Train and val CSVs
labeled_df = dataset_df[dataset_df["masks"] != "None"]

train_df, validation_df = train_test_split(
    labeled_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df.to_csv(csv_dir / "train.csv", index=False)
validation_df.to_csv(csv_dir / "validation.csv", index=False)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")

# Bar chart with number of samples in each CSV
split_names = ["Dataset", "Train", "Validation", "Test"]

sample_counts = [
    len(dataset_df),
    len(train_df),
    len(validation_df),
    len(test_df)
]

plt.bar(split_names, sample_counts)

plt.xlabel("CSV file")
plt.ylabel("Number of samples")
plt.title("Dataset Split Sizes")

plt.tight_layout()
plt.savefig(outputs_dir / "csv_sample_counts.png")
plt.show()
