from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF


class KneeDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        xray_path = row["xrays"]
        xray_image = Image.open(xray_path).convert("L")

        mask_path = row["masks"]
        mask_image = Image.open(mask_path).convert("L")

        xray_tensor = TF.to_tensor(xray_image)
        mask_tensor = TF.to_tensor(mask_image)

        return xray_tensor, mask_tensor
