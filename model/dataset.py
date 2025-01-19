import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
# from build_dataset import dataset_path
from get_images import path
class GeoGuessrDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # Read coordinates from CSV instead of .npy
        self.coords_df = pd.read_csv(os.path.join(data_dir, 'coords.csv'), 
                                   header=None, 
                                   names=['latitude', 'longitude'])
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, idx):
        # Images are named as n.png
        img_path = os.path.join(self.data_dir, f'{idx}.png')
        
        # Load and transform image
        img = self.pil_loader(img_path)
        if self.transform:
            img = self.transform(img)
        
        # Get coordinates as target
        coords = self.coords_df.iloc[idx]
        target = torch.tensor([coords.latitude, coords.longitude], dtype=torch.float)
        
        return img, target
    
    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
        

full_dataset = GeoGuessrDataset(path)

# Set sizes
train_size = 1000
val_size = 100
test_size = 100

# Create subsets using indices
train_dataset = Subset(full_dataset, range(0, train_size))
val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
test_dataset = Subset(full_dataset, range(train_size + val_size, train_size + val_size + test_size))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print("Loading training data...")
# for batch in tqdm(train_loader, desc="Training batches"):
#     images, coords = batch
#     pass
