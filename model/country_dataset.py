import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from model.get_images import path

# Update path to look in compressed_dataset subdirectory
dataset_path = os.path.join(path, "compressed_dataset")

class CountryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with country folders containing images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to standard size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Get list of countries (folders)
        self.countries = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        if not self.countries:
            raise ValueError(f"No country folders found in {root_dir}. Please ensure the dataset is properly organized with country folders.")
        
        print(f"Found {len(self.countries)} countries in the dataset:")
        for country in self.countries:
            country_path = os.path.join(root_dir, country)
            num_images = len([f for f in os.listdir(country_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"- {country}: {num_images} images")
        
        # Create country to index mapping
        self.country_to_idx = {country: idx for idx, country in enumerate(self.countries)}
        
        # Get all image paths and their labels
        self.image_paths = []
        self.labels = []
        
        for country in self.countries:
            country_path = os.path.join(root_dir, country)
            for img_name in os.listdir(country_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(country_path, img_name))
                    self.labels.append(self.country_to_idx[country])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_num_classes(self):
        return len(self.countries)
    
    def get_class_names(self):
        return self.countries


# attempt to load the dataset from cache kaggle
try:
    # print(f"Looking for dataset in: {dataset_path}")
    full_dataset = CountryDataset(dataset_path)
    # print(f"\nTotal number of images in dataset: {len(full_dataset)}")
    # print(f"Number of classes (countries): {full_dataset.get_num_classes()}")
    # print("\nClass names (countries):")
    # for country in full_dataset.get_class_names():
    #     print(f"- {country}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    print("\nPlease ensure the dataset is properly organized with country folders.")
    print("Each country should have its own folder containing the images for that country.")
