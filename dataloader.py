from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class ScatterPointDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir,transform=None):
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.transform = transform
        if transform is None: 
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL to tensor [0, 1]
            ])
        else:
            self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        heatmap_path = os.path.join(self.heatmap_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        heatmap = Image.open(heatmap_path).convert('L')

        image = self.transform(image)
        heatmap = self.transform(heatmap)
        
        return image, heatmap

        
