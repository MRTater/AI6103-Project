import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms 


class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transform = None, target_transform=None):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.transform = transform
        self.target_transform = target_transform
        # self.imagefolder = os.path.join(root, "pokemon/pokemon")
        self.image_folder = img_folder
        self.imagenames = []
        for root, files, names in os.walk(self.image_folder):
            for name in names:
                if name.endswith("png") or name.endswith("jpg"):
                    self.imagenames.append(os.path.join(root, name))

    def __getitem__(self, index):
        img_name = self.imagenames[index]
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imagenames)

def load_transformed_dataset(dataset_folder, img_size, bs, num_workers=4):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)
    data = PokemonDataset(dataset_folder, transform=data_transform)
    dataloader = DataLoader(data, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers)
    return dataloader


