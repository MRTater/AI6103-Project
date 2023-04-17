import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets


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

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, img_floder, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.img_folder = img_floder
        self.img_names = []

        for root, file, names in os.walk(self.img_folder):
            for name in names:
                if name.endswith("png") or name.endswith("jpg"):
                    self.img_names.append(os.path.join(root, name))
        print(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
        print("-------")
        print(type(img))
        return img

    def __len__(self):
        print(len(self.img_names))
        return len(self.img_names)

def lamfunc(x):
    return x * 2.0 - 1.0

def load_FaceDataset(dataset_folder, img_size, bs, num_workers):
    data_transforms = transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)], [0.5 for _ in range(3)]
            ),
            transforms.Lambda(lamfunc)
        ]
    )

    dataset = FaceDataset(img_floder=dataset_folder, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    datas = load_FaceDataset("data/a tiny sample (30 images)", 256, 3, 1)
    print(type(datas))
    for i, data in enumerate(datas):
        print(i)
        print(data)