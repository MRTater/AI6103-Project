import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms


def load_transformed_dataset(img_size, batch_size, num_workers=4):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)
    train = torchvision.datasets.StanfordCars(root="/home/msai/zfu009/dataset/", download=False, transform=data_transform)

    test = torchvision.datasets.StanfordCars(root="/home/msai/zfu009/dataset/", download=False, transform=data_transform, split='test')
    data = torch.utils.data.ConcatDataset([train, test])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return dataloader
