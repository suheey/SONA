import os
import random
import pickle
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset
import torchvision.datasets as D
import torchvision.transforms as T

class ImageListDataset(Dataset):
    def __init__(self, root, data_dict, transform=None, resolution=512):
        self.root = root
        self.data_dict = data_dict
        self.transform = transform
        self.resolution = resolution 

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_path, label = list(self.data_dict.items())[idx]

        # img_path = list(self.data_dict.keys())[idx]
        # label = list(self.data_dict.values())[idx]
        # print(img_path, label)
        
        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((self.resolution, self.resolution), resample=Image.BILINEAR)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32) # [-2, 0]
        image = torch.from_numpy(image).permute(2, 0, 1) # [-1, 1]
        return image, label, img_path

class CustomDataset(D.ImageFolder):
    def __init__(self, root, transform, resolution):
        super().__init__(root, transform=transform)
        self.resolution = resolution

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)

        image = image.convert('RGB')
        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((self.resolution, self.resolution), resample=Image.BILINEAR)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label, path


def load_dataset(name, root, split="train", resolution=256):
    if name.startswith("imagenet"):
        # transform = T.Compose([
        #     T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
        if split == "train" or split == "test":
            root = os.path.join(root, "train" if split == "train" else "test")
            dataset = D.ImageFolder(root, transform=None)
            dataset.num_classes = len(dataset.classes)
            
        elif split =="sun" or split =="places":
            root = os.path.join(root, "ood/places" if split == "places" else "ood/sun")
            dataset = D.ImageFolder(root, transform=None)
            dataset = get_subset_with_len(dataset, length=3000, shuffle=True)
        
        elif split == "generate":
            root = os.path.join(root)
            dataset = CustomDataset(root, transform=None, resolution=resolution)
            # dataset.num_classes = len(dataset.classes)
            # data_root = '/workspace/sanghyu.yoon/nfsdata/home/sanghyu.yoon/code/research/research_24/neurips2024/OpenOOD/data/images_largescale/'
            # with open('./pred_labels/label_None.pkl', 'rb') as f:
            #     loaded_object = pickle.load(f)

            
            # dataset = ImageListDataset(data_root, loaded_object, resolution=resolution) 
            dataset.num_classes = 200

        dataset.name = split
        return dataset

    raise NotImplementedError


def get_subset_with_len(dataset, length, shuffle=False):

    dataset_size = len(dataset)
    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)
    assert len(subset) == length

    return subset
