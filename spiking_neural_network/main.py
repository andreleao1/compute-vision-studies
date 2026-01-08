import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
from custom_dataset import CustomDataset

lif1 = snn.Leaky(beta=0.9)

oxford_town_dataset = '../oxford_datasets'
caviar_dataset = '../caviar_datasets'

batch_size = 128
data_path = oxford_town_dataset

img_dir_train = os.path.join(data_path, 'images/train')
label_dir_train = os.path.join(data_path, 'labels/train')

img_dir_test = os.path.join(data_path, 'images/test')
label_dir_test = os.path.join(data_path, 'labels/test')

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

dataset_train = CustomDataset(img_dir=img_dir_train, label_dir=label_dir_train, transform=transform)
dataset_test = CustomDataset(img_dir=img_dir_test, label_dir=label_dir_test, transform=transform)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Dataset de treino carregado com {len(dataset_train)} amostras.")
