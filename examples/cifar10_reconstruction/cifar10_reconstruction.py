# mnist_denoising_example.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from random_tensor_laboratory.networks import DenseNet

from random_tensor_laboratory.tasks import ImageReconstructionTask
from random_tensor_laboratory.distributions import DatasetSampler

# Hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
iterations_per_epoch = 100
noise_level = 0.4

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(3, 32, 32))  # Ensuring shape [batch_size, 1, 28, 28]
])

train_dataset = datasets.CIFAR10(root='/home/random_tensor_laboratory/data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

train_dataset_sampler = DatasetSampler(train_dataset, batch_size=batch_size)


