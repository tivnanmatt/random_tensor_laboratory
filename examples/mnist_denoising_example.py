# mnist_denoising_example.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from random_tensor_laboratory.networks import DenseNet

# Hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
iterations_per_epoch = 10
noise_level = 0.1

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 28, 28))  # Ensuring shape [batch_size, 1, 28, 28]
])

train_dataset = datasets.MNIST(root='/home/random_tensor_laboratory/data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the DenseNet model
input_shape = (1, 28, 28)
output_shape = (1, 28, 28)
hidden_channels_list = [1024]
activation = 'prelu'

model = DenseNet(input_shape=input_shape, output_shape=output_shape, hidden_channels_list=hidden_channels_list, activation=activation)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, _) in enumerate(train_loader):

        noise = noise_level * torch.randn_like(images)
        noisy_images = images + noise
        
        # Forward pass
        outputs = model(noisy_images)
        loss = loss_fn(outputs, noise)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        
        if i == iterations_per_epoch:
            break
    
    epoch_loss = running_loss / (iterations_per_epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Training completed.')