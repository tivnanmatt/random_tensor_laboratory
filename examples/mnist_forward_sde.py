# mnist_sde_example.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from random_tensor_laboratory.diffusion.sde import SongVariancePreservingProcess
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hyperparameters
batch_size = 64
num_timesteps = 32
beta = 5.0
noise_level = 0.1
output_file = '/home/random_tensor_laboratory/examples/mnist_sde_forward_process.mp4'

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 28, 28)),  # Ensuring shape [batch_size, 1, 28, 28]
    transforms.Lambda(lambda x: x*1)
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the SongVariancePreservingProcess SDE
sde = SongVariancePreservingProcess(beta=beta)

# Get a batch of MNIST images
images, _ = next(iter(train_loader))
images = images[:16]  # Taking only the first 16 images for visualization

# Define timesteps
timesteps = torch.linspace(0, 1, num_timesteps)

# Sample from the forward process
all_samples = sde.sample(images, timesteps, sampler='euler', return_all=True)

# Create a video of the forward process
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

def update_frame(timestep_index):
    for ax in axes.flatten():
        ax.clear()
        ax.axis('off')
    
    current_images = all_samples[timestep_index]
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(current_images[i].squeeze().cpu().detach().numpy(), cmap='gray', vmin=-5, vmax=5)
    
    print('Animating frame ', timestep_index, ' of ', len(timesteps))
    return axes

ani = animation.FuncAnimation(fig, update_frame, frames=len(timesteps), interval=100, repeat=False)
writer = animation.FFMpegWriter(fps=10)
ani.save(output_file, writer=writer)

print('Video saved to', output_file)