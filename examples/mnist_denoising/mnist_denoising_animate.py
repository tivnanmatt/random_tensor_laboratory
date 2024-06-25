import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from random_tensor_laboratory.networks import DenseNet

# Hyperparameters
batch_size = 64
noise_level = 0.4

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 28, 28))  # Ensuring shape [batch_size, 1, 28, 28]
])

test_dataset = datasets.MNIST(root='/home/random_tensor_laboratory/data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the DenseNet model
input_shape = (1, 28, 28)
output_shape = (1, 28, 28)
hidden_channels_list = [1024, 2048, 4096, 2048, 1024]
activation = 'prelu'

model = DenseNet(input_shape=input_shape, output_shape=output_shape, hidden_channels_list=hidden_channels_list, activation=activation)

# Load the trained model weights
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'mnist_denoising.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# Get some test images
data_iter = iter(test_loader)
images, _ = next(data_iter)

# Add noise to the images
noise = noise_level * torch.randn_like(images)
noisy_images = images + noise

# Denoise the images using the model
with torch.no_grad():
    denoised_images = noisy_images - model(noisy_images)

# Convert images to numpy for plotting
images_np = images.numpy()
noisy_images_np = noisy_images.numpy()
denoised_images_np = denoised_images.numpy()

# Create a figure for the animation
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Initialize the image handles
ground_truth_im = axes[0].imshow(images_np[0].reshape(28, 28), cmap='gray', animated=True)
noisy_im = axes[1].imshow(noisy_images_np[0].reshape(28, 28), cmap='gray', animated=True)
denoised_im = axes[2].imshow(denoised_images_np[0].reshape(28, 28), cmap='gray', animated=True)

# set the clim to (0, 1) for all images
ground_truth_im.set_clim(0, 1)
noisy_im.set_clim(0, 1)
denoised_im.set_clim(0, 1)

# Set titles for the subplots
axes[0].set_title('Ground Truth')
axes[1].set_title('Noisy')
axes[2].set_title('Denoised')

# Hide axis ticks
for ax in axes:
    ax.axis('off')

def init():
    return ground_truth_im, noisy_im, denoised_im

def animate(i):
    ground_truth_im.set_data(images_np[i].reshape(28, 28))
    noisy_im.set_data(noisy_images_np[i].reshape(28, 28))
    denoised_im.set_data(denoised_images_np[i].reshape(28, 28))
    return ground_truth_im, noisy_im, denoised_im

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(images_np), interval=200, blit=True)


# Save the animation
output_path = os.path.join(script_dir, 'mnist_denoising_animation.mp4')
writer = FFMpegWriter(fps=10)
anim.save(output_path, writer=writer)

print('Animation saved to', output_path)