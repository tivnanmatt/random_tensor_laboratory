import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random_tensor_laboratory as rtl

# Define the drift and diffusion terms
def drift(x, t):
    return torch.zeros_like(x)

def diffusion(x, t):
    g = 1.0  # constant diffusion rate
    return rtl.diffusion.sde.ScalarLinearOperator(g)

# Create an instance of the SDE
sde = rtl.diffusion.sde.StochasticDifferentialEquation(f=drift, G=diffusion)

# Define initial condition and time steps
x0 = torch.zeros(100)  # 100 initial conditions
timesteps = torch.linspace(0, 1, 100)**2.0

# Sample from the SDE
samples = sde.sample(x0, timesteps, sampler='euler', return_all=True)

# Convert samples to numpy for plotting
samples_np = [s.numpy() for s in samples]

# Create a figure for the animation
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(-3, 3)
ax.set_title('Forward Scalar SDE: $dx_t = 0 dt + 1 dw_t$')
ax.set_xlabel('Time')
ax.set_ylabel('Position (x)')

# Initialization function for the animation
def init():
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)
    return []

# Animation function
def animate(i):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)
    x = timesteps[:i].numpy()
    for j in range(100):
        y = [sample[j] for sample in samples_np[:i]]
        ax.plot(x, y, lw=1)
    return []

# Determine the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(timesteps), interval=100, blit=True)

# Save the animation
output_path = os.path.join(script_dir, 'forward_scalar_sde.mp4')
writer = FFMpegWriter(fps=10)
anim.save(output_path, writer=writer)
