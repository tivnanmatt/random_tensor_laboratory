import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random_tensor_laboratory as rtl

# Define the drift and diffusion terms for the forward process
def forward_drift(x, t):
    return torch.zeros_like(x)

def forward_diffusion(x, t):
    g = 1.0  # constant diffusion rate
    return rtl.diffusion.sde.ScalarLinearOperator(g)

# Create an instance of the forward SDE
forward_sde = rtl.diffusion.sde.StochasticDifferentialEquation(f=forward_drift, G=forward_diffusion)

# Define the score function for the reverse SDE
def score_function(x, t):
    g = 1.0  # constant diffusion rate
    return -x / (g**2 * t)

# Create an instance of the reverse SDE using the score function
reverse_sde = forward_sde.reverse_SDE_given_score_estimator(score_function)

# Define initial condition at t=1 (standard normal distribution)
x0 = torch.randn(100)  # 100 initial conditions
reverse_timesteps = torch.linspace(1, 0, 101)**2.0

# Sample from the reverse SDE
reverse_samples = reverse_sde.sample(x0, reverse_timesteps, sampler='euler', return_all=True)

# Convert reverse samples to numpy for plotting
reverse_samples_np = [s.numpy() for s in reverse_samples]

# Create a figure for the animation
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(-3, 3)
# ax.set_title('Reverse Scalar SDE: $dx_t = 0 dt + 1 dw_t$')
ax.set_title('Reverse Scalar SDE: $dx_t = (-x / (1^2) t) dt + 1 dw_t$')
ax.set_xlabel('Time')
ax.set_ylabel('Position (x)')

# Reverse the x-axis to show the reverse process moving from right to left
ax.invert_xaxis()

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
    # Plot reverse process
    x_rev = reverse_timesteps[:i].numpy()
    for j in range(100):
        y_rev = [sample[j] for sample in reverse_samples_np[:i]]
        ax.plot(x_rev, y_rev, lw=1)
    return []

# Determine the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(reverse_timesteps), interval=100, blit=True)

# Save the animation
output_path = os.path.join(script_dir, 'reverse_scalar_sde.mp4')
writer = FFMpegWriter(fps=10)
anim.save(output_path, writer=writer)

