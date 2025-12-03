#!/usr/bin/env python3
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import random

# Number of random 5s to save
N = 2

# Output directory
out_dir = Path("mnist_5_examples")
out_dir.mkdir(exist_ok=True)

# Load MNIST with no transform yet
raw_mnist = datasets.MNIST(root="~/MNIST_data", train=True, download=True)

# Get all indices where label == 5
indices_of_5 = [i for i, (_, label) in enumerate(raw_mnist) if label == 5]

# Randomly pick 2 different 5s
random_indices = random.sample(indices_of_5, N)

print("Random selected indices:", random_indices)

for idx_num, mnist_idx in enumerate(random_indices):
    img, label = raw_mnist[mnist_idx]

    # Convert PIL image to numpy and save
    img = img.resize((28, 28)).convert("L")
    img_np = np.array(img, dtype=np.uint8)
    img_pil = Image.fromarray(img_np, mode="L")

    filepath = out_dir / f"5_random_{idx_num}.png"
    img_pil.save(filepath)
    print(f"Saved: {filepath}")

print("Done!")

