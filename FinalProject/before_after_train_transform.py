#!/usr/bin/env python3
"""
Generate PNG examples showing the effect of this transform:

    transforms.RandomAffine(
        degrees=10,
        translate=(0.2, 0.2)
    )

Outputs:
    before_after_examples/before_0.png
    before_after_examples/after_0.png
    before_after_examples/before_1.png
    before_after_examples/after_1.png
    ...
"""

from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import random
import torch

# ------------------------------------------
# Configuration
# ------------------------------------------
N = 5  # number of examples to save
DIGIT = 5  # pick any digit you want: 0–9
out_dir = Path("before_after_examples")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------
# Original (no augmentation) transform
# ------------------------------------------
base_transform = transforms.ToTensor()

# ------------------------------------------
# Your augmentation transform
# ------------------------------------------
aug_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.2, 0.2)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------------------------------
# Load MNIST once
# ------------------------------------------
mnist = datasets.MNIST(
    root="~/MNIST_data",
    train=True,
    download=True
)

# Find all indices of digit = DIGIT
digit_indices = [i for i in range(len(mnist)) if mnist[i][1] == DIGIT]

# Sample N random examples
random_indices = random.sample(digit_indices, N)

print("Selected MNIST indices:", random_indices)

# ------------------------------------------
# Process examples
# ------------------------------------------
for k, idx in enumerate(random_indices):

    # Original PIL image
    img_pil, label = mnist[idx]

    # Save BEFORE (raw MNIST)
    before_path = out_dir / f"before_{k}.png"
    img_pil.save(before_path)
    print("Saved:", before_path)

    # Convert to tensor, apply augmentation ON THE PIL
    # Important: RandomAffine expects a PIL image
    aug_img_tensor = aug_transform(img_pil)  # (1, 28, 28), normalized

    # Undo normalization for visualization: [-1,1] → [0,255]
    aug_np = aug_img_tensor.squeeze().numpy()
    aug_np = (aug_np * 0.5 + 0.5) * 255
    aug_np = aug_np.clip(0, 255).astype("uint8")

    # Save AFTER
    after_path = out_dir / f"after_{k}.png"
    Image.fromarray(aug_np, mode="L").save(after_path)
    print("Saved:", after_path)

print("\nDone! Check the before_after_examples/ folder.")

