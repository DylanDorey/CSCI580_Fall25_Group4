#!/usr/bin/env python3
"""
visualize_digits_norm.py

Generate before/after PNGs for class-collected digit images in ../digits
showing the effect of the test-time transform:

    transforms.ToTensor()
    transforms.Normalize((0.5,), (0.5,))

Outputs (in normalized_class_examples/):
    before_0_digit_X.png
    after_0_digit_X.png
    before_1_digit_Y.png
    after_1_digit_Y.png
    ...
"""

from pathlib import Path
from PIL import Image
import numpy as np
import random
from torchvision import transforms

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
N_SAMPLES = 6                     # how many examples to visualize
DIGITS_DIR = Path("../digits")    # class digit folder
OUT_DIR = Path("normalized_class_examples")
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)

# ---------------------------------------------------------------------
# This matches your TEST transform:
# ToTensor() -> [0,1], then Normalize(0.5, 0.5) -> [-1,1]
# ---------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------------------------------------------------------
# Collect all digit PNGs from ../digits
# ---------------------------------------------------------------------
png_paths = sorted(DIGITS_DIR.glob("*.png"))
if not png_paths:
    raise SystemExit(f"No PNG files found in {DIGITS_DIR}")

# sample up to N_SAMPLES
if len(png_paths) <= N_SAMPLES:
    chosen_paths = png_paths
else:
    chosen_paths = random.sample(png_paths, N_SAMPLES)

print("Selected files:")
for p in chosen_paths:
    print("  ", p.name)

# ---------------------------------------------------------------------
# Helper: parse label from filename like "3-4-1.png" -> 3
# ---------------------------------------------------------------------
def get_label_from_name(path: Path):
    stem = path.stem  # e.g., "3-4-1"
    parts = stem.split("-")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None

# ---------------------------------------------------------------------
# Main loop: for each chosen image, save before/after
# ---------------------------------------------------------------------
for k, png_path in enumerate(chosen_paths):
    label = get_label_from_name(png_path)
    label_str = "unk" if label is None else str(label)

    # -------- BEFORE (raw class digit image, resized 28x28) --------
    img_pil = Image.open(png_path).convert("L")
    img_pil = img_pil.resize((28, 28))

    before_path = OUT_DIR / f"before_{k}_digit_{label_str}.png"
    img_pil.save(before_path)
    print("Saved:", before_path)

    # -------- AFTER (apply test transform: ToTensor + Normalize) --------
    # Apply transform: result is tensor (1, 28, 28) in [-1, 1]
    transformed = test_transform(img_pil)

    # Undo normalization so we can save as a normal PNG:
    # x_norm = (x - 0.5)/0.5  => x = 0.5*x_norm + 0.5
    transformed_np = transformed.squeeze().numpy()        # (28, 28), [-1, 1]
    transformed_np = transformed_np * 0.5 + 0.5           # back to [0, 1]
    transformed_np = (transformed_np * 255.0).clip(0, 255).astype("uint8")

    after_img_pil = Image.fromarray(transformed_np, mode="L")
    after_path = OUT_DIR / f"after_{k}_digit_{label_str}.png"
    after_img_pil.save(after_path)
    print("Saved:", after_path)

print("\nDone! Check the 'normalized_class_examples/' folder for before/after PNGs.")

