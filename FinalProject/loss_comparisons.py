import matplotlib.pyplot as plt

# -------------------------------------------------
# Loss values you provided
# -------------------------------------------------

loss_with_aug = [
    1.5736, 0.9843, 0.8269, 0.7477, 0.6881,
    0.6494, 0.6232, 0.5841, 0.5804, 0.5552,
    0.5535, 0.5304, 0.5249, 0.5059, 0.4939
]

loss_without_aug = [
    0.4260, 0.1945, 0.1543, 0.1354, 0.1190,
    0.1053, 0.0958, 0.0884, 0.0859, 0.0794,
    0.0765, 0.0724, 0.0667, 0.0680, 0.0596
]

epochs = range(1, 16)

# -------------------------------------------------
# Create the plot
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    loss_with_aug,
    marker="o",
    label="Training Loss (With Augmentations)",
    linewidth=2
)

plt.plot(
    epochs,
    loss_without_aug,
    marker="s",
    label="Training Loss (No Augmentations)",
    linewidth=2
)

# Labels and aesthetics
plt.title("Training Loss Comparison: With vs. Without Data Augmentation", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss (Cross-Entropy)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12)

# Save the figure
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=200)

print("Saved plot as loss_comparison.png")

