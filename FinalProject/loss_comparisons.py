import matplotlib.pyplot as plt

# -------------------------------------------------
# Loss values you provided
# -------------------------------------------------

# WITH AUGMENTATIONS (cold colors)
loss_aug_lr_1e2 = [
    1.5960, 1.3617, 1.3450, 1.3230, 1.3287,
    1.3114, 1.3229, 1.3159, 1.3256, 1.3219,
    1.3227, 1.3257, 1.3390, 1.3257, 1.3223
]

loss_aug_lr_1e3 = [
    0.9511, 0.4571, 0.3729, 0.3202, 0.2978,
    0.2807, 0.2672, 0.2516, 0.2387, 0.2335,
    0.2295, 0.2173, 0.2172, 0.2168, 0.2088
]

loss_aug_lr_1e4 = [
    1.5643, 0.9540, 0.6519, 0.5020, 0.4320,
    0.3750, 0.3449, 0.3151, 0.2975, 0.2768,
    0.2605, 0.2530, 0.2372, 0.2269, 0.2236
]

# WITHOUT AUGMENTATIONS (warm colors)
loss_noaug_lr_1e2 = [
    0.6821, 0.4846, 0.4674, 0.4567, 0.4695,
    0.4444, 0.4292, 0.4582, 0.4455, 0.4239,
    0.4573, 0.4565, 0.4609, 0.4705, 0.4475
]

loss_noaug_lr_1e3 = [
    0.4230, 0.1932, 0.1533, 0.1305, 0.1189,
    0.1033, 0.0997, 0.0898, 0.0825, 0.0812,
    0.0774, 0.0742, 0.0669, 0.0668, 0.0624
]

loss_noaug_lr_1e4 = [
    0.8466, 0.3540, 0.2719, 0.2189, 0.1846,
    0.1594, 0.1397, 0.1228, 0.1127, 0.1023,
    0.0936, 0.0869, 0.0816, 0.0759, 0.0701
]

epochs = range(1, 16)

# -------------------------------------------------
# Create the main plot (NO LEGEND)
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# ---- With augmentations (cold colors) ----
ax.plot(
    epochs,
    loss_aug_lr_1e2,
    marker="o",
    linestyle=":",
    label="With Augmentations (lr=1e-2)",
    linewidth=2,
    color="tab:cyan",
    alpha=0.7
)

ax.plot(
    epochs,
    loss_aug_lr_1e3,
    marker="o",
    linestyle="-",
    label="With Augmentations (lr=1e-3)",
    linewidth=3,
    color="tab:blue",
)

ax.plot(
    epochs,
    loss_aug_lr_1e4,
    marker="o",
    linestyle=":",
    label="With Augmentations (lr=1e-4)",
    linewidth=2,
    color="tab:purple",
    alpha=0.7
)

# ---- Without augmentations (warm colors) ----
ax.plot(
    epochs,
    loss_noaug_lr_1e2,
    marker="s",
    linestyle=":",
    label="No Augmentations (lr=1e-2)",
    linewidth=2,
    color="tab:orange",
    alpha=0.7
)

ax.plot(
    epochs,
    loss_noaug_lr_1e3,
    marker="s",
    linestyle="-",
    label="No Augmentations (lr=1e-3)",
    linewidth=3,
    color="tab:red",
)

ax.plot(
    epochs,
    loss_noaug_lr_1e4,
    marker="s",
    linestyle=":",
    label="No Augmentations (lr=1e-4)",
    linewidth=2,
    color="tab:olive",
    alpha=0.7
)

# Titles, labels, grid
ax.set_title("Training Loss vs Epoch\nDifferent Learning Rates and Augmentation", fontsize=14)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss (Cross-Entropy)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xticks(list(epochs))

# IMPORTANT: do NOT call ax.legend() here -> no legend on main plot

plt.tight_layout()
fig.savefig("loss_comparison_all_runs_no_legend.png", dpi=200)
print("Saved main plot (no legend) as loss_comparison_all_runs_no_legend.png")

# -------------------------------------------------
# Create separate legend-only figure
# -------------------------------------------------

# Get handles and labels from the main axes
handles, labels = ax.get_legend_handles_labels()

legend_fig, legend_ax = plt.subplots(figsize=(4, 6))
legend_ax.axis("off")

legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=True,
    fontsize=10,
)

legend_fig.tight_layout()
legend_fig.savefig("loss_comparison_legend_only.png", dpi=200, bbox_inches="tight")
print("Saved legend-only figure as loss_comparison_legend_only.png")

