# final_project.py
# authors: dylan dorey, ethan harris, halin gailey, jake baartman

# trains a simple fully connected neural network (MLP) on MNIST
# - evaluates it on both MNIST and a class digits dataset
# - computes per class loss and confusion matrices
# - generates per group performance stats (loss and acc)

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms


# -------------------------------------------------------------------- #
# -------------------- Global config / device ------------------------ #
# -------------------------------------------------------------------- #

# digits 0 - 9
NUM_CLASSES = 10

# use the GPU here if it is available, if not use CPU like usual
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)


# -------------------------------------------------------------------- #
# --------------------- Visualization helpers ------------------------ #
# - These functions are for plotting and presentaiton purposes only -- #
# -- confusion matrix with counts ------------------------------------ #
# -- per class loss table -------------------------------------------- #
# -- per group accuracy and loss bar charts -------------------------- #
# -------------------------------------------------------------------- #

# plot and save confusion matrix with numeric labels  
def save_confusion_matrix(cm, filename, title):

    fig, ax = plt.subplots(figsize=(6, 5))
    # use blue colormap 
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(range(NUM_CLASSES))
    ax.set_yticklabels(range(NUM_CLASSES))

    # colorbar to show
    #plt.colorbar(im, ax=ax)
 
    # add the raw count text inside each cell
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j, i,
                str(value),
                ha="center",
                va="center",
                # high values use white text and low values use black
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {filename}")

# plot a table for per class average loss
# - row 1: digits 0 - 1
# - row 2: per class loss 
def save_loss_table(class_avg_loss, filename):
    
    class_avg_loss = np.asarray(class_avg_loss)

    # visualization formatting: no leading 0 for decimal values < 1 and only 2 sf
    loss_row = [f"{loss:.2f}"[1:] if loss < 1 else f"{loss:.2f}" for loss in class_avg_loss]

    digit_row = [str(d) for d in range(NUM_CLASSES)]
    table_data = [digit_row, loss_row]

    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )

    # visualization formatting: parameters for compactness
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 0.6)

    # header row: light blue
    # data row: white and small padding
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.05
        if row == 0:
            cell.set_facecolor("#dbe9ff")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

        cell.set_height(0.15)

    plt.tight_layout(pad=0.1)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss table to {filename}")

# plot per gorup accuracy and loss as side by side bar chart
# blue bars: accuracy %
# orange bars: loss
def plot_group_results(group_ids, group_losses, group_accs,
                       filename_plot="group_results.png",
                       filename_legend="group_results_legend.png"):
    
    group_ids = list(group_ids)
    group_losses = list(group_losses)
    group_accs = list(group_accs)

    x = np.arange(len(group_ids))
    # thinner bars to fit better on the slides
    width = 0.25
    
    # main square figure no legend
    fig, ax1 = plt.subplots(figsize=(6, 6))
   
    # use twin axies bc accurcay and loss are on different scales
    # - left y axis: accuracy %
    acc_bars = ax1.bar(
        x - width / 2,
        [a * 100.0 for a in group_accs],
        width,
        label="Accuracy (%)",
        color="tab:blue",
        edgecolor="black"
    )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)

    # - right y axis: loss values
    ax2 = ax1.twinx()
    loss_bars = ax2.bar(
        x + width / 2,
        group_losses,
        width,
        label="Loss",
        color="tab:orange",
        edgecolor="black"
    )
    ax2.set_ylabel("Loss")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(g) for g in group_ids], fontsize=10)
    ax1.set_xlabel("Group")
    ax1.set_title("Per Group Accuracy and Loss", fontsize=13)

    # handles for separate legend figure
    # - remvoe legend form the main plot to keep it readable
    handles = [acc_bars, loss_bars]
    labels = ["Accuracy (%)", "Loss"]

    fig.tight_layout()
    fig.savefig(filename_plot, dpi=200)
    plt.close(fig)
    print(f"Saved per group plot as {filename_plot}")

    # create separate legend figure so the main plor remains uncluttered
    # - this is useful for positioning stuff on our slides
    legend_fig, legend_ax = plt.subplots(figsize=(3, 2))
    legend_ax.axis("off")

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        fontsize=11
    )

    legend_fig.tight_layout()
    legend_fig.savefig(filename_legend, dpi=200, bbox_inches="tight")
    plt.close(legend_fig)
    print(f"Saved per group plot legend as {filename_legend}")


# -------------------------------------------------------------------- #
# ------------- Data loading helpers + dataset wrappers -------------- #
# - wrap the png files into np arrays and PyTorch datasets ----------- #
# -------------------------------------------------------------------- #

# loads all class digits from the directory into np arrays
# - expects filenames of the form: digit-group-member.png
# - only the first field is used as a label here (the digit)
# - filters out data based on: too faint, low contrast, too few bright pixels
def project_data_loader(digits_dir="../digits"):

    digits_path = Path(digits_dir)
    image_list = []
    label_list = []

    for png_path in sorted(digits_path.glob("*.png")):
        
        # filenames follow as convention: digit-group-member.png
        # - we just need the digit here (first field)
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 1:
            continue
        try:
            label = int(parts[0])
        except ValueError:
            continue

        # load and normalize to 28x28 grayscale to match MNIST format
        img = Image.open(png_path).convert("L")
        # resize to 28x28 to match MNIST dimensions and model input requirements
        img = img.resize((28, 28))

        arr = np.array(img, dtype=np.float32)

        # quality filtering heuristics:
        # - max_pix < 70: the digit is too faint to be used
        # - foregroung_pixels: ensures the image isnt blank or just noise
        # - contrast < 30: aoids extremely low contrast digits that confuse the model
        max_pix = arr.max()
        foreground_pixels = (arr > 60).sum()
        contrast = arr.max() - arr.min()

        # these threshold prevent extremely low quality or blank digits
        # from corrupting the test results
        if max_pix < 70:
            print(f"Skipping {png_path.name}: too faint (max={max_pix})")
            continue

        if foreground_pixels < 10:
            print(f"Skipping {png_path.name}: almost blank ({foreground_pixels} bright pixels)")
            continue

        if contrast < 30:
            print(f"Skipping {png_path.name}: low contrast (range={contrast})")
            continue

        image_list.append(arr)
        label_list.append(label)

    # stack into (N, 28, 28)
    # if no images, return empty np array with same shape
    images = np.stack(image_list, axis=0) if image_list else np.empty((0, 28, 28), dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return images, labels

# load digits and group them by groupID
# - expected filenames: digit-group-member.png
# returns dict[group_id] = (images_np, labels_np)
# applies the same heuristic filters as project_data_loader() above
def load_grouped_digits(digits_dir="../digits"):

    digits_path = Path(digits_dir)
    group_images = {}
    group_labels = {}

    for png_path in sorted(digits_path.glob("*.png")):
        
        # filenames follow as convention: digit-group-member.png
        # - we just need the digit here (first field)
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 2:
            continue
        try:
            label = int(parts[0])
            group = int(parts[1])
        except ValueError:
            continue

        # load and normalize to 28x28 grayscale to match MNIST format
        img = Image.open(png_path).convert("L")
        # resize to 28x28 to match MNIST dimensions and model input requirements
        img = img.resize((28, 28))

        arr = np.array(img, dtype=np.float32)

        # quality filtering heuristics:
        # - max_pix < 70: the digit is too faint to be used
        # - foregroung_pixels: ensures the image isnt blank or just noise
        # - contrast < 30: aoids extremely low contrast digits that confuse the model
        max_pix = arr.max()
        foreground_pixels = (arr > 60).sum()
        contrast = arr.max() - arr.min()

        # these threshold prevent extremely low quality or blank digits
        # from corrupting the test results
        if max_pix < 70:
            continue
        if foreground_pixels < 20:
            continue
        if contrast < 30:
            continue

        group_images.setdefault(group, []).append(arr)
        group_labels.setdefault(group, []).append(label)

    grouped = {}
    for g in group_images:
        images = np.stack(group_images[g], axis=0).astype(np.float32)
        labels = np.array(group_labels[g], dtype=np.int64)
        grouped[g] = (images, labels)

    return grouped

# wraps the class digits form project_data_loader() so we cna use them
# in PyTorch dataloader with the same trasnforms as MNIST
class project_digits_dataset(torch.utils.data.Dataset):

    def __init__(self, digits_dir="../digits", transform=None):
        self.images_np, self.labels_np = project_data_loader(digits_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        img_arr = self.images_np[idx]
        label = int(self.labels_np[idx])
        
        # convert np array back to PIL bc torchvision transforms operate on PIL images
        img = Image.fromarray(img_arr.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# dataset wrapper for pre loaded np image arrays and labels 
# used for per group evaluation
class numpy_digits_dataset(torch.utils.data.Dataset):

    def __init__(self, images_np, labels_np, transform=None):
        self.images_np = images_np
        self.labels_np = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        img_arr = self.images_np[idx]
        label = int(self.labels_np[idx])
        
        # convert np array back to PIL bc torchvision transforms operate on PIL images
        img = Image.fromarray(img_arr.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# -------------------------------------------------------------------- #
# ------------ Model + evaluation + training helpers ----------------- #
# -------------------------------------------------------------------- #

# fully connected MLP for 28x28 digits
# architecture:
# - input: 784 dim flattened image
# - hidden layers: 512 -> 256 -> 128 with ReLU + dropout
# - output: 10 logits (1 for each digit class)
class MLP(nn.Module):

    # layer sizes chosen to balane capacity and training speed
    # tried 256 -> 128 -> 64 and it didnt perform as well
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, NUM_CLASSES)
        # dropout added to reduce overfitting
        # - important for training on MNIST
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten the 28x28 image into 784 legnth vector for fully connected layers
        # - flatten from (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        # dropout adds regularization and reduces overfitting on MNIST
        # - important bc the model is large for a simple dataset
        # - apply dropout after each hidden layer to improve the generalization
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

# evaluate the model on a given DataLoader
# returns:
# - avg_loss: mean cross entropy over all samples
# - accuracy: fraction of correctly classified samples
# - cm: confusion matrix
# - class_avg_loss: per class average loss
def evaluate(model, dataloader, device, criterion, compute_confusion=False, num_classes=NUM_CLASSES):
    
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    cm = None
    if compute_confusion:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # for per class loss analysis
    class_loss_sum = np.zeros(num_classes, dtype=np.float64)
    class_loss_count = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            # predicted class indices
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # update confusion matrix if requested:
            # - increment [true_label, predicted_label]
            if compute_confusion:
                true_np = labels.cpu().numpy()
                pred_np = preds.cpu().numpy()
                for t, p in zip(true_np, pred_np):
                    cm[t, p] += 1
            
            # compute loss per individual sample rather than batch mean
            # this allows us to accumulate oer class average loss later
            # this helps indentify whcih digits the model struggles with
            per_item_loss = nn.functional.cross_entropy(
                logits, labels, reduction="none"
            ).cpu().numpy()

            for lbl, loss_val in zip(labels.cpu().numpy(), per_item_loss):
                class_loss_sum[lbl] += loss_val
                class_loss_count[lbl] += 1

    if total == 0:
        if compute_confusion:
            return float("nan"), float("nan"), cm
        else:
            return float("nan"), float("nan")

    # compute how many predictions exactly match the gorund truth labels
    avg_loss = running_loss / total
    accuracy = correct / total
    class_avg_loss = class_loss_sum / np.maximum(class_loss_count, 1)

    if compute_confusion:
        return avg_loss, accuracy, cm, class_avg_loss
    else:
        return avg_loss, accuracy

# supervised training loop on MNIST
# - tracks and returns training loss per epoch for plotting later
# - uses Adam + cross entropy on augmented MNIST data
def train_model(model, trainloader, device, criterion, optimizer, epochs):
    
    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        # track total loss over entire epoch to report average at the end of each
        running_loss = 0.0
        start_time = time.time()

        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # clear old gradients
            # - pytorch does not do thsi automatically
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            # compute gradient of loss wrt model parameters
            loss.backward()
            # update model parameters using ADam optimizer
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # average loss over all sample this epoch - monitoring convergence
        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f} - {elapsed:.1f}s")

    print("Training complete.")
    return train_losses

# evaluate the trianed model separately for each group (second number in filename)
# - helps us identify whether certain groups draw differently
# - useful for disgnosing bias or harder writing style
# - for each group:
# -- build a small dataset from that groups images
# -- run evaluate() to get loss + accuracy
# - returns: group_ids, group_losses, group_accs
def evaluate_by_group(model, device, criterion, test_transform, digits_dir="../digits"):

    # load digits grouped by the second number in filename
    # - we cna see if certain groups draw digits differently
    grouped_digits = load_grouped_digits(digits_dir)

    group_ids = sorted(grouped_digits.keys())
    group_losses = []
    group_accs = []

    for g in group_ids:
        images_np, labels_np = grouped_digits[g]
        if len(labels_np) == 0:
            continue

        group_dataset = numpy_digits_dataset(images_np, labels_np, transform=test_transform)
        group_loader = torch.utils.data.DataLoader(
            group_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        g_loss, g_acc = evaluate(
            model, group_loader, device, criterion, compute_confusion=False
        )

        # store group accuracy and loss to use in plot later
        group_losses.append(g_loss)
        group_accs.append(g_acc)
        print(f"Group {g}: loss = {g_loss:.4f}, acc = {100.0 * g_acc:.2f}% (N={len(group_dataset)})")

    return group_ids, group_losses, group_accs


# -------------------------------------------------------------------- #
# -------------------------- Main script ----------------------------- #
# - define transforms and load MNIST train/test sets ----------------- #
# - train MLP on MNIST with light data augmentation ------------------ #
# - evaluate on MNIST test set (loss and accuracy) ------------------- #
# - evaluate on class digits dataset as a whole (loss and accuracy) -- #
# - get the confusion matrix and per class average loss for each test  #
# - evaluate per group and plot group performance (loss and acc) ----- #
# -------------------------------------------------------------------- #

def main():

    batch_size = 128

    # data augmentation for trianing
    # - small random affine distortion (rotation/translation/scale/sher)
    # - helps simulate natural handwriting variation and improves robustness
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=5,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=5
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # test transform: just tensor + normalization
    # - no augmentation so evaulaiton is fair and consistent
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST training set / dataloader
    trainset = datasets.MNIST(
        root="~/MNIST_data",
        train=True,
        download=True,
        transform=train_transform
    )
    # MNISt is only sued for traiing and baseline evaluation
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # MNIST test set / dataloader
    testset = datasets.MNIST(
        root="~/MNIST_data",
        train=False,
        download=True,
        transform=test_transform
    )
    # MNIST is only used for training and baseline evaluation
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # model, loss, optimizer
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train model on MBIST
    epochs = 15

    print("--------------------------------------------------------------")
    print("\nTraining model on MNIST train set...\n")
    train_losses = train_model(model, trainloader, device, criterion, optimizer, epochs)

    # Plot training loss vs epoch for THIS single run
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", linewidth=2)
    plt.title("Training Loss vs Epoch (single run)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss (Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(1, epochs + 1))

    plt.show()

    # ---------------------------- Evaluate on MNIST test set -------------------------------- #
    # - baseline evaluation ------------------------------------------------------------------ #
    test_loss, test_acc, test_cm, mnist_avg_loss = evaluate(
        model, testloader, device, criterion, compute_confusion=True
    )
    
    print("\nTesting on MNIST test set...\n")
    save_loss_table(mnist_avg_loss, "mnist_loss_table.png")
    save_confusion_matrix(test_cm, "mnist_confusion.png", "MNIST Test Confusion Matrix")
    print("\n")
    print(f"MNIST Test loss: {test_loss:.4f}, accuracy: {100 * test_acc:.2f}%")
    print("--------------------------------------------------------------")
    # ---------------------------------------------------------------------------------------- #

    # --------------------------- Evaluate on all class digits ------------------------------- #
    # - real world performance --------------------------------------------------------------- #
    print("\nTesting on Class handwritten data...\n")
    project_dataset = project_digits_dataset("../digits", transform=test_transform)
    project_loader = torch.utils.data.DataLoader(
        project_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    proj_loss, proj_acc, proj_cm, proj_avg_loss = evaluate(
        model, project_loader, device, criterion, compute_confusion=True
    )

    print("\n")
    save_loss_table(proj_avg_loss, "class_loss_table.png")
    save_confusion_matrix(proj_cm, "project_confusion.png", "Class Digits Confusion Matrix")
    print(f"\nClass handwritten digits - loss: {proj_loss:.4f}, accuracy: {100 * proj_acc:.2f}%")
    print("--------------------------------------------------------------")
    # ---------------------------------------------------------------------------------------- #
    
    # ------------------------ Evaluate per group class digits ------------------------------- #
    # - analyze performance differences due to style varaints -------------------------------- #
    print("\nTesting on individual group digit sets...\n")
    group_ids, group_losses, group_accs = evaluate_by_group(
        model, device, criterion, test_transform, digits_dir="../digits"
    )

    print("\n")
    plot_group_results(
        group_ids,
        group_losses,
        group_accs,
        filename_plot="group_results.png",
        filename_legend="group_results_legend.png"
    )
    print("--------------------------------------------------------------")
    # ----------------------------------------------------------------------------------------- #

# ---------- Entry point ---------- #
if __name__ == "__main__":
    main()
# --------------------------------- #

