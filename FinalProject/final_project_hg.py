import torch, time 
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

def save_confusion_matrix(cm, filename, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    plt.colorbar(im, ax=ax)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j, i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {filename}")

def save_loss_table(class_avg_loss, filename):
    class_avg_loss = np.asarray(class_avg_loss)

    loss_row = [f"{loss:.2f}"[1:] if loss < 1 else f"{loss:.2f}" for loss in class_avg_loss]

    digit_row = [str(d) for d in range(10)]
    table_data = [digit_row, loss_row]

    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 0.6)

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
    print(f"Saved compact 2x10 loss table to {filename}")

def plot_group_results(group_ids, group_losses, group_accs,
                       filename_plot="group_results.png",
                       filename_legend="group_results_legend.png"):
    group_ids = list(group_ids)
    group_losses = list(group_losses)
    group_accs = list(group_accs)

    x = np.arange(len(group_ids))
    width = 0.25  # thin bars

    fig, ax1 = plt.subplots(figsize=(6, 6))

    acc_bars = ax1.bar(
        x - width/2,
        [a * 100.0 for a in group_accs],
        width,
        label="Accuracy (%)",
        color="tab:blue",
        edgecolor="black"
    )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    loss_bars = ax2.bar(
        x + width/2,
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
    ax1.set_title("Per-Group Accuracy and Loss", fontsize=13)
    
    handles, labels = [], []
    handles.extend([acc_bars, loss_bars])
    labels.extend(["Accuracy (%)", "Loss"])

    fig.tight_layout()
    fig.savefig(filename_plot, dpi=200)
    plt.close(fig)
    print(f"Saved square thin-bar plot WITHOUT legend as {filename_plot}")

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
    print(f"Saved standalone legend as {filename_legend}")


# i am running on the A100
# use the GPU here if it is available, if not use CPU like usual
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def ProjectDataLoader(digits_dir="../digits"):
    digits_path = Path(digits_dir)
    image_list = []
    label_list = []
    
    for png_path in sorted(digits_path.glob("*.png")):
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 1:
            continue
        try:
            label = int(parts[0])
        except ValueError:
            continue

        img = Image.open(png_path).convert("L")
        img = img.resize((28, 28))

        arr = np.array(img, dtype=np.float32)
        
        max_pix = arr.max()
        foreground_pixels = (arr > 60).sum()
        contrast = arr.max() - arr.min()

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

    images = np.stack(image_list, axis=0) if image_list else np.empty((0, 28, 28), dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return images, labels

def load_grouped_digits(digits_dir="../digits"):
    digits_path = Path(digits_dir)
    group_images = {}
    group_labels = {}

    for png_path in sorted(digits_path.glob("*.png")):
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 2:
            continue
        try:
            label = int(parts[0])
            group = int(parts[1])
        except ValueError:
            continue

        img = Image.open(png_path).convert("L")
        img = img.resize((28, 28))

        arr = np.array(img, dtype=np.float32)
        max_pix = arr.max()
        foreground_pixels = (arr > 60).sum()
        contrast = arr.max() - arr.min()

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

class ProjectDigitsDataset(torch.utils.data.Dataset):

    def __init__(self, digits_dir="../digits", transform=None):
        self.images_np, self.labels_np = ProjectDataLoader(digits_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        img_arr = self.images_np[idx]
        label = int(self.labels_np[idx])

        img = Image.fromarray(img_arr.astype(np.uint8), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class NumpyDigitsDataset(torch.utils.data.Dataset):
    def __init__(self, images_np, labels_np, transform=None):
        self.images_np = images_np
        self.labels_np = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        img_arr = self.images_np[idx]
        label = int(self.labels_np[idx])

        img = Image.fromarray(img_arr.astype(np.uint8), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def evaluate(model, dataloader, device, criterion, compute_confusion=False, num_classes=10):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    cm = None
    if compute_confusion:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    class_loss_sum = np.zeros(num_classes, dtype=np.float64)
    class_loss_count = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if compute_confusion:
                true_np = labels.cpu().numpy()
                pred_np = preds.cpu().numpy()
                for t, p in zip(true_np, pred_np):
                    cm[t, p] += 1
            
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

    avg_loss = running_loss / total
    accuracy = correct / total
    class_avg_loss = class_loss_sum / np.maximum(class_loss_count, 1)

    if compute_confusion:
        return avg_loss, accuracy, cm, class_avg_loss
    else:
        return avg_loss, accuracy

def main():

    batch_size = 128

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
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST(
            root="~/MNIST_data",
            train=True,
            download=True,
            transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
    )

    testset = datasets.MNIST(
            root="~/MNIST_data",
            train=False,
            download=True,
            transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
    )

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
    
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
         
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
    
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f} - {elapsed:.1f}s")

    print("Training complete.")

    test_loss, test_acc, test_cm, mnist_avg_loss = evaluate(model, testloader, device, criterion, compute_confusion=True)
    save_loss_table(mnist_avg_loss, "mnist_loss_table.png")
    print(f"MNIST Test loss: {test_loss:.4f}, accuracy: {100*test_acc:.2f}%")

    images_np, labels_np = ProjectDataLoader("../digits")

    project_dataset = ProjectDigitsDataset("../digits", transform=test_transform)
    project_loader = torch.utils.data.DataLoader(
        project_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    proj_loss, proj_acc, proj_cm, proj_avg_loss = evaluate(model, project_loader, device, criterion, compute_confusion=True)
    save_loss_table(proj_avg_loss, "class_loss_table.png")
    print(f"Class handwritten digits - loss: {proj_loss:.4f}, accuracy: {100*proj_acc:.2f}%")

    save_confusion_matrix(test_cm, "mnist_confusion.png", "MNIST Test Confusion Matrix")
    save_confusion_matrix(proj_cm, "project_confusion.png", "Class Digits Confusion Matrix")

    grouped_digits = load_grouped_digits("../digits")

    group_ids = sorted(grouped_digits.keys())
    group_losses = []
    group_accs = []

    for g in group_ids:
        images_np, labels_np = grouped_digits[g]
        if len(labels_np) == 0:
            continue

        group_dataset = NumpyDigitsDataset(images_np, labels_np, transform=test_transform)
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
        group_losses.append(g_loss)
        group_accs.append(g_acc)
        print(f"Group {g}: loss = {g_loss:.4f}, acc = {100.0 * g_acc:.2f}% (N={len(group_dataset)})")

    plot_group_results(group_ids, group_losses, group_accs, filename_plot="group_results.png",
                       filename_legend="group_results_legend.png")


if __name__ == "__main__":
    main()
