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

    # Format losses as ".12"
    loss_row = [f"{loss:.2f}"[1:] if loss < 1 else f"{loss:.2f}" for loss in class_avg_loss]

    digit_row = [str(d) for d in range(10)]
    table_data = [digit_row, loss_row]

    # Smaller figure for minimal whitespace
    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )

    # Compact styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 0.6)   # shrink table overall

    # Reduce padding by controlling each cell height
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.05  # minimal padding
        if row == 0:
            cell.set_facecolor("#dbe9ff")  # light blue header
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

# load the classes test digit data
# Helper to load the class digits dataset:
# - reads all the png files
# - extracts the labels form filenames
# - return N x 28 x 28 grayscale images and labels
# use the "../digits" path so it works for anyone
def ProjectDataLoader(digits_dir="../digits"):
    digits_path = Path(digits_dir)
    image_list = []
    label_list = []
    
    # sort for deterministic ordering so we can recreate runs
    # good for debugging and alaysis in our report
    for png_path in sorted(digits_path.glob("*.png")):
        fname = png_path.stem
        parts = fname.split("-")
        if len(parts) < 1:
            # skip any file that doesnt follow the naming convention
            continue
        try:
            # first part of filename is the digit label we want
            label = int(parts[0])
        except ValueError:
            # skip if this isnt an integer
            continue

        # open image and force it to grayscale("L") to match MINST format
        img = Image.open(png_path).convert("L")
        # ensure that the size is actualy 28 x 28
        # maybe this could be a problem but i dont think so, i think shen made sure
        # maybe change this so it confirms that its 28x28 instead of tryign to resize it
        # again if its not
        img = img.resize((28, 28))

        # convert to flost32 np array
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

    # stack into a single np array of shape (N, 28, 28)
    # return empty array if there are no images with this smae shape
    images = np.stack(image_list, axis=0) if image_list else np.empty((0, 28, 28), dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return images, labels

def load_grouped_digits(digits_dir="../digits"):
    digits_path = Path(digits_dir)
    group_images = {}
    group_labels = {}

    for png_path in sorted(digits_path.glob("*.png")):
        fname = png_path.stem  # e.g. "1-2-3"
        parts = fname.split("-")
        # need at least digit + group
        if len(parts) < 2:
            continue
        try:
            label = int(parts[0])      # digit
            group = int(parts[1])      # group ID
        except ValueError:
            continue

        # load & preprocess exactly like ProjectDataLoader
        img = Image.open(png_path).convert("L")
        img = img.resize((28, 28))

        arr = np.array(img, dtype=np.float32)
        max_pix = arr.max()
        foreground_pixels = (arr > 60).sum()
        contrast = arr.max() - arr.min()

        if max_pix < 70:
            # too faint
            continue
        if foreground_pixels < 20:
            # almost blank
            continue
        if contrast < 30:
            # low contrast
            continue

        group_images.setdefault(group, []).append(arr)
        group_labels.setdefault(group, []).append(label)

    grouped = {}
    for g in group_images:
        images = np.stack(group_images[g], axis=0).astype(np.float32)
        labels = np.array(group_labels[g], dtype=np.int64)
        grouped[g] = (images, labels)

    return grouped

# Create the dataset of the classes digits pngs to use in testing the mlp
# class datset wrapper for the ProjectDataLoader function above
# lets us plug in the class digit set into a PyTorch DataLoader
# where we can apply the same transforms (ToTensor/Normalize) that we do to MINST
class ProjectDigitsDataset(torch.utils.data.Dataset):

    def __init__(self, digits_dir="../digits", transform=None):
        # load all the images abd labels intp np arrays
        self.images_np, self.labels_np = ProjectDataLoader(digits_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        # get the 28 x 28 np image and label oh shapr (28, 28)
        img_arr = self.images_np[idx]
        label = int(self.labels_np[idx])

        # convert the np array back to a PIL image so that we can apply
        # torchvision transforms (ToTensor/Normalize)
        # using mode="L" is just 8-bit grayscale
        img = Image.fromarray(img_arr.astype(np.uint8), mode="L")

        # apply the same transform pipeline as MNIST to img
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


# Define mlp 
# later tune: Hidden sizes, Dropout rate, Learning rate, Batch size, Number of epochs
# mlp model architecture: fuuly connected feed forward network
# - input: 28*28 = 784 features (flattened image)
# - hidden layers: 256, 128, 64 w/ 0.3 (81%) with ReLU - modified to 512, 256, 128 (86%) w/ 0.3
# - dropout: 0.2 to reduce the amount of overfitting (86%) w/ above modified
# triend with 512, 256, 128, 64, with dropout = 0.2% and got 84%
# tried with same as above with 0.3% and got 84%
# - output: 10 logits (one per digit)
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # flatten from (batch, 1, 28, 28) to (batch, 784)
        x = x.view(x.size(0), -1)
        # apply fully connected layers with ReLU and dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        # final layer outputs raw logits
        # CroddEntropyLoss handles the softmax internally
        x = self.fc4(x)
        return x

# evaluate the model on the MINST test set now
# Evaluation helper used for both MNIST and class data
# computes average loss and accuracy over the given DataLoader
# keep this funcitong eneric by being able to pass in "criterion"
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
            # move batch to GPU/CPU if s=using the A100 like i am
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            # accumulate total (sum of loss * batch size)
            running_loss += loss.item() * images.size(0)

            # compute predictions and count correct ones
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if compute_confusion:
                # move to CPU and numpy for counting
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


    # avoid divion by zero is dataloader is empty
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

# main training and evaluation calls
# Main funciton:
# - load and preprocesss MNIST train and test data
# - train the MLP on MNISt data
# - evaluate on MNIST test set
# - load class digits set, preprocess it, and evaluate it on MLP
def main():

    batch_size = 128

    # define transforms for image preprocessing:
    # - ToTensor: [0, 255] -> [0, 1]
    # - Normalize(mean=0.5, std=0.5): [0, 1] -> [-1, 1]
    # use a different trasnform for triaing than for testing so you can let the model see soem
    # invariance liek oratiations and shifts it wouldnt see witht he regular MNISt data that it
    # would in out class data
    # use these same transforms for both testing MNIST and class data so its consistent for the MLP
    # increasing translation form 0.1 to 0.2 went form 85.57% to 86.43% w/ 10 deg
    # smae thing with 15% went from 86.43% to 86.07%
    # tried with 0.15 translation and 10 deg and got 85.71%
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

    # download MNIST training set: used for training the MLP
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

    # download MNISt test set: used to evaluate loss/accuracy
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

    # initalize model, loss, and optimizer
    # - MLP() is the fully connected network
    # - CrossEntropyLoss = softmax + log loss
    # - Adam optimizer with lr=1e-3 (this is pretty common) gets 
    # modified to 5e-4 get 85% no difference from 1e-3 except more consistnent
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop:
    # - run for fixed number of epochs (we should change this and graph results)
    # - for each epoch: iterate iver all the training batches
    # - compute loss, backprop, and update weights
    # - track and print the average training loss per epoch: shows us if the model is improving
    # 15 eopchs seems to be a good number, doesnt overfit and doesnt underfit (85%),
    # when i tried anything smaller it was slgihtly underfit but not too bad and when i
    # trien 20 it was uncessesary and did not improve the results (i also tried 30 and same thing)
    epochs = 15
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
    
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
            # zero out gradients from previous step
            optimizer.zero_grad()
            # forward pass
            logits = model(images)
            # compute training loss
            loss = criterion(logits, labels)
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()
         
            # accumulate the total loss
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
    
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f} - {elapsed:.1f}s")

    print("Training complete.")

    # Evaluate the trained MLP on the MNISt test set
    # this shows us the baseline performance on the stndardized data
    # then we will have to compare this to the class digits performance 
    test_loss, test_acc, test_cm, mnist_avg_loss = evaluate(model, testloader, device, criterion, compute_confusion=True)
    save_loss_table(mnist_avg_loss, "mnist_loss_table.png")
    print(f"MNIST Test loss: {test_loss:.4f}, accuracy: {100*test_acc:.2f}%")

    # load the class digits from "../digits" and check shapes/labels
    images_np, labels_np = ProjectDataLoader("../digits")

    # build Dataset and DataLoader for the class digits using the same transforms as MNIST
    # now we cna reuse the same evaluation code that we sued for MNIST
    project_dataset = ProjectDigitsDataset("../digits", transform=test_transform)
    project_loader = torch.utils.data.DataLoader(
        project_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # evaluate the smae trained MLP on the class digits
    proj_loss, proj_acc, proj_cm, proj_avg_loss = evaluate(model, project_loader, device, criterion, compute_confusion=True)
    save_loss_table(proj_avg_loss, "class_loss_table.png")
    print(f"Class handwritten digits - loss: {proj_loss:.4f}, accuracy: {100*proj_acc:.2f}%")

    save_confusion_matrix(test_cm, "mnist_confusion.png", "MNIST Test Confusion Matrix")
    save_confusion_matrix(proj_cm, "project_confusion.png", "Class Digits Confusion Matrix")

    # -------------------------------------------------
    # Per-group evaluation
    # -------------------------------------------------
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

    # Plot bar chart of per-group accuracy & loss
    plot_group_results(group_ids, group_losses, group_accs, filename_plot="group_results.png",
                       filename_legend="group_results_legend.png")


# entry point
if __name__ == "__main__":
    main()
