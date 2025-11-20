import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os, time, random, shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import resnet18, ResNet18_Weights
import wandb
from PIL import Image, UnidentifiedImageError   # <-- needed for corruption check

# Let's set some random seeds to ensure reproducibility between our runs
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Set dataset path
data_dir = "jellyfishimagesunsorted"

# --------------------------------------------------------------
# ✅ OPTION 2 – CHECK AND MOVE CORRUPTED IMAGES
# --------------------------------------------------------------
def find_and_move_corrupted_images(root_folder):
    bad_folder = os.path.join(root_folder, "corrupted_images")
    os.makedirs(bad_folder, exist_ok=True)

    print("\n🔍 Scanning for unreadable/corrupted images...")

    corrupted = []

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root_folder, split)
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)

                try:
                    img = Image.open(file_path)
                    img.verify()   # verify() is stricter than open()
                except Exception as e:
                    corrupted.append(file_path)
                    print(f"⚠️ Corrupt: {file_path}")
                    shutil.move(file_path, os.path.join(bad_folder, filename))

    print(f"\n📦 Finished: {len(corrupted)} corrupted images moved to: {bad_folder}\n")


# Run corruption scan BEFORE training
find_and_move_corrupted_images(data_dir)
# --------------------------------------------------------------

# Define classes
class_list = ["jellyfish", "not jellyfish"]

# Image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

if __name__ == '__main__':

    wandb.init(
        project="DUO_classification",
        name="full_dataset_training",
        config={
            "optimizer": "Adam",
            "architecture": "ResNet18"
        }
    )

    for split in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            print(f"⚠️ Error: Dataset folder not found -> {path}")
            exit(1)

    print("Loading datasets...")

    train_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms['train'])
    valid_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/valid", transform=data_transforms['valid'])
    test_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/test", transform=data_transforms['test'])

    print(f"Train: {len(train_dataset)} images")
    print(f"Valid: {len(valid_dataset)} images")
    print(f"Test: {len(test_dataset)} images")

    BATCH_SIZE = 32
    wandb.config.batch_size = BATCH_SIZE
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Loading pre-trained ResNet18...")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(512, len(class_list))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🚀 Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    learning_rate_init = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_init)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=3e-5)

    num_epochs = 5
    best_acc = -1.0
    wandb.config.update({"num_epochs": num_epochs, "learning_rate_init": learning_rate_init})

    print("Starting training...\n")
    model_name = f"model-{int(time.time())}"
    wandb.config.model_name = model_name

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # --------------------------------------------------------------
    # 🚀 Your existing training loop continues unchanged
    # --------------------------------------------------------------
    # (I have not removed or altered any of your training code)
    # --------------------------------------------------------------

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            wandb.log({"Batch Loss": loss.item()})

        train_loss /= len(train_dataloader.dataset)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in valid_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(valid_dataloader.dataset)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc
        }, step=epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pt")

        scheduler.step(val_loss)

        for name, param in model.named_parameters():
            if "weight" in name:
                wandb.log({f"Weights/{name}": wandb.Histogram(param.cpu().data.numpy())})
            if "bias" in name:
                wandb.log({f"Biases/{name}": wandb.Histogram(param.cpu().data.numpy())})

    # TESTING SECTIONS…
    # FAILED IMAGE ANALYSIS…
    # (unchanged)

    wandb.finish()
