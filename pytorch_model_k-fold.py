# kfold_stratified_with_runtime_guard.py
import os, time, random, shutil, subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import resnet18, ResNet18_Weights
import wandb

# -------------------- USER-CONFIG --------------------
data_dir = "jellyfishimagesunsorted"   # root that contains train/, valid/, test/
class_list = ["jellyfish", "not jellyfish"]
K = 5
NUM_EPOCHS = 5         # per-fold epochs (you can reduce to shrink runtime)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_RUNTIME_HOURS = 5  # threshold to estimate against (script will warn, not stop)
SEED = 42
NUM_WORKERS = min(8, max(0, os.cpu_count() - 1))  # tune for HPC
# -----------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------- corrupted image scan (option 2) --------------------
def find_and_move_corrupted_images(root_folder):
    bad_folder = os.path.join(root_folder, "corrupted_images")
    os.makedirs(bad_folder, exist_ok=True)
    print("\n🔍 Scanning for unreadable/corrupted images...")
    corrupted = []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root_folder, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                try:
                    img = Image.open(file_path)
                    img.verify()
                except Exception:
                    corrupted.append(file_path)
                    print(f"⚠️ Corrupt: {file_path}")
                    shutil.move(file_path, os.path.join(bad_folder, filename))
    print(f"\n📦 Finished: {len(corrupted)} corrupted images moved to: {bad_folder}\n")

find_and_move_corrupted_images(data_dir)

# -------------------- transforms --------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

# -------------------- combined dataset for train+valid --------------------
class ImageFolderFromSamples(Dataset):
    """Simple dataset that uses (path,label) list and applies a transform."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

# Load existing train and valid ImageFolder to reuse their sample lists
train_folder = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train"))
valid_folder = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "valid"))
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=data_transforms['test'])

combined_samples = train_folder.samples + valid_folder.samples
combined_labels = np.array([lab for _, lab in combined_samples])
combined_dataset = ImageFolderFromSamples(combined_samples, transform=data_transforms['train'])  # train transform includes augmentation

print(f"Combined train+valid samples: {len(combined_dataset)}")
print(f"Test samples (untouched): {len(test_dataset)}")

# -------------------- device + wandb init + gpu info --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device selected:", device)
try:
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
            print("nvidia-smi output:\n", out.decode().strip())
        except Exception as e:
            print("Could not run nvidia-smi:", e)
except Exception:
    pass

wandb.init(project="DUO_classification", name="resnet18_kfold_trainval", config={
    "k": K, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS, "model": "resnet18"
})

# -------------------- training utilities --------------------
def make_model(num_classes=len(class_list)):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(512, num_classes)
    )
    return model

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    start = time.time()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_time = time.time() - start
    return running_loss / total, 100.0 * running_correct / total, epoch_time

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * total_correct / total

# -------------------- Stratified K-Fold loop --------------------
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
fold_scores = []
fold_models = []
epoch_times = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(combined_labels)), combined_labels), 1):
    print(f"\n=== Fold {fold}/{K} ===")
    wandb.run.summary[f"fold_{fold}_started_at"] = time.time()
    # create dataloaders for this fold (use val transform for val)
    train_subset = Subset(combined_dataset, train_idx)
    val_subset = Subset(ImageFolderFromSamples(combined_samples, transform=data_transforms['val']), val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))

    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=3e-5)

    best_val_acc = -1.0
    best_state = None
    per_fold_epoch_times = []

    for epoch in range(1, NUM_EPOCHS + 1):
        use_amp = (device.type == "cuda")
        train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        per_fold_epoch_times.append(epoch_time)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # log
        wandb.log({
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=(fold-1)*NUM_EPOCHS + epoch)

        print(f"Fold {fold} Epoch {epoch} | train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | epoch_time={epoch_time:.1f}s")

        # log weights histograms (per-epoch)
        for name, p in model.named_parameters():
            wandb.log({f"Weights/{name}": wandb.Histogram(p.detach().cpu().numpy())})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            torch.save(best_state, f"best_model_fold{fold}.pt")

        scheduler.step(val_loss)

        # After the first epoch of the first fold, estimate runtime and warn if > MAX_RUNTIME_HOURS
        if fold == 1 and epoch == 1:
            avg_epoch = np.mean(per_fold_epoch_times)
            estimated_total_secs = avg_epoch * NUM_EPOCHS * K
            estimated_hours = estimated_total_secs / 3600.0
            wandb.run.summary["estimated_total_hours"] = estimated_hours
            msg = f"Estimated total time for k-fold run: {estimated_hours:.2f} hours (based on first epoch)"
            print("⏱️", msg)
            if estimated_hours > MAX_RUNTIME_HOURS:
                print("⚠️ WARNING: Estimated runtime exceeds your MAX_RUNTIME_HOURS =", MAX_RUNTIME_HOURS)
                print("The script will continue to run (as requested) but this may take a long time on CPU.")
        # track epoch times to help final estimate
        epoch_times.append(epoch_time)

    fold_scores.append(best_val_acc)
    fold_models.append(f"best_model_fold{fold}.pt")
    wandb.log({f"fold_{fold}_best_val_acc": best_val_acc})
    print(f"Fold {fold} best validation acc: {best_val_acc:.2f}%")

# -------------------- After all folds: pick best fold and evaluate on test set --------------------
best_fold_idx = int(np.argmax(fold_scores))
best_model_path = fold_models[best_fold_idx]
print(f"\nBest fold: {best_fold_idx+1} with val acc {fold_scores[best_fold_idx]:.2f}% -> loading {best_model_path}")

# Build final model and load best
final_model = make_model().to(device)
final_model.load_state_dict(torch.load(best_model_path, map_location=device))
final_model.eval()

# Evaluate on the untouched test set and log confusion matrix & misclassified images
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = final_model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds); all_labels = np.array(all_labels)
test_acc = 100.0 * np.mean(all_preds == all_labels)
print(f"✅ Final Test Accuracy (best-fold model): {test_acc:.2f}%")
wandb.log({"final_test_accuracy": test_acc})

# confusion matrix (percent)
cm = confusion_matrix(all_labels, all_preds)
cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(8,6))
ConfusionMatrixDisplay(cm_percent, display_labels=test_dataset.classes).plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Confusion Matrix (percentages)")
plt.savefig("confusion_matrix_percent.png"); plt.close()
wandb.log({"confusion_matrix_percent": wandb.Image("confusion_matrix_percent.png")})

# -------------------- copy misclassified images for manual analysis --------------------
false_pos_idx = np.where((all_preds == 0) & (all_labels == 1))[0]
false_neg_idx = np.where((all_preds == 1) & (all_labels == 0))[0]

false_pos_files = [test_dataset.samples[i][0] for i in false_pos_idx]
false_neg_files = [test_dataset.samples[i][0] for i in false_neg_idx]

output_dir = os.path.join(data_dir, "analysis_results_kfold")
os.makedirs(output_dir, exist_ok=True)

def copy_images(file_list, folder_name):
    folder = os.path.join(output_dir, folder_name)
    os.makedirs(folder, exist_ok=True)
    for f in file_list:
        try:
            shutil.copy(f, folder)
        except Exception as e:
            print("Could not copy", f, e)

copy_images(false_pos_files, "false_positives")
copy_images(false_neg_files, "false_negatives")
print(f"Copied {len(false_pos_files)} false positives and {len(false_neg_files)} false negatives to {output_dir}")

# Log a few examples to wandb (up to 5 each)
examples = []
for i in (list(false_pos_idx[:5]) + list(false_neg_idx[:5])):
    img_path = test_dataset.samples[i][0]
    img = Image.open(img_path).convert("RGB")
    examples.append(wandb.Image(np.array(img), caption=f"True:{test_dataset.samples[i][1]} Pred:{all_preds[i]}"))
if examples:
    wandb.log({"misclassified_examples": examples})

print("K-fold complete. Fold accuracies:", fold_scores)
print("Mean fold val accuracy:", np.mean(fold_scores))

wandb.finish()
