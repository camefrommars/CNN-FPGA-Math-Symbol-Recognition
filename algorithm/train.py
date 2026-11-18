
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import MathSymbolCNN


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RESULT_DIR = BASE_DIR / "algorithm" / "results"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

CLASS_TO_IDX = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "division": 3,
    "equal": 4
}


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cpu")

print("Training device:", device)


# ==========================================
# DATASET CLASS
# ==========================================

class MathSymbolDataset(datasets.ImageFolder):

    def find_classes(self, directory):

        classes = list(CLASS_TO_IDX.keys())

        for class_name in classes:
            if not (Path(directory) / class_name).is_dir():
                raise FileNotFoundError(
                    f"Missing dataset class: {class_name}"
                )

        return classes, CLASS_TO_IDX


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# ==========================================
# LOAD DATASET
# ==========================================

train_dataset = MathSymbolDataset(
    root=DATASET_DIR / "train",
    transform=transform
)

test_dataset = MathSymbolDataset(
    root=DATASET_DIR / "test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nDataset information:")
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))
print("Class mapping:", train_dataset.class_to_idx)


# ==========================================
# INITIALIZE CNN MODEL
# ==========================================

model = MathSymbolCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING PROCESS
# ==========================================

print("\nStarting CNN training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Training Accuracy: {train_accuracy:.2f}%"
    )


# ==========================================
# TESTING PROCESS
# ==========================================

print("\nEvaluating CNN model...")

model.eval()

correct = 0
total = 0

class_correct = [0] * len(CLASS_TO_IDX)
class_total = [0] * len(CLASS_TO_IDX)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):

            label = labels[i].item()

            class_total[label] += 1

            if predicted[i] == labels[i]:
                class_correct[label] += 1


test_accuracy = 100 * correct / total

print("\n===== TEST RESULTS =====")
print(f"Total testing images: {total}")
print(f"Correct predictions: {correct}")
print(f"Test accuracy: {test_accuracy:.2f}%")

print("\nAccuracy per class:")

for class_name, class_idx in CLASS_TO_IDX.items():

    accuracy = (
        100 * class_correct[class_idx] /
        class_total[class_idx]
    )

    print(f"{class_name}: {accuracy:.2f}%")


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_path = RESULT_DIR / "math_symbol_cnn.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "test_accuracy": test_accuracy,
        "epochs": EPOCHS
    },
    model_path
)

print("\nTraining completed!")
print("Model saved to:", model_path)
