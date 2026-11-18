
from pathlib import Path

import math
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import MathSymbolCNN


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "algorithm" / "results" / "math_symbol_cnn.pth"


# ==========================================
# 2. CLASS MAPPING
# Harus sama dengan mapping saat training
# ==========================================

CLASS_TO_IDX = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "division": 3,
    "equal": 4
}

CLASS_NAMES = [
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "equal"
]


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# ==========================================
# 4. CUSTOM DATASET
# Menggunakan urutan kelas saat training
# ==========================================

class MathSymbolDataset(datasets.ImageFolder):

    def find_classes(self, directory):

        classes = list(CLASS_TO_IDX.keys())

        for class_name in classes:

            class_path = Path(directory) / class_name

            if not class_path.is_dir():
                raise FileNotFoundError(
                    f"Folder kelas tidak ditemukan: {class_path}"
                )

        return classes, CLASS_TO_IDX


dataset = MathSymbolDataset(
    root=str(DATASET_DIR),
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. LOAD TRAINED CNN MODEL
# ==========================================

device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = MathSymbolCNN()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ==========================================
# 6. PREDICTION AND ERROR DETECTION
# ==========================================

errors = []

correct_predictions = 0
total_predictions = 0

with torch.no_grad():

    offset = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            total_predictions += 1

            if actual == predicted:

                correct_predictions += 1

            else:

                errors.append({
                    "image": images[i].squeeze().cpu().numpy(),
                    "actual": CLASS_NAMES[actual],
                    "predicted": CLASS_NAMES[predicted],
                    "filename": Path(
                        dataset.samples[offset + i][0]
                    ).name
                })

        offset += len(labels)


# ==========================================
# 7. DISPLAY EVALUATION RESULTS
# ==========================================

accuracy = (
    correct_predictions / total_predictions
) * 100

print("\n========== CNN EVALUATION ==========")

print("Total test images:", total_predictions)

print("Correct predictions:", correct_predictions)

print("Incorrect predictions:", len(errors))

print(f"Accuracy: {accuracy:.2f}%")

print("====================================\n")


# ==========================================
# 8. VISUALIZE MISCLASSIFIED IMAGES
# ==========================================

if len(errors) == 0:

    print("No misclassified images found!")

else:

    columns = 4

    rows = math.ceil(
        len(errors) / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows)
    )

    axes = axes.flatten()

    for i, error in enumerate(errors):

        axes[i].imshow(
            error["image"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i].set_title(
            f"Actual: {error['actual']}\n"
            f"Predicted: {error['predicted']}",
            fontsize=9
        )

        axes[i].axis("off")

    # Hide unused plot spaces
    for i in range(len(errors), len(axes)):

        axes[i].axis("off")

    plt.suptitle(
        "CNN Misclassified Images",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()
