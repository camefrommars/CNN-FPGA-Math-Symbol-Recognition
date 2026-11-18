
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import MathSymbolCNN


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "algorithm" / "results" / "math_symbol_cnn.pth"

CLASS_TO_IDX = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "division": 3,
    "equal": 4
}

CLASS_NAMES = list(CLASS_TO_IDX.keys())


class MathSymbolDataset(datasets.ImageFolder):

    def find_classes(self, directory):
        return list(CLASS_TO_IDX.keys()), CLASS_TO_IDX


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


dataset = MathSymbolDataset(
    root=DATASET_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# Load trained model
device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = MathSymbolCNN().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Confusion matrix
num_classes = len(CLASS_NAMES)

confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

misclassified = []

with torch.no_grad():

    for images, labels in loader:

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            confusion_matrix[actual][predicted] += 1

            if actual != predicted:
                misclassified.append({
                    "actual": CLASS_NAMES[actual],
                    "predicted": CLASS_NAMES[predicted],
                    "file": dataset.samples[
                        len(misclassified) + sum(
                            confusion_matrix.flatten()
                        ) - len(misclassified) - 1
                    ][0]
                })


# Display confusion matrix
print("\n===== CONFUSION MATRIX =====")
print("Rows = Actual | Columns = Predicted\n")

print("Actual/Pred", *[
    name[:8] for name in CLASS_NAMES
])

for i, row in enumerate(confusion_matrix):
    print(
        f"{CLASS_NAMES[i]:15}",
        *row
    )


# Display total errors
print("\n===== MISCLASSIFICATION SUMMARY =====")
print("Total incorrect predictions:", len(misclassified))

for actual in range(num_classes):

    errors = confusion_matrix[actual].sum() - confusion_matrix[actual][actual]

    print(
        f"{CLASS_NAMES[actual]}: {errors} incorrect"
    )

print("\nEvaluation completed!")
