
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import MathSymbolCNN


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "algorithm" / "results" / "math_symbol_cnn.pth"

FLOAT_DIR = BASE_DIR / "weights" / "float"
QUANTIZED_DIR = BASE_DIR / "weights" / "quantized"

BATCH_SIZE = 32
SCALE = 256

CLASS_TO_IDX = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "division": 3,
    "equal": 4
}

CLASS_NAMES = list(CLASS_TO_IDX.keys())

device = torch.device("cpu")


# ==========================================
# DATASET
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


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

test_dataset = MathSymbolDataset(
    root=DATASET_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# LOAD ORIGINAL MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if checkpoint["class_to_idx"] != CLASS_TO_IDX:
    raise ValueError("Class mapping checkpoint tidak sesuai!")

original_model = MathSymbolCNN().to(device)

original_model.load_state_dict(
    checkpoint["model_state_dict"]
)

original_model.eval()


# ==========================================
# CREATE DEQUANTIZED MODEL
# ==========================================

quantized_model = MathSymbolCNN().to(device)

quantized_state = {
    key: value.clone()
    for key, value in checkpoint["model_state_dict"].items()
}

parameter_mapping = {
    "conv1.weight": "conv1_weight",
    "conv1.bias": "conv1_bias",
    "conv2.weight": "conv2_weight",
    "conv2.bias": "conv2_bias",
    "fc.weight": "fc_weight",
    "fc.bias": "fc_bias"
}

for model_key, file_name in parameter_mapping.items():

    integer_weights = np.load(
        QUANTIZED_DIR / f"{file_name}.npy"
    )

    # Kembalikan integer Q8.8 menjadi desimal
    restored_weights = integer_weights.astype(
        np.float32
    ) / SCALE

    quantized_state[model_key] = torch.from_numpy(
        restored_weights
    )

quantized_model.load_state_dict(quantized_state)
quantized_model.eval()


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, model_name):

    correct = 0
    total = 0

    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):

                label = labels[i].item()

                class_total[label] += 1

                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    accuracy = 100 * correct / total

    print(f"\n===== {model_name} =====")
    print(f"Total images : {total}")
    print(f"Correct      : {correct}")
    print(f"Accuracy     : {accuracy:.2f}%")

    print("\nAccuracy per class:")

    for class_name, class_idx in CLASS_TO_IDX.items():

        class_accuracy = (
            100 * class_correct[class_idx] /
            class_total[class_idx]
        )

        print(f"{class_name}: {class_accuracy:.2f}%")

    return accuracy


# ==========================================
# COMPARE RESULTS
# ==========================================

print("========================================")
print(" ORIGINAL VS QUANTIZED MODEL")
print("========================================")

print(f"Test images: {len(test_dataset)}")

original_accuracy = evaluate_model(
    original_model,
    "ORIGINAL MODEL"
)

quantized_accuracy = evaluate_model(
    quantized_model,
    "QUANTIZED MODEL (DEQUANTIZED)"
)

difference = quantized_accuracy - original_accuracy

print("\n========================================")
print("          COMPARISON RESULTS")
print("========================================")

print(f"Original accuracy  : {original_accuracy:.2f}%")
print(f"Quantized accuracy : {quantized_accuracy:.2f}%")
print(f"Accuracy difference: {difference:+.2f}%")

print("========================================")
