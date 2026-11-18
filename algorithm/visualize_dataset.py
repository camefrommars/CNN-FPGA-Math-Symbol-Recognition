from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

CLASSES = [
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "equal"
]

random.seed(42)

fig, axes = plt.subplots(
    5, 5,
    figsize=(9, 9)
)

for row, class_name in enumerate(CLASSES):

    image_dir = DATASET_DIR / "train" / class_name
    image_files = list(image_dir.glob("*.png"))

    samples = random.sample(image_files, 5)

    for col, image_path in enumerate(samples):

        image = Image.open(image_path)

        axes[row, col].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                class_name,
                fontsize=10
            )

plt.suptitle(
    "Mathematical Symbol Dataset Samples",
    fontsize=14
)

plt.tight_layout()
plt.show()