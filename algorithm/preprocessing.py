from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# Standard image dimensions for CNN input
IMAGE_SIZE = 32
SYMBOL_SIZE = 28


def preprocess_image(image_path, output_path):
    """
    Convert an input image into a 32x32 grayscale image.

    Processing steps:
    1. Convert image to grayscale.
    2. Make the background black.
    3. Crop unnecessary empty areas.
    4. Resize the symbol while maintaining its aspect ratio.
    5. Center the symbol on a 32x32 black canvas.
    """

    # Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Detect background brightness using the four corners
    pixels = np.array(image)

    corners = [
        pixels[0, 0],
        pixels[0, -1],
        pixels[-1, 0],
        pixels[-1, -1]
    ]

    # Invert image if its background is predominantly white
    if np.mean(corners) > 127:
        image = ImageOps.invert(image)

    # Convert image into a NumPy array
    pixels = np.array(image)

    # Find the symbol area
    foreground = pixels > 20

    if not np.any(foreground):
        raise ValueError(f"No symbol detected in {image_path}")

    # Get symbol boundaries
    rows = np.where(np.any(foreground, axis=1))[0]
    cols = np.where(np.any(foreground, axis=0))[0]

    top, bottom = rows[0], rows[-1] + 1
    left, right = cols[0], cols[-1] + 1

    # Crop the symbol
    image = image.crop((left, top, right, bottom))

    # Resize while maintaining aspect ratio
    image.thumbnail(
        (SYMBOL_SIZE, SYMBOL_SIZE),
        Image.Resampling.LANCZOS
    )

    # Create a 32x32 black background
    canvas = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=0
    )

    # Calculate position to center the symbol
    x = (IMAGE_SIZE - image.width) // 2
    y = (IMAGE_SIZE - image.height) // 2

    # Place symbol in the center
    canvas.paste(image, (x, y))

    # Create output directory if it does not exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed image
    canvas.save(output_path)

    print(f"Preprocessing completed: {output_path}")


if __name__ == "__main__":
    print("CNN Image Preprocessing Module")
    print(f"Target image size: {IMAGE_SIZE}x{IMAGE_SIZE}")