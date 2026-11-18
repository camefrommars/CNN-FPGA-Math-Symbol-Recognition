from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


# ==========================================
# DATASET CONFIGURATION
# ==========================================

IMAGE_SIZE = 32
HIGH_RES_SIZE = 128
SYMBOL_SIZE = 88

TRAIN_PER_CLASS = 400
TEST_PER_CLASS = 100

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

CLASSES = {
    "addition": "+",
    "subtraction": "−",
    "multiplication": "×",
    "division": "÷",
    "equal": "="
}

FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/cambria.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
]


# ==========================================
# FONT LOADER
# ==========================================

def get_available_fonts():
    fonts = []

    for font_path in FONT_PATHS:
        if Path(font_path).exists():
            fonts.append(font_path)

    if not fonts:
        raise FileNotFoundError(
            "No supported Windows fonts were found."
        )

    return fonts


# ==========================================
# IMAGE GENERATOR
# ==========================================

def generate_symbol_image(symbol, font_paths):
    # Create high-resolution black canvas
    image = Image.new(
        "L",
        (HIGH_RES_SIZE, HIGH_RES_SIZE),
        color=0
    )

    draw = ImageDraw.Draw(image)

    # Random font and font size
    font_path = random.choice(font_paths)
    font_size = random.randint(65, 100)

    font = ImageFont.truetype(
        font_path,
        font_size
    )

    # Random symbol thickness
    stroke_width = random.randint(0, 3)

    # Calculate symbol boundaries
    bbox = draw.textbbox(
        (0, 0),
        symbol,
        font=font,
        stroke_width=stroke_width
    )

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Center symbol in the high-resolution image
    x = (HIGH_RES_SIZE - width) // 2 - bbox[0]
    y = (HIGH_RES_SIZE - height) // 2 - bbox[1]

    draw.text(
        (x, y),
        symbol,
        fill=255,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=255
    )

    # Crop unnecessary black area
    bbox = image.getbbox()

    if bbox is None:
        raise ValueError(f"Failed to render symbol: {symbol}")

    image = image.crop(bbox)

    # Resize while maintaining aspect ratio
    image.thumbnail(
        (SYMBOL_SIZE, SYMBOL_SIZE),
        Image.Resampling.LANCZOS
    )

    # Place symbol on a new black canvas
    canvas = Image.new(
        "L",
        (HIGH_RES_SIZE, HIGH_RES_SIZE),
        color=0
    )

    # Random position variation
    shift_x = random.randint(-8, 8)
    shift_y = random.randint(-8, 8)

    x = (HIGH_RES_SIZE - image.width) // 2 + shift_x
    y = (HIGH_RES_SIZE - image.height) // 2 + shift_y

    canvas.paste(image, (x, y))

    # Random rotation
    angle = random.uniform(-15, 15)

    canvas = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0
    )

    # Convert to final 32x32 image
    canvas = canvas.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    return canvas


# ==========================================
# DATASET GENERATOR
# ==========================================

def generate_dataset():
    random.seed(RANDOM_SEED)

    font_paths = get_available_fonts()

    print("Starting dataset generation...")
    print(f"Available fonts: {len(font_paths)}")

    for class_name, symbol in CLASSES.items():

        for split, total_images in [
            ("train", TRAIN_PER_CLASS),
            ("test", TEST_PER_CLASS)
        ]:

            output_dir = (
                DATASET_DIR / split / class_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for index in range(total_images):

                image = generate_symbol_image(
                    symbol,
                    font_paths
                )

                output_path = (
                    output_dir / f"{class_name}_{index:04d}.png"
                )

                image.save(output_path)

            print(
                f"Generated {total_images} images "
                f"for {class_name} ({split})"
            )

    print("\nDataset generation completed!")
    print("Total images: 2500")
    print(f"Dataset location: {DATASET_DIR}")


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":
    generate_dataset()