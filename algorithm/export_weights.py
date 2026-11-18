
import torch
import numpy as np
from pathlib import Path

# Lokasi file model hasil training
CHECKPOINT_PATH = Path("algorithm/results/math_symbol_cnn.pth")

# Folder untuk menyimpan weights
OUTPUT_DIR = Path("weights/float")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

# Mengambil state dictionary
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Daftar parameter yang akan diekspor
parameters = [
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc.weight",
    "fc.bias"
]

print("=== CNN WEIGHT EXPORT ===")

for name in parameters:
    tensor = state_dict[name]
    array = tensor.detach().cpu().numpy()

    # Simpan dalam format NPY
    np.save(OUTPUT_DIR / f"{name.replace('.', '_')}.npy", array)

    # Simpan dalam format TXT
    np.savetxt(
        OUTPUT_DIR / f"{name.replace('.', '_')}.txt",
        array.reshape(1, -1),
        delimiter=",",
        fmt="%.8f"
    )

    print(f"\nParameter: {name}")
    print(f"Shape    : {array.shape}")
    print(f"Min      : {array.min():.6f}")
    print(f"Max      : {array.max():.6f}")

print("\nExport berhasil!")
print(f"File tersimpan di: {OUTPUT_DIR}")
