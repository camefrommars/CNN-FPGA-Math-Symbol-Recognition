
import numpy as np
from pathlib import Path

# =====================================
# KONFIGURASI QUANTIZATION
# =====================================

# Fixed-Point Q8.8
SCALE = 256

# Batas signed 16-bit
MIN_VALUE = -32768
MAX_VALUE = 32767

# Lokasi file
INPUT_DIR = Path("weights/float")
OUTPUT_DIR = Path("weights/quantized")

# Membuat folder output jika belum ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter CNN yang akan dikonversi
parameters = [
    "conv1_weight",
    "conv1_bias",
    "conv2_weight",
    "conv2_bias",
    "fc_weight",
    "fc_bias"
]


# =====================================
# PROSES QUANTIZATION
# =====================================

print("=================================")
print("     CNN WEIGHT QUANTIZATION")
print("=================================")
print("Format : Q8.8")
print(f"Scale  : {SCALE}")

for name in parameters:

    # Membaca data floating point
    input_file = INPUT_DIR / f"{name}.npy"
    data = np.load(input_file)

    # Mengubah floating point menjadi integer
    quantized = np.round(data * SCALE)

    # Memeriksa batas nilai signed 16-bit
    if np.any(quantized < MIN_VALUE) or np.any(quantized > MAX_VALUE):
        raise ValueError(
            f"Nilai parameter {name} melebihi batas signed 16-bit!"
        )

    # Mengubah menjadi integer 16-bit
    quantized = quantized.astype(np.int16)

    # Menyimpan file NPY
    np.save(
        OUTPUT_DIR / f"{name}.npy",
        quantized
    )

    # Menyimpan file TXT
    np.savetxt(
        OUTPUT_DIR / f"{name}.txt",
        quantized.reshape(1, -1),
        delimiter=",",
        fmt="%d"
    )

    # Menampilkan informasi
    print("\nParameter :", name)
    print("Shape     :", quantized.shape)
    print("Min       :", quantized.min())
    print("Max       :", quantized.max())

print("\n=================================")
print("Quantization berhasil!")
print(f"Hasil tersimpan di: {OUTPUT_DIR}")
print("=================================")
