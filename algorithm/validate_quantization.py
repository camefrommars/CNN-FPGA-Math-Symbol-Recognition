
import numpy as np
from pathlib import Path

# =====================================
# KONFIGURASI
# =====================================

SCALE = 256

FLOAT_DIR = Path("weights/float")
QUANTIZED_DIR = Path("weights/quantized")

parameters = [
    "conv1_weight",
    "conv1_bias",
    "conv2_weight",
    "conv2_bias",
    "fc_weight",
    "fc_bias"
]

print("========================================")
print("       CNN QUANTIZATION VALIDATION")
print("========================================")


# =====================================
# VALIDASI SETIAP PARAMETER
# =====================================

all_valid = True

for name in parameters:

    # Membaca data asli
    original = np.load(FLOAT_DIR / f"{name}.npy")

    # Membaca data hasil quantization
    quantized = np.load(QUANTIZED_DIR / f"{name}.npy")

    # Mengembalikan integer ke floating point
    restored = quantized.astype(np.float32) / SCALE

    # Memastikan bentuk data sama
    if original.shape != quantized.shape:
        print(f"\nERROR: Shape {name} tidak cocok!")
        all_valid = False
        continue

    # Menghitung selisih
    error = original - restored

    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    print(f"\nParameter : {name}")
    print(f"Shape     : {original.shape}")
    print(f"Max Error : {max_error:.8f}")
    print(f"Mean Error: {mean_error:.8f}")
    print(f"RMSE      : {rmse:.8f}")

    # Memeriksa batas error teoritis pembulatan
    if max_error <= (0.5 / SCALE) + 1e-7:
        print("Status    : PASS")
    else:
        print("Status    : FAIL")
        all_valid = False


# =====================================
# HASIL AKHIR
# =====================================

print("\n========================================")

if all_valid:
    print("VALIDASI QUANTIZATION BERHASIL!")
else:
    print("VALIDASI QUANTIZATION GAGAL!")

print("========================================")
