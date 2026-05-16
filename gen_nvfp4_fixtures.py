#!/usr/bin/env python3
"""
Generate NVFP4 reference fixtures from a real ComfyUI-quantized model.

Extracts the first 128 rows × 256 logical columns of
layers.0.mlp.gate_proj from ernieImageQuants_turboNVFP4.safetensors,
then dequantizes using the same formula as dequantizeFp4Cluster() in
ScaledQuant.zig to produce the expected F32 output.

Choosing ROWS=128, COLS=256 is deliberate:
  - r0 = row // 128 is 0 for all rows → exercises only the first 128-row tile,
    keeping the fixture small while still covering all 4 cuBLAS column-block
    groups (n_col_blocks = COLS/16 / 4 = 4).
  - Every scale index 0..2047 is used exactly once, so the scale fixture is
    a clean 2048-byte prefix of the scale tensor with no unused gaps.

Outputs (all in src/test_fixtures/):
  nvfp4_weight.u8          – packed nibble bytes  [128 rows × 128 packed bytes]
  nvfp4_weight_scale.u8    – F8_E4M3 scale bytes in cuBLAS tiled order [2048 bytes]
  nvfp4_weight_scale_2.f32 – global F32 scalar [4 bytes, little-endian]
  nvfp4_expected.f32       – dequantized F32 values [128*256 = 32768 values]

Run from the project root:
  venv/bin/python3 gen_nvfp4_fixtures.py
"""

import struct
import json
import numpy as np
import ml_dtypes
import os

MODEL_PATH = "test-models/ernieImageQuants_turboNVFP4.safetensors"
OUT_DIR = "src/test_fixtures"
os.makedirs(OUT_DIR, exist_ok=True)

ROWS = 128
COLS = 256  # logical columns; packed storage uses COLS // 2 = 128 bytes per row

# FP4 E2M1 LUT — must match lut_fp4_e2m1[] in DataTransform.zig
fp4_lut = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

# F8_E4M3 LUT: decode all 256 byte values — mirrors lut_e4m3[] in DataTransform.zig
f8_lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)

# ---------------------------------------------------------------------------
# Load the safetensors file
# ---------------------------------------------------------------------------

with open(MODEL_PATH, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header_json = f.read(header_len)
    data_start = 8 + header_len

header = json.loads(header_json)

prefix = "layers.0.mlp.gate_proj"
weight_info  = header[f"{prefix}.weight"]
scale_info   = header[f"{prefix}.weight_scale"]
scale2_info  = header[f"{prefix}.weight_scale_2"]

assert weight_info["dtype"] == "U8",      "weight dtype changed"
assert scale_info["dtype"]  == "F8_E4M3", "weight_scale dtype changed"
assert scale2_info["dtype"] == "F32",     "weight_scale_2 dtype changed"

full_packed_cols = weight_info["shape"][1]  # 2048 (= 4096 logical / 2)

with open(MODEL_PATH, "rb") as f:
    # Weight: extract first ROWS rows, first COLS//2 packed columns each.
    # The weight tensor is row-major [12288, 2048], so each row is contiguous.
    w_base = data_start + weight_info["data_offsets"][0]
    rows_bytes = []
    for row in range(ROWS):
        f.seek(w_base + row * full_packed_cols)
        rows_bytes.append(f.read(COLS // 2))
    weight_bytes = b"".join(rows_bytes)

    # Scale: first 2048 bytes of the scale tensor.
    # For ROWS=128, COLS=256 the cuBLAS tiling produces scale indices 0..2047
    # (every index is hit exactly once), so a 2048-byte prefix is sufficient.
    s_base = data_start + scale_info["data_offsets"][0]
    f.seek(s_base)
    scale_bytes = f.read(2048)

    # Global scale: little-endian F32 scalar.
    gs_base = data_start + scale2_info["data_offsets"][0]
    f.seek(gs_base)
    global_scale_bytes = f.read(4)

global_scale = struct.unpack("<f", global_scale_bytes)[0]

print(f"Global scale : {global_scale}")
print(f"Weight bytes : {len(weight_bytes)}  (expected {ROWS * COLS // 2})")
print(f"Scale bytes  : {len(scale_bytes)}  (expected 2048)")

# ---------------------------------------------------------------------------
# Dequantize — mirrors dequantizeFp4Cluster() in ScaledQuant.zig exactly
# ---------------------------------------------------------------------------

num_scale_cols = COLS // 16   # = 16
n_col_blocks   = (num_scale_cols + 3) // 4  # = 4

weight_arr = np.frombuffer(weight_bytes, dtype=np.uint8).reshape(ROWS, COLS // 2)
scale_arr  = np.frombuffer(scale_bytes,  dtype=np.uint8)

expected = np.zeros(ROWS * COLS, dtype=np.float32)

for row in range(ROWS):
    r0 = row // 128
    r1 = row % 128
    for col in range(COLS):
        # Nibble packing: even cols → HIGH nibble, odd cols → LOW nibble
        byte_val = int(weight_arr[row, col // 2])
        nibble = (byte_val >> 4) & 0xF if col % 2 == 0 else byte_val & 0xF
        fp4_val = fp4_lut[nibble]

        # cuBLAS tiled scale index — matches the formula in ScaledQuant.zig
        scale_col = col // 16
        c0 = scale_col // 4
        c1 = scale_col % 4
        scale_idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 // 32) * 4 + c1

        block_scale = f8_lut[scale_arr[scale_idx]]
        expected[row * COLS + col] = fp4_val * block_scale * global_scale

# ---------------------------------------------------------------------------
# Sanity-check: verify all 2048 scale indices were exercised
# ---------------------------------------------------------------------------

used_indices = set()
for row in range(ROWS):
    r0 = row // 128
    r1 = row % 128
    for col in range(COLS):
        scale_col = col // 16
        c0 = scale_col // 4
        c1 = scale_col % 4
        used_indices.add((r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 // 32) * 4 + c1)

assert used_indices == set(range(2048)), "scale index coverage check failed"
print("Scale index coverage: 0..2047 all hit exactly once ✓")

# ---------------------------------------------------------------------------
# Write fixtures
# ---------------------------------------------------------------------------

def write_fixture(name, data: bytes):
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    print(f"Wrote {path}  ({len(data)} bytes)")

write_fixture("nvfp4_weight.u8",          weight_bytes)
write_fixture("nvfp4_weight_scale.u8",    scale_bytes)
write_fixture("nvfp4_weight_scale_2.f32", global_scale_bytes)
write_fixture("nvfp4_expected.f32",       expected.view(np.uint8).tobytes())

print(f"\nExpected F32 range : [{float(expected.min()):.4f}, {float(expected.max()):.4f}]")
print(f"First 8 values     : {expected[:8].tolist()}")
print(f"Non-zero elements  : {int((expected != 0).sum())} / {ROWS * COLS}")
