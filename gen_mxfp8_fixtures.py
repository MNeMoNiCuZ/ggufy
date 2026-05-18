#!/usr/bin/env python3
"""
Generate MXFP8 reference fixtures from a real ComfyUI-quantized model.

Extracts the first 128 rows × 128 columns of layers.0.mlp.gate_proj from
etCenterV1_v10MXFP8.safetensors, then dequantizes using the same formula as
dequantizeMxfp8Raw() in ScaledQuant.zig to produce the expected F32 output.

MXFP8 layout (OCP MX spec, ComfyUI Kitchen converter):
  weight      F8_E4M3  [rows, cols]       — one byte per element
  weight_scale  U8     [rows, cols/32]    — E8M0 scale, one per 32-element block
  Scale is stored in linear row-major order (no cuBLAS tiling).

Outputs (all in src/test_fixtures/):
  mxfp8_weight.u8        – F8_E4M3 bytes  [128 × 128 = 16384 bytes]
  mxfp8_weight_scale.u8  – E8M0 bytes     [128 × 4   =   512 bytes]
  mxfp8_expected.f32     – dequantized F32 values [128*128 = 16384 values]

Run from the project root:
  venv/bin/python3 gen_mxfp8_fixtures.py
"""

import struct
import json
import numpy as np
import ml_dtypes
import os

MODEL_PATH = "test-models/etCenterV1_v10MXFP8.safetensors"
OUT_DIR = "src/test_fixtures"
os.makedirs(OUT_DIR, exist_ok=True)

ROWS = 128
COLS = 128  # must be a multiple of 32

# F8_E4M3 LUT: decode all 256 byte values — mirrors lut_e4m3[] in DataTransform.zig
f8_lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)

# E8M0 LUT: decode all 256 byte values — mirrors e8m0_to_f32() in DataTransform.zig
e8m0_lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu).astype(np.float32)

# ---------------------------------------------------------------------------
# Load the safetensors file
# ---------------------------------------------------------------------------

with open(MODEL_PATH, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header_json = f.read(header_len)
    data_start = 8 + header_len

header = json.loads(header_json)

prefix = "layers.0.mlp.gate_proj"
weight_info = header[f"{prefix}.weight"]
scale_info  = header[f"{prefix}.weight_scale"]

assert weight_info["dtype"] == "F8_E4M3", f"unexpected weight dtype: {weight_info['dtype']}"
assert scale_info["dtype"]  == "U8",      f"unexpected scale dtype: {scale_info['dtype']}"

full_cols      = weight_info["shape"][1]   # 4096
full_scale_cols = scale_info["shape"][1]   # 128 (= 4096 / 32)

assert full_cols // 32 == full_scale_cols, "scale shape inconsistent with weight shape"
assert COLS % 32 == 0, "COLS must be a multiple of 32"

with open(MODEL_PATH, "rb") as f:
    # Weight: first ROWS rows, first COLS bytes of each row.
    w_base = data_start + weight_info["data_offsets"][0]
    rows_bytes = []
    for row in range(ROWS):
        f.seek(w_base + row * full_cols)
        rows_bytes.append(f.read(COLS))
    weight_bytes = b"".join(rows_bytes)

    # Scale: first ROWS rows, first COLS//32 bytes of each row.
    s_base = data_start + scale_info["data_offsets"][0]
    scale_rows = []
    for row in range(ROWS):
        f.seek(s_base + row * full_scale_cols)
        scale_rows.append(f.read(COLS // 32))
    scale_bytes = b"".join(scale_rows)

print(f"Weight bytes : {len(weight_bytes)}  (expected {ROWS * COLS})")
print(f"Scale bytes  : {len(scale_bytes)}  (expected {ROWS * COLS // 32})")

# ---------------------------------------------------------------------------
# Dequantize — mirrors dequantizeMxfp8Raw() in ScaledQuant.zig exactly
# ---------------------------------------------------------------------------

num_scale_cols = COLS // 32  # = 4

weight_arr = np.frombuffer(weight_bytes, dtype=np.uint8)
scale_arr  = np.frombuffer(scale_bytes,  dtype=np.uint8)

expected = np.zeros(ROWS * COLS, dtype=np.float32)

for flat_idx in range(ROWS * COLS):
    row = flat_idx // COLS
    col = flat_idx % COLS
    scale_idx = row * num_scale_cols + col // 32
    scale = e8m0_lut[scale_arr[scale_idx]]
    expected[flat_idx] = f8_lut[weight_arr[flat_idx]] * scale

# ---------------------------------------------------------------------------
# Write fixtures
# ---------------------------------------------------------------------------

def write_fixture(name, data: bytes):
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    print(f"Wrote {path}  ({len(data)} bytes)")

write_fixture("mxfp8_weight.u8",       weight_bytes)
write_fixture("mxfp8_weight_scale.u8", scale_bytes)
write_fixture("mxfp8_expected.f32",    expected.view(np.uint8).tobytes())

print(f"\nExpected F32 range : [{float(expected.min()):.4f}, {float(expected.max()):.4f}]")
print(f"First 8 values     : {expected[:8].tolist()}")
print(f"Non-zero elements  : {int((expected != 0).sum())} / {ROWS * COLS}")
