#!/usr/bin/env python3
"""Generate int4 (int4_tensorwise) test fixtures with an independent NumPy reference.

Unlike the int8 fixtures, there is no upstream ComfyUI/comfy_kitchen loader for this
4-bit layout — it is ggufy's own symmetric per-row int4 (a 4-bit sibling of
int8_tensorwise). So the "known-good" reference here is this self-contained NumPy
implementation of the exact spec, which the Zig code is validated against:

    scale[r] = max(amax(row[r]) / 7, 1e-30)          (computed in float32)
    q[r,c]   = clamp(round_half_even(row[r,c]/scale[r]), -8, 7)
    packed:  element 2k -> low nibble of byte k, element 2k+1 -> high nibble,
             each nibble = value's two's-complement low 4 bits.
  With convrot, the row is first rotated group-wise by the normalized regular
  Hadamard matrix (same one used by the int8 convrot path) before quantization.

The input is the real krea2 weight slice already committed as convrot_expected.f32
(16 x 6144 f32), so these fixtures exercise real model data.

Run: python3 gen_int4_fixtures.py

Outputs (into src/test_fixtures/):
    int4_plain_weight.u8       packed int4 weight, plain          [ROWS x COLS/2]
    int4_plain_scale.f32       per-row scale, plain               [ROWS]
    int4_plain_expected.f32    plain dequant                      [ROWS x COLS]
    int4_convrot_weight.u8     packed int4 weight, convrot        [ROWS x COLS/2]
    int4_convrot_scale.f32     per-row scale, convrot             [ROWS]
    int4_convrot_expected.f32  convrot dequant (un-rotated)       [ROWS x COLS]
    int4_meta.json             {rows, cols, group_size}
"""
import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "test_fixtures")
GROUP_SIZE = 256


def build_hadamard(size: int) -> np.ndarray:
    """Normalized regular Hadamard of power-of-4 `size`, matching Zig's buildHadamard."""
    assert size >= 4 and (size & (size - 1)) == 0 and (size.bit_length() - 1) % 2 == 0
    h4 = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32
    )
    h = h4.copy()
    while h.shape[0] < size:
        h = np.kron(h, h4)
    return (h * (1.0 / np.sqrt(np.float32(size)))).astype(np.float32)


def rotate_groupwise(mat: np.ndarray, group_size: int) -> np.ndarray:
    """Apply H @ g to each contiguous group of `group_size` along the columns (float32)."""
    rows, cols = mat.shape
    assert cols % group_size == 0
    h = build_hadamard(group_size)
    n_groups = cols // group_size
    g = mat.reshape(rows, n_groups, group_size).astype(np.float32)
    # (H @ g) for each group vector g == groups @ H.T; H is symmetric.
    rot = np.einsum("rgc,dc->rgd", g, h, dtype=np.float32)
    return rot.reshape(rows, cols).astype(np.float32)


def quantize_int4(mat: np.ndarray):
    """Symmetric per-row int4. Returns (packed_u8 [rows, cols/2], scale_f32 [rows])."""
    mat = mat.astype(np.float32)
    rows, cols = mat.shape
    assert cols % 2 == 0
    finite = np.where(np.isfinite(mat), np.abs(mat), np.float32(0.0))
    amax = finite.max(axis=1).astype(np.float32)
    scale = np.maximum(amax / np.float32(7.0), np.float32(1e-30)).astype(np.float32)
    # round-half-to-even (numpy default), computed in float32.
    q = np.rint((mat / scale[:, None]).astype(np.float32)).astype(np.float32)
    q = np.clip(q, -8.0, 7.0).astype(np.int8)
    nib = (q.astype(np.int32) & 0x0F).astype(np.uint8)  # two's-complement low 4 bits
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed, scale


def dequantize_int4(packed: np.ndarray, scale: np.ndarray, cols: int, convrot: bool):
    rows = packed.shape[0]
    lo = packed & 0x0F
    hi = packed >> 4

    def sign_ext(n):
        n = n.astype(np.int16)
        return np.where(n >= 8, n - 16, n).astype(np.float32)

    out = np.empty((rows, cols), dtype=np.float32)
    out[:, 0::2] = sign_ext(lo)
    out[:, 1::2] = sign_ext(hi)
    out *= scale[:, None]
    if convrot:
        out = rotate_groupwise(out, GROUP_SIZE)
    return out.astype(np.float32)


def main():
    src = os.path.join(OUT_DIR, "convrot_expected.f32")
    meta = json.load(open(os.path.join(OUT_DIR, "convrot_meta.json")))
    rows, cols = int(meta["rows"]), int(meta["cols"])
    inp = np.fromfile(src, dtype=np.float32).reshape(rows, cols)
    assert cols % GROUP_SIZE == 0 and cols % 2 == 0
    print(f"input {src} shape=({rows},{cols})  gs={GROUP_SIZE}")

    # Plain int4.
    pw, ps = quantize_int4(inp)
    pexp = dequantize_int4(pw, ps, cols, convrot=False)
    pw.tofile(os.path.join(OUT_DIR, "int4_plain_weight.u8"))
    ps.astype(np.float32).tofile(os.path.join(OUT_DIR, "int4_plain_scale.f32"))
    pexp.tofile(os.path.join(OUT_DIR, "int4_plain_expected.f32"))

    # ConvRot int4: rotate the row group-wise, then quantize.
    rot = rotate_groupwise(inp, GROUP_SIZE)
    cw, cs = quantize_int4(rot)
    cexp = dequantize_int4(cw, cs, cols, convrot=True)  # un-rotates back to input space
    cw.tofile(os.path.join(OUT_DIR, "int4_convrot_weight.u8"))
    cs.astype(np.float32).tofile(os.path.join(OUT_DIR, "int4_convrot_scale.f32"))
    cexp.tofile(os.path.join(OUT_DIR, "int4_convrot_expected.f32"))

    with open(os.path.join(OUT_DIR, "int4_meta.json"), "w") as f:
        json.dump({"rows": rows, "cols": cols, "group_size": GROUP_SIZE}, f)

    plain_err = float(np.abs(pexp - inp).mean())
    convrot_err = float(np.abs(cexp - inp).mean())
    print(f"wrote int4 fixtures to {OUT_DIR}")
    print(f"  plain   mean|err| = {plain_err:.6f}")
    print(f"  convrot mean|err| = {convrot_err:.6f}  (should be < plain for outlier-heavy data)")


if __name__ == "__main__":
    main()
