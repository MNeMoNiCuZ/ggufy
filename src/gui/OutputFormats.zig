const ggufy = @import("ggufy");
const types = ggufy.types;

/// One entry in the combined, user-facing output-format catalog. Used both to
/// render the multi-select checkbox list and the format-visibility settings
/// modal, and as the source of truth for persisted visibility (keyed by `label`).
pub const OutputFormat = struct {
    filetype: types.FileType,
    dtype: types.DataType,
    /// User-facing display name AND persisted settings key. Must be unique.
    label: []const u8,
};

/// Combined catalog: SafeTensors formats first, GGUF formats last (per user
/// request to keep GGUF toward the bottom of the list). This single array
/// replaces the old separate gguf_target_types/st_target_types split — there
/// is no longer a filetype dropdown gating which sub-list is shown.
pub const all_formats = [_]OutputFormat{
    .{ .filetype = .safetensors, .dtype = .F32, .label = "F32" },
    .{ .filetype = .safetensors, .dtype = .F16, .label = "F16" },
    .{ .filetype = .safetensors, .dtype = .BF16, .label = "BF16" },
    .{ .filetype = .safetensors, .dtype = .F8_E4M3, .label = "F8_E4M3" },
    .{ .filetype = .safetensors, .dtype = .SCALED_F8_E4M3, .label = "Scaled F8_E4M3" },
    .{ .filetype = .safetensors, .dtype = .F8_E5M2, .label = "F8_E5M2" },
    .{ .filetype = .safetensors, .dtype = .MXFP8_E4M3, .label = "MXFP8_E4M3" },
    .{ .filetype = .safetensors, .dtype = .NVFP4, .label = "NVFP4" },
    .{ .filetype = .safetensors, .dtype = .INT8, .label = "INT8" },
    .{ .filetype = .safetensors, .dtype = .INT8_CONVROT, .label = "INT8 ConvRot" },
    .{ .filetype = .safetensors, .dtype = .INT4_CONVROT, .label = "INT4 ConvRot" },
    .{ .filetype = .safetensors, .dtype = .INT4_CONVROT_SR, .label = "INT4 ConvRot SR" },

    .{ .filetype = .gguf, .dtype = .f32, .label = "GGUF f32" },
    .{ .filetype = .gguf, .dtype = .f16, .label = "GGUF f16" },
    .{ .filetype = .gguf, .dtype = .bf16, .label = "GGUF bf16" },
    .{ .filetype = .gguf, .dtype = .q2_k, .label = "GGUF q2_k" },
    .{ .filetype = .gguf, .dtype = .q3_k, .label = "GGUF q3_k" },
    .{ .filetype = .gguf, .dtype = .q4_0, .label = "GGUF q4_0" },
    .{ .filetype = .gguf, .dtype = .q4_1, .label = "GGUF q4_1" },
    .{ .filetype = .gguf, .dtype = .q4_k, .label = "GGUF q4_k" },
    .{ .filetype = .gguf, .dtype = .q5_0, .label = "GGUF q5_0" },
    .{ .filetype = .gguf, .dtype = .q5_1, .label = "GGUF q5_1" },
    .{ .filetype = .gguf, .dtype = .q5_k, .label = "GGUF q5_k" },
    .{ .filetype = .gguf, .dtype = .q6_k, .label = "GGUF q6_k" },
    .{ .filetype = .gguf, .dtype = .q8_0, .label = "GGUF q8_0" },
};

/// Filename-safe tag used by the renamer and as the naive `-<tag>` suffix
/// fallback — just the underlying DataType's tag name (e.g. "f16", "q4_k",
/// "F8_E4M3"), matching what the naming scheme already produced before this
/// feature existed.
pub fn fileTag(fmt: OutputFormat) []const u8 {
    return @tagName(fmt.dtype);
}

/// Index of the first GGUF entry — all_formats is grouped contiguously
/// (SafeTensors block, then GGUF block), so this is also the SafeTensors
/// block's length. Used to slice all_formats into its two groups for the
/// multi-column layout, keeping SafeTensors/GGUF grouping without a
/// per-item filetype scan.
pub const gguf_start_index = blk: {
    for (all_formats, 0..) |f, i| {
        if (f.filetype == .gguf) break :blk i;
    }
    break :blk all_formats.len;
};

pub fn indexOfLabel(label: []const u8) ?usize {
    for (all_formats, 0..) |f, i| {
        if (std.mem.eql(u8, f.label, label)) return i;
    }
    return null;
}

const std = @import("std");
