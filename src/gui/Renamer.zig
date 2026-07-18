//! Automatic output-filename renamer.
//!
//! Port of the precision-swap logic from the user's existing
//! SafetensorsModelPrecisionConverter (src/converter.py:
//! detect_precision_from_filename / generate_output_filename): if the
//! original filename already names a precision (fp32, fp16, bf16, fp8, and
//! the float32/float16/bfloat16/float8 spellings), swap that token in place
//! for the new format's tag. Otherwise fall back to the naive `-<tag>` suffix
//! that ggufy has always used.
const std = @import("std");

/// Precision keywords we recognize in an existing filename, longest-first so
/// a boundary-checked scan doesn't need to worry about one being a substring
/// of another (e.g. "float16" inside "bfloat16" is still rejected because the
/// preceding character 'b' isn't a separator).
const keywords = [_][]const u8{
    "bfloat16", "float32", "float16", "float8", "bf16", "fp32", "fp16", "fp8",
};

fn isSep(c: u8) bool {
    return c == '_' or c == '-' or c == '.';
}

pub const Match = struct { start: usize, end: usize };

/// Find the leftmost recognized precision token in `stem`, bounded on both
/// sides by `_`, `-`, `.`, or start/end of string. Case-insensitive.
pub fn detectPrecisionToken(allocator: std.mem.Allocator, stem: []const u8) !?Match {
    const lower = try std.ascii.allocLowerString(allocator, stem);
    defer allocator.free(lower);

    var best: ?Match = null;
    for (keywords) |kw| {
        var search_from: usize = 0;
        while (std.mem.indexOfPos(u8, lower, search_from, kw)) |idx| {
            const before_ok = idx == 0 or isSep(stem[idx - 1]);
            const after_idx = idx + kw.len;
            const after_ok = after_idx == stem.len or isSep(stem[after_idx]);
            if (before_ok and after_ok) {
                if (best == null or idx < best.?.start) best = .{ .start = idx, .end = after_idx };
                break;
            }
            search_from = idx + 1;
        }
    }
    return best;
}

/// Build the output stem for `original_stem` targeting `new_tag`:
/// - If a precision token is found, replace just that token in place,
///   preserving the surrounding separators and the rest of the name.
/// - Otherwise append "-{new_tag}" (today's existing naive behavior).
pub fn renameForFormat(allocator: std.mem.Allocator, original_stem: []const u8, new_tag: []const u8) ![]u8 {
    if (try detectPrecisionToken(allocator, original_stem)) |m| {
        return std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            original_stem[0..m.start], new_tag, original_stem[m.end..],
        });
    }
    return std.fmt.allocPrint(allocator, "{s}-{s}", .{ original_stem, new_tag });
}

test "detects and swaps a separator-bounded token" {
    const a = std.testing.allocator;
    const out = try renameForFormat(a, "model_fp16_v2", "q4_k");
    defer a.free(out);
    try std.testing.expectEqualStrings("model_q4_k_v2", out);
}

test "detects an end-anchored token" {
    const a = std.testing.allocator;
    const out = try renameForFormat(a, "model-bf16", "F8_E4M3");
    defer a.free(out);
    try std.testing.expectEqualStrings("model-F8_E4M3", out);
}

test "is case-insensitive and does not confuse bfloat16 with float16" {
    const a = std.testing.allocator;
    const out = try renameForFormat(a, "model_BFLOAT16", "f16");
    defer a.free(out);
    try std.testing.expectEqualStrings("model_f16", out);
}

test "falls back to naive suffix when no precision token is present" {
    const a = std.testing.allocator;
    const out = try renameForFormat(a, "my_cool_model", "q6_k");
    defer a.free(out);
    try std.testing.expectEqualStrings("my_cool_model-q6_k", out);
}
