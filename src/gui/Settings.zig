//! Persisted GUI format settings. Plain JSON, no new dependency.
const std = @import("std");

pub const Settings = struct {
    /// gpa-owned strings, gpa-owned slice. Values are OutputFormat.label keys.
    hidden_formats: [][]u8 = &.{},
    formats: []FormatSetting = &.{},
    architecture_filter_enabled: bool = true,

    pub fn deinit(self: *Settings, allocator: std.mem.Allocator) void {
        for (self.hidden_formats) |f| allocator.free(f);
        allocator.free(self.hidden_formats);
        for (self.formats) |f| {
            allocator.free(f.key);
            allocator.free(f.output_name);
            for (f.unavailable_architectures) |name| allocator.free(name);
            allocator.free(f.unavailable_architectures);
        }
        allocator.free(self.formats);
        self.* = .{};
    }

    pub fn isHidden(self: Settings, label: []const u8) bool {
        for (self.hidden_formats) |f| {
            if (std.mem.eql(u8, f, label)) return true;
        }
        return false;
    }
};

pub const FormatSetting = struct {
    key: []u8,
    output_name: []u8,
    order: usize,
    unavailable_architectures: [][]u8,
};

pub const JsonFormatSetting = struct {
    key: []const u8,
    output_name: []const u8,
    order: usize,
    unavailable_architectures: []const []const u8 = &.{},
    flux: ?bool = null,
    zit: ?bool = null,
    ideogram4: ?bool = null,
};

const JsonShape = struct {
    hidden_formats: []const []const u8 = &.{},
    formats: []const JsonFormatSetting = &.{},
    architecture_filter_enabled: bool = true,
};

/// Resolve the settings file path: `%APPDATA%\ggufy\settings.json` if
/// APPDATA is available, else `<exe_dir>/.ggufy/settings.json`.
pub fn resolvePath(allocator: std.mem.Allocator, appdata: ?[]const u8, exe_dir: ?[]const u8) ![]u8 {
    if (appdata) |ad| {
        return std.fs.path.join(allocator, &.{ ad, "ggufy", "settings.json" });
    }
    return std.fs.path.join(allocator, &.{ exe_dir orelse ".", ".ggufy", "settings.json" });
}

/// Load settings from `path`. Any failure (missing file, corrupt JSON, etc.)
/// silently yields defaults (nothing hidden) rather than propagating an error
/// — a missing settings file on first run is the expected common case.
pub fn load(io: std.Io, allocator: std.mem.Allocator, path: []const u8) Settings {
    const file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch return .{};
    defer file.close(io);

    var read_buf: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    const data = reader.interface.allocRemaining(allocator, .unlimited) catch return .{};
    defer allocator.free(data);

    var parsed = std.json.parseFromSlice(JsonShape, allocator, data, .{ .ignore_unknown_fields = true }) catch return .{};
    defer parsed.deinit();

    var out: std.ArrayList([]u8) = .empty;
    for (parsed.value.hidden_formats) |f| {
        const dup = allocator.dupe(u8, f) catch continue;
        out.append(allocator, dup) catch {
            allocator.free(dup);
            continue;
        };
    }
    var formats: std.ArrayList(FormatSetting) = .empty;
    for (parsed.value.formats) |f| {
        const key = allocator.dupe(u8, f.key) catch continue;
        const output_name = allocator.dupe(u8, f.output_name) catch {
            allocator.free(key);
            continue;
        };
        var unavailable: std.ArrayList([]u8) = .empty;
        for (f.unavailable_architectures) |name| {
            const dup = allocator.dupe(u8, name) catch continue;
            unavailable.append(allocator, dup) catch allocator.free(dup);
        }
        if (f.flux == false) {
            const dup = allocator.dupe(u8, "flux") catch null;
            if (dup) |name| unavailable.append(allocator, name) catch allocator.free(name);
        }
        if (f.zit == false) {
            const dup = allocator.dupe(u8, "lumina2") catch null;
            if (dup) |name| unavailable.append(allocator, name) catch allocator.free(name);
        }
        const unavailable_owned = unavailable.toOwnedSlice(allocator) catch allocator.alloc([]u8, 0) catch unreachable;
        formats.append(allocator, .{
            .key = key, .output_name = output_name, .order = f.order,
            .unavailable_architectures = unavailable_owned,
        }) catch {
            allocator.free(key);
            allocator.free(output_name);
            for (unavailable_owned) |name| allocator.free(name);
            allocator.free(unavailable_owned);
        };
    }
    return .{
        .hidden_formats = out.toOwnedSlice(allocator) catch &.{},
        .formats = formats.toOwnedSlice(allocator) catch &.{},
        .architecture_filter_enabled = parsed.value.architecture_filter_enabled,
    };
}

pub fn save(io: std.Io, allocator: std.mem.Allocator, path: []const u8, hidden_formats: []const []const u8, formats: []const JsonFormatSetting, architecture_filter_enabled: bool) !void {
    if (std.fs.path.dirname(path)) |dir| {
        std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    }
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    const json = try std.json.Stringify.valueAlloc(allocator, JsonShape{
        .hidden_formats = hidden_formats,
        .formats = formats,
        .architecture_filter_enabled = architecture_filter_enabled,
    }, .{ .whitespace = .indent_2 });
    defer allocator.free(json);

    var write_buf: [8192]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    try writer.interface.writeAll(json);
    try writer.interface.flush();
}
