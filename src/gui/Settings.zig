//! Minimal persisted GUI settings — currently just which output formats the
//! user has hidden from the combined format checklist. Plain JSON, no new
//! dependency. There is no other settings/config persistence in this repo
//! (checked before adding this).
const std = @import("std");

pub const Settings = struct {
    /// gpa-owned strings, gpa-owned slice. Values are OutputFormat.label keys.
    hidden_formats: [][]u8 = &.{},

    pub fn deinit(self: *Settings, allocator: std.mem.Allocator) void {
        for (self.hidden_formats) |f| allocator.free(f);
        allocator.free(self.hidden_formats);
        self.* = .{};
    }

    pub fn isHidden(self: Settings, label: []const u8) bool {
        for (self.hidden_formats) |f| {
            if (std.mem.eql(u8, f, label)) return true;
        }
        return false;
    }
};

const JsonShape = struct {
    hidden_formats: []const []const u8 = &.{},
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
    return .{ .hidden_formats = out.toOwnedSlice(allocator) catch &.{} };
}

pub fn save(io: std.Io, allocator: std.mem.Allocator, path: []const u8, hidden_formats: []const []const u8) !void {
    if (std.fs.path.dirname(path)) |dir| {
        std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    }
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    const json = try std.json.Stringify.valueAlloc(allocator, JsonShape{ .hidden_formats = hidden_formats }, .{ .whitespace = .indent_2 });
    defer allocator.free(json);

    var write_buf: [8192]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    try writer.interface.writeAll(json);
    try writer.interface.flush();
}
