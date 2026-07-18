const std = @import("std");
const guiState = @import("gui_state.zig");
const ggufy = @import("ggufy");
const conv = ggufy.convert;
const SDLBackend = @import("backend");
const OutputFormats = @import("OutputFormats.zig");
const Renamer = @import("Renamer.zig");

// Wakeup helper

fn pushWakeupEvent(state: *guiState.State) void {
    var ev: SDLBackend.c.SDL_Event = std.mem.zeroes(SDLBackend.c.SDL_Event);
    ev.type = state.wakeup_event_type;
    _ = SDLBackend.c.SDL_PushEvent(&ev);
}

// File loading

fn dominantType(tf: *const ggufy.fileLoader.TensorFile) ?ggufy.types.DataType {
    var best: ?ggufy.types.DataType = null;
    var best_count: usize = 0;
    var it = tf.type_counts.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* > best_count) {
            best_count = entry.value_ptr.*;
            best = entry.key_ptr.*;
        }
    }
    return best;
}

/// Load the whole pending batch (state.pending_paths) into state.input_files
/// as one atomic operation, then compute the cross-file architecture-mismatch
/// flag. Aborts the whole batch on the first file that fails to load (mirrors
/// the old single-file error behavior). Runs on a detached thread.
pub fn loadInputBatch(gpa: std.mem.Allocator, arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    state.load_state.store(.loading, .release);
    const paths = state.pending_paths.items;

    var i: usize = 0;
    while (i < paths.len) : (i += 1) {
        const p = paths[i];
        const tf = ggufy.fileLoader.TensorFile.loadFile(state.io, gpa, arena_alloc, p) catch |err| {
            gpa.free(p);
            for (paths[i + 1 ..]) |rest| gpa.free(rest);
            state.pending_paths.clearRetainingCapacity();
            state.load_error = err;
            state.load_state.store(.err, .release);
            pushWakeupEvent(state);
            return;
        };
        var inf = guiState.InputFile{ .path = p };
        inf.file = tf;
        inf.arch_name = if (tf.arch) |a| a.name else null;
        inf.dominant_type = dominantType(&tf);
        state.input_files.append(gpa, inf) catch {
            gpa.free(p);
        };
    }
    state.pending_paths.clearRetainingCapacity();

    // Cross-file architecture-consistency check.
    var mismatch = false;
    var first_name: ?[]const u8 = null;
    for (state.input_files.items) |f| {
        if (f.arch_name) |n| {
            if (first_name) |fname| {
                if (!std.mem.eql(u8, fname, n)) {
                    mismatch = true;
                    break;
                }
            } else first_name = n;
        }
    }
    state.arch_mismatch = mismatch;

    state.load_state.store(.done, .release);
    pushWakeupEvent(state);
}

/// SDL open-file dialog callback (multi-select) — stages the chosen paths
/// and signals the main loop to load them as one batch.
pub fn fileDialogCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.file_dialog_open = false;

    const files = filelist orelse {
        std.log.err("Dialog error: {s}", .{SDLBackend.c.SDL_GetError()});
        return;
    };
    if (files[0] == null) {
        std.log.info("File open dialog cancelled", .{});
        return;
    }

    var i: usize = 0;
    while (files[i] != null) : (i += 1) {
        const path = std.mem.span(files[i]);
        std.log.info("Selected: {s}", .{path});
        const dup = state.gpa.dupe(u8, path) catch continue;
        state.pending_paths.append(state.gpa, dup) catch state.gpa.free(dup);
    }
    if (state.pending_paths.items.len > 0) {
        state.pending_ready.store(true, .release);
    }
}

// Output-folder dialog callback

pub fn folderDialogCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.folder_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return; // cancelled

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.target_folder_buf.len - 1);
    @memcpy(state.target_folder_buf[0..len], path[0..len]);
    state.target_folder_buf[len] = 0;
    pushWakeupEvent(state);
}

// Sensitivity file dialog callback

pub fn sensitivityFileCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.sensitivity_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.sensitivity_path_buf.len - 1);
    @memcpy(state.sensitivity_path_buf[0..len], path[0..len]);
    state.sensitivity_path_buf[len] = 0;
    state.sensitivity_path = state.sensitivity_path_buf[0..len];
    // Custom sensitivity and built-in sensitivity are mutually exclusive.
    state.template_path = null;
    state.skip_sensitivity = false;
    pushWakeupEvent(state);
}

// Template file dialog callback

pub fn templateFileCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.template_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.template_path_buf.len - 1);
    @memcpy(state.template_path_buf[0..len], path[0..len]);
    state.template_path_buf[len] = 0;
    state.template_path = state.template_path_buf[0..len];
    // Selecting a template clears any sensitivity file
    state.sensitivity_path = null;
    pushWakeupEvent(state);
}

// Conversion progress/cancel callbacks

fn progressCallback(
    ctx: ?*anyopaque,
    done: u32,
    total: u32,
    name: []const u8,
    src_type: []const u8,
    dst_type: []const u8,
    n_elements: u64,
) void {
    const state: *guiState.State = @ptrCast(@alignCast(ctx));

    // Write tensor info before the .release store so the main thread
    // observes consistent values after its .acquire load of convert_progress.
    const name_len = @min(name.len, state.convert_tensor_name_buf.len);
    @memcpy(state.convert_tensor_name_buf[0..name_len], name[0..name_len]);
    state.convert_tensor_name_len = name_len;

    const src_len = @min(src_type.len, state.convert_tensor_src_type_buf.len);
    @memcpy(state.convert_tensor_src_type_buf[0..src_len], src_type[0..src_len]);
    state.convert_tensor_src_type_len = src_len;

    const dst_len = @min(dst_type.len, state.convert_tensor_dst_type_buf.len);
    @memcpy(state.convert_tensor_dst_type_buf[0..dst_len], dst_type[0..dst_len]);
    state.convert_tensor_dst_type_len = dst_len;

    state.convert_tensor_elements = n_elements;
    state.convert_total = total;

    // Release store: signals all writes above are visible to main thread.
    state.convert_progress.store(done, .release);
    pushWakeupEvent(state);
}

fn cancelCallback(ctx: ?*anyopaque) bool {
    const state: *guiState.State = @ptrCast(@alignCast(ctx));
    return state.cancel_requested.load(.acquire);
}

// Export template / generate sensitivities
// These tools operate on the first file of the current batch only.

fn setToolStatus(state: *guiState.State, is_error: bool, comptime fmt: []const u8, args: anytype) void {
    const msg = std.fmt.bufPrint(&state.tool_status_buf, fmt, args) catch "(message too long)";
    state.tool_status_len = msg.len;
    state.tool_status_is_error = is_error;
}

pub fn exportTemplateCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.export_template_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.export_template_path_buf.len - 1);
    @memcpy(state.export_template_path_buf[0..len], path[0..len]);
    state.export_template_path_buf[len] = 0;
    state.export_template_path = state.export_template_path_buf[0..len];
    state.export_template_requested = true;
    pushWakeupEvent(state);
}

pub fn genSensitivitiesCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.gen_sensitivities_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.gen_sensitivities_path_buf.len - 1);
    @memcpy(state.gen_sensitivities_path_buf[0..len], path[0..len]);
    state.gen_sensitivities_path_buf[len] = 0;
    state.gen_sensitivities_path = state.gen_sensitivities_path_buf[0..len];
    state.gen_sensitivities_requested = true;
    pushWakeupEvent(state);
}

pub fn doExportTemplate(arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    const path = state.export_template_path.?;
    const inf = &state.input_files.items[0];
    const loaded_file = &inf.file.?;
    const arch_opt: ?*const ggufy.imageArch.Arch = if (loaded_file.arch != null) &(loaded_file.arch.?) else null;
    const reverse_dims = loaded_file.type == .safetensors;

    const out_file = std.Io.Dir.cwd().createFile(state.io, path, .{ .truncate = true }) catch |err| {
        setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
        return;
    };
    defer out_file.close(state.io);

    var write_buf: [8192]u8 = undefined;
    var file_writer = out_file.writer(state.io, &write_buf);
    const writer = &file_writer.interface;

    // Re-open the source file so cluster detection can read `.comfy_quant` markers and
    // collapse ComfyUI cluster layouts into a single logical quant-typed entry — the
    // TensorFile loaded in `state` holds no open handle. The arena backs both temporary
    // allocations and the file object for this one-shot export.
    const src_path = inf.path;
    switch (loaded_file.type) {
        .safetensors => {
            var f = ggufy.safetensor.init(src_path, state.io, arena_alloc, arena_alloc, false, false) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
            defer f.deinit();
            conv.writeTemplateFromFile(&f, arch_opt, reverse_dims, writer, arena_alloc, arena_alloc) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
        },
        .gguf => {
            var f = ggufy.gguf.init(src_path, state.io, arena_alloc, arena_alloc, false) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
            defer f.deinit();
            conv.writeTemplateFromFile(&f, arch_opt, reverse_dims, writer, arena_alloc, arena_alloc) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
        },
    }
    writer.flush() catch {};
    setToolStatus(state, false, "Template exported to {s}", .{std.fs.path.basename(path)});
}

pub fn doGenSensitivities(arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    const path = state.gen_sensitivities_path.?;
    const inf = &state.input_files.items[0];
    const loaded_file = &inf.file.?;
    const arch_opt: ?*const ggufy.imageArch.Arch = if (loaded_file.arch != null) &(loaded_file.arch.?) else null;
    const threshold: u64 = if (arch_opt) |a| (a.threshhold orelse conv.QUANTIZATION_THRESHOLD) else conv.QUANTIZATION_THRESHOLD;

    const out_file = std.Io.Dir.cwd().createFile(state.io, path, .{ .truncate = true }) catch |err| {
        setToolStatus(state, true, "Failed: {s}", .{@errorName(err)});
        return;
    };
    defer out_file.close(state.io);

    var write_buf: [8192]u8 = undefined;
    var file_writer = out_file.writer(state.io, &write_buf);
    const writer = &file_writer.interface;

    conv.generateSensitivitiesFromTensors(
        loaded_file.tensors.items,
        arch_opt,
        threshold,
        writer,
        arena_alloc,
    ) catch |err| {
        setToolStatus(state, true, "Failed: {s}", .{@errorName(err)});
        return;
    };
    writer.flush() catch {};
    setToolStatus(state, false, "Sensitivities written to {s}", .{std.fs.path.basename(path)});
}

// Conversion options / prediction / batch conversion

/// Build a ConvertOptions for one (input file, output format) pair.
pub fn buildConvertOptionsForPair(
    state: *guiState.State,
    input_path: []const u8,
    fmt: OutputFormats.OutputFormat,
    output_name: []const u8,
) conv.ConvertOptions {
    const folder = state.targetFolder();
    return conv.ConvertOptions{
        .io = state.io,
        .path = input_path,
        .filetype = fmt.filetype,
        .datatype = fmt.dtype,
        .output_dir = if (folder.len > 0) folder else null,
        .output_name = output_name,
        .threads = state.target_threads,
        .skip_sensitivity = state.skip_sensitivity,
        .sensitivities_path = state.sensitivity_path,
        .template_path = state.template_path,
        .quantization_aggressiveness = @as(f32, @floatFromInt(state.target_aggressiveness)),
        .model_only = state.model_only,
        .allow_unknown_arch = state.allow_unknown_arch,
        .allow_upscale = state.allow_upscale,
        .arch_override = state.archOverride(),
    };
}

fn predictSizeForPair(pa: std.mem.Allocator, io: std.Io, path: []const u8, opts: conv.ConvertOptions) ?u64 {
    const file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch return null;
    var read_buf: [8]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const file_type = ggufy.types.FileType.detect_from_file(&file_reader.interface, pa) catch ggufy.types.FileType.safetensors;
    file.close(io);

    switch (file_type) {
        .safetensors => {
            var f = ggufy.safetensor.init(path, io, pa, pa, false, false) catch return null;
            defer f.deinit();
            return conv.predictOutputSize(&f, opts, pa, pa) catch null;
        },
        .gguf => {
            var f = ggufy.gguf.init(path, io, pa, pa, false) catch return null;
            defer f.deinit();
            return conv.predictOutputSize(&f, opts, pa, pa) catch null;
        },
    }
}

/// Recompute the per-pair output filename + predicted size preview for every
/// currently-selected (input file, output format) combination. Replaces
/// state.pair_previews in place. Called only when predictionSignature changes.
pub fn rebuildPairPreviews(gpa: std.mem.Allocator, state: *guiState.State) void {
    state.clearPairPreviews(gpa);
    if (state.input_files.items.len == 0) return;

    var pred_arena = std.heap.ArenaAllocator.init(gpa);
    defer pred_arena.deinit();

    for (state.input_files.items) |inf| {
        for (OutputFormats.all_formats, 0..) |fmt, i| {
            if (!state.target_formats[i] or state.hidden_formats[i]) continue;
            _ = pred_arena.reset(.retain_capacity);
            const pa = pred_arena.allocator();

            const stem = std.fs.path.stem(inf.path);
            const out_name = Renamer.renameForFormat(pa, stem, OutputFormats.fileTag(fmt)) catch continue;
            const opts = buildConvertOptionsForPair(state, inf.path, fmt, out_name);
            const out_path_tmp = conv.computeOutputPath(opts, pa) catch continue;
            const out_path = gpa.dupe(u8, out_path_tmp) catch continue;
            const size = predictSizeForPair(pa, state.io, inf.path, opts);

            state.pair_previews.append(gpa, .{
                .input_path = inf.path,
                .format_label = fmt.label,
                .output_path = out_path,
                .predicted_size = size,
            }) catch gpa.free(out_path);
        }
    }
}

fn detectFileType(io: std.Io, path: []const u8) ggufy.types.FileType {
    const file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch return .safetensors;
    defer file.close(io);
    var read_buf: [8]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    return ggufy.types.FileType.detect_from_file(&file_reader.interface, std.heap.page_allocator) catch .safetensors;
}

fn recordPairResult(
    gpa: std.mem.Allocator,
    state: *guiState.State,
    input_path: []const u8,
    format_label: []const u8,
    output_path: []const u8,
    err: ?anyerror,
) void {
    const dup = gpa.dupe(u8, output_path) catch return;
    state.batch_results.append(gpa, .{
        .input_path = input_path,
        .format_label = format_label,
        .output_path = dup,
        .err = err,
    }) catch gpa.free(dup);
    pushWakeupEvent(state);
}

/// Precompute all output paths that would be overwritten, and detect any
/// (file, format) pair whose output path equals one of the input paths.
/// Also checks for upscaling across every selected pair. Sets
/// state.same_file_error / state.overwrite_pending_paths / state.upscale_pending
/// as appropriate, and sets state.convert_requested = true only if none of
/// those need user confirmation first.
pub fn prepareBatchLaunch(gpa: std.mem.Allocator, state: *guiState.State) void {
    state.clearOverwritePending(gpa);
    state.same_file_error = false;
    state.same_file_error_len = 0;

    if (state.pair_previews.items.len == 0) return;

    for (state.pair_previews.items) |p| {
        for (state.input_files.items) |inf| {
            if (std.mem.eql(u8, p.output_path, inf.path)) {
                state.same_file_error = true;
                const msg = std.fmt.bufPrint(&state.same_file_error_buf, "{s}", .{p.output_path}) catch "(path too long)";
                state.same_file_error_len = msg.len;
                return;
            }
        }
    }

    if (!state.allow_upscale) {
        for (state.input_files.items) |inf| {
            const tf = inf.file orelse continue;
            for (OutputFormats.all_formats, 0..) |fmt, i| {
                if (!state.target_formats[i] or state.hidden_formats[i]) continue;
                if (conv.detectUpscaling(tf.tensors.items, fmt.dtype)) {
                    state.upscale_pending = true;
                    return;
                }
            }
        }
    }

    for (state.pair_previews.items) |p| {
        const exists = blk: {
            std.Io.Dir.cwd().access(state.io, p.output_path, .{}) catch break :blk false;
            break :blk true;
        };
        if (exists) {
            const dup = gpa.dupe(u8, p.output_path) catch continue;
            state.overwrite_pending_paths.append(gpa, dup) catch gpa.free(dup);
        }
    }

    if (state.overwrite_pending_paths.items.len > 0) return; // modal shown by the GUI

    state.convert_requested = true;
}

/// Convert every currently-selected (input file, output format) pair.
/// Continue-on-error: one pair failing doesn't stop the rest of the batch.
/// Cancellation stops the outer loop promptly (checked between pairs, on top
/// of the existing per-tensor cancel callback). Runs on a detached thread.
pub fn convertAll(gpa: std.mem.Allocator, arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    _ = arena_alloc; // per-pair scratch uses a dedicated local arena instead (see below)
    state.convert_state.store(.converting, .release);
    pushWakeupEvent(state);

    var pairs: std.ArrayList(struct { file_idx: usize, fmt_idx: usize }) = .empty;
    defer pairs.deinit(gpa);
    for (state.input_files.items, 0..) |_, fi| {
        for (OutputFormats.all_formats, 0..) |_, ti| {
            if (state.target_formats[ti] and !state.hidden_formats[ti]) {
                pairs.append(gpa, .{ .file_idx = fi, .fmt_idx = ti }) catch {};
            }
        }
    }

    state.convert_pair_total = pairs.items.len;
    state.convert_pair_idx = 0;

    var conv_arena = std.heap.ArenaAllocator.init(gpa);
    defer conv_arena.deinit();

    const convert_start_ts = std.Io.Clock.Timestamp.now(state.io, .awake);

    for (pairs.items, 0..) |pair, idx| {
        if (state.cancel_requested.load(.acquire)) {
            state.cancel_requested.store(false, .release);
            break;
        }
        state.convert_pair_idx = idx;
        state.convert_progress.store(0, .release);
        state.convert_tensor_name_len = 0;
        pushWakeupEvent(state);

        _ = conv_arena.reset(.retain_capacity);
        const ca = conv_arena.allocator();

        const inf = &state.input_files.items[pair.file_idx];
        const fmt = OutputFormats.all_formats[pair.fmt_idx];

        const stem = std.fs.path.stem(inf.path);
        const out_name = Renamer.renameForFormat(ca, stem, OutputFormats.fileTag(fmt)) catch |err| {
            recordPairResult(gpa, state, inf.path, fmt.label, inf.path, err);
            continue;
        };

        var opts = buildConvertOptionsForPair(state, inf.path, fmt, out_name);
        opts.callbacks = .{
            .progress_fn = progressCallback,
            .progress_ctx = state,
            .cancel_fn = cancelCallback,
            .cancel_ctx = state,
        };

        const output_path_str = conv.computeOutputPath(opts, ca) catch |err| {
            recordPairResult(gpa, state, inf.path, fmt.label, inf.path, err);
            continue;
        };

        const file_type = detectFileType(state.io, inf.path);

        const maybe_err: ?anyerror = blk: {
            switch (file_type) {
                .safetensors => {
                    var f = ggufy.safetensor.init(inf.path, state.io, gpa, ca, false, false) catch |err| break :blk err;
                    defer f.deinit();
                    conv.convert(&f, opts, gpa, ca) catch |err| break :blk err;
                },
                .gguf => {
                    var f = ggufy.gguf.init(inf.path, state.io, gpa, ca, false) catch |err| break :blk err;
                    defer f.deinit();
                    conv.convert(&f, opts, gpa, ca) catch |err| break :blk err;
                },
            }
            break :blk null;
        };

        if (maybe_err) |err| {
            if (err == error.Cancelled) {
                state.cancel_requested.store(false, .release);
                break;
            }
            recordPairResult(gpa, state, inf.path, fmt.label, output_path_str, err);
        } else {
            recordPairResult(gpa, state, inf.path, fmt.label, output_path_str, null);
        }
    }

    const elapsed = convert_start_ts.durationTo(std.Io.Clock.Timestamp.now(state.io, .awake));
    state.convert_elapsed_ns = @intCast(@max(@as(i96, 0), elapsed.raw.nanoseconds));

    state.convert_state.store(.done, .release);
    pushWakeupEvent(state);
}
