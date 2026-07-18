const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const SDLBackend = @import("backend");
const ggufy = @import("ggufy");
const guiState = @import("gui_state.zig");
const fileHandling = @import("file_handling.zig");
const OutputFormats = @import("OutputFormats.zig");
const SettingsMod = @import("Settings.zig");
const conv = ggufy.convert;
const build_options = @import("build_options");

comptime {
    std.debug.assert(@hasDecl(SDLBackend, "SDLBackend"));
}

const window_icon_png = @embedFile("gg.png");

var gpa_instance = std.heap.DebugAllocator(.{}){};
const gpa = gpa_instance.allocator();

var arena = std.heap.ArenaAllocator.init(gpa);
const arena_alloc = arena.allocator();

var state: guiState.State = .{};

var g_backend: ?SDLBackend = null;
var g_win: ?*dvui.Window = null;

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").os.tag == .windows) {
        dvui.Backend.Common.windowsAttachConsole() catch {};
    }
    SDLBackend.enableSDLLogging();
    std.log.info("SDL version: {f}", .{SDLBackend.getSDLVersion()});

    defer if (gpa_instance.deinit() != .ok) @panic("Memory leak on exit!");
    defer arena.deinit();
    defer state.clearInputFiles(gpa);
    defer state.clearBatchResults(gpa);
    defer state.clearPairPreviews(gpa);
    defer state.clearOverwritePending(gpa);

    // Populate CPU count and thread default before first frame.
    state.io = init.io;
    state.gpa = gpa;
    state.cpu_count = std.Thread.getCpuCount() catch 4;
    state.target_threads = state.cpu_count;

    // Resolve + load persisted settings (which output formats are hidden).
    {
        const appdata = init.environ_map.get("APPDATA");
        const exe_dir = std.process.executableDirPathAlloc(init.io, gpa) catch null;
        defer if (exe_dir) |d| gpa.free(d);
        if (SettingsMod.resolvePath(gpa, appdata, exe_dir) catch null) |sp| {
            defer gpa.free(sp);
            const len = @min(sp.len, state.settings_path_buf.len);
            @memcpy(state.settings_path_buf[0..len], sp[0..len]);
            state.settings_path_len = len;

            var settings = SettingsMod.load(state.io, gpa, sp);
            defer settings.deinit(gpa);
            for (settings.hidden_formats) |label| {
                if (OutputFormats.indexOfLabel(label)) |idx| state.hidden_formats[idx] = true;
            }
        }
    }

    var backend = try SDLBackend.initWindow(.{
        .allocator = gpa,
        .io = init.io,
        .size = .{ .w = 800.0, .h = 600.0 },
        .min_size = .{ .w = 250.0, .h = 350.0 },
        .vsync = true,
        .title = "ggufy",
        .icon = window_icon_png,
    });
    g_backend = backend;
    defer backend.deinit();

    _ = SDLBackend.c.SDL_EnableScreenSaver();
    state.wakeup_event_type = SDLBackend.c.SDL_RegisterEvents(1);

    var win = try dvui.Window.init(@src(), gpa, backend.backend(), .{
        .theme = switch (backend.preferredColorScheme() orelse .dark) {
            .light => dvui.Theme.builtin.adwaita_light,
            .dark => dvui.Theme.builtin.adwaita_dark,
        },
    });
    defer win.deinit();
    g_win = &win;

    // Register a synchronous event watch for drag-and-drop.  An event watch
    // fires the moment SDL pumps an OS event into its queue - before any
    // SDL_PollEvent caller (including addAllEvents) can consume it.  This
    // guarantees we never miss DROP_FILE regardless of frame timing.
    _ = SDLBackend.c.SDL_AddEventWatch(dropEventWatch, &state);

    var interrupted = false;

    main_loop: while (true) {
        const nstime = win.beginWait(interrupted);

        // Batch-load trigger — fires once per new selection/drop, whether it
        // carries one file or many. New files are ADDED to any already-loaded
        // input_files (drop/"Open Files" again to load more); only a load into
        // an empty list resets the shared arena and per-batch conversion state,
        // since earlier files' tensor/arch data would otherwise dangle mid-load.
        // Use "Unload all" to explicitly clear the current batch first.
        if (state.pending_ready.load(.acquire)) {
            state.pending_ready.store(false, .release);
            state.load_state.store(.loading, .release);
            const adding_to_existing = state.input_files.items.len > 0;
            if (!adding_to_existing) {
                _ = arena.reset(.free_all);
                state.clearBatchResults(gpa);
                state.clearPairPreviews(gpa);
                state.clearOverwritePending(gpa);
                state.convert_options_initialized = false;
                state.convert_state.store(.idle, .release);
                state.convert_progress.store(0, .release);
                state.batch_fatal_error = null;
                state.sensitivity_path = null;
                state.template_path = null;
                state.skip_sensitivity = false;
                state.allow_unknown_arch = false;
                state.allow_upscale = false;
                state.upscale_pending = false;
                state.arch_override_buf = std.mem.zeroes([64]u8);
                state.tool_status_len = 0;
                state.same_file_error = false;
                state.same_file_error_len = 0;
                state.target_folder_buf = std.mem.zeroes([std.fs.max_path_bytes]u8);
                @memset(&state.target_formats, false);
            }
            state.prev_pred_signature = null; // force preview recompute
            const thread = std.Thread.spawn(.{ .allocator = gpa }, fileHandling.loadInputBatch, .{ gpa, arena_alloc, &state }) catch |err| {
                state.load_error = err;
                state.load_state.store(.err, .release);
                continue :main_loop;
            };
            thread.detach();
        }

        // Conversion trigger
        if (state.convert_requested) {
            state.convert_requested = false;
            state.convert_progress.store(0, .release);
            state.convert_tensor_name_len = 0;
            state.convert_tensor_src_type_len = 0;
            state.convert_tensor_dst_type_len = 0;
            state.convert_tensor_elements = 0;
            state.batch_fatal_error = null;
            state.clearBatchResults(gpa);
            const thread = std.Thread.spawn(.{ .allocator = gpa }, fileHandling.convertAll, .{ gpa, arena_alloc, &state }) catch |err| {
                state.batch_fatal_error = err;
                state.convert_state.store(.err, .release);
                continue :main_loop;
            };
            thread.detach();
        }

        // Export template trigger (synchronous — fast file write)
        if (state.export_template_requested) {
            state.export_template_requested = false;
            fileHandling.doExportTemplate(arena_alloc, &state);
        }

        // Generate sensitivities trigger (synchronous — fast file write)
        if (state.gen_sensitivities_requested) {
            state.gen_sensitivities_requested = false;
            fileHandling.doGenSensitivities(arena_alloc, &state);
        }

        try win.begin(nstime);

        // Let dvui's backend consume all pending SDL events (mouse, keyboard, etc.).
        // Drop events are handled by dropEventWatch above and are ignored here.
        try backend.addAllEvents(&win);

        _ = SDLBackend.c.SDL_SetRenderDrawColor(backend.renderer, 0, 0, 0, 0);
        _ = SDLBackend.c.SDL_RenderClear(backend.renderer);

        const keep_running = gui_frame();
        if (!keep_running) break :main_loop;

        const end_micros = try win.end(.{});
        try backend.setCursor(win.cursorRequested());
        try backend.textInputRect(win.textInputRequested());
        try backend.renderPresent();
        const wait_event_micros = win.waitTime(end_micros);
        interrupted = try backend.waitEventTimeout(wait_event_micros);
    }
}

// Top-level frame

fn gui_frame() bool {
    // Menu bar
    {
        var hbox = dvui.box(@src(), .{ .dir = .horizontal }, .{ .style = .window, .background = true, .expand = .horizontal, .name = "main" });
        defer hbox.deinit();

        var m = dvui.menu(@src(), .horizontal, .{});
        defer m.deinit();

        if (dvui.menuItemLabel(@src(), "File", .{ .submenu = true }, .{})) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r }, .{});
            defer fw.deinit();

            if (dvui.menuItemLabel(@src(), "Open Files...", .{}, .{ .expand = .horizontal }) != null) {
                if (!state.file_dialog_open) {
                    state.file_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFileDialog(
                        fileHandling.fileDialogCallback,
                        &state,
                        g_backend.?.window,
                        &file_filters,
                        file_filters.len,
                        null,
                        true, // allow multiple selection
                    );
                }
            }

            if (dvui.menuItemLabel(@src(), "Formats...", .{}, .{ .expand = .horizontal }) != null) {
                @memcpy(&state.settings_temp_hidden, &state.hidden_formats);
                state.settings_dialog_open = true;
            }

            if (dvui.menuItemLabel(@src(), "Exit", .{}, .{ .expand = .horizontal }) != null) {
                return false;
            }
        }

        if (dvui.menuItemLabel(@src(), "Help", .{ .submenu = true }, .{})) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r }, .{});
            defer fw.deinit();
            if (dvui.menuItemLabel(@src(), "About ggufy", .{}, .{ .expand = .horizontal }) != null) {
                state.show_about = true;
            }
            if (dvui.menuItemLabel(@src(), "GitHub Page", .{}, .{ .expand = .horizontal }) != null) {
                _ = SDLBackend.c.SDL_OpenURL("https://github.com/qskousen/ggufy");
            }
        }
    }

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();

    // Drop-zone highlight
    const dropping = state.dropping;
    const border_color: dvui.Color = .{ .r = if (dropping) 120 else 0, .g = if (dropping) 120 else 0, .b = if (dropping) 230 else 0, .a = 240 };
    const border: ?dvui.Rect = if (dropping) dvui.Rect.all(1) else null;
    const background_color: dvui.Color = .{ .r = if (dropping) 120 else 0, .g = if (dropping) 120 else 0, .b = if (dropping) 230 else 0, .a = 80 };

    var box = dvui.box(@src(), .{}, .{ .expand = .both, .color_border = border_color, .border = border, .color_fill = background_color, .background = true });
    defer box.deinit();

    switch (state.load_state.load(.acquire)) {
        .idle => showIntro(),
        .loading => showLoading(),
        .done => switch (state.convert_state.load(.acquire)) {
            .idle => showInputFile(),
            .converting => showConverting(),
            .done => showBatchSummary(),
            .err => showBatchFatalError(),
        },
        .err => showLoadError(),
    }

    // Overwrite dialog floats on top of everything
    if (state.overwrite_pending_paths.items.len > 0) showOverwriteDialog();

    // Upscale warning dialog
    if (state.upscale_pending) showUpscaleDialog();

    // About modal
    if (state.show_about) showAboutModal();

    // Format-visibility settings modal
    if (state.settings_dialog_open) showFormatSettingsModal();

    // Check for quit events
    for (dvui.events()) |*e| {
        if (e.evt == .window and e.evt.window.action == .close) return false;
        if (e.evt == .app and e.evt.app.action == .quit) return false;
    }

    return true;
}

// Helper: frame arena

fn frameArena() std.mem.Allocator {
    return dvui.currentWindow().arena();
}

// Formatting helpers

fn formatWithCommas(value: u64, buf: []u8) []u8 {
    var tmp: [20]u8 = undefined;
    var tmp_len: usize = 0;
    if (value == 0) { buf[0] = '0'; return buf[0..1]; }
    var v = value;
    while (v > 0) {
        tmp[tmp_len] = '0' + @as(u8, @intCast(v % 10));
        tmp_len += 1;
        v /= 10;
    }
    var out_idx: usize = 0;
    for (0..tmp_len) |i| {
        const digit_pos = tmp_len - 1 - i;
        if (i > 0 and (tmp_len - i) % 3 == 0) { buf[out_idx] = ','; out_idx += 1; }
        buf[out_idx] = tmp[digit_pos];
        out_idx += 1;
    }
    return buf[0..out_idx];
}

fn formatBytes(value: u64, buf: []u8) []u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB", "PB" };
    var idx: usize = 0;
    var scaled = @as(f64, @floatFromInt(value));
    while (scaled >= 1024.0 and idx < units.len - 1) { scaled /= 1024.0; idx += 1; }
    if (idx == 0) return std.fmt.bufPrint(buf, "{d}{s}", .{ value, units[idx] }) catch buf[0..0];
    if (scaled >= 100.0) return std.fmt.bufPrint(buf, "{d:.0}{s}", .{ scaled, units[idx] }) catch buf[0..0];
    if (scaled >= 10.0)  return std.fmt.bufPrint(buf, "{d:.1}{s}", .{ scaled, units[idx] }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "{d:.2}{s}", .{ scaled, units[idx] }) catch buf[0..0];
}

// Screen: intro

fn showIntro() void {
    var box_inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer box_inner.deinit();

    dvui.label(@src(), "Convert safetensors or gguf files", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });

    if (dvui.button(@src(), "Select Files", .{}, .{ .gravity_x = 0.5 })) {
        if (!state.file_dialog_open) {
            state.file_dialog_open = true;
            SDLBackend.c.SDL_ShowOpenFileDialog(
                fileHandling.fileDialogCallback,
                &state,
                g_backend.?.window,
                &file_filters,
                file_filters.len,
                null,
                true, // allow multiple selection
            );
        }
    }
    dvui.label(@src(), "Or drag and drop one or more files", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });
}

// Screen: loading

fn showLoading() void {
    var box_inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer box_inner.deinit();
    dvui.label(@src(), "Loading...", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });
}

// Screen: load error

fn showLoadError() void {
    var box_inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer box_inner.deinit();
    dvui.label(@src(), "Error loading file: {}", .{state.load_error.?}, .{ .gravity_x = 0.5, .font = .theme(.title) });
    if (dvui.button(@src(), "Back", .{}, .{ .gravity_x = 0.5 })) {
        state.load_state.store(.idle, .release);
        state.load_error = null;
    }
}

// Screen: input files + conversion options

fn showInputFile() void {
    const fa = frameArena();

    // Auto-populate output folder once on first display of a batch.
    const first_init = !state.convert_options_initialized;
    if (first_init) {
        state.convert_options_initialized = true;
        if (state.target_folder_buf[0] == 0 and state.input_files.items.len > 0) {
            const dir = std.fs.path.dirname(state.input_files.items[0].path) orelse ".";
            const dir_len = @min(dir.len, state.target_folder_buf.len - 1);
            @memcpy(state.target_folder_buf[0..dir_len], dir[0..dir_len]);
            state.target_folder_buf[dir_len] = 0;
        }
    }

    const dim_color = dvui.themeGet().color(.control, .text).opacity(0.45);
    const warn_color = dvui.Color{ .r = 200, .g = 140, .b = 0, .a = 255 };
    const err_color = dvui.Color{ .r = 220, .g = 60, .b = 60, .a = 255 };

    // Selected input files
    {
        var list_box = dvui.box(@src(), .{}, .{ .expand = .horizontal, .border = dvui.Rect.all(1), .margin = .all(4), .padding = .all(6) });
        defer list_box.deinit();

        dvui.label(@src(), "Input files ({d})", .{state.input_files.items.len}, .{ .font = .theme(.title) });

        for (state.input_files.items, 0..) |inf, i| {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 4, .w = 0, .h = 0 }, .id_extra = i });
            defer row.deinit();

            const arch_name = inf.arch_name orelse "unknown";
            const orig_type: []const u8 = if (inf.dominant_type) |dt| @tagName(dt) else "?";
            var size_buf: [16]u8 = undefined;
            const size_str = if (inf.file) |f| formatBytes(f.sizeInBytes, &size_buf) else "?";
            const filetype_name: []const u8 = if (inf.file) |f| @tagName(f.type) else "?";

            const line = std.fmt.allocPrint(fa, "{s}   [{s}]   Arch: {s}   Original: {s}   Size: {s}", .{
                std.fs.path.basename(inf.path), filetype_name, arch_name, orig_type, size_str,
            }) catch inf.path;
            dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .gravity_y = 0.5 });

            if (dvui.button(@src(), "Remove", .{}, .{ .gravity_y = 0.5, .id_extra = i })) {
                removeInputFile(i);
                return; // list mutated mid-iteration; redraw next frame
            }
        }
    }

    // Architecture-mismatch warning (hard block — not dismissible)
    if (state.arch_mismatch) {
        var warn_box = dvui.box(@src(), .{}, .{
            .expand = .horizontal, .border = dvui.Rect.all(1),
            .margin = .{ .x = 0, .y = 6, .w = 0, .h = 2 }, .padding = .all(6),
            .color_border = err_color,
        });
        defer warn_box.deinit();
        dvui.label(@src(), "The selected files have different architectures. Remove files until they all share one architecture before converting.", .{}, .{
            .color_text = err_color,
        });
    }

    // Conversion options
    {
        var opts_box = dvui.box(@src(), .{}, .{ .expand = .horizontal, .margin = .{ .x = 4, .y = 8, .w = 4, .h = 4 } });
        defer opts_box.deinit();

        // Combined output-format checklist — SafeTensors formats first, GGUF last,
        // laid out in 4 columns (same column layout as the Formats settings modal,
        // so a format sits in the same visual slot in both places).
        {
            dvui.label(@src(), "Output formats", .{}, .{ .font = .theme(.title), .margin = .{ .x = 0, .y = 4, .w = 0, .h = 4 } });

            dvui.label(@src(), "SafeTensors formats", .{}, .{ .color_text = dim_color, .margin = .{ .x = 0, .y = 6, .w = 0, .h = 2 } });
            showFormatGrid(0, OutputFormats.all_formats[0..OutputFormats.gguf_start_index], &state.target_formats, &state.hidden_formats);

            dvui.label(@src(), "GGUF formats", .{}, .{ .color_text = dim_color, .margin = .{ .x = 0, .y = 6, .w = 0, .h = 2 } });
            showFormatGrid(OutputFormats.gguf_start_index, OutputFormats.all_formats[OutputFormats.gguf_start_index..], &state.target_formats, &state.hidden_formats);
        }

        // Output folder row
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 2 } });
            defer row.deinit();
            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Output folder", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
            dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                "Directory where output files will be written. Defaults to the first input file's directory.", .{}, .{});

            var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = &state.target_folder_buf } }, .{ .expand = .horizontal, .gravity_y = 0.5 });
            te.deinit();

            if (dvui.button(@src(), "Browse...", .{}, .{ .gravity_y = 0.5 })) {
                if (!state.folder_dialog_open) {
                    state.folder_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFolderDialog(
                        fileHandling.folderDialogCallback,
                        &state,
                        g_backend.?.window,
                        null,
                        false,
                    );
                }
            }
        }

        // Advanced section (accordion, collapsed by default)
        {
            var adv_tree = dvui.TreeWidget.tree(@src(), .{ .enable_reordering = false }, .{ .expand = .horizontal });
            defer adv_tree.deinit();
            const adv_branch = adv_tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal });
            defer adv_branch.deinit();
            const adv_caret: []const u8 = if (adv_branch.expanded) "v " else "> ";
            const adv_header = std.fmt.allocPrint(fa, "{s}Advanced", .{adv_caret}) catch "> Advanced";
            dvui.labelNoFmt(@src(), adv_header, .{}, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 2 } });
            if (adv_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
                // These options apply per-pair based on that pair's own output
                // filetype (a batch may mix SafeTensors and GGUF targets), so
                // nothing here is dimmed based on a single global filetype anymore.
                const first_arch: ?ggufy.imageArch.Arch = if (state.input_files.items.len > 0)
                    (if (state.input_files.items[0].file) |f| f.arch else null)
                else
                    null;
                const has_sensitivities = if (first_arch) |a| a.sensitivities.len > 1 else false;

                // Model only checkbox — applies to SafeTensors-target pairs
                {
                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();
                    var cwd: dvui.WidgetData = undefined;
                    _ = dvui.checkbox(@src(), &state.model_only, "Model only", .{ .gravity_y = 0.5, .data_out = &cwd });
                    dvui.tooltip(@src(), .{ .active_rect = cwd.borderRectScale().r },
                        "Only convert the main model (UNet/transformer). CLIP, VAE, and other components are excluded. Only applies to SafeTensors output formats.", .{}, .{});
                }

                // Architecture name override — applies to GGUF-target pairs
                {
                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();
                    var lwd: dvui.WidgetData = undefined;
                    dvui.label(@src(), "Architecture", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
                    dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                        "Override the architecture name written to GGUF metadata. Leave blank to use the auto-detected name. Only applies to GGUF output formats.", .{}, .{});
                    var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = &state.arch_override_buf } }, .{ .expand = .horizontal, .gravity_y = 0.5 });
                    te.deinit();
                }

                // Skip sensitivity — shown when the shared arch has built-in data
                if (has_sensitivities and state.sensitivity_path == null) {
                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();
                    var cwd: dvui.WidgetData = undefined;
                    _ = dvui.checkbox(@src(), &state.skip_sensitivity, "Skip built-in sensitivity", .{ .gravity_y = 0.5, .data_out = &cwd });
                    dvui.tooltip(@src(), .{ .active_rect = cwd.borderRectScale().r },
                        "By default, per-layer sensitivity scores preserve precision on important layers. Check this to quantize all eligible layers uniformly. Only applies to GGUF output formats.", .{}, .{});
                }

                // Sensitivity file row
                {
                    const blocked_by_template = state.template_path != null;
                    const row_color: ?dvui.Color = if (blocked_by_template) dim_color else null;

                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();
                    dvui.label(@src(), "Sensitivity file", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .color_text = row_color });

                    const sens_display = if (state.sensitivity_path) |p| p else "none";
                    dvui.labelNoFmt(@src(), sens_display, .{}, .{ .expand = .horizontal, .gravity_y = 0.5, .color_text = row_color });

                    if (state.sensitivity_path != null and !blocked_by_template) {
                        if (dvui.button(@src(), "Clear", .{}, .{ .gravity_y = 0.5 })) {
                            state.sensitivity_path = null;
                        }
                    }
                    var bwd: dvui.WidgetData = undefined;
                    if (dvui.button(@src(), "Browse...", .{}, .{ .gravity_y = 0.5, .color_text = row_color, .data_out = &bwd })) {
                        if (!blocked_by_template and !state.sensitivity_dialog_open) {
                            state.sensitivity_dialog_open = true;
                            SDLBackend.c.SDL_ShowOpenFileDialog(
                                fileHandling.sensitivityFileCallback,
                                &state,
                                g_backend.?.window,
                                &json_filters,
                                json_filters.len,
                                null,
                                false,
                            );
                        }
                    }
                    if (blocked_by_template) {
                        dvui.tooltip(@src(), .{ .active_rect = bwd.borderRectScale().r },
                            "Cannot use a sensitivity file while a template is selected.", .{}, .{});
                    }
                }

                // Template file row
                {
                    const blocked_by_sens = state.sensitivity_path != null;
                    const row_color: ?dvui.Color = if (blocked_by_sens) dim_color else null;

                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();
                    dvui.label(@src(), "Template file", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .color_text = row_color });

                    const tmpl_display = if (state.template_path) |p| p else "none";
                    dvui.labelNoFmt(@src(), tmpl_display, .{}, .{ .expand = .horizontal, .gravity_y = 0.5, .color_text = row_color });

                    if (state.template_path != null and !blocked_by_sens) {
                        if (dvui.button(@src(), "Clear", .{}, .{ .gravity_y = 0.5 })) {
                            state.template_path = null;
                        }
                    }
                    var bwd: dvui.WidgetData = undefined;
                    if (dvui.button(@src(), "Browse...", .{}, .{ .gravity_y = 0.5, .color_text = row_color, .data_out = &bwd })) {
                        if (!blocked_by_sens and !state.template_dialog_open) {
                            state.template_dialog_open = true;
                            SDLBackend.c.SDL_ShowOpenFileDialog(
                                fileHandling.templateFileCallback,
                                &state,
                                g_backend.?.window,
                                &json_filters,
                                json_filters.len,
                                null,
                                false,
                            );
                        }
                    }
                    if (blocked_by_sens) {
                        dvui.tooltip(@src(), .{ .active_rect = bwd.borderRectScale().r },
                            "Cannot use a template while a sensitivity file is selected.", .{}, .{});
                    }
                }

                // Aggressiveness slider
                {
                    const sens_active = !state.skip_sensitivity and (has_sensitivities or state.sensitivity_path != null);
                    const agg_color: ?dvui.Color = if (!sens_active) dim_color else null;

                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();

                    var lwd: dvui.WidgetData = undefined;
                    dvui.label(@src(), "Aggressiveness", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .color_text = agg_color, .data_out = &lwd });
                    dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                        "How aggressively to quantize sensitive layers. Higher = smaller file, lower = better quality. Only applies to GGUF output formats.", .{}, .{});

                    var agg_label_buf: [8]u8 = undefined;
                    const agg_label = std.fmt.bufPrint(&agg_label_buf, "{d}", .{state.target_aggressiveness}) catch "?";
                    dvui.labelNoFmt(@src(), agg_label, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 28 }, .color_text = agg_color });

                    var agg_frac: f32 = (@as(f32, @floatFromInt(state.target_aggressiveness)) - 1.0) / 99.0;
                    if (sens_active) {
                        if (dvui.slider(@src(), .{ .fraction = &agg_frac }, .{ .expand = .horizontal, .gravity_y = 0.5 })) {
                            const raw: u8 = @intFromFloat(@round(agg_frac * 99.0));
                            state.target_aggressiveness = std.math.clamp(raw + 1, 1, 100);
                        }
                    } else {
                        dvui.progress(@src(), .{ .percent = agg_frac }, .{ .expand = .horizontal, .gravity_y = 0.5, .color_fill = dim_color });
                    }
                }

                // Threads slider
                {
                    const cpu_f: f32 = @floatFromInt(state.cpu_count);
                    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
                    defer row.deinit();

                    var lwd: dvui.WidgetData = undefined;
                    dvui.label(@src(), "Threads", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
                    dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                        "Number of CPU threads to use during quantization.", .{}, .{});

                    var thr_label_buf: [8]u8 = undefined;
                    const thr_label = std.fmt.bufPrint(&thr_label_buf, "{d}", .{state.target_threads}) catch "?";
                    dvui.labelNoFmt(@src(), thr_label, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 28 } });

                    var thr_frac: f32 = (@as(f32, @floatFromInt(state.target_threads)) - 1.0) / (cpu_f - 1.0);
                    if (dvui.slider(@src(), .{ .fraction = &thr_frac }, .{ .expand = .horizontal, .gravity_y = 0.5 })) {
                        const raw = @as(usize, @intFromFloat(@round(thr_frac * (cpu_f - 1.0))));
                        state.target_threads = std.math.clamp(raw + 1, 1, state.cpu_count);
                    }
                }
            }
        }

        // Unrecognized-architecture warning + override checkbox (per-file, dismissible)
        var any_unknown = false;
        for (state.input_files.items) |inf| {
            if (inf.arch_name == null) { any_unknown = true; break; }
        }
        if (any_unknown and !state.arch_mismatch) {
            var warn_box = dvui.box(@src(), .{}, .{
                .expand = .horizontal, .border = dvui.Rect.all(1),
                .margin = .{ .x = 0, .y = 6, .w = 0, .h = 2 }, .padding = .all(6),
                .color_border = warn_color,
            });
            defer warn_box.deinit();
            dvui.label(@src(), "Warning: One or more files have an unrecognized architecture. Results may be suboptimal.", .{}, .{
                .color_text = warn_color, .margin = .{ .x = 0, .y = 0, .w = 0, .h = 4 },
            });
            _ = dvui.checkbox(@src(), &state.allow_unknown_arch, "Convert anyway", .{});
        }

        // Predicted output preview — recompute only when a size/name-affecting option changes.
        {
            const sig = state.predictionSignature();
            if (state.prev_pred_signature == null or state.prev_pred_signature.? != sig) {
                state.prev_pred_signature = sig;
                fileHandling.rebuildPairPreviews(gpa, &state);
            }

            if (state.pair_previews.items.len > 0) {
                var total: u64 = 0;
                var total_known = true;
                for (state.pair_previews.items) |p| {
                    if (p.predicted_size) |s| total += s else total_known = false;
                }

                if (total_known) {
                    var out_buf: [16]u8 = undefined;
                    const out_str = formatBytes(total, &out_buf);
                    var out_commas: [32]u8 = undefined;
                    const out_bytes_str = formatWithCommas(total, &out_commas);
                    dvui.label(@src(), "Total estimated output: {s} ({s} bytes) across {d} file(s)", .{ out_str, out_bytes_str, state.pair_previews.items.len }, .{
                        .margin = .{ .x = 0, .y = 8, .w = 0, .h = 2 }, .font = .theme(.body),
                    });
                } else {
                    dvui.label(@src(), "Total estimated output: unavailable for one or more pairs", .{}, .{
                        .margin = .{ .x = 0, .y = 8, .w = 0, .h = 2 }, .color_text = dim_color,
                    });
                }

                var pv_tree = dvui.TreeWidget.tree(@src(), .{ .enable_reordering = false }, .{ .expand = .horizontal });
                defer pv_tree.deinit();
                const pv_branch = pv_tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal });
                defer pv_branch.deinit();
                const pv_caret: []const u8 = if (pv_branch.expanded) "v " else "> ";
                const pv_header = std.fmt.allocPrint(fa, "{s}Output files ({d})", .{ pv_caret, state.pair_previews.items.len }) catch "Output files";
                dvui.labelNoFmt(@src(), pv_header, .{}, .{ .expand = .horizontal });
                if (pv_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
                    for (state.pair_previews.items, 0..) |p, i| {
                        var sbuf: [16]u8 = undefined;
                        const size_str = if (p.predicted_size) |s| formatBytes(s, &sbuf) else "?";
                        const line = std.fmt.allocPrint(fa, "{s} [{s}]  ->  {s}  ({s})", .{
                            std.fs.path.basename(p.input_path), p.format_label, std.fs.path.basename(p.output_path), size_str,
                        }) catch p.output_path;
                        dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .id_extra = i });
                    }
                }
            }
        }

        // Action buttons
        if (state.same_file_error) {
            dvui.label(@src(), "One of the generated output paths matches an input file: {s}", .{state.sameFileErrorMessage()}, .{
                .color_text = err_color, .margin = .{ .x = 0, .y = 4, .w = 0, .h = 2 },
            });
        }
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 10, .w = 0, .h = 4 } });
            defer row.deinit();

            var any_selected = false;
            for (state.target_formats) |sel| { if (sel) { any_selected = true; break; } }

            const convert_blocked = state.arch_mismatch or (any_unknown and !state.allow_unknown_arch) or !any_selected or state.input_files.items.len == 0;
            if (convert_blocked) {
                _ = dvui.button(@src(), "Convert", .{}, .{ .gravity_y = 0.5, .color_text = dim_color });
            } else if (dvui.button(@src(), "Convert", .{}, .{ .gravity_y = 0.5 })) {
                fileHandling.prepareBatchLaunch(gpa, &state);
            }

            // Gap between Convert and the export/generate buttons
            {
                var gap = dvui.box(@src(), .{}, .{ .min_size_content = .{ .w = 16 } });
                defer gap.deinit();
            }

            if (dvui.button(@src(), "Export Template", .{}, .{ .gravity_y = 0.5 })) {
                if (!state.export_template_dialog_open and state.input_files.items.len > 0) {
                    state.export_template_dialog_open = true;
                    state.tool_status_len = 0;
                    SDLBackend.c.SDL_ShowSaveFileDialog(
                        fileHandling.exportTemplateCallback,
                        &state,
                        g_backend.?.window,
                        &json_filters,
                        json_filters.len,
                        null,
                    );
                }
            }

            if (dvui.button(@src(), "Generate Sensitivities", .{}, .{ .gravity_y = 0.5 })) {
                if (!state.gen_sensitivities_dialog_open and state.input_files.items.len > 0) {
                    state.gen_sensitivities_dialog_open = true;
                    state.tool_status_len = 0;
                    SDLBackend.c.SDL_ShowSaveFileDialog(
                        fileHandling.genSensitivitiesCallback,
                        &state,
                        g_backend.?.window,
                        &json_filters,
                        json_filters.len,
                        null,
                    );
                }
            }

            // Spacer pushes Unload to the right
            {
                var spacer = dvui.box(@src(), .{}, .{ .expand = .horizontal });
                defer spacer.deinit();
            }

            if (dvui.button(@src(), "Unload All", .{}, .{ .gravity_y = 0.5 })) {
                unloadAll();
            }
        }

        // Status message from tool operations (template export, sensitivities gen)
        {
            const status = state.toolStatus();
            if (status.len > 0) {
                const color: dvui.Color = if (state.tool_status_is_error) err_color else .{ .r = 80, .g = 180, .b = 80, .a = 255 };
                dvui.labelNoFmt(@src(), status, .{}, .{ .color_text = color, .margin = .{ .x = 0, .y = 2, .w = 0, .h = 2 } });
            }
        }
    }

    // Divider + model internals preview (first file in the batch only)
    if (state.input_files.items.len > 0) {
        {
            var divider = dvui.box(@src(), .{}, .{
                .expand = .horizontal, .min_size_content = .{ .h = 1 }, .background = true,
                .color_fill = dim_color, .margin = .{ .x = 4, .y = 8, .w = 4, .h = 8 },
            });
            defer divider.deinit();
        }
        if (state.input_files.items.len > 1) {
            dvui.label(@src(), "Showing details for the first selected file ({d} more selected).", .{state.input_files.items.len - 1}, .{ .color_text = dim_color });
        }
        showModelInternals();
    }
}

const format_grid_columns = 4;

/// Render one contiguous group of OutputFormats.all_formats (a SafeTensors or
/// GGUF block) as a checkbox grid with `format_grid_columns` columns. Only
/// entries that will actually be rendered (hidden ones excluded) are counted
/// when dealing out columns, so the columns stay evenly sized regardless of
/// how many entries in the group are hidden.
///
/// `bound` is indexed by absolute catalog index and holds the checkbox state.
/// `hidden`, if given, suppresses rendering entirely (not just visually) for
/// entries where hidden[i] is true.
fn showFormatGrid(group_start: usize, group: []const OutputFormats.OutputFormat, bound: *[OutputFormats.all_formats.len]bool, hidden: ?*const [OutputFormats.all_formats.len]bool) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .id_extra = group_start });
    defer row.deinit();

    var visible_local: [OutputFormats.all_formats.len]usize = undefined;
    var visible_count: usize = 0;
    for (0..group.len) |local| {
        const i = group_start + local;
        if (hidden) |h| {
            if (h[i]) continue;
        }
        visible_local[visible_count] = local;
        visible_count += 1;
    }

    const base = visible_count / format_grid_columns;
    const extra = visible_count % format_grid_columns; // first `extra` columns get one more entry

    var col: usize = 0;
    var next: usize = 0;
    while (col < format_grid_columns) : (col += 1) {
        var colbox = dvui.box(@src(), .{}, .{ .expand = .horizontal, .id_extra = group_start + col });
        defer colbox.deinit();

        const count = base + (if (col < extra) @as(usize, 1) else 0);
        var n: usize = 0;
        while (n < count) : (n += 1) {
            const local = visible_local[next];
            next += 1;
            const i = group_start + local;
            _ = dvui.checkbox(@src(), &bound[i], group[local].label, .{ .margin = .{ .x = 4, .y = 1, .w = 4, .h = 1 }, .id_extra = i });
        }
    }
}

fn removeInputFile(idx: usize) void {
    var f = state.input_files.orderedRemove(idx);
    if (f.file) |*lf| lf.deinit();
    gpa.free(f.path);

    var mismatch = false;
    var first_name: ?[]const u8 = null;
    for (state.input_files.items) |inf| {
        if (inf.arch_name) |n| {
            if (first_name) |fname| {
                if (!std.mem.eql(u8, fname, n)) { mismatch = true; break; }
            } else first_name = n;
        }
    }
    state.arch_mismatch = mismatch;
    state.prev_pred_signature = null; // force preview recompute

    if (state.input_files.items.len == 0) {
        state.load_state.store(.idle, .release);
    }
}

fn unloadAll() void {
    state.clearInputFiles(gpa);
    _ = arena.reset(.free_all);
    state.clearBatchResults(gpa);
    state.clearPairPreviews(gpa);
    state.clearOverwritePending(gpa);
    state.convert_options_initialized = false;
    state.convert_state.store(.idle, .release);
    state.convert_progress.store(0, .release);
    state.batch_fatal_error = null;
    state.same_file_error = false;
    state.same_file_error_len = 0;
    state.tool_status_len = 0;
    state.prev_pred_signature = null;
    state.load_state.store(.idle, .release);
}

// Screen: converting

fn showConverting() void {
    var outer = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5, .expand = .both });
    defer outer.deinit();

    var inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .expand = .horizontal, .gravity_y = 0.5, .min_size_content = .{ .w = 400 } });
    defer inner.deinit();

    // Use .acquire so we read tensor info written before this store.
    const done = state.convert_progress.load(.acquire);
    const total = state.convert_total;
    const pair_idx = state.convert_pair_idx;
    const pair_total = state.convert_pair_total;

    dvui.label(@src(), "Converting...", .{}, .{ .gravity_x = 0.5, .font = .theme(.title), .margin = .{ .x = 0, .y = 0, .w = 0, .h = 4 } });
    if (pair_total > 0) {
        dvui.label(@src(), "File/format {d} of {d}", .{ pair_idx + 1, pair_total }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 0, .w = 0, .h = 8 } });
    }

    // Progress bar
    const frac: f32 = if (total > 0)
        @as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total))
    else
        0.0;
    dvui.progress(@src(), .{ .percent = frac }, .{ .expand = .horizontal, .margin = .{ .x = 50, .y = 4, .w = 50, .h = 4 } });

    // Tensor count
    dvui.label(@src(), "{d} / {d} tensors", .{ done, total }, .{ .gravity_x = 0.5 });

    // Current tensor info
    if (done > 0) {
        const tensor_name = state.currentTensorName();
        const src_type = state.currentTensorSrcType();
        const dst_type = state.currentTensorDstType();
        const n_elem = state.convert_tensor_elements;

        if (tensor_name.len > 0) {
            var elem_buf: [32]u8 = undefined;
            const elem_str = formatWithCommas(n_elem, &elem_buf);

            dvui.labelNoFmt(@src(), tensor_name, .{}, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 6, .w = 0, .h = 2 } });
            dvui.label(@src(), "{s} -> {s}  |  {s} elements", .{ src_type, dst_type, elem_str }, .{ .gravity_x = 0.5 });
        }
    }

    // Cancel button
    if (dvui.button(@src(), "Cancel", .{}, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 12, .w = 0, .h = 0 } })) {
        state.cancel_requested.store(true, .release);
    }
}

// Screen: batch conversion summary

fn showBatchSummary() void {
    const fa = frameArena();
    var outer = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer outer.deinit();

    var n_ok: usize = 0;
    var n_err: usize = 0;
    for (state.batch_results.items) |r| {
        if (r.err == null) n_ok += 1 else n_err += 1;
    }

    dvui.label(@src(), "Conversion complete", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });
    dvui.label(@src(), "{d} succeeded, {d} failed", .{ n_ok, n_err }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 4, .w = 0, .h = 4 } });

    {
        const ns = state.convert_elapsed_ns;
        const secs = @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
        if (secs >= 60.0) {
            const mins: u64 = ns / std.time.ns_per_min;
            const rem_s = (ns % std.time.ns_per_min) / std.time.ns_per_s;
            dvui.label(@src(), "Completed in {d}m {d}s", .{ mins, rem_s }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 0, .w = 0, .h = 12 } });
        } else {
            dvui.label(@src(), "Completed in {d:.2}s", .{secs}, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 0, .w = 0, .h = 12 } });
        }
    }

    {
        var tree = dvui.TreeWidget.tree(@src(), .{ .enable_reordering = false }, .{ .expand = .horizontal, .min_size_content = .{ .w = 500 } });
        defer tree.deinit();
        const branch = tree.branch(@src(), .{ .expanded = n_err > 0 }, .{ .expand = .horizontal });
        defer branch.deinit();
        const caret: []const u8 = if (branch.expanded) "v " else "> ";
        const header = std.fmt.allocPrint(fa, "{s}Results ({d})", .{ caret, state.batch_results.items.len }) catch "Results";
        dvui.labelNoFmt(@src(), header, .{}, .{ .expand = .horizontal });
        if (branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
            for (state.batch_results.items, 0..) |r, i| {
                const line = if (r.err) |e|
                    std.fmt.allocPrint(fa, "FAILED  {s} [{s}]: {s}", .{ std.fs.path.basename(r.input_path), r.format_label, @errorName(e) }) catch r.input_path
                else
                    std.fmt.allocPrint(fa, "OK  {s} [{s}] -> {s}", .{ std.fs.path.basename(r.input_path), r.format_label, std.fs.path.basename(r.output_path) }) catch r.input_path;
                const color: ?dvui.Color = if (r.err != null) dvui.Color{ .r = 220, .g = 60, .b = 60, .a = 255 } else null;
                dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .id_extra = i, .color_text = color });
            }
        }
    }

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 0 } });
        defer row.deinit();

        if (dvui.button(@src(), "Convert Again", .{}, .{ .margin = .all(4) })) {
            state.convert_state.store(.idle, .release);
            state.convert_progress.store(0, .release);
        }

        if (dvui.button(@src(), "Open New Files", .{}, .{ .margin = .all(4) })) {
            unloadAll();
        }
    }
}

// Screen: batch could not start at all (rare — e.g. thread spawn failure)

fn showBatchFatalError() void {
    var outer = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer outer.deinit();

    dvui.label(@src(), "Conversion failed to start: {}", .{state.batch_fatal_error.?}, .{ .gravity_x = 0.5, .font = .theme(.title) });

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 0 } });
        defer row.deinit();

        if (dvui.button(@src(), "Try Again", .{}, .{ .margin = .all(4) })) {
            state.convert_state.store(.idle, .release);
            state.batch_fatal_error = null;
        }

        if (dvui.button(@src(), "Open New Files", .{}, .{ .margin = .all(4) })) {
            unloadAll();
        }
    }
}

// Overwrite confirmation dialog (aggregate — lists every conflicting output path)

fn showOverwriteDialog() void {
    const fa = frameArena();
    var float = dvui.floatingWindow(@src(), .{ .modal = true }, .{ .min_size_content = .{ .w = 420, .h = 220 } });
    defer float.deinit();

    var content = dvui.box(@src(), .{}, .{ .expand = .both, .padding = .all(16) });
    defer content.deinit();

    dvui.label(@src(), "{d} file(s) already exist and will be overwritten:", .{state.overwrite_pending_paths.items.len}, .{ .font = .theme(.title) });

    {
        var s = dvui.scrollArea(@src(), .{}, .{ .expand = .horizontal, .min_size_content = .{ .h = 100 }, .max_size_content = .height(100) });
        defer s.deinit();
        for (state.overwrite_pending_paths.items, 0..) |p, i| {
            const line = std.fmt.allocPrint(fa, "{s}", .{std.fs.path.basename(p)}) catch p;
            dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .id_extra = i });
        }
    }

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 1.0, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 0 } });
        defer row.deinit();

        if (dvui.button(@src(), "Cancel", .{}, .{ .margin = .all(4) })) {
            state.clearOverwritePending(gpa);
        }

        if (dvui.button(@src(), "Overwrite", .{}, .{ .margin = .all(4) })) {
            state.clearOverwritePending(gpa);
            state.convert_requested = true;
        }
    }
}

// Upscale warning dialog

fn showUpscaleDialog() void {
    var float = dvui.floatingWindow(@src(), .{ .modal = true }, .{ .min_size_content = .{ .w = 420, .h = 160 } });
    defer float.deinit();

    var content = dvui.box(@src(), .{}, .{ .expand = .both, .padding = .all(16) });
    defer content.deinit();

    dvui.label(@src(), "Precision warning", .{}, .{ .font = .theme(.title) });
    dvui.label(
        @src(),
        "One or more source files contain lossy-quantized tensors. Converting to a higher-precision\n" ++
        "format will NOT recover lost information — the extra bits are fill-in only.",
        .{},
        .{ .margin = .{ .x = 0, .y = 8, .w = 0, .h = 12 } },
    );

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 1.0 });
        defer row.deinit();

        if (dvui.button(@src(), "Cancel", .{}, .{ .margin = .all(4) })) {
            state.upscale_pending = false;
        }

        if (dvui.button(@src(), "Convert anyway", .{}, .{ .margin = .all(4) })) {
            state.upscale_pending = false;
            state.allow_upscale = true;
            fileHandling.prepareBatchLaunch(gpa, &state);
        }
    }
}

// About modal

fn showAboutModal() void {
    var float = dvui.floatingWindow(@src(), .{ .modal = true, .resize = .none }, .{ .min_size_content = .{ .w = 360, .h = 160 } });
    defer float.deinit();
    float.dragAreaSet(.{});

    var content = dvui.box(@src(), .{}, .{ .expand = .both, .padding = .all(20) });
    defer content.deinit();

    dvui.label(@src(), "ggufy", .{}, .{ .font = .theme(.title), .margin = .{ .x = 0, .y = 0, .w = 0, .h = 6 } });
    dvui.label(@src(), "Version: {s}", .{build_options.version}, .{ .margin = .{ .x = 0, .y = 0, .w = 0, .h = 4 } });
    dvui.label(@src(), "Convert ML model files between safetensors and GGUF formats.", .{}, .{ .margin = .{ .x = 0, .y = 0, .w = 0, .h = 4 } });
    dvui.labelNoFmt(@src(), "https://github.com/qskousen/ggufy", .{}, .{ .margin = .{ .x = 0, .y = 0, .w = 0, .h = 16 } });

    if (dvui.button(@src(), "Close", .{}, .{ .gravity_x = 0.5 })) {
        state.show_about = false;
    }
}

// Format-visibility settings modal

fn saveFormatSettings() void {
    @memcpy(&state.hidden_formats, &state.settings_temp_hidden);
    state.settings_dialog_open = false;
    state.prev_pred_signature = null; // force preview recompute against new visibility

    const fa = frameArena();
    var labels: std.ArrayList([]const u8) = .empty;
    for (OutputFormats.all_formats, 0..) |fmt, i| {
        if (state.hidden_formats[i]) labels.append(fa, fmt.label) catch {};
    }
    SettingsMod.save(state.io, gpa, state.settingsPath(), labels.items) catch |err| {
        std.log.err("Failed to save settings: {}", .{err});
    };
}

fn showFormatSettingsModal() void {
    var float = dvui.floatingWindow(@src(), .{ .modal = true }, .{ .min_size_content = .{ .w = 480, .h = 460 } });
    defer float.deinit();

    // ESC or a click outside the dialog closes it (discarding changes, same as Cancel).
    const win_rect = float.data().rectScale().r;
    for (dvui.events()) |*e| {
        switch (e.evt) {
            .key => |ke| {
                if (ke.action == .down and ke.code == .escape) {
                    e.handle(@src(), float.data());
                    state.settings_dialog_open = false;
                }
            },
            .mouse => |me| {
                if (me.action == .press and !win_rect.contains(me.p)) {
                    state.settings_dialog_open = false;
                }
            },
            else => {},
        }
    }
    if (!state.settings_dialog_open) return;

    var content = dvui.box(@src(), .{}, .{ .expand = .both, .padding = .all(16) });
    defer content.deinit();

    const hint_color = dvui.themeGet().color(.control, .text).opacity(0.6);

    dvui.label(@src(), "Visible formats", .{}, .{ .font = .theme(.title), .margin = .{ .x = 0, .y = 0, .w = 0, .h = 4 } });
    dvui.label(@src(), "Uncheck a format to hide it from the output list.", .{}, .{ .color_text = hint_color, .margin = .{ .x = 0, .y = 0, .w = 0, .h = 8 } });

    var visible: [OutputFormats.all_formats.len]bool = undefined;
    for (0..visible.len) |i| visible[i] = !state.settings_temp_hidden[i];

    {
        var s = dvui.scrollArea(@src(), .{}, .{ .expand = .horizontal, .min_size_content = .{ .h = 300 }, .max_size_content = .height(300) });
        defer s.deinit();

        dvui.label(@src(), "SafeTensors formats", .{}, .{ .color_text = hint_color, .margin = .{ .x = 0, .y = 4, .w = 0, .h = 2 } });
        showFormatGrid(0, OutputFormats.all_formats[0..OutputFormats.gguf_start_index], &visible, null);

        dvui.label(@src(), "GGUF formats", .{}, .{ .color_text = hint_color, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 2 } });
        showFormatGrid(OutputFormats.gguf_start_index, OutputFormats.all_formats[OutputFormats.gguf_start_index..], &visible, null);
    }

    for (0..visible.len) |i| state.settings_temp_hidden[i] = !visible[i];

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .gravity_x = 1.0, .margin = .{ .x = 0, .y = 12, .w = 0, .h = 0 } });
        defer row.deinit();

        if (dvui.button(@src(), "Close", .{}, .{ .margin = .all(4) })) {
            state.settings_dialog_open = false;
        }

        if (dvui.button(@src(), "Save", .{}, .{ .margin = .all(4) })) {
            saveFormatSettings();
        }
    }
}

// Model internals tree (first file in the batch)

fn showModelInternals() void {
    const file = &state.input_files.items[0].file.?;
    const fa = frameArena();

    var tree = dvui.TreeWidget.tree(@src(), .{ .enable_reordering = false }, .{ .expand = .horizontal });
    defer tree.deinit();

    // Metadata branch
    if (file.metadata) |meta| {
        const meta_branch = tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal });
        defer meta_branch.deinit();
        const meta_caret: []const u8 = if (meta_branch.expanded) "v " else "> ";
        const header = std.fmt.allocPrint(fa, "{s}Metadata ({d})", .{ meta_caret, meta.count() }) catch "Metadata";
        dvui.labelNoFmt(@src(), header, .{}, .{ .expand = .horizontal });
        if (meta_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
            var it = meta.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) {
                const type_name = @tagName(entry.value_ptr.*);
                const entry_branch = tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal, .id_extra = i });
                defer entry_branch.deinit();
                const entry_caret: []const u8 = if (entry_branch.expanded) "v " else "> ";
                const entry_label = std.fmt.allocPrint(fa, "{s}{s}  [{s}]", .{ entry_caret, entry.key_ptr.*, type_name }) catch entry.key_ptr.*;
                dvui.labelNoFmt(@src(), entry_label, .{}, .{ .expand = .horizontal });
                if (entry_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal, .id_extra = i })) {
                    showJsonValue(entry.value_ptr.*);
                }
            }
        }
    }

    // Tensors branch
    {
        const n = file.tensors.items.len;
        const tensors_branch = tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal });
        defer tensors_branch.deinit();
        const tensors_caret: []const u8 = if (tensors_branch.expanded) "v " else "> ";
        const header = std.fmt.allocPrint(fa, "{s}Tensors ({d})", .{ tensors_caret, n }) catch "Tensors";
        dvui.labelNoFmt(@src(), header, .{}, .{ .expand = .horizontal });
        if (tensors_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
            showTensorsVirtual(file, tensors_branch.data().id, n);
        }
    }
}

fn showTensorsVirtual(file: *const ggufy.fileLoader.TensorFile, parent_id: dvui.Id, n: usize) void {
    if (n == 0) return;

    // Row height in logical content pixels — derived from the current body font,
    // so it adapts to theme changes without needing a measurement frame.
    const row_h: f32 = dvui.themeGet().font_body.lineHeight() + 2.0;

    // Place a zero-height marker to record the physical Y where the list starts.
    var marker_wd: dvui.WidgetData = undefined;
    {
        var marker = dvui.box(@src(), .{}, .{ .min_size_content = .{ .h = 0 }, .expand = .horizontal, .data_out = &marker_wd });
        marker.deinit();
    }

    const scale = dvui.windowNaturalScale();
    const row_h_phys = row_h * scale;
    const list_y = marker_wd.borderRectScale().r.y; // physical Y of list top
    const clip = dvui.clipGet(); // physical visible rect

    // How far (in physical px) the viewport top is below the list top.
    // Negative means the viewport starts above the list (list not yet scrolled to).
    const rel_top = clip.y - list_y;
    const rel_bot = (clip.y + clip.h) - list_y;

    // Visible row range with one row of overdraw on each side.
    const first: usize = if (rel_top > row_h_phys)
        @intFromFloat(@floor((rel_top - row_h_phys) / row_h_phys))
    else
        0;
    const last_raw: usize = if (rel_bot > 0)
        @intFromFloat(@ceil((rel_bot + row_h_phys) / row_h_phys))
    else
        0;
    const last: usize = @min(n, last_raw);

    // Top spacer — represents the rows above the viewport.
    if (first > 0) {
        var spacer = dvui.box(@src(), .{}, .{
            .min_size_content = .{ .h = @as(f32, @floatFromInt(first)) * row_h },
            .expand = .horizontal,
            // stable id so dvui doesn't confuse this with real tensor rows
            .id_extra = std.math.maxInt(usize),
        });
        spacer.deinit();
    }

    // Visible tensors.
    for (file.tensors.items[first..last], first..) |tensor, i| {
        showTensorBranch(tensor, i);
    }

    // Bottom spacer — represents the rows below the viewport.
    const tail = n - last;
    if (tail > 0) {
        var spacer = dvui.box(@src(), .{}, .{
            .min_size_content = .{ .h = @as(f32, @floatFromInt(tail)) * row_h },
            .expand = .horizontal,
            .id_extra = std.math.maxInt(usize) - 1,
        });
        spacer.deinit();
    }

    // Keep the stored parent id alive so the data key is stable.
    _ = parent_id;
}

fn showJsonValue(value: std.json.Value) void {
    const json_str = std.json.Stringify.valueAlloc(frameArena(), value, .{ .whitespace = .indent_2 }) catch {
        dvui.labelNoFmt(@src(), "(error serializing value)", .{}, .{});
        return;
    };
    var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
    tl.addText(json_str, .{});
    tl.deinit();
}

fn showTensorBranch(tensor: ggufy.types.Tensor, idx: usize) void {
    const fa = frameArena();

    var dims_buf: [256]u8 = undefined;
    var dims_writer = std.Io.Writer.fixed(&dims_buf);
    const w = &dims_writer;
    w.writeByte('[') catch {};
    for (tensor.dims, 0..) |d, j| {
        if (j > 0) w.writeAll(" x ") catch {};
        w.print("{d}", .{d}) catch {};
    }
    w.writeByte(']') catch {};

    var n: u64 = 1;
    for (tensor.dims) |d| n *= d;

    var size_buf: [32]u8 = undefined;
    const size_str = formatBytes(tensor.size, &size_buf);

    const line = std.fmt.allocPrint(fa, "{s}  [{s}]  Dimensions: {s}  ({d} total), Size: {s}  Offset: {d}", .{
        tensor.name, tensor.type, dims_writer.buffered(), n, size_str, tensor.offset,
    }) catch tensor.name;

    dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .id_extra = idx });
}

// Drop events
// SDL_AddEventWatch fires synchronously when SDL pumps an OS event - before
// any SDL_PollEvent caller (dvui's addAllEvents included) can consume it.
// This guarantees we never lose drop events to timing races. Multiple
// DROP_FILE events between DROP_BEGIN and DROP_COMPLETE (one multi-file drop)
// all accumulate into pending_paths; the batch is staged for load only once,
// at DROP_COMPLETE.

fn dropEventWatch(userdata: ?*anyopaque, event: [*c]SDLBackend.c.SDL_Event) callconv(.c) bool {
    const s: *guiState.State = @ptrCast(@alignCast(userdata));
    const ev = event.*;
    switch (ev.type) {
        SDLBackend.c.SDL_EVENT_DROP_BEGIN,
        SDLBackend.c.SDL_EVENT_DROP_POSITION => {
            std.log.debug("Drop begin/position", .{});
            s.dropping = true;
        },
        SDLBackend.c.SDL_EVENT_DROP_FILE => {
            const load_busy = s.load_state.load(.acquire) == .loading;
            const conv_busy = s.convert_state.load(.acquire) == .converting;
            if (load_busy or conv_busy) {
                std.log.info("Drop ignored: load/conversion in progress", .{});
            } else {
                const path = std.mem.span(ev.drop.data);
                std.log.info("Dropped: {s}", .{path});
                if (s.gpa.dupe(u8, path) catch null) |dup| {
                    s.pending_paths.append(s.gpa, dup) catch s.gpa.free(dup);
                }
            }
        },
        SDLBackend.c.SDL_EVENT_DROP_COMPLETE => {
            std.log.debug("Drop finished", .{});
            s.dropping = false;
            if (s.pending_paths.items.len > 0) {
                s.pending_ready.store(true, .release);
            }
            dvui.refresh(g_win, @src(), null);
        },
        else => {},
    }
    return true; // never filter events out
}

// File dialog filters

const file_filters = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "Compatible files", .pattern = "gguf;safetensors" },
    .{ .name = "GGUF files",       .pattern = "gguf" },
    .{ .name = "Safetensors files",.pattern = "safetensors" },
    .{ .name = "All files",        .pattern = "*" },
};

const json_filters = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "JSON files", .pattern = "json" },
    .{ .name = "All files",  .pattern = "*" },
};
