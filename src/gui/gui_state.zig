const std = @import("std");
const ggufy = @import("ggufy");
const OutputFormats = @import("OutputFormats.zig");
const SettingsMod = @import("Settings.zig");

pub const LoadState = enum(u8) { idle, loading, done, err };
pub const ConvertState = enum(u8) { idle, converting, done, err };

const n_formats = OutputFormats.all_formats.len;

/// One selected/loaded input file in the current batch.
pub const InputFile = struct {
    /// gpa-owned copy of the path.
    path: []u8,
    file: ?ggufy.fileLoader.TensorFile = null,
    arch_name: ?[]const u8 = null,
    /// Highest-count entry in the file's type_counts — the "original precision".
    dominant_type: ?ggufy.types.DataType = null,
};

/// Result of converting one (input file, output format) pair.
pub const PairResult = struct {
    input_path: []const u8, // borrowed from an InputFile still alive in input_files
    format_label: []const u8, // borrowed from OutputFormats.all_formats (static)
    output_path: []u8, // gpa-owned
    err: ?anyerror,
};

/// A previewed (input file, output format) pair shown before conversion:
/// generated output filename + predicted size.
pub const PairPreview = struct {
    input_path: []const u8, // borrowed from input_files
    format_label: []const u8, // borrowed from OutputFormats.all_formats
    output_path: []u8, // gpa-owned
    predicted_size: ?u64,
};

pub const State = struct {
    io: std.Io = undefined,
    /// General-purpose allocator, set once at startup. Needed by SDL callbacks
    /// (file dialog, drag-drop) that only receive `state` as userdata and must
    /// allocate owned copies of dropped/selected paths.
    gpa: std.mem.Allocator = undefined,

    // File load
    load_state: std.atomic.Value(LoadState) = .init(.idle),
    dropping: bool = false,
    file_dialog_open: bool = false,
    load_error: ?anyerror = null,
    wakeup_event_type: u32 = 0,

    /// Paths collected from the file dialog or a drag-drop batch, staged for
    /// one atomic load. Set by fileDialogCallback / dropEventWatch, consumed
    /// by the main loop, which resets the arena exactly once and loads all of
    /// them before clearing this list. gpa-owned.
    pending_paths: std.ArrayList([]u8) = .empty,
    pending_ready: std.atomic.Value(bool) = .init(false),

    /// The currently loaded batch of input files. Selecting/dropping a new
    /// batch always replaces this list (no incremental add in this version).
    input_files: std.ArrayList(InputFile) = .empty,
    /// True when 2+ files in input_files have a detected architecture and
    /// those architectures don't all match. Hard-blocks conversion.
    arch_mismatch: bool = false,

    // Conversion options
    convert_options_initialized: bool = false,
    /// Which catalog entries (by index into OutputFormats.all_formats) are checked.
    target_formats: [n_formats]bool = std.mem.zeroes([n_formats]bool),
    /// Which catalog entries are hidden from the checklist (mirrors Settings; edited
    /// via the format-visibility modal through a temp copy, see settings_temp_hidden).
    hidden_formats: [n_formats]bool = std.mem.zeroes([n_formats]bool),
    settings_dialog_open: bool = false,
    settings_temp_hidden: [n_formats]bool = std.mem.zeroes([n_formats]bool),
    settings_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    settings_path_len: usize = 0,

    /// Output folder — null-terminated; dvui textEntry writes here directly.
    target_folder_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    /// CPU thread count for quantization. Populated at startup with getCpuCount().
    target_threads: usize = 4,
    cpu_count: usize = 4,
    /// 1-100: how aggressively to quantize sensitivity-scaled layers.
    target_aggressiveness: u8 = 50,
    skip_sensitivity: bool = false,
    model_only: bool = false,
    allow_unknown_arch: bool = false,
    /// Free-form architecture name to write as `general.architecture` in GGUF output.
    /// Null-terminated; empty string means "use auto-detected name".
    arch_override_buf: [64]u8 = std.mem.zeroes([64]u8),
    sensitivity_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    sensitivity_path: ?[]u8 = null,
    template_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    template_path: ?[]u8 = null,
    /// Separate open-flag for each dialog so they don't interfere.
    folder_dialog_open: bool = false,
    sensitivity_dialog_open: bool = false,
    template_dialog_open: bool = false,
    export_template_dialog_open: bool = false,
    gen_sensitivities_dialog_open: bool = false,

    // Export template / generate sensitivities
    export_template_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    export_template_path: ?[]u8 = null,
    gen_sensitivities_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    gen_sensitivities_path: ?[]u8 = null,
    export_template_requested: bool = false,
    gen_sensitivities_requested: bool = false,
    // Status message shown after a tool operation (template export / sensitivities gen)
    tool_status_buf: [256]u8 = std.mem.zeroes([256]u8),
    tool_status_len: usize = 0,
    tool_status_is_error: bool = false,

    // Conversion progress (batch)
    convert_state: std.atomic.Value(ConvertState) = .init(.idle),
    /// Index of the (file, format) pair currently in flight / last completed.
    convert_pair_idx: usize = 0,
    convert_pair_total: usize = 0,
    /// Index of the last completed tensor within the *current* pair. Written
    /// with .release so all preceding plain writes (tensor name/type/elements)
    /// are visible to the main thread after it loads this with .acquire.
    convert_progress: std.atomic.Value(u32) = .init(0),
    convert_total: u32 = 0,
    /// Set true in the GUI to request cancellation; cleared by the convert thread.
    cancel_requested: std.atomic.Value(bool) = .init(false),
    /// Set true in the main loop to spawn the convert thread on the next iteration.
    convert_requested: bool = false,
    /// Fatal error that aborted the whole batch (as opposed to a single pair
    /// failing, which is recorded per-pair in batch_results and does not stop
    /// the rest of the batch).
    batch_fatal_error: ?anyerror = null,
    convert_elapsed_ns: u64 = 0,
    /// gpa-owned; freed and rebuilt at the start of each batch run.
    batch_results: std.ArrayList(PairResult) = .empty,

    // Current tensor info — written by the convert thread BEFORE the
    // convert_progress .release store.  The main thread reads these fields
    // after a .acquire load of convert_progress, so no extra sync needed.
    convert_tensor_name_buf: [256]u8 = undefined,
    convert_tensor_name_len: usize = 0,
    convert_tensor_src_type_buf: [32]u8 = undefined,
    convert_tensor_src_type_len: usize = 0,
    convert_tensor_dst_type_buf: [32]u8 = undefined,
    convert_tensor_dst_type_len: usize = 0,
    convert_tensor_elements: u64 = 0,

    // Predicted per-pair output preview, recomputed only when a size/name-affecting
    // option changes, tracked via `prev_pred_signature`. gpa-owned; rebuilt in place.
    pair_previews: std.ArrayList(PairPreview) = .empty,
    prev_pred_signature: ?u64 = null,

    // Aggregate overwrite confirmation — all output paths that already exist,
    // computed once before starting a batch (rather than prompting per pair).
    overwrite_pending_paths: std.ArrayList([]u8) = .empty,

    // Same-file conflict (an output path would equal one of the input paths)
    same_file_error: bool = false,
    same_file_error_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    same_file_error_len: usize = 0,

    // Upscale confirmation
    upscale_pending: bool = false,
    allow_upscale: bool = false,

    // Misc UI state
    show_about: bool = false,

    // Helpers
    pub fn targetFolder(self: *const State) []const u8 {
        return std.mem.sliceTo(&self.target_folder_buf, 0);
    }

    pub fn archOverride(self: *const State) ?[]const u8 {
        const s = std.mem.sliceTo(&self.arch_override_buf, 0);
        return if (s.len > 0) s else null;
    }

    pub fn currentTensorName(self: *const State) []const u8 {
        return self.convert_tensor_name_buf[0..self.convert_tensor_name_len];
    }

    pub fn currentTensorSrcType(self: *const State) []const u8 {
        return self.convert_tensor_src_type_buf[0..self.convert_tensor_src_type_len];
    }

    pub fn currentTensorDstType(self: *const State) []const u8 {
        return self.convert_tensor_dst_type_buf[0..self.convert_tensor_dst_type_len];
    }

    pub fn toolStatus(self: *const State) []const u8 {
        return self.tool_status_buf[0..self.tool_status_len];
    }

    pub fn settingsPath(self: *const State) []const u8 {
        return self.settings_path_buf[0..self.settings_path_len];
    }

    pub fn sameFileErrorMessage(self: *const State) []const u8 {
        return self.same_file_error_buf[0..self.same_file_error_len];
    }

    /// Free everything owned by the current input batch and reset batch-derived
    /// state. Does NOT touch pending_paths/pending_ready.
    pub fn clearInputFiles(self: *State, allocator: std.mem.Allocator) void {
        for (self.input_files.items) |*f| {
            if (f.file) |*lf| lf.deinit();
            allocator.free(f.path);
        }
        self.input_files.clearRetainingCapacity();
        self.arch_mismatch = false;
    }

    pub fn clearBatchResults(self: *State, allocator: std.mem.Allocator) void {
        for (self.batch_results.items) |r| allocator.free(r.output_path);
        self.batch_results.clearRetainingCapacity();
    }

    pub fn clearPairPreviews(self: *State, allocator: std.mem.Allocator) void {
        for (self.pair_previews.items) |p| allocator.free(p.output_path);
        self.pair_previews.clearRetainingCapacity();
    }

    pub fn clearOverwritePending(self: *State, allocator: std.mem.Allocator) void {
        for (self.overwrite_pending_paths.items) |p| allocator.free(p);
        self.overwrite_pending_paths.clearRetainingCapacity();
    }

    /// A hash of every option that affects the predicted per-pair output
    /// (filenames + sizes). When it changes, previews are recomputed; otherwise
    /// the cached list is reused so the (relatively cheap, but non-trivial)
    /// prediction doesn't run every frame.
    pub fn predictionSignature(self: *const State) u64 {
        var h = std.hash.Wyhash.init(0);
        for (self.input_files.items) |f| h.update(f.path);
        for (self.target_formats, 0..) |sel, i| if (sel) std.hash.autoHash(&h, i);
        std.hash.autoHash(&h, self.target_aggressiveness);
        std.hash.autoHash(&h, self.skip_sensitivity);
        std.hash.autoHash(&h, self.model_only);
        std.hash.autoHash(&h, self.allow_unknown_arch);
        std.hash.autoHash(&h, self.allow_upscale);
        h.update(std.mem.sliceTo(&self.arch_override_buf, 0));
        if (self.template_path) |p| h.update(p);
        if (self.sensitivity_path) |p| h.update(p);
        h.update(self.targetFolder());
        return h.final();
    }
};
