const std = @import("std");

const max_concurrent = 512;

pub const WaitGroup = struct {
    buf: [max_concurrent]std.Thread = undefined,
    count: usize = 0,

    pub fn wait(wg: *WaitGroup) void {
        for (wg.buf[0..wg.count]) |t| t.join();
        wg.count = 0;
    }
};

const Threads = struct { len: usize };

pub const ThreadPool = struct {
    threads: Threads = .{ .len = 1 },

    pub fn init(pool: *ThreadPool, opts: struct {
        allocator: std.mem.Allocator,
        n_jobs: usize,
    }) !void {
        pool.threads = .{ .len = @min(opts.n_jobs, max_concurrent) };
    }

    pub fn deinit(pool: *ThreadPool) void {
        _ = pool;
    }

    pub fn spawnWg(
        pool: *ThreadPool,
        wg: *WaitGroup,
        comptime func: anytype,
        args: anytype,
    ) void {
        _ = pool;
        std.debug.assert(wg.count < max_concurrent);
        const t = std.Thread.spawn(.{}, func, args) catch {
            @call(.auto, func, args);
            return;
        };
        wg.buf[wg.count] = t;
        wg.count += 1;
    }
};
