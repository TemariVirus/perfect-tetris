const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dir = Io.Dir;
const assert = std.debug.assert;
const os = std.os;
const SIG = os.linux.SIG;

const engine = @import("engine");
const GameState = engine.GameState(SevenBag);
const PieceKind = engine.pieces.PieceKind;
const SevenBag = engine.bags.SevenBag;

const root = @import("perfect-tetris");
const SequenceIterator = root.next.SequenceIterator;
const pc = root.pc;
const PCSolution = root.PCSolution;
const Placement = root.Placement;

// Height of perfect clears to find
const HEIGHT = 4;
comptime {
    assert(HEIGHT % 2 == 0);
}
const NEXT_LEN = HEIGHT * 10 / 4;
// Number of threads to use
const THREADS = 6;

const SAVE_PATH = std.fmt.comptimePrint("pc-data/{}", .{HEIGHT});

// Count of threads saving to disk to make sure all threads finish saving
// before exiting.
var saving_threads: std.atomic.Value(i32) = .init(0);

const Timer = struct {
    last: Io.Timestamp,

    pub fn start(io: Io) Timer {
        return .{ .last = Io.Timestamp.now(io, .real) };
    }

    pub fn lap(self: *Timer, io: Io) Io.Duration {
        const now = Io.Timestamp.now(io, .real);
        const dur = self.last.durationTo(now);
        self.last = now;
        return dur;
    }
};

/// Thread-safe ring buffer for distributing work, and storing and writing
/// solutions to disk.
const SolutionBuffer = struct {
    const CHUNK_SIZE = 64;
    const CHUNKS = THREADS * 8;
    pub const Iterator = SequenceIterator(NEXT_LEN + 1, @min(6, NEXT_LEN));
    pub const AtomicLength = std.atomic.Value(isize);

    io: Io,
    write_lock: Io.Mutex = .init,
    read_lock: Io.Mutex = .init,
    solved: u64 = 0,
    count: u64 = 0,
    last_count: u64 = 0,
    timer: Timer,
    iter: Iterator,

    write_idx: u32 = 0,
    read_idx: std.atomic.Value(u32) = .init(0),
    lengths: [CHUNKS]AtomicLength = @splat(.init(-1)),
    sequences: [CHUNKS][CHUNK_SIZE]u48,
    solutions: [CHUNKS][CHUNK_SIZE][NEXT_LEN]Placement,

    /// Initializes a new SolutionBuffer with the given allocator.
    pub fn init(io: Io, allocator: Allocator) SolutionBuffer {
        return .{
            .io = io,
            .timer = .start(io),
            .iter = .init(allocator),
            .sequences = undefined,
            .solutions = undefined,
        };
    }

    /// Loads a SolutionBuffer with data from disk or initializes a new one if
    /// the files don't exist.
    pub fn loadOrInit(io: Io, allocator: Allocator, path: []const u8) !SolutionBuffer {
        const pc_path = try std.fmt.allocPrint(allocator, "{s}.pc", .{path});
        defer allocator.free(pc_path);
        const count_path = try std.fmt.allocPrint(
            allocator,
            "{s}.count",
            .{path},
        );
        defer allocator.free(count_path);

        var self = init(io, allocator);

        // Get number of solves
        blk: {
            const file = Dir.cwd().openFile(io, pc_path, .{}) catch |e| {
                // Leave self.solved as 0 if the file doesn't exist
                if (e != Io.File.OpenError.FileNotFound) {
                    return e;
                }
                break :blk;
            };
            defer file.close(io);

            const stat = try file.stat(io);
            const SOLUTION_SIZE = 8 + NEXT_LEN;
            self.solved = @divExact(stat.size, SOLUTION_SIZE);
        }

        // Get count
        const file = Dir.cwd().openFile(io, count_path, .{}) catch |e| {
            // Leave self.conut as 0 if the file doesn't exist
            if (e != Io.File.OpenError.FileNotFound) {
                return e;
            }
            return self;
        };
        defer file.close(io);

        const max_len = comptime std.math.log10_int(@as(u64, std.math.maxInt(u64))) + 1;
        var buf: [max_len]u8 = undefined;
        const buf_len = try file.readPositionalAll(io, &buf, 0);

        self.count = try std.fmt.parseInt(u64, buf[0..buf_len], 10);
        self.last_count = self.count;
        // Sync iterator state with count
        for (0..self.count) |_| {
            _ = try self.iter.next();
        }

        return self;
    }

    pub fn deinit(self: *SolutionBuffer) void {
        const old_cancel_protect = self.io.swapCancelProtection(.blocked);
        defer _ = self.io.swapCancelProtection(old_cancel_protect);

        self.write_lock.lock(self.io) catch unreachable;
        defer self.write_lock.unlock(self.io);
        self.iter.deinit();
    }

    fn mask(index: u32) u32 {
        return index % CHUNKS;
    }

    fn mask2(index: u32) u32 {
        return index % (2 * CHUNKS);
    }

    /// Gets the next available chunk of sequences to solve.
    pub fn nextChunk(
        self: *SolutionBuffer,
    ) !?struct { *AtomicLength, []u48, [][NEXT_LEN]Placement } {
        try self.write_lock.lock(self.io);
        defer self.write_lock.unlock(self.io);

        if (self.isFull()) {
            std.debug.print(
                "INFO: solution buffer is full. Consider increasing the number of chunks or the chunk size\n",
                .{},
            );
            // Wait for space to become available
            try self.io.futexWait(u32, &self.read_idx.raw, mask2(self.write_idx + CHUNKS));
        }

        if (self.iter.done()) {
            return null;
        }

        const index = mask(self.write_idx);
        self.lengths[index].store(-1, .monotonic);

        var len: usize = 0;
        while (try self.iter.next()) |pieces| {
            var next: PCSolution.NextArray = .{ .len = pieces.len };
            @memcpy(next.slice(), &pieces);
            self.sequences[index][len] = PCSolution.packNext(next);
            len += 1;
            if (len >= CHUNK_SIZE) {
                break;
            }
        }

        self.write_idx = mask2(self.write_idx + 1);
        return .{
            &self.lengths[index],
            self.sequences[index][0..len],
            self.solutions[index][0..len],
        };
    }

    /// Write all non-blocked finished chunks to disk, and free space in the
    /// ring buffer.
    pub fn writeDoneChunks(
        self: *SolutionBuffer,
        path: []const u8,
    ) !void {
        try self.read_lock.lock(self.io);
        defer self.read_lock.unlock(self.io);

        // Check if there's anything to write
        if (self.isEmpty() or
            self.lengths[mask(self.read_idx.raw)].load(.monotonic) < 0)
        {
            return;
        }

        // Multiple threads can save at the same time, but no threads should
        // start saving when we are exiting
        if (saving_threads.load(.monotonic) < 0) {
            return;
        }

        _ = saving_threads.fetchAdd(1, .monotonic);
        defer _ = saving_threads.fetchSub(1, .monotonic);

        const allocator = self.iter.seen.allocator;

        const pc_file = blk: {
            const pc_path = try std.fmt.allocPrint(
                allocator,
                "{s}.pc",
                .{path},
            );
            defer allocator.free(pc_path);

            break :blk Dir.cwd().openFile(
                self.io,
                pc_path,
                .{ .mode = .write_only },
            ) catch |e| {
                // Create file if it doesn't exist
                if (e != Io.File.OpenError.FileNotFound) {
                    return e;
                }
                try Dir.cwd().createDirPath(
                    self.io,
                    std.fs.path.dirname(pc_path) orelse return error.InvalidPath,
                );
                break :blk try Dir.cwd().createFile(self.io, pc_path, .{});
            };
        };
        defer pc_file.close(self.io);
        // Seek to end to append to file
        var buf: [4096]u8 = undefined;
        var pc_writer = pc_file.writer(self.io, &buf);
        try pc_writer.seekToUnbuffered(try pc_file.length(self.io));

        var wrote = false;
        // A negative length indicates that the chunk is not done yet
        while (!self.isEmpty() and
            self.lengths[mask(self.read_idx.raw)].load(.monotonic) >= 0)
        {
            const len: usize = @intCast(self.lengths[mask(self.read_idx.raw)]
                .load(.monotonic));
            self.solved += len;
            // NOTE: for the last chunk, the count value may become larger than
            // it actually is. This isn't an issue as the iterator is already
            // exhausted.
            self.count += CHUNK_SIZE;
            try self.saveAppend(&pc_writer.interface, len);

            wrote = true;
            self.read_idx.store(mask2(self.read_idx.raw + 1), .monotonic);
            self.io.futexWake(u32, &self.read_idx.raw, 1);
        }
        try pc_writer.flush();

        const count_path = try std.fmt.allocPrint(
            allocator,
            "{s}.count",
            .{path},
        );
        defer allocator.free(count_path);
        var count_file = try Dir.cwd().createFileAtomic(
            self.io,
            count_path,
            .{ .make_path = true, .replace = true },
        );
        defer count_file.deinit(self.io);

        var count_writer = count_file.file.writer(self.io, &.{});
        try count_writer.interface.print("{d}", .{self.count});
        try count_file.replace(self.io);

        if (wrote) {
            try self.printStatsAndBackup(path);
        }
    }

    fn saveAppend(
        self: *SolutionBuffer,
        pc_writer: *Io.Writer,
        len: usize,
    ) !void {
        assert(NEXT_LEN <= 16);

        for (
            self.sequences[mask(self.read_idx.raw)][0..len],
            self.solutions[mask(self.read_idx.raw)][0..len],
        ) |seq, sol| {
            var holds: u16 = 0;
            var placements: [NEXT_LEN]u8 = @splat(0);

            const sequence = PCSolution.unpackNext(seq);

            var hold = sequence.buffer[0];
            var current = sequence.buffer[1];
            for (sol, 0..) |placement, i| {
                // Use canonical position so that the position is always in the
                // range [0, 59]
                const canon_pos = placement.piece
                    .canonicalPosition(placement.pos);
                const pos = canon_pos.y * 10 + canon_pos.x;
                assert(pos < 60);
                placements[i] = @intFromEnum(placement.piece.facing) |
                    (@as(u8, pos) << 2);

                if (current != placement.piece.kind) {
                    holds |= @as(u16, 1) << @intCast(i);
                    hold = current;
                }
                // Only update current if it's not the last piece
                if (i < sol.len - 1) {
                    current = sequence.buffer[i + 2];
                }
            }

            try pc_writer.writeInt(u48, seq, .little);
            try pc_writer.writeInt(u16, holds, .little);
            try pc_writer.writeAll(&placements);
        }
    }

    fn printStatsAndBackup(self: *SolutionBuffer, path: []const u8) !void {
        std.debug.print(
            "Solved {} out of {}\n",
            .{ self.solved, self.count },
        );

        const count = self.count - self.last_count;
        if (count >= THREADS * 1024) {
            // Create backup files
            const allocator = self.iter.seen.allocator;
            {
                const pc_path = try std.fmt.allocPrint(
                    allocator,
                    "{s}.pc",
                    .{path},
                );
                defer allocator.free(pc_path);
                const backup_path = try std.fmt.allocPrint(
                    allocator,
                    "{s}-backup.pc",
                    .{path},
                );
                defer allocator.free(backup_path);

                try Dir.cwd().copyFile(pc_path, .cwd(), backup_path, self.io, .{});
            }
            {
                const count_path = try std.fmt.allocPrint(
                    allocator,
                    "{s}.count",
                    .{path},
                );
                defer allocator.free(count_path);
                const backup_path = try std.fmt.allocPrint(
                    allocator,
                    "{s}-backup.count",
                    .{path},
                );
                defer allocator.free(backup_path);

                try Dir.cwd().copyFile(count_path, .cwd(), backup_path, self.io, .{});
            }

            std.debug.print(
                "Time per sequence per thread: {f}\n\n",
                .{Io.Duration.fromNanoseconds(@divTrunc(self.timer.lap(self.io).toNanoseconds() * THREADS, count))},
            );
            self.last_count = self.count;
        }
    }

    fn isEmpty(self: SolutionBuffer) bool {
        return self.write_idx == self.read_idx.load(.monotonic);
    }

    fn isFull(self: SolutionBuffer) bool {
        return mask2(self.write_idx + CHUNKS) == self.read_idx.load(.monotonic);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    setupExitHandler();

    var buf: SolutionBuffer = try .loadOrInit(io, allocator, SAVE_PATH);
    defer buf.deinit();

    var threads: [THREADS]std.Thread = undefined;
    for (0..threads.len) |i| {
        threads[i] = try .spawn(
            .{ .allocator = allocator },
            solveThread,
            .{ allocator, &buf },
        );
    }
    for (threads) |thread| {
        thread.join();
    }

    // Wait for saves to finish
    while (saving_threads.load(.monotonic) > 0) {
        io.sleep(.fromMilliseconds(1), .cpu_process) catch {};
    }
}

const handle_signals = [_]SIG{
    SIG.ABRT,
    SIG.INT,
    SIG.QUIT,
    SIG.TERM,
};
fn setupExitHandler() void {
    if (@import("builtin").os.tag == .windows) {
        const signal = struct {
            extern "c" fn signal(
                sig: c_int,
                func: *const fn (c_int, c_int) callconv(os.windows.WINAPI) void,
            ) callconv(.c) *anyopaque;
        }.signal;
        for (handle_signals) |sig| {
            _ = signal(sig, handleExitWindows);
        }
    } else {
        const action: os.linux.Sigaction = .{
            .handler = .{ .handler = handleExit },
            .mask = os.linux.sigemptyset(),
            .flags = 0,
        };
        for (handle_signals) |sig| {
            _ = os.linux.sigaction(sig, &action, null);
        }
    }
}

fn handleExit(sig: SIG) callconv(.c) void {
    if (std.mem.containsAtLeast(SIG, &handle_signals, 1, &.{sig})) {
        // Set to -1 to signal saves to stop and then wait for saves to finish
        const saving_count = saving_threads.swap(-1, .monotonic);
        while (saving_threads.load(.monotonic) >= -saving_count) {
            std.Thread.yield() catch {};
        }
        std.process.exit(0);
    }
}

fn handleExitWindows(sig: c_int, _: c_int) callconv(.C) void {
    handleExit(sig);
}

fn solveThread(allocator: Allocator, buf: *SolutionBuffer) !void {
    const nn = try root.defaultNN(allocator);
    defer nn.deinit(allocator);

    while (try buf.nextChunk()) |tuple| {
        const sol_count, const sequences, const solutions = tuple;

        var solved: usize = 0;
        for (sequences) |seq| {
            var next_pieces = PCSolution.unpackNext(seq);
            _ = pc.findPc(
                .{
                    .allocator = allocator,
                    .playfield = .{},
                    .pieces = next_pieces.slice(),
                    .kicks = engine.kicks.srs,
                    .min_height = HEIGHT,
                    .nn = nn,
                },
                &solutions[solved],
            ) catch |e| if (e == root.FindPcError.SolutionTooLong) {
                continue;
            } else {
                return e;
            };

            sequences[solved] = seq;
            solved += 1;
        }

        sol_count.store(@intCast(solved), .monotonic);
        try buf.writeDoneChunks(SAVE_PATH);
    }
}
