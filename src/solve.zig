const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const fs = std.fs;
const Mutex = Thread.Mutex;
const os = std.os;
const SIG = os.linux.SIG;
const Thread = std.Thread;
const Timer = std.time.Timer;

const engine = @import("engine");
const GameState = engine.GameState(SevenBag);
const PieceKind = engine.pieces.PieceKind;
const SevenBag = engine.bags.SevenBag;

const root = @import("perfect-tetris");
const SequenceIterator = root.next.SequenceIterator;
const NN = root.NN;
const pc = root.pc;
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

// Count of threads saving to disk to make sure all threads finish saving before exiting.
var saving_threads = std.atomic.Value(i32).init(0);

/// Thread-safe ring buffer for distributing work, and storing and writing
/// solutions to disk.
const SolutionBuffer = struct {
    const CHUNK_SIZE = 64;
    const CHUNKS = THREADS * 8;
    pub const Iterator = SequenceIterator(NEXT_LEN + 1, @min(6, NEXT_LEN));
    pub const AtomicLength = std.atomic.Value(isize);

    mutex: Mutex = .{},
    solved: u64 = 0,
    count: u64 = 0,
    last_count: u64 = 0,
    timer: Timer,
    iter: Iterator,

    write_idx: usize = 0,
    read_idx: usize = 0,
    lengths: [CHUNKS]AtomicLength = [_]AtomicLength{AtomicLength.init(-1)} ** CHUNKS,
    sequences: [CHUNKS][CHUNK_SIZE]u48,
    solutions: [CHUNKS][CHUNK_SIZE][NEXT_LEN]Placement,

    /// Initializes a new SolutionBuffer with the given allocator.
    pub fn init(allocator: Allocator) !SolutionBuffer {
        return .{
            .timer = try Timer.start(),
            .iter = Iterator.init(allocator),
            .sequences = undefined,
            .solutions = undefined,
        };
    }

    /// Loads a SolutionBuffer with data from disk or initializes a new one if
    /// the files don't exist.
    pub fn loadOrInit(allocator: Allocator, path: []const u8) !SolutionBuffer {
        const pc_path = try std.fmt.allocPrint(allocator, "{s}.pc", .{path});
        defer allocator.free(pc_path);
        const count_path = try std.fmt.allocPrint(allocator, "{s}.count", .{path});
        defer allocator.free(count_path);

        var self = try init(allocator);

        // Get number of solves
        blk: {
            const file = fs.cwd().openFile(pc_path, .{}) catch |e| {
                // Leave self.solved as 0 if the file doesn't exist
                if (e != fs.File.OpenError.FileNotFound) {
                    return e;
                }
                break :blk;
            };
            defer file.close();

            const stat = try file.stat();
            const SOLUTION_SIZE = 8 + NEXT_LEN;
            self.solved = @divExact(stat.size, SOLUTION_SIZE);
        }

        // Get count
        const file = fs.cwd().openFile(count_path, .{}) catch |e| {
            // Leave self.conut as 0 if the file doesn't exist
            if (e != fs.File.OpenError.FileNotFound) {
                return e;
            }
            return self;
        };
        defer file.close();

        const max_len = comptime std.math.log10_int(@as(u64, std.math.maxInt(u64))) + 1;
        var buf = [_]u8{undefined} ** max_len;
        const buf_len = try file.readAll(&buf);

        self.count = try std.fmt.parseInt(u64, buf[0..buf_len], 10);
        self.last_count = self.count;
        // Sync iterator state with count
        for (0..self.count) |_| {
            _ = try self.iter.next();
        }

        return self;
    }

    pub fn deinit(self: *SolutionBuffer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.iter.deinit();
    }

    fn mask(index: usize) usize {
        return index % CHUNKS;
    }

    fn mask2(index: usize) usize {
        return index % (2 * CHUNKS);
    }

    /// Gets the next available chunk of sequences to solve.
    pub fn nextChunk(
        self: *SolutionBuffer,
    ) !?struct { *AtomicLength, []u48, [][NEXT_LEN]Placement } {
        // Wait for space to become available
        while (true) {
            self.mutex.lock();
            if (!self.isFull()) {
                break;
            }

            self.mutex.unlock();
            std.debug.print(
                "INFO: solution buffer is full. Consider increasing the number of chunks or the chunk size\n",
                .{},
            );
            std.time.sleep(std.time.ns_per_s);
        }
        defer self.mutex.unlock();

        if (self.iter.done()) {
            return null;
        }

        const index = mask(self.write_idx);
        self.lengths[index].store(-1, .monotonic);

        var len: usize = 0;
        while (try self.iter.next()) |pieces| {
            self.sequences[index][len] = packSequence(&pieces);
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
    /// ring buffer. Returns true if any chunks were written.
    pub fn writeDoneChunks(
        self: *SolutionBuffer,
        path: []const u8,
    ) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if there's anything to write
        if (self.isEmpty() or self.lengths[mask(self.read_idx)].load(.monotonic) < 0) {
            return false;
        }

        // Multiple threads can save at the same time, but no threads should start
        // saving when we are exiting
        if (saving_threads.load(.monotonic) < 0) {
            return false;
        }

        _ = saving_threads.fetchAdd(1, .monotonic);
        defer _ = saving_threads.fetchSub(1, .monotonic);

        const allocator = self.iter.seen.allocator;

        const pc_file = blk: {
            const pc_path = try std.fmt.allocPrint(allocator, "{s}.pc", .{path});
            defer allocator.free(pc_path);

            break :blk fs.cwd().openFile(
                pc_path,
                .{ .mode = .write_only },
            ) catch |e| {
                // Create file if it doesn't exist
                if (e != fs.File.OpenError.FileNotFound) {
                    return e;
                }
                try fs.cwd().makePath(fs.path.dirname(pc_path) orelse return error.InvalidPath);
                break :blk try fs.cwd().createFile(pc_path, .{});
            };
        };
        defer pc_file.close();
        // Seek to end to append to file
        try pc_file.seekFromEnd(0);
        var buf_writer = std.io.bufferedWriter(pc_file.writer());
        const pc_writer = buf_writer.writer();

        var wrote = false;
        // A negative length indicates that the chunk is not done yet
        while (!self.isEmpty() and self.lengths[mask(self.read_idx)].load(.monotonic) >= 0) {
            const len: usize = @intCast(self.lengths[mask(self.read_idx)].load(.monotonic));
            self.solved += len;
            // NOTE: for the last chunk, the count value may become larger than
            // it actually is. This isn't an issue as the iterator is already exhausted.
            self.count += CHUNK_SIZE;
            try self.saveAppend(pc_writer, len);

            wrote = true;
            self.read_idx = mask2(self.read_idx + 1);
        }
        try buf_writer.flush();

        const count_path = try std.fmt.allocPrint(allocator, "{s}.count", .{path});
        defer allocator.free(count_path);
        var count_file = try fs.cwd().atomicFile(count_path, .{ .make_path = true });
        defer count_file.deinit();

        try count_file.file.writer().print("{d}", .{self.count});
        try count_file.finish();

        return wrote;
    }

    fn saveAppend(
        self: *SolutionBuffer,
        pc_writer: anytype,
        len: usize,
    ) !void {
        assert(NEXT_LEN <= 16);

        for (
            self.sequences[mask(self.read_idx)][0..len],
            self.solutions[mask(self.read_idx)][0..len],
        ) |seq, sol| {
            var holds: u16 = 0;
            var placements = [_]u8{0} ** NEXT_LEN;

            const sequence = unpackSequence(seq, NEXT_LEN + 1);

            var hold = sequence[0];
            var current = sequence[1];
            for (sol, 0..) |placement, i| {
                // Use canonical position so that the position is always in the range [0, 59]
                const canon_pos = placement.piece.canonicalPosition(placement.pos);
                const pos = canon_pos.y * 10 + canon_pos.x;
                assert(pos < 60);
                placements[i] = @intFromEnum(placement.piece.facing) | (@as(u8, pos) << 2);

                if (current != placement.piece.kind) {
                    holds |= @as(u16, 1) << @intCast(i);
                    hold = current;
                }
                // Only update current if it's not the last piece
                if (i < sol.len - 1) {
                    current = sequence[i + 2];
                }
            }

            try pc_writer.writeInt(u48, seq, .little);
            try pc_writer.writeInt(u16, holds, .little);
            try pc_writer.writeAll(&placements);
        }
    }

    pub fn printStatsAndBackup(self: *SolutionBuffer, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        std.debug.print(
            "Solved {} out of {}\n",
            .{ self.solved, self.count },
        );

        const count = self.count - self.last_count;
        if (count >= THREADS * 512) {
            // Create backup files
            const allocator = self.iter.seen.allocator;
            {
                const pc_path = try std.fmt.allocPrint(allocator, "{s}.pc", .{path});
                defer allocator.free(pc_path);
                const backup_path = try std.fmt.allocPrint(allocator, "{s}-backup.pc", .{path});
                defer allocator.free(backup_path);

                try fs.cwd().copyFile(pc_path, fs.cwd(), backup_path, .{});
            }
            {
                const count_path = try std.fmt.allocPrint(allocator, "{s}.count", .{path});
                defer allocator.free(count_path);
                const backup_path = try std.fmt.allocPrint(allocator, "{s}-backup.count", .{path});
                defer allocator.free(backup_path);

                try fs.cwd().copyFile(count_path, fs.cwd(), backup_path, .{});
            }

            std.debug.print(
                "Time per sequence per thread: {}\n\n",
                .{std.fmt.fmtDuration(self.timer.lap() * THREADS / count)},
            );
            self.last_count = self.count;
        }
    }

    fn isEmpty(self: SolutionBuffer) bool {
        return self.write_idx == self.read_idx;
    }

    fn isFull(self: SolutionBuffer) bool {
        return mask2(self.write_idx + CHUNKS) == self.read_idx;
    }
};

pub fn main() !void {
    setupExitHandler();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var buf = try SolutionBuffer.loadOrInit(allocator, SAVE_PATH);
    defer buf.deinit();

    var threads = [_]Thread{undefined} ** THREADS;
    for (0..threads.len) |i| {
        threads[i] = try Thread.spawn(
            .{ .allocator = allocator },
            solveThread,
            .{&buf},
        );
    }
    for (threads) |thread| {
        thread.join();
    }

    // Wait for saves to finish
    while (saving_threads.load(.monotonic) > 0) {
        std.time.sleep(std.time.ns_per_ms);
    }
}

const handle_signals = [_]c_int{ SIG.ABRT, SIG.INT, SIG.QUIT, SIG.STOP, SIG.TERM };
fn setupExitHandler() void {
    if (@import("builtin").os.tag == .windows) {
        const signal = struct {
            extern "c" fn signal(
                sig: c_int,
                func: *const fn (c_int, c_int) callconv(os.windows.WINAPI) void,
            ) callconv(.C) *anyopaque;
        }.signal;
        for (handle_signals) |sig| {
            _ = signal(sig, handleExitWindows);
        }
    } else {
        const action = os.linux.Sigaction{
            .handler = .{ .handler = handleExit },
            .mask = os.linux.empty_sigset,
            .flags = 0,
        };
        for (handle_signals) |sig| {
            _ = os.linux.sigaction(@intCast(sig), &action, null);
        }
    }
}

fn handleExit(sig: c_int) callconv(.C) void {
    if (std.mem.containsAtLeast(c_int, &handle_signals, 1, &.{sig})) {
        // Set to -1 to signal saves to stop and then wait for saves to finish
        const saving_count = saving_threads.swap(-1, .monotonic);
        while (saving_threads.load(.monotonic) >= -saving_count) {
            std.time.sleep(std.time.ns_per_ms);
        }
        std.process.exit(0);
    }
}

fn handleExitWindows(sig: c_int, _: c_int) callconv(.C) void {
    handleExit(sig);
}

fn solveThread(buf: *SolutionBuffer) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const nn = try NN.load(allocator, "NNs/Fast2.json");
    defer nn.deinit(allocator);

    while (try buf.nextChunk()) |tuple| {
        const sol_count, const sequences, const solutions = tuple;

        var solved: usize = 0;
        for (sequences) |seq| {
            _ = pc.findPc(
                SevenBag,
                allocator,
                gameWithPieces(&unpackSequence(seq, NEXT_LEN + 1)),
                nn,
                HEIGHT,
                &solutions[solved],
            ) catch |e| if (e == pc.FindPcError.SolutionTooLong) {
                continue;
            } else {
                return e;
            };

            sequences[solved] = seq;
            solved += 1;
        }

        sol_count.store(@intCast(solved), .monotonic);
        if (try buf.writeDoneChunks(SAVE_PATH)) {
            try buf.printStatsAndBackup(SAVE_PATH);
        }
    }
}

/// Pack a sequence of pieces into a u48.
fn packSequence(pieces: []const PieceKind) u48 {
    assert(pieces.len <= 48 / 3);

    var seq: u48 = 0;
    for (pieces, 0..) |piece, i| {
        seq |= @as(u48, @intFromEnum(piece)) << @intCast(3 * i);
    }
    // Fill remaining bits with 1s
    seq |= @truncate(~@as(u64, 0) << @intCast(3 * pieces.len));
    return seq;
}

/// Unpack a u48 into a sequence of pieces.
fn unpackSequence(seq: u48, comptime len: usize) [len]PieceKind {
    var pieces = [_]PieceKind{undefined} ** len;
    for (0..pieces.len) |i| {
        pieces[i] = @enumFromInt(@as(u3, @truncate(seq >> @intCast(3 * i))));
    }
    return pieces;
}

fn gameWithPieces(pieces: []const PieceKind) GameState {
    var game = GameState.init(SevenBag.init(0), engine.kicks.srs);
    game.hold_kind = pieces[0];
    game.current.kind = pieces[1];

    for (0..@min(pieces.len - 2, game.next_pieces.len)) |i| {
        game.next_pieces[i] = pieces[i + 2];
    }
    game.bag.context.index = 0;
    for (0..pieces.len -| 9) |i| {
        game.bag.context.pieces[i] = pieces[i + 9];
    }

    return game;
}
