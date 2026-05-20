const std = @import("std");
const Duration = std.Io.Duration;
const Timestamp = std.Io.Timestamp;

const engine = @import("engine");
const GameState = engine.GameState(SevenBag);
const SevenBag = engine.bags.SevenBag;

const root = @import("perfect-tetris");
const NN = root.NN;
const pathfind = root.pathfind;
const pc = root.pc;
const pc_slow = root.pc_slow;

const io = std.Io.Threaded.global_single_threaded.io();

pub fn main() !void {
    // Benchmark command:
    // zig build bench -Dcpu=x86_64_v3

    // Mean: 10.342ms ± 25.7ms
    // Max:  231.386ms
    // Mean memory usage: 138.722KiB
    // Max memory usage:  3.199MiB
    try pcBenchmark(4, "NNs/Fast3.json", false);

    // Mean: 17.139ms ± 41.975ms
    // Max:  371.552ms
    // Mean memory usage: 847.522KiB
    // Max memory usage:  26.424MiB
    try pcBenchmark(4, "NNs/Fast3.json", true);

    // Mean: 9.022ms ± 20.68ms
    // Max:  211.971ms
    // Mean memory usage: 120.016KiB
    // Max memory usage:  3.279MiB
    try pcBenchmark(6, "NNs/Fast3.json", false);

    // Mean: 7.254us ± 4.819us
    // Max:  58.031us
    try pathfindBenchmark();

    // Mean: 42ns
    getFeaturesBenchmark();
}

fn mean(T: type, values: []const T) T {
    var sum: T = 0;
    for (values) |v| {
        sum += v;
    }
    return @divTrunc(sum, std.math.lossyCast(T, values.len));
}

fn max(T: type, values: []const T) T {
    return std.sort.max(T, values, {}, std.sort.asc(T)) orelse
        @panic("max: empty array");
}

fn standardDeviation(T: type, values: []const T) !f64 {
    const m = mean(T, values);
    var sum: f64 = 0;
    for (values) |v| {
        const diff: f64 = @floatCast(v - m);
        sum += diff * diff;
    }
    return @sqrt(sum / @as(f64, @floatFromInt(values.len)));
}

pub fn pcBenchmark(
    comptime height: u8,
    nn_path: []const u8,
    comptime slow: bool,
) !void {
    const RUN_COUNT = 200;

    std.debug.print(
        \\
        \\------------------------
        \\      PC Benchmark
        \\------------------------
        \\Height: {}
        \\NN:     {s}
        \\Slow:   {}
        \\------------------------
        \\
    , .{ height, std.fs.path.stem(nn_path), slow });

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const nn = try NN.load(io, allocator, nn_path);
    defer nn.deinit(allocator);

    const placements = try allocator.alloc(root.Placement, height * 10 / 4);
    defer allocator.free(placements);

    var times: [RUN_COUNT]i96 = undefined;
    var mems: [RUN_COUNT]u64 = undefined;
    for (0..RUN_COUNT) |i| {
        const gamestate: GameState = .init(
            SevenBag.init(i),
            &engine.kicks.srsPlus,
        );
        const pieces = try root.getPieces(
            SevenBag,
            allocator,
            gamestate,
            1 + height * 10 / 4,
        );
        defer allocator.free(pieces);
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        const solve_start: Timestamp = .now(io, .cpu_process);
        const solution = if (slow)
            try pc_slow.findPc(
                .{
                    .allocator = arena.allocator(),
                    .playfield = gamestate.playfield,
                    .pieces = pieces,
                    .kicks = gamestate.kicks,
                    .min_height = height,
                    .nn = nn,
                },
                placements,
            )
        else
            try pc.findPc(
                .{
                    .allocator = arena.allocator(),
                    .playfield = gamestate.playfield,
                    .pieces = pieces,
                    .kicks = gamestate.kicks,
                    .min_height = height,
                    .nn = nn,
                },
                placements,
            );
        times[i] = solve_start.untilNow(io, .cpu_process).toNanoseconds();
        std.mem.doNotOptimizeAway(solution);

        mems[i] = arena.queryCapacity();
    }

    const avg_time = mean(i96, &times);
    const max_time = max(i96, &times);
    var times_f: [RUN_COUNT]f64 = undefined;
    for (0..RUN_COUNT) |i| {
        times_f[i] = @floatFromInt(times[i]);
    }
    const time_std: i96 = @intFromFloat(try standardDeviation(f64, &times_f));
    std.debug.print("Mean: {f} ± {f}\n", .{
        Duration.fromNanoseconds(avg_time),
        Duration.fromNanoseconds(time_std),
    });
    std.debug.print("Max:  {f}\n", .{Duration.fromNanoseconds(max_time)});

    const avg_mem = mean(u64, &mems);
    const max_mem = max(u64, &mems);
    std.debug.print("Mean memory usage: {Bi:.3}\n", .{avg_mem});
    std.debug.print("Max memory usage:  {Bi:.3}\n", .{max_mem});
}

pub fn pathfindBenchmark() !void {
    const RUN_COUNT = 200;
    const HEIGHT = 6;
    const SOL_LEN = HEIGHT * 10 / 4;
    const TIME_COUNT = RUN_COUNT * SOL_LEN;

    std.debug.print(
        \\
        \\------------------------
        \\   Pathfind Benchmark
        \\------------------------
        \\
    , .{});

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const nn = try root.defaultNN(allocator);
    defer nn.deinit(allocator);

    const placements = try allocator.alloc(root.Placement, SOL_LEN);
    defer allocator.free(placements);

    var times: [TIME_COUNT]i96 = undefined;
    for (0..RUN_COUNT) |i| {
        var gamestate: GameState = .init(
            SevenBag.init(i),
            &engine.kicks.srsPlus,
        );
        const pieces = try root.getPieces(
            SevenBag,
            allocator,
            gamestate,
            1 + SOL_LEN,
        );
        defer allocator.free(pieces);

        const solution = try pc.findPc(
            .{
                .allocator = allocator,
                .playfield = gamestate.playfield,
                .pieces = pieces,
                .kicks = gamestate.kicks,
                .min_height = HEIGHT,
                .nn = nn,
            },
            placements,
        );

        for (solution, 0..) |placement, j| {
            if (gamestate.current.kind != placement.piece.kind) {
                gamestate.hold();
            }

            const solve_start: Timestamp = .now(io, .cpu_process);
            const path = pathfind.pathfind(
                gamestate.playfield,
                gamestate.kicks,
                .{ .piece = gamestate.current, .pos = gamestate.pos },
                placement,
            );
            const time_taken = solve_start.untilNow(io, .cpu_process).toNanoseconds();
            times[i * SOL_LEN + j] = time_taken;

            std.mem.doNotOptimizeAway(path);
            gamestate.current = placement.piece;
            gamestate.pos = placement.pos;
            _ = gamestate.lockCurrent(-1);
            gamestate.nextPiece();
        }
    }

    const avg_time = mean(i96, &times);
    const max_time = max(i96, &times);
    var times_f: [TIME_COUNT]f64 = undefined;
    for (0..TIME_COUNT) |i| {
        times_f[i] = @floatFromInt(times[i]);
    }
    const time_std: i96 = @intFromFloat(try standardDeviation(f64, &times_f));
    std.debug.print("Mean: {f} ± {f}\n", .{
        Duration.fromNanoseconds(avg_time),
        Duration.fromNanoseconds(time_std),
    });
    std.debug.print("Max:  {f}\n", .{Duration.fromNanoseconds(max_time)});
}

pub fn getFeaturesBenchmark() void {
    const RUN_COUNT = 100_000_000;

    std.debug.print(
        \\
        \\--------------------------------
        \\  Feature Extraction Benchmark
        \\--------------------------------
        \\
    , .{});

    // Randomly place 3 pieces
    var xor: std.Random.Xoroshiro128 = .init(0);
    const rand = xor.random();
    var game: GameState = .init(.init(xor.next()), &engine.kicks.srsPlus);
    for (0..3) |_| {
        game.current.facing = rand.enumValue(engine.pieces.Facing);
        game.pos.x = rand.intRangeAtMost(
            i8,
            game.current.minX(),
            game.current.maxX(),
        );
        _ = game.dropToGround();
        _ = game.lockCurrent(-1);
    }
    const playfield: root.bit_masks.BoardMask = .from(game.playfield);

    const start: Timestamp = .now(io, .cpu_process);
    for (0..RUN_COUNT) |_| {
        std.mem.doNotOptimizeAway(
            pc.getFeatures(playfield, 6, @splat(true)),
        );
    }
    const time_taken = start.untilNow(io, .cpu_process).toNanoseconds();

    std.debug.print(
        "Mean: {f}\n",
        .{Duration.fromNanoseconds(@divTrunc(time_taken, RUN_COUNT))},
    );
}
