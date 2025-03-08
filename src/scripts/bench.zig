const std = @import("std");
const time = std.time;

const engine = @import("engine");
const GameState = engine.GameState(SevenBag);
const SevenBag = engine.bags.SevenBag;

const root = @import("perfect-tetris");
const NN = root.NN;
const pc = root.pc;
const pc_slow = root.pc_slow;

pub fn main() !void {
    // Benchmark command:
    // zig build bench -Dcpu=baseline

    // Mean: 11.864ms ± 29.254ms
    // Max: 257.017ms
    try pcBenchmark(4, "NNs/Fast3.json", false);

    // Mean: 15.939ms ± 38.706ms
    // Max: 338.054ms
    try pcBenchmark(4, "NNs/Fast3.json", true);

    // Mean: 10.035ms ± 22.749ms
    // Max: 233.465ms
    try pcBenchmark(6, "NNs/Fast3.json", false);

    // Mean: 60ns
    getFeaturesBenchmark();
}

fn mean(T: type, values: []T) T {
    var sum: T = 0;
    for (values) |v| {
        sum += v;
    }
    return sum / std.math.lossyCast(T, values.len);
}

fn max(T: type, values: []T) T {
    var maximum: T = values[0];
    for (values) |v| {
        maximum = @max(v, maximum);
    }
    return maximum;
}

fn standardDeviation(T: type, values: []T) !f64 {
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
    slow: bool,
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

    // For some reason std.heap.smp_allocator takes forever when height = 6
    const allocator = std.heap.page_allocator;

    const nn = try NN.load(allocator, nn_path);
    defer nn.deinit(allocator);

    const placements = try allocator.alloc(root.Placement, height * 10 / 4);
    defer allocator.free(placements);

    var times: [RUN_COUNT]u64 = undefined;
    for (0..RUN_COUNT) |i| {
        const gamestate = GameState.init(
            SevenBag.init(i),
            &engine.kicks.srsPlus,
        );

        const solve_start = time.nanoTimestamp();
        const solution = if (slow)
            try pc_slow.findPc(
                SevenBag,
                allocator,
                gamestate,
                nn,
                height,
                placements,
                null,
            )
        else
            try pc.findPc(
                SevenBag,
                allocator,
                gamestate,
                nn,
                height,
                placements,
                null,
            );
        times[i] = @intCast(time.nanoTimestamp() - solve_start);
        std.mem.doNotOptimizeAway(solution);
    }

    const avg_time: u64 = mean(u64, &times);
    const max_time: u64 = max(u64, &times);
    var times_f: [RUN_COUNT]f64 = undefined;
    for (0..RUN_COUNT) |i| {
        times_f[i] = @floatFromInt(times[i]);
    }
    const time_std: u64 = @intFromFloat(try standardDeviation(f64, &times_f));
    std.debug.print("Mean: {} ± {}\n", .{
        std.fmt.fmtDuration(avg_time),
        std.fmt.fmtDuration(time_std),
    });
    std.debug.print("Max: {}\n", .{std.fmt.fmtDuration(max_time)});
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
    var game: GameState = .init(SevenBag.init(xor.next()), &engine.kicks.srsPlus);
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

    const start = time.nanoTimestamp();
    for (0..RUN_COUNT) |_| {
        std.mem.doNotOptimizeAway(
            pc.getFeatures(playfield, 6, @splat(true)),
        );
    }
    const time_taken: u64 = @intCast(time.nanoTimestamp() - start);

    std.debug.print(
        "Mean: {}\n",
        .{std.fmt.fmtDuration(time_taken / RUN_COUNT)},
    );
}
