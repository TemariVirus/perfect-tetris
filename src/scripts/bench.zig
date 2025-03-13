const std = @import("std");
const time = std.time;

const engine = @import("engine");
const GameState = engine.GameState;
const SevenBag = engine.bags.SevenBag;

const root = @import("perfect-tetris");
const FixedBag = root.FixedBag;
const NN = root.NN;
const pc = root.pc;
const pc_slow = root.pc_slow;

pub fn main() !void {
    // Benchmark command:
    // zig build bench -Dcpu=x86_64_v3

    // Mean: 10.697ms ± 26.627ms
    // Max: 237.634ms
    // try pcBenchmark(4, "NNs/Fast3.json", false);

    // Mean: 14.696ms ± 35.902ms
    // Max: 322.321ms
    // try pcBenchmark(4, "NNs/Fast3.json", true);

    // Mean: 9.253ms ± 21.252ms
    // Max: 217.802ms
    // try pcBenchmark(6, "NNs/Fast3.json", false);

    // /Sturctures with same shape and same piece kinds are considered the same (may look different to human)
    // Solutions: 118 (not sure if correct)
    // Time: 21.001s
    // /Only solves that looks same to human
    // Solutions: 8622 (should be 1315)
    // Time: 53.674s
    try pcAllBenchmark(
        4,
        &.{ .t, .o, .l, .s, .z, .j, .i, .o, .t, .s, .z },
    );

    // Mean: 43ns
    // getFeaturesBenchmark();
}

fn mean(T: type, values: []const T) T {
    var sum: T = 0;
    for (values) |v| {
        sum += v;
    }
    return sum / std.math.lossyCast(T, values.len);
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

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const nn = try NN.load(allocator, nn_path);
    defer nn.deinit(allocator);

    const placements = try allocator.alloc(root.Placement, height * 10 / 4);
    defer allocator.free(placements);

    var times: [RUN_COUNT]u64 = undefined;
    for (0..RUN_COUNT) |i| {
        const gamestate: GameState(SevenBag) = .init(
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

pub fn pcAllBenchmark(comptime height: u8, pieces: []const engine.pieces.PieceKind) !void {
    std.debug.print(
        \\
        \\------------------------
        \\    PC all Benchmark
        \\------------------------
        \\Height: {}
        \\------------------------
        \\
    , .{height});

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const placements = try allocator.alloc(root.Placement, height * 10 / 4);
    defer allocator.free(placements);

    const bag: FixedBag = .{ .pieces = pieces };
    const game: GameState(FixedBag) = .init(bag, engine.kicks.srs);
    var iter: pc.SolutionIterator = try .init(
        FixedBag,
        allocator,
        game,
        height,
        placements,
        null,
    );
    defer iter.deinit();

    const f = try std.fs.cwd().createFile("all.pc", .{});
    defer f.close();
    const writer = f.writer();

    const start = std.time.nanoTimestamp();

    var count: usize = 0;
    while (try iter.next()) |sol| {
        count += 1;

        var holds: u16 = 0;
        var ps: [height * 10 / 4]u8 = @splat(0);

        const sequence = try root.PCSolution.NextArray.fromSlice(pieces);

        var hold = sequence.buffer[0];
        var current = sequence.buffer[1];
        for (sol, 0..) |placement, i| {
            // Use canonical position so that the position is always in the
            // range [0, 59]
            const canon_pos = placement.piece
                .canonicalPosition(placement.pos);
            const pos = canon_pos.y * 10 + canon_pos.x;
            std.debug.assert(pos < 60);
            ps[i] = @intFromEnum(placement.piece.facing) |
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

        try writer.writeInt(u48, root.PCSolution.packNext(sequence), .little);
        try writer.writeInt(u16, holds, .little);
        try writer.writeAll(&ps);
    }

    const t: u64 = @intCast(std.time.nanoTimestamp() - start);
    std.debug.print("Solutions: {}\n", .{count});
    std.debug.print("Time: {}\n", .{std.fmt.fmtDuration(t)});
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
    var game: GameState(SevenBag) = .init(.init(xor.next()), &engine.kicks.srsPlus);
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
