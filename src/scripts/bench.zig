const std = @import("std");
const time = std.time;

const engine = @import("engine");
const GameState = engine.GameState(FixedBag);

const root = @import("perfect-tetris");
const pc = root.pc;
const FixedBag = root.FixedBag;

pub fn main() !void {
    // Solutions: 1315
    // Time: 21.734s
    try pcBenchmark(4);
}

pub fn pcBenchmark(comptime height: u8) !void {
    std.debug.print(
        \\
        \\------------------------
        \\      PC Benchmark
        \\------------------------
        \\Height: {}
        \\------------------------
        \\
    , .{height});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const start = std.time.nanoTimestamp();

    const bag = FixedBag{
        .pieces = &.{
            .t, .o, .l, .s, .z, .j, .i, .o, .t, .s, .z,
        },
    };
    const game = GameState.init(bag, engine.kicks.srs);
    const solutions = try pc.findAllPcs(FixedBag, allocator, game, 4);
    defer {
        for (solutions) |solution| {
            allocator.free(solution);
        }
        allocator.free(solutions);
    }

    const t: u64 = @intCast(std.time.nanoTimestamp() - start);
    std.debug.print("Solutions: {}\n", .{solutions.len});
    std.debug.print("Time: {}\n", .{std.fmt.fmtDuration(t)});
}
