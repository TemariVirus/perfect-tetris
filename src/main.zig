const std = @import("std");
const Allocator = std.mem.Allocator;

const engine = @import("engine");
const GameState = engine.GameState(FixedBag);

const root = @import("perfect-tetris");
pub const FixedBag = root.FixedBag;
const pc = root.pc;
const PCSolution = root.PCSolution;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const start = std.time.nanoTimestamp();

    const bag = FixedBag{
        .pieces = &.{
            //TOLSZJISTLZ
            .t, .o, .l, .s, .z, .j, .i, .s, .t, .l, .z,
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

    const file = try std.fs.cwd().createFile("pc-data/all.pc", .{});
    defer file.close();

    // Nothing to write
    if (solutions.len == 0) {
        return;
    }

    var bf = std.io.bufferedWriter(file.writer());
    const writer = bf.writer().any();

    const pieces = try pc.getPieces(FixedBag, allocator, game, solutions[0].len + 1);
    defer allocator.free(pieces);
    for (solutions) |solution| {
        var s = PCSolution{};
        s.next.len = @intCast(pieces.len);
        @memcpy(s.next.slice(), pieces);
        s.placements.len = @intCast(solution.len);
        @memcpy(s.placements.slice(), solution);
        try s.write(writer);
    }
    try bf.flush();
}
