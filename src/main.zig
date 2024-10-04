const std = @import("std");
const Allocator = std.mem.Allocator;

const pc = @import("perfect-tetris").pc;

const engine = @import("engine");
const GameState = engine.GameState(FixedBag);
const PieceKind = engine.pieces.PieceKind;

pub const FixedBag = struct {
    pieces: []const PieceKind,
    index: usize = 0,

    pub fn init(seed: u64) FixedBag {
        _ = seed; // autofix
        return .{ .pieces = &.{} };
    }

    pub fn next(self: *FixedBag) PieceKind {
        if (self.index >= self.pieces.len) {
            return undefined;
        }

        defer self.index += 1;
        return self.pieces[self.index];
    }

    pub fn setSeed(self: *FixedBag, seed: u64) void {
        _ = seed; // autofix
        self.index = 0;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const bag = FixedBag{
        .pieces = &.{
            //TOLSZJISTLS
            .t, .o, .l, .s, .z, .j, .i, .s, .t, .l, .s,
        },
    };
    _ = try pc.findPc(
        FixedBag,
        allocator,
        GameState.init(bag, engine.kicks.srsPlus),
        4,
    );
}
