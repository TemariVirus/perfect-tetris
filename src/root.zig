pub const bit_masks = @import("bit_masks.zig");
pub const movegen = @import("movegen.zig");
pub const movegen_slow = @import("slow/movegen.zig");
pub const next = @import("next.zig");
pub const pc = @import("pc.zig");
pub const pc_slow = @import("slow/pc.zig");

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const expect = std.testing.expect;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const Facing = engine.pieces.Facing;
const GameState = engine.GameState;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;

const NNInner = @import("zmai").genetic.neat.NN;

/// The maximum length of a next sequence in a `.pc` file.
pub const PC_MAX_SEQ_LEN = 16;

pub const Placement = struct {
    piece: Piece,
    pos: Position,
};

pub const NN = struct {
    pub const INPUT_COUNT = 9;

    net: NNInner,
    inputs_used: [INPUT_COUNT]bool,

    pub fn load(allocator: Allocator, path: []const u8) !NN {
        var inputs_used: [INPUT_COUNT]bool = undefined;
        const nn = try NNInner.load(allocator, path, &inputs_used);
        return .{
            .net = nn,
            .inputs_used = inputs_used,
        };
    }

    pub fn deinit(self: NN, allocator: Allocator) void {
        self.net.deinit(allocator);
    }

    pub fn predict(self: NN, input: [INPUT_COUNT]f32) f32 {
        var output: [1]f32 = undefined;
        self.net.predict(&input, &output);
        return output[0];
    }
};

pub const PCSolution = struct {
    pub const NextArray = std.BoundedArray(PieceKind, PC_MAX_SEQ_LEN);
    pub const PlacementArray = std.BoundedArray(Placement, PC_MAX_SEQ_LEN - 1);

    next: NextArray = NextArray{},
    placements: PlacementArray = PlacementArray{},

    /// Reads the next perfect clear solution from the reader, assuming the
    /// reader is reading a `.pc` file. If the reader is at the end of the
    /// file, this function returns `null`.
    pub fn readOne(reader: anytype) !?PCSolution {
        const seq = reader.readInt(u48, .little) catch |e| {
            if (e == error.EndOfStream) {
                return null;
            }
            return e;
        };
        const holds = try reader.readInt(u16, .little);

        var solution = PCSolution{};
        while (solution.next.len < PC_MAX_SEQ_LEN) {
            const shift = 3 * @as(usize, solution.next.len);
            const p: u3 = @truncate(seq >> @intCast(shift));
            // 0b111 is the sentinel value for the end of the sequence
            if (p == 0b111) {
                break;
            }
            solution.next.appendAssumeCapacity(@enumFromInt(p));
        }

        var hold: PieceKind = solution.next.buffer[0];
        for (1..solution.next.len) |i| {
            // Check if the piece is held
            var current = solution.next.buffer[i];
            if ((holds >> @intCast(i - 1)) & 1 == 1) {
                const temp = hold;
                hold = current;
                current = temp;
            }

            const placement = try reader.readByte();
            const facing: Facing = @enumFromInt(@as(u2, @truncate(placement)));
            const piece = Piece{ .facing = facing, .kind = current };
            const canon_pos = placement >> 2;
            const pos = piece.fromCanonicalPosition(.{
                .x = @intCast(canon_pos % 10),
                .y = @intCast(canon_pos / 10),
            });

            solution.placements.appendAssumeCapacity(.{
                .piece = piece,
                .pos = pos,
            });
        }

        return solution;
    }
};

pub const FindPcError = error{
    ImpossibleSaveHold,
    NoPcExists,
    SolutionTooLong,
};

/// Returns the minimum possible height and number of pieces needed for a
/// perfect clear given the playfield. There is no guarantee that a perfect
/// clear exists at the returned height. Returns `null` if a perfect clear is
/// impossible.
pub fn minPcInfo(playfield: BoardMask) ?struct {
    height: u6,
    pieces_needed: u8,
} {
    var height: u32 = playfield.height();
    const bits_set = blk: {
        var set: usize = 0;
        for (0..height) |i| {
            set += @popCount(playfield.rows[i] & ~BoardMask.EMPTY_ROW);
        }
        break :blk set;
    };
    const empty_cells = BoardMask.WIDTH * height - bits_set;

    // Assumes that all pieces have 4 cells and that the playfield is 10 cells wide.
    // Thus, an odd number of empty cells means that a perfect clear is impossible.
    if (empty_cells % 2 == 1) {
        return null;
    }

    var pieces_needed = if (empty_cells % 4 == 2)
        // If the number of empty cells is not a multiple of 4, we need to fill
        // an extra so that it becomes a multiple of 4
        // 2 + 10 = 12 which is a multiple of 4
        (empty_cells + 10) / 4
    else
        empty_cells / 4;
    // Don't return an empty solution
    if (pieces_needed == 0) {
        pieces_needed = 5;
        height += 2;
    }

    return .{
        .height = @intCast(height),
        .pieces_needed = @intCast(pieces_needed),
    };
}

/// Finds a perfect clear with the least number of pieces possible for the
/// given game state, and returns the sequence of placements required to
/// achieve it.
///
/// `min_height` determines the minimum height of the perfect clear.
///
/// `max_len` determines the maximum number of placements.
///
/// `save_hold` determines what piece must be in the hold slot at the end of
/// the sequence. If `save_hold` is `null`, this constraint is ignored.
///
/// Returns an error if no perfect clear exists, or if the number of pieces
/// needed exceeds `max_pieces`.
pub fn findPcAuto(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    nn: NN,
    min_height: u7,
    max_len: usize,
    save_hold: ?PieceKind,
) ![]Placement {
    const placements = try allocator.alloc(Placement, max_len);
    errdefer allocator.free(placements);
    const height = @max(
        min_height,
        (minPcInfo(game.playfield) orelse return FindPcError.NoPcExists).height,
    );

    // Use fast pc if possible
    if (height <= 6) {
        if (pc.findPc(
            BagType,
            allocator,
            game,
            nn,
            @intCast(height),
            placements,
            save_hold,
        )) |solution| {
            assert(allocator.resize(placements, solution.len));
            return solution;
        } else |e| if (e != FindPcError.SolutionTooLong) {
            return e;
        }
    }

    const solution = try pc_slow.findPc(
        BagType,
        allocator,
        game,
        nn,
        @max(7, height),
        placements,
        save_hold,
    );
    assert(allocator.resize(placements, solution.len));
    return solution;
}

test {
    std.testing.refAllDecls(@This());
}
