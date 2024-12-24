const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const Order = std.math.Order;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const Facing = engine.pieces.Facing;
const KickFn = engine.kicks.KickFn;
const Position = engine.pieces.Position;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Rotation = engine.kicks.Rotation;

const root = @import("../root.zig");
const NN = root.NN;
const Move = root.movegen.Move;
const Placement = root.Placement;

const PiecePosSet = @import("../PiecePosSet.zig").PiecePosSet(.{ 10, 24, 4 });

// 200 cells * 4 rotations = 800 intermediate placements at most
const PlacementStack = std.BoundedArray(PiecePosition, 800);
const PiecePosition = packed struct {
    y: i8,
    x: i6,
    facing: Facing,

    pub fn pack(piece: Piece, pos: Position) PiecePosition {
        return PiecePosition{
            .y = pos.y,
            .x = @intCast(pos.x),
            .facing = piece.facing,
        };
    }

    pub fn unpack(self: PiecePosition, piece_kind: PieceKind) Placement {
        return .{
            .piece = .{ .kind = piece_kind, .facing = self.facing },
            .pos = .{ .y = self.y, .x = self.x },
        };
    }
};

/// Returns the set of all placements where the top of the piece does not
/// exceed `max_height`. Assumes that no cells in the playfield are higher than
/// `max_height`.
pub fn allPlacements(
    playfield: BoardMask,
    do_o_rotations: bool,
    kicks: *const KickFn,
    piece_kind: PieceKind,
    max_height: u7,
) PiecePosSet {
    return root.movegen.allPlacementsRaw(
        PiecePosSet,
        PiecePosition,
        PlacementStack,
        BoardMask,
        playfield,
        do_o_rotations,
        kicks,
        piece_kind,
        max_height,
    );
}

pub const MoveNode = struct {
    placement: Placement,
    score: f32,
};
const compareFn = (struct {
    fn cmp(_: void, a: MoveNode, b: MoveNode) Order {
        return std.math.order(b.score, a.score);
    }
}).cmp;
pub const MoveQueue = std.PriorityQueue(MoveNode, void, compareFn);

/// Scores and orders the moves in `moves` based on the `scoreFn`, removing
/// placements where `validFn` returns `false`. Higher scores are dequeued
/// first.
pub fn orderMoves(
    queue: *MoveQueue,
    playfield: BoardMask,
    piece: PieceKind,
    moves: PiecePosSet,
    max_height: u7,
    comptime validFn: fn ([]const u16) bool,
    nn: NN,
    comptime scoreFn: fn ([]const u16, NN) f32,
) void {
    var iter = moves.iterator(piece);
    while (iter.next()) |placement| {
        var board = playfield;
        board.place(placement.piece.mask(), placement.pos);
        const cleared = board.clearLines(placement.pos.y);
        const new_height = max_height - cleared;
        if (!validFn(board.rows[0..@min(BoardMask.HEIGHT, new_height)])) {
            continue;
        }

        queue.add(.{
            .placement = placement,
            .score = scoreFn(
                board.rows[0..@min(BoardMask.HEIGHT, new_height)],
                nn,
            ),
        }) catch unreachable;
    }
}

test allPlacements {
    var playfield = BoardMask{};
    playfield.rows[3] |= 0b0111111110_0;
    playfield.rows[2] |= 0b0010000000_0;
    playfield.rows[1] |= 0b0000001000_0;
    playfield.rows[0] |= 0b0000000001_0;

    const PIECE = PieceKind.l;
    const placements = allPlacements(
        playfield,
        false,
        &engine.kicks.srs,
        PIECE,
        5,
    );

    var iter = placements.iterator(PIECE);
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try expect(count == 25);
}
