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
const PiecePosition = root.movegen.PiecePosition;
const Placement = root.Placement;

const PiecePosSet = @import("../PiecePosSet.zig").PiecePosSet(.{ 10, 24, 4 });

// Every intermediate placement needs a previous placement, so the maximum stack
// size is `PiecePosSet.len * moves.len / (moves.len + 1)`
const PlacementStack = std.BoundedArray(
    PiecePosition,
    PiecePosSet.len * Move.moves.len / (Move.moves.len + 1),
);

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
) error{OutOfMemory}!void {
    var iter = moves.iterator(piece);
    while (iter.next()) |placement| {
        var board = playfield;
        board.place(placement.piece.mask(), placement.pos);
        const cleared = board.clearLines(placement.pos.y);
        const new_height = max_height - cleared;
        if (!validFn(board.rows[0..@min(BoardMask.HEIGHT, new_height)])) {
            continue;
        }

        try queue.add(.{
            .placement = placement,
            .score = scoreFn(
                board.rows[0..@min(BoardMask.HEIGHT, new_height)],
                nn,
            ),
        });
    }
}

test allPlacements {
    var playfield: BoardMask = .{};
    playfield.rows[3] |= 0b0111111110_0;
    playfield.rows[2] |= 0b0010000000_0;
    playfield.rows[1] |= 0b0000001000_0;
    playfield.rows[0] |= 0b0000000001_0;

    const PIECE: PieceKind = .l;
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

test "No placements" {
    var playfield: BoardMask = .{};
    playfield.rows[2] |= 0b1111111110_0;
    playfield.rows[1] |= 0b1111111110_0;
    playfield.rows[0] |= 0b1111111100_0;

    const PIECE: PieceKind = .j;
    const placements = allPlacements(
        playfield,
        false,
        &engine.kicks.none,
        PIECE,
        3,
    );

    var iter = placements.iterator(PIECE);
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try expect(count == 0);
}
