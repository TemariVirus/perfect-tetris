const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const Order = std.math.Order;

const engine = @import("engine");
const Facing = engine.pieces.Facing;
const KickFn = engine.kicks.KickFn;
const Position = engine.pieces.Position;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Rotation = engine.kicks.Rotation;

const root = @import("root.zig");
const BoardMask = root.bit_masks.BoardMask;
const NN = root.NN;
const PieceMask = root.bit_masks.PieceMask;
const Placement = root.Placement;

const small = @import("options").small;

pub const PiecePosSet = @import("PiecePosSet.zig").PiecePosSet(.{
    10,
    (if (small) 4 else 6) + 3,
    4,
});

// 40 cells * 4 rotations = 160 intermediate placements at most
// 60 cells * 4 rotations = 240 intermediate placements at most
const PlacementStack = std.BoundedArray(
    PiecePosition,
    if (small) 160 else 240,
);
const PiecePosition = if (small) packed struct {
    facing: Facing,
    pos: u6,

    pub fn pack(piece: Piece, pos: Position) PiecePosition {
        const x = pos.x - piece.minX();
        const y = pos.y - piece.minY();
        return PiecePosition{
            .facing = piece.facing,
            .pos = @intCast(y * BoardMask.WIDTH + x),
        };
    }

    pub fn unpack(self: PiecePosition, piece_kind: PieceKind) Placement {
        const piece = Piece{ .kind = piece_kind, .facing = self.facing };
        return .{
            .piece = piece,
            .pos = .{
                .x = @intCast(self.pos % BoardMask.WIDTH + piece.minX()),
                .y = @intCast(self.pos / BoardMask.WIDTH + piece.minY()),
            },
        };
    }
} else packed struct {
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

pub const Move = enum {
    left,
    right,
    rotate_cw,
    rotate_double,
    rotate_ccw,
    drop,

    pub const moves = std.enums.values(Move);
};

/// Intermediate game state when searching for possible placements.
pub const Intermediate = struct {
    playfield: BoardMask,
    current: Piece,
    pos: Position,
    do_o_rotations: bool,
    kicks: *const KickFn,

    /// Returns `true` if the move was successful. Otherwise, `false`.
    pub fn makeMove(self: *Intermediate, move: Move) bool {
        return switch (move) {
            .left => blk: {
                if (self.pos.x <= self.current.minX()) {
                    break :blk false;
                }
                break :blk self.tryTranspose(.{
                    .x = self.pos.x - 1,
                    .y = self.pos.y,
                });
            },
            .right => blk: {
                if (self.pos.x >= self.current.maxX()) {
                    break :blk false;
                }
                break :blk self.tryTranspose(.{
                    .x = self.pos.x + 1,
                    .y = self.pos.y,
                });
            },
            .rotate_cw => self.tryRotate(.quarter_cw),
            .rotate_double => self.tryRotate(.half),
            .rotate_ccw => self.tryRotate(.quarter_ccw),
            .drop => blk: {
                if (self.pos.y <= self.current.minY()) {
                    break :blk false;
                }
                break :blk self.tryTranspose(.{
                    .x = self.pos.x,
                    .y = self.pos.y - 1,
                });
            },
        };
    }

    /// Returns `true` if the piece was successfully transposed. Otherwise, `false`.
    fn tryTranspose(self: *Intermediate, pos: Position) bool {
        if (self.playfield.collides(self.current, pos)) {
            return false;
        }
        self.pos = pos;
        return true;
    }

    /// Returns `true` if the piece was successfully rotated. Otherwise, `false`.
    fn tryRotate(self: *Intermediate, rotation: Rotation) bool {
        if (self.current.kind == .o and !self.do_o_rotations) {
            return false;
        }

        const new_piece = Piece{
            .facing = self.current.facing.rotate(rotation),
            .kind = self.current.kind,
        };

        for (self.kicks(self.current, rotation)) |kick| {
            const kicked_pos = self.pos.add(kick);
            if (!self.playfield.collides(new_piece, kicked_pos)) {
                self.current = new_piece;
                self.pos = kicked_pos;
                return true;
            }
        }

        return false;
    }

    /// Returns `true` if the piece is on the ground. Otherwise, `false`.
    pub fn onGround(self: Intermediate) bool {
        return self.playfield.collides(
            self.current,
            Position{ .x = self.pos.x, .y = self.pos.y - 1 },
        );
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
    max_height: u3,
) PiecePosSet {
    return allPlacementsRaw(
        PiecePosSet,
        PiecePosition,
        PlacementStack,
        BoardMask,
        Intermediate,
        playfield,
        do_o_rotations,
        kicks,
        piece_kind,
        max_height,
    );
}

pub fn allPlacementsRaw(
    comptime TPiecePosSet: type,
    comptime TPiecePosition: type,
    comptime TPlacementStack: type,
    comptime TBoardMask: type,
    comptime TIntermediate: type,
    playfield: TBoardMask,
    do_o_rotations: bool,
    kicks: *const KickFn,
    piece_kind: PieceKind,
    max_height: u7,
) TPiecePosSet {
    var seen = TPiecePosSet.init();
    var placements = TPiecePosSet.init();
    var stack = TPlacementStack.init(0) catch unreachable;

    // Start right above `max_height`
    for (std.enums.values(Facing)) |facing| {
        const piece = Piece{
            .facing = facing,
            .kind = piece_kind,
        };
        const pos = Position{
            .x = 0,
            .y = @as(i8, max_height) + piece.minY(),
        };

        if (!do_o_rotations and piece_kind == .o and facing != .up) {
            continue;
        }
        stack.append(TPiecePosition.pack(piece, pos)) catch unreachable;
    }

    while (stack.popOrNull()) |placement| {
        const piece, const pos = blk: {
            const temp = placement.unpack(piece_kind);
            break :blk .{ temp.piece, temp.pos };
        };
        if (seen.putGet(piece, pos)) {
            continue;
        }

        for (Move.moves) |move| {
            var new_game = TIntermediate{
                .playfield = playfield,
                .current = piece,
                .pos = pos,
                .do_o_rotations = do_o_rotations,
                .kicks = kicks,
            };

            // Skip if piece was unable to move
            if (!new_game.makeMove(move)) {
                continue;
            }
            if (seen.contains(new_game.current, new_game.pos)) {
                continue;
            }

            // Branch out after movement if the piece is not too high
            if (new_game.pos.y > @as(i8, max_height) + new_game.current.minY()) {
                continue;
            }
            stack.append(
                TPiecePosition.pack(new_game.current, new_game.pos),
            ) catch unreachable;

            // Skip this placement if the piece is too high, or if it's not on
            // the ground
            if (new_game.pos.y + @as(i8, new_game.current.top()) > max_height or
                !new_game.onGround())
            {
                continue;
            }
            placements.put(new_game.current, new_game.pos);
        }
    }

    return placements;
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
/// placements where `validFn` returns `false`. Higher scores are dequeued first.
pub fn orderMoves(
    queue: *MoveQueue,
    playfield: BoardMask,
    piece: PieceKind,
    moves: PiecePosSet,
    max_height: u3,
    comptime validFn: fn (BoardMask, u3) bool,
    nn: NN,
    comptime scoreFn: fn (BoardMask, u3, NN) f32,
) void {
    var iter = moves.iterator(piece);
    while (iter.next()) |placement| {
        var board = playfield;
        board.place(PieceMask.from(placement.piece), placement.pos);
        const cleared = board.clearLines(placement.pos.y);
        const new_height = max_height - cleared;
        if (!validFn(board, new_height)) {
            continue;
        }

        queue.add(.{
            .placement = placement,
            .score = scoreFn(board, max_height, nn),
        }) catch unreachable;
    }
}

test allPlacements {
    var playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0111111110) << 30;
    playfield.mask |= @as(u64, 0b0010000000) << 20;
    playfield.mask |= @as(u64, 0b0000001000) << 10;
    playfield.mask |= @as(u64, 0b0000000001);

    const PIECE = PieceKind.l;
    const placements = allPlacements(
        playfield,
        false,
        engine.kicks.srs,
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
