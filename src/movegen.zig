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

pub const PiecePosSet = @import("PiecePosSet.zig").PiecePosSet(.{ 10, 6, 4 });

// Every intermediate placement needs a previous placement, so the maximum stack
// size is `PiecePosSet.len * moves.len / (moves.len + 1)`
const PlacementStack = std.BoundedArray(
    PiecePosition,
    PiecePosSet.len * Move.moves.len / (Move.moves.len + 1),
);
pub const PiecePosition = packed struct {
    y: i8,
    x: i6,
    facing: Facing,

    pub fn pack(piece: Piece, pos: Position) PiecePosition {
        return .{
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
pub fn Intermediate(comptime TPiecePosSet: type) type {
    return struct {
        collision_set: *const TPiecePosSet,
        current: Piece,
        pos: Position,
        do_o_rotations: bool,
        max_height: i8,
        kicks: *const KickFn,

        const Self = @This();

        /// Returns `true` if the move was successful. Otherwise, `false`.
        pub fn makeMove(self: *Self, move: Move) bool {
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

        /// Returns `true` if the piece would collide with the playfield at the
        /// given position. Otherwise, `false`.
        pub fn collides(self: Self, piece: Piece, pos: Position) bool {
            // Out of bounds
            if (pos.x > piece.maxX() or
                pos.x < piece.minX() or
                pos.y < piece.minY())
            {
                return true;
            }
            // Piece is completely above the playfield
            if (pos.y >= self.max_height + piece.minY()) {
                return false;
            }
            return self.collision_set.contains(piece, pos);
        }

        /// Returns `true` if the piece was successfully transposed. Otherwise, `false`.
        pub fn tryTranspose(self: *Self, pos: Position) bool {
            if (self.collides(self.current, pos)) {
                return false;
            }
            self.pos = pos;
            return true;
        }

        /// Returns `true` if the piece was successfully rotated. Otherwise, `false`.
        pub fn tryRotate(self: *Self, rotation: Rotation) bool {
            if (self.current.kind == .o and !self.do_o_rotations) {
                return false;
            }

            const new_piece: Piece = .{
                .facing = self.current.facing.rotate(rotation),
                .kind = self.current.kind,
            };

            for (self.kicks(self.current, rotation)) |kick| {
                const kicked_pos = self.pos.add(kick);
                if (!self.collides(new_piece, kicked_pos)) {
                    self.current = new_piece;
                    self.pos = kicked_pos;
                    return true;
                }
            }

            return false;
        }

        /// Returns `true` if the piece is on the ground. Otherwise, `false`.
        pub fn onGround(self: Self) bool {
            return self.collides(
                self.current,
                .{ .x = self.pos.x, .y = self.pos.y - 1 },
            );
        }
    };
}

/// Returns the set of all placements where the piece collides with the playfield.
/// Assumes that no cells in the playfield are higher than `max_height`.
pub fn collisionSet(
    comptime TPiecePosSet: type,
    comptime TBoardMask: type,
    playfield: TBoardMask,
    do_o_rotations: bool,
    current: Piece,
    max_height: u7,
) TPiecePosSet {
    var collision_set = TPiecePosSet.init();
    for (std.enums.values(Facing)) |facing| {
        // Skip collisions for facings that cannot be reached
        if (current.kind == .o and
            !do_o_rotations and
            current.facing != facing)
        {
            continue;
        }

        const piece: Piece = .{
            .facing = facing,
            .kind = current.kind,
        };
        var x = piece.minX();
        while (x <= piece.maxX()) : (x += 1) {
            var y = piece.minY();
            while (y < @as(i8, max_height) + piece.minY()) : (y += 1) {
                if (playfield.collides(piece, .{ .x = x, .y = y })) {
                    collision_set.put(piece, .{ .x = x, .y = y });
                }
            }
        }
    }

    return collision_set;
}

/// Returns the set of all placements where the top of the piece does not
/// exceed `max_height`. Assumes that no cells in the playfield are higher than
/// `max_height`.
pub fn allPlacements(
    playfield: BoardMask,
    do_o_rotations: bool,
    kicks: *const KickFn,
    piece_kind: PieceKind,
    max_height: u6,
) error{Overflow}!PiecePosSet {
    return try allPlacementsRaw(
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

pub fn allPlacementsRaw(
    comptime TPiecePosSet: type,
    comptime TPiecePosition: type,
    comptime TPlacementStack: type,
    comptime TBoardMask: type,
    playfield: TBoardMask,
    do_o_rotations: bool,
    kicks: *const KickFn,
    piece_kind: PieceKind,
    max_height: u7,
) error{Overflow}!TPiecePosSet {
    var seen: TPiecePosSet = .init();
    var placements: TPiecePosSet = .init();
    var stack = TPlacementStack.init(0) catch unreachable;
    const collision_set = collisionSet(
        TPiecePosSet,
        TBoardMask,
        playfield,
        do_o_rotations,
        Piece{ .kind = piece_kind, .facing = .up },
        max_height,
    );

    // Start right above `max_height`
    for (std.enums.values(Facing)) |facing| {
        if (!do_o_rotations and piece_kind == .o and facing != .up) {
            continue;
        }
        const piece: Piece = .{
            .facing = facing,
            .kind = piece_kind,
        };

        var x = piece.minX();
        while (x <= piece.maxX()) : (x += 1) {
            try stack.append(TPiecePosition.pack(piece, .{
                .x = x,
                .y = @as(i8, max_height) + piece.minY(),
            }));
        }
    }

    while (stack.len > 0) {
        const placement = stack.buffer[stack.len - 1];
        stack.len -= 1;

        const piece, const pos = blk: {
            const temp = placement.unpack(piece_kind);
            break :blk .{ temp.piece, temp.pos };
        };

        for (Move.moves) |move| {
            var new_game: Intermediate(TPiecePosSet) = .{
                .collision_set = &collision_set,
                .current = piece,
                .pos = pos,
                .do_o_rotations = do_o_rotations,
                .max_height = max_height,
                .kicks = kicks,
            };

            // Skip if piece was unable to move or is completely above the playfield
            if (!new_game.makeMove(move) or
                new_game.pos.y >= max_height + new_game.current.minY())
            {
                continue;
            }
            if (seen.putGet(new_game.current, new_game.pos)) {
                continue;
            }

            // Branch out after movement
            try stack.append(
                TPiecePosition.pack(new_game.current, new_game.pos),
            );

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

    fn cmp(_: void, a: MoveNode, b: MoveNode) Order {
        return std.math.order(b.score, a.score);
    }
};
pub const MoveQueue = std.PriorityQueue(MoveNode, void, MoveNode.cmp);

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
) error{OutOfMemory}!void {
    var iter = moves.iterator(piece);
    while (iter.next()) |placement| {
        var board = playfield;
        board.place(.from(placement.piece), placement.pos);
        const cleared = board.clearLines(placement.pos.y);
        const new_height = max_height - cleared;
        if (!validFn(board, new_height)) {
            continue;
        }

        try queue.add(.{
            .placement = placement,
            .score = scoreFn(board, max_height, nn),
        });
    }
}

test allPlacements {
    var playfield: BoardMask = .{};
    playfield.mask |= @as(u64, 0b0111111110) << 30;
    playfield.mask |= @as(u64, 0b0010000000) << 20;
    playfield.mask |= @as(u64, 0b0000001000) << 10;
    playfield.mask |= @as(u64, 0b0000000001);

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
    playfield.mask |= @as(u64, 0b1111111110) << 20;
    playfield.mask |= @as(u64, 0b1111111110) << 10;
    playfield.mask |= @as(u64, 0b1111111100);

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

test "No placements 2" {
    var playfield: BoardMask = .{};
    playfield.mask |= @as(u64, 0b1111111110) << 30;
    playfield.mask |= @as(u64, 0b1111111100) << 20;
    playfield.mask |= @as(u64, 0b1111111000) << 10;
    playfield.mask |= @as(u64, 0b1111111100);

    const PIECE: PieceKind = .j;
    const placements = allPlacements(
        playfield,
        false,
        &engine.kicks.srsPlus,
        PIECE,
        4,
    );

    var iter = placements.iterator(PIECE);
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try expect(count == 0);
}

test "No placements 3" {
    var playfield: BoardMask = .{};
    playfield.mask |= @as(u64, 0b1111111110) << 20;
    playfield.mask |= @as(u64, 0b1111111100) << 10;
    playfield.mask |= @as(u64, 0b1111111101);

    const PIECE: PieceKind = .z;
    const placements = allPlacements(
        playfield,
        false,
        &engine.kicks.srsPlus,
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
