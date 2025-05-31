const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const KickFn = engine.kicks.KickFn;
const Piece = engine.pieces.Piece;
const Rotation = engine.kicks.Rotation;

const root = @import("root.zig");
const collisionSet = root.movegen.collisionSet;
const Intermediate = root.movegen.Intermediate(CollisionSet);
const Move = root.movegen.Move;
const PieceMask = root.bit_masks.PieceMask;
const PiecePosition = root.movegen.PiecePosition;
const Placement = root.Placement;

const CollisionSet = @import("PiecePosSet.zig").PiecePosSet(.{ 10, 24, 4 });
const DistanceMap = @import("PiecePosSet.zig").PiecePosMap(.{ 10, 24, 4 }, u16);
const PlacementQueue = @import("ring_queue.zig").RingQueue(PiecePosition);

// 240 cells * 4 rotations - 1 = 959 moves at most
pub const Path = std.BoundedArray(Move, 959);

/// Finds the sequence of moves with the least key presses that brings a piece from
/// `from` to `to`, or to a position that is visually indistinguishable from `to`.
/// Consecutive drops are considered a single key press.
/// Returns `null` if no path exists. Asserts that `from.piece.kind == to.piece.kind`.
pub fn pathfind(
    playfield: BoardMask,
    kicks: *const KickFn,
    from: Placement,
    to: Placement,
) ?Path {
    assert(from.piece.kind == to.piece.kind);
    const piece_kind = from.piece.kind;
    const do_o_rotations = root.pc.hasOKicks(kicks);
    const height = playfield.height();

    var distances: DistanceMap = .{ .data = @splat(std.math.maxInt(u16)) };
    var queue_buf: [CollisionSet.len]PiecePosition = undefined;
    var queue: PlacementQueue = .{ .data = &queue_buf };
    const collision_set = collisionSet(
        CollisionSet,
        BoardMask,
        playfield,
        do_o_rotations,
        from.piece,
        height,
    );

    distances.put(from.piece, from.pos, 0);
    queue.enqueueAssumeCapacity(.pack(from.piece, from.pos));

    const end: Placement = outer: while (queue.dequeue()) |placement| {
        const piece, const pos = blk: {
            const temp = placement.unpack(piece_kind);
            break :blk .{ temp.piece, temp.pos };
        };
        const distance = distances.get(piece, pos) + 1;

        for (Move.moves) |move| {
            var new_game: Intermediate = .{
                .collision_set = &collision_set,
                .current = piece,
                .pos = pos,
                .do_o_rotations = do_o_rotations,
                .max_height = height,
                .kicks = kicks,
            };

            while (true) {
                // Skip if piece was unable to move
                if (!new_game.makeMove(move)) {
                    break;
                }

                // Stop if we reached the target
                if (PieceMask.minosEqual(.from(to.piece), to.pos, .from(new_game.current), new_game.pos)) {
                    distances.put(new_game.current, new_game.pos, distance);
                    break :outer .{ .piece = new_game.current, .pos = new_game.pos };
                }

                // Skip if the piece is too high or if the distance is higher than previously found
                if (new_game.pos.y - new_game.current.minY() >= DistanceMap.height or
                    distances.get(new_game.current, new_game.pos) <= distance)
                {
                    switch (move) {
                        .drop => continue,
                        else => break,
                    }
                }

                // Branch out after movement
                distances.put(new_game.current, new_game.pos, distance);
                queue.enqueue(.pack(new_game.current, new_game.pos)) catch unreachable;

                // Keep dropping as it's still a single key press
                if (move != .drop) {
                    break;
                }
            }
        }
    } else {
        return null;
    };

    // Reconstruct the path
    var path = Path.init(0) catch unreachable;
    var game: Intermediate = .{
        .collision_set = &collision_set,
        .current = end.piece,
        .pos = end.pos,
        .do_o_rotations = do_o_rotations,
        .max_height = height,
        .kicks = kicks,
    };
    while (game.current != from.piece or game.pos != from.pos) {
        const old_distance = distances.get(game.current, game.pos);
        var moves_it = std.mem.reverseIterator(Move.moves);
        while (moves_it.next()) |move| {
            if (!unmakeMove(&game, move, distances)) {
                continue;
            }
            if (old_distance - 1 == distances.get(game.current, game.pos) or
                // Drops are allowed to add 0 distance
                (move == .drop and old_distance == distances.get(game.current, game.pos)))
            {
                path.append(move) catch unreachable;
                break;
            }
            assert(game.makeMove(move));
        } else unreachable;
    }

    std.mem.reverse(Move, path.slice());
    return path;
}

fn unmakeMove(self: *Intermediate, move: Move, distances: DistanceMap) bool {
    return switch (move) {
        .left => self.makeMove(.right),
        .right => self.makeMove(.left),
        .rotate_cw => tryUnrotate(self, .quarter_cw, distances),
        .rotate_double => tryUnrotate(self, .half, distances),
        .rotate_ccw => tryUnrotate(self, .quarter_ccw, distances),
        .drop => blk: {
            if (self.pos.y >= self.current.maxY()) {
                break :blk false;
            }
            break :blk self.tryTranspose(.{
                .x = self.pos.x,
                .y = self.pos.y + 1,
            });
        },
    };
}

fn tryUnrotate(self: *Intermediate, rotation: Rotation, distances: DistanceMap) bool {
    if (self.current.kind == .o and !self.do_o_rotations) {
        return false;
    }

    const current = self.current;
    const pos = self.pos;
    const old_piece: Piece = .{
        .facing = self.current.facing.rotate(switch (rotation) {
            .quarter_cw => .quarter_ccw,
            .half => .half,
            .quarter_ccw => .quarter_cw,
        }),
        .kind = self.current.kind,
    };

    const distance = distances.get(current, pos);
    for (self.kicks(old_piece, rotation)) |kick| {
        const old_pos = pos.sub(kick);
        if (self.collides(old_piece, old_pos)) {
            continue;
        }
        if (distances.get(old_piece, old_pos) >= distance) {
            continue;
        }

        self.current = old_piece;
        self.pos = old_pos;
        _ = self.tryRotate(rotation);
        if (self.current == current and self.pos == pos) {
            self.current = old_piece;
            self.pos = old_pos;
            return true;
        }
    }

    self.current = current;
    self.pos = pos;
    return false;
}

test pathfind {
    var playfield: BoardMask = .{};
    playfield.rows[3] = 0b0111111110_0 | BoardMask.EMPTY_ROW;
    playfield.rows[2] = 0b0001000000_0 | BoardMask.EMPTY_ROW;
    playfield.rows[1] = 0b0000001000_0 | BoardMask.EMPTY_ROW;
    playfield.rows[0] = 0b0000001001_0 | BoardMask.EMPTY_ROW;
    try expect(pathfind(
        playfield,
        &engine.kicks.srs,
        .{
            .piece = .{ .kind = .t, .facing = .up },
            .pos = .{ .x = 4, .y = 3 },
        },
        .{
            .piece = .{ .kind = .t, .facing = .right },
            .pos = .{ .x = 3, .y = 0 },
        },
    ) == null);
    try testPathfind(
        playfield,
        &engine.kicks.srs,
        .{
            .piece = .{ .kind = .j, .facing = .up },
            .pos = .{ .x = 4, .y = 3 },
        },
        .{
            .piece = .{ .kind = .j, .facing = .right },
            .pos = .{ .x = 4, .y = 0 },
        },
    );
}

fn testPathfind(
    playfield: BoardMask,
    kicks: *const KickFn,
    from: Placement,
    to: Placement,
) !void {
    const GameState = engine.GameState(engine.bags.SevenBag);

    var path = pathfind(playfield, kicks, from, to).?;
    var game: GameState = .init(.init(0), kicks);
    game.playfield = playfield;
    game.current = from.piece;
    game.pos = from.pos;
    for (path.slice()) |move| {
        switch (move) {
            .left => _ = try expect(game.slide(-1) == 1),
            .right => _ = try expect(game.slide(1) == 1),
            .rotate_cw => try expect(game.rotate(.quarter_cw) != -1),
            .rotate_double => try expect(game.rotate(.half) != -1),
            .rotate_ccw => try expect(game.rotate(.quarter_ccw) != -1),
            .drop => try expect(game.drop(1) == 1),
        }
    }

    try expect(PieceMask.minosEqual(.from(to.piece), to.pos, .from(game.current), game.pos));
}
