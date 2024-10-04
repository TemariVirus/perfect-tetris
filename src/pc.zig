const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const expect = std.testing.expect;

const engine = @import("engine");
const Facing = engine.pieces.Facing;
const GameState = engine.GameState;
const KickFn = engine.kicks.KickFn;
const PieceKind = engine.pieces.PieceKind;
const Rotation = engine.kicks.Rotation;
const SevenBag = engine.bags.SevenBag;

const root = @import("root.zig");
const BoardMask = root.bit_masks.BoardMask;
const movegen = root.movegen;
const NN = root.NN;
const PieceMask = root.bit_masks.PieceMask;
const Placement = root.Placement;

const SearchNode = packed struct {
    playfield: u40,
    held: PieceKind,
    height: u3,
};
// TODO: compare performance iwth AutoHashMap
const NodeSet = std.AutoArrayHashMap(SearchNode, void);

const FindPcError = root.FindPcError;

pub fn findPc(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    height: u3,
) ![][]Placement {
    if (height > 4) {
        return FindPcError.SolutionTooLong;
    }

    const playfield = BoardMask.from(game.playfield);
    const pc_info = root.minPcInfo(game.playfield) orelse
        return FindPcError.NoPcExists;
    if (pc_info.height > height or
        (height - pc_info.height) % 2 != 0)
    {
        return FindPcError.NoPcExists;
    }
    const pieces_needed = pc_info.pieces_needed + (height - pc_info.height) / 2 * 5;

    const pieces = try getPieces(BagType, allocator, game, pieces_needed + 1);
    defer allocator.free(pieces);

    const do_o_rotations = hasOKicks(game.kicks);

    var curr = NodeSet.init(allocator);
    defer curr.deinit();
    var next = NodeSet.init(allocator);
    defer next.deinit();

    // Root node
    try curr.put(.{
        .playfield = @truncate(playfield.mask),
        .held = pieces[0],
        .height = height,
    }, {});

    for (pieces[1..]) |current| {
        while (curr.popOrNull()) |kv| {
            const pf = BoardMask{ .mask = kv.key.playfield };
            const hold = kv.key.held;
            const h = kv.key.height;

            for ([_]PieceKind{ current, hold }) |p| {
                const moves = movegen.allPlacements(
                    pf,
                    do_o_rotations,
                    game.kicks,
                    p,
                    h,
                );

                var iter = moves.iterator(p);
                while (iter.next()) |placement| {
                    var board = pf;
                    board.place(PieceMask.from(placement.piece), placement.pos);
                    const cleared = board.clearLines(placement.pos.y);
                    const new_h = h - cleared;
                    if (!isPcPossible(board, new_h)) {
                        continue;
                    }

                    try next.put(.{
                        .playfield = @truncate(board.mask),
                        .held = if (p == hold) current else hold,
                        .height = new_h,
                    }, {});
                }

                if (current == hold) {
                    break;
                }
            }
        }

        std.mem.swap(NodeSet, &curr, &next);
        next.clearRetainingCapacity();
        std.debug.print("{}\n", .{curr.count()});
    }

    return &.{};
}

/// Extracts `pieces_count` pieces from the game state, in the format
/// [hold, current, next...].
pub fn getPieces(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    pieces_count: usize,
) ![]PieceKind {
    if (pieces_count == 0) {
        return &.{};
    }

    var pieces = try allocator.alloc(PieceKind, pieces_count);

    const start: usize = if (game.hold_kind) |hold| blk: {
        pieces[0] = hold;
        break :blk 1;
    } else 0;
    if (pieces_count <= start) {
        return pieces;
    }

    pieces[start] = game.current.kind;
    for (game.next_pieces, start + 1..) |piece, i| {
        if (i >= pieces.len) {
            break;
        }
        pieces[i] = piece;
    }

    // If next pieces are not enough, fill the rest from the bag
    var bag_copy = game.bag;
    for (@min(pieces.len, start + 1 + game.next_pieces.len)..pieces.len) |i| {
        pieces[i] = bag_copy.next();
    }

    return pieces;
}

/// Returns `true` if an O piece could be affected by kicks. Otherwise, `false`.
pub fn hasOKicks(kicks: *const KickFn) bool {
    // Check if 1st O kick could be something other than (0, 0)
    for (std.enums.values(Facing)) |facing| {
        for (std.enums.values(Rotation)) |rot| {
            const k = kicks(
                .{ .kind = .o, .facing = facing },
                rot,
            );
            if (k.len > 0 and (k[0].x != 0 or k[0].y != 0)) {
                return true;
            }
        }
    }
    return false;
}

/// A fast check to see if a perfect clear is possible by making sure every
/// empty "segment" of the playfield has a multiple of 4 cells. Assumes the
/// total number of empty cells is a multiple of 4.
pub fn isPcPossible(playfield: BoardMask, max_height: u3) bool {
    assert(playfield.mask >> (@as(u6, max_height) * BoardMask.WIDTH) == 0);
    assert((@as(u6, max_height) * BoardMask.WIDTH - @popCount(playfield.mask)) % 4 == 0);

    var walls: u64 = (1 << BoardMask.WIDTH) - 1;
    for (0..max_height) |y| {
        const row = playfield.row(@intCast(y));
        walls &= row | (row << 1);
    }
    walls &= walls ^ (walls << 1); // Reduce consecutive walls to 1 wide

    while (walls != 0) {
        // A mask of all the bits before the first wall
        var right_of_wall = root.bit_masks.lsb(walls) - 1;
        // Duplicate to all rows
        for (@min(1, max_height)..max_height) |_| {
            right_of_wall |= right_of_wall << BoardMask.WIDTH;
        }

        // Each "segment" separated by a wall must have a multiple of 4 empty cells,
        // as pieces can only be placed in one segment (each piece occupies 4 cells).
        // All of the other segments to the right are confirmed to have a
        // multiple of 4 empty cells, so it doesn't matter if we count them again.
        const empty_count = @popCount(~playfield.mask & right_of_wall);
        if (empty_count % 4 != 0) {
            return false;
        }

        // Clear lowest bit
        walls &= walls - 1;
    }

    // The remaining empty cells must also be a multiple of 4, so we don't need
    // to  check the leftmost segment
    return true;
}
