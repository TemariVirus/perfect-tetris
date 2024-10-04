const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const expect = std.testing.expect;

const engine = @import("engine");
const Facing = engine.pieces.Facing;
const GameState = engine.GameState;
const KickFn = engine.kicks.KickFn;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;
const Rotation = engine.kicks.Rotation;
const SevenBag = engine.bags.SevenBag;

const root = @import("root.zig");
const BoardMask = root.bit_masks.BoardMask;
const movegen = root.movegen;
const PieceMask = root.bit_masks.PieceMask;
const Placement = root.Placement;

const Node = packed struct {
    playfield: u40,
    held: PieceKind,
    height: u3,
};
const NodeParents = std.ArrayListUnmanaged(struct { Node, Placement });
const Graph = std.AutoHashMap(Node, NodeParents);

const ColorArray = std.PackedIntArray(u4, 40);

const FindPcError = root.FindPcError;
pub fn findAllPcs(
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

    var arena = std.heap.ArenaAllocator.init(allocator);
    const arena_alloc = arena.allocator();
    defer arena.deinit();

    const pieces_needed = pc_info.pieces_needed + (height - pc_info.height) / 2 * 5;
    const pieces = try getPieces(BagType, arena_alloc, game, pieces_needed + 1);
    const graph, const leaves = try generateGraph(
        arena_alloc,
        @truncate(playfield.mask),
        pieces,
        height,
        game.kicks,
    );

    return try getSolutions(allocator, graph, leaves);
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
            const ks = kicks(
                .{ .kind = .o, .facing = facing },
                rot,
            );
            if (ks.len > 0 and (ks[0].x != 0 or ks[0].y != 0)) {
                return true;
            }
        }
    }
    return false;
}

pub fn generateGraph(
    allocator: Allocator,
    root_mask: u40,
    pieces: []PieceKind,
    max_height: u3,
    kicks: *const KickFn,
) !struct { Graph, []Node } {
    var arena = std.heap.ArenaAllocator.init(allocator);
    const arena_alloc = arena.allocator();
    errdefer arena.deinit();

    var graph = Graph.init(arena_alloc);
    var curr = std.ArrayList(Node).init(arena_alloc);
    var next = std.ArrayList(Node).init(allocator);
    defer next.deinit();

    // Root node
    {
        const n = Node{
            .playfield = root_mask,
            .held = pieces[0],
            .height = max_height,
        };
        try curr.append(n);
        try graph.putNoClobber(n, NodeParents{});
    }

    const do_o_rotations = hasOKicks(kicks);
    for (pieces[1..], 1..) |current, i| {
        _ = i; // autofix
        for (curr.items) |node| {
            const pf = BoardMask{ .mask = node.playfield };
            const hold = node.held;
            const h = node.height;

            var is_leaf = true;
            for ([_]PieceKind{ current, hold }) |p| {
                const moves = movegen.allPlacements(
                    pf,
                    do_o_rotations,
                    kicks,
                    p,
                    h,
                );

                var iter = moves.iterator(p);
                while (iter.next()) |placement| {
                    var new_pf = pf;
                    new_pf.place(PieceMask.from(placement.piece), placement.pos);
                    const cleared = new_pf.clearLines(placement.pos.y);
                    const new_h = h - cleared;
                    if (!isPcPossible(new_pf, new_h)) {
                        continue;
                    }

                    const n = Node{
                        .playfield = @truncate(new_pf.mask),
                        .held = if (p == hold) current else hold,
                        .height = new_h,
                    };
                    const result = try graph.getOrPut(n);
                    if (!result.found_existing) {
                        try next.append(n);
                        result.value_ptr.* = NodeParents{};
                    }
                    try result.value_ptr.append(arena_alloc, .{
                        node,
                        placement,
                    });
                    is_leaf = false;
                }

                if (current == hold) {
                    break;
                }
            }

            // Remove unnecessary leaf nodes to save RAM
            if (is_leaf) {
                assert(graph.remove(node));
            }
        }

        std.mem.swap(std.ArrayList(Node), &curr, &next);
        next.clearRetainingCapacity();
    }

    return .{ graph, try curr.toOwnedSlice() };
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

pub fn getSolutions(allocator: Allocator, graph: Graph, leaves: []Node) ![][]Placement {
    if (leaves.len == 0) {
        return &.{};
    }

    var solutions = std.ArrayList([]Placement).init(allocator);
    errdefer {
        for (solutions.items) |solution| {
            allocator.free(solution);
        }
        solutions.deinit();
    }

    const solution_len = blk: {
        var l: u8 = 0;
        var parents = graph.get(leaves[0]).?.items;
        while (parents.len > 0) : (l += 1) {
            parents = graph.get(parents[0][0]).?.items;
        }
        break :blk l;
    };
    const sol = try allocator.alloc(Placement, solution_len);
    defer allocator.free(sol);

    var stack = std.ArrayList(struct { Node, Placement, u8 }).init(allocator);
    defer stack.deinit();
    for (leaves) |node| {
        for (graph.get(node).?.items) |item| {
            try stack.append(.{
                item[0],
                item[1],
                1,
            });
        }
    }

    // Get all paths that lead to the root to extract solutions
    var seen = std.AutoHashMap(ColorArray, void).init(allocator);
    defer seen.deinit();
    while (stack.popOrNull()) |kv| {
        const node = kv[0];
        const placement = kv[1];
        const d = kv[2];

        for (graph.get(node).?.items) |item| {
            try stack.append(.{
                item[0],
                item[1],
                d + 1,
            });
        }

        sol[sol.len - d] = placement;
        // Check if we reached the root
        if (d == sol.len) {
            const board = drawMatrix(sol);
            if ((try seen.getOrPut(board)).found_existing) {
                continue;
            }
            try solutions.append(try allocator.dupe(Placement, sol));
        }
    }

    return try solutions.toOwnedSlice();
}

/// Get the positions of the minos of a piece relative to the bottom left
/// corner.
fn getMinos(piece: Piece) [4]Position {
    const mask = piece.mask().rows;
    var minos: [4]Position = undefined;
    var i: usize = 0;

    // Make sure minos are sorted highest first
    var y: i8 = 3;
    while (y >= 0) : (y -= 1) {
        for (0..10) |x| {
            if ((mask[@intCast(y)] >> @intCast(10 - x)) & 1 == 1) {
                minos[i] = .{ .x = @intCast(x), .y = y };
                i += 1;
            }
        }
    }
    assert(i == 4);

    return minos;
}

fn drawMatrixPiece(
    board: *ColorArray,
    row_occupancy: []u8,
    piece: Piece,
    pos: Position,
) void {
    const minos = getMinos(piece);
    for (minos) |mino| {
        const cleared = blk: {
            var cleared: i8 = 0;
            var top = pos.y + mino.y;
            var i: usize = 0;
            // Any clears below the mino will push it up.
            while (i <= top) : (i += 1) {
                if (row_occupancy[i] >= 10) {
                    cleared += 1;
                    top += 1;
                }
            }
            break :blk cleared;
        };

        const mino_x: u8 = @intCast(pos.x + mino.x);
        // The y coordinate is flipped when converting to nterm coordinates.
        const mino_y: u8 = @intCast(pos.y + mino.y + cleared);
        board.set(mino_y * 10 + mino_x, @intFromEnum(piece.kind));

        row_occupancy[mino_y] += 1;
    }
}

fn drawMatrix(placements: []const Placement) ColorArray {
    var board = ColorArray.initAllTo(7);
    var row_occupancy = [_]u8{0} ** 4;
    for (placements) |p| {
        drawMatrixPiece(&board, &row_occupancy, p.piece, p.pos);
    }
    return board;
}

test isPcPossible {
    var playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0111111110) << 30;
    playfield.mask |= @as(u64, 0b0010000000) << 20;
    playfield.mask |= @as(u64, 0b0000001000) << 10;
    playfield.mask |= @as(u64, 0b0000001001);
    try expect(isPcPossible(playfield, 4));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0000000000) << 30;
    playfield.mask |= @as(u64, 0b0010011000) << 20;
    playfield.mask |= @as(u64, 0b0000011000) << 10;
    playfield.mask |= @as(u64, 0b0000011001);
    try expect(isPcPossible(playfield, 4));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0010011100) << 20;
    playfield.mask |= @as(u64, 0b0000011000) << 10;
    playfield.mask |= @as(u64, 0b0000011011);
    try expect(!isPcPossible(playfield, 3));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0010010000) << 20;
    playfield.mask |= @as(u64, 0b0000001000) << 10;
    playfield.mask |= @as(u64, 0b0000001011);
    try expect(isPcPossible(playfield, 3));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0100011100) << 20;
    playfield.mask |= @as(u64, 0b0010001000) << 10;
    playfield.mask |= @as(u64, 0b0111111011);
    try expect(!isPcPossible(playfield, 3));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0100010000) << 20;
    playfield.mask |= @as(u64, 0b0010011000) << 10;
    playfield.mask |= @as(u64, 0b0100011011);
    try expect(isPcPossible(playfield, 3));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0100111000) << 20;
    playfield.mask |= @as(u64, 0b0011011100) << 10;
    playfield.mask |= @as(u64, 0b1100111000);
    try expect(!isPcPossible(playfield, 3));

    playfield = BoardMask{};
    playfield.mask |= @as(u64, 0b0100111000) << 20;
    playfield.mask |= @as(u64, 0b0011111100) << 10;
    playfield.mask |= @as(u64, 0b0100111000);
    try expect(isPcPossible(playfield, 3));
}
