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
    playfield: u60,
    held: PieceKind,
};
const NodeSet = std.AutoHashMap(SearchNode, void);

const FindPcError = root.FindPcError;

pub fn findPc(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    nn: NN,
    min_height: u3,
    placements: []Placement,
    save_hold: ?PieceKind,
) FindPcError![]Placement {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const playfield: BoardMask = .from(game.playfield);
    const pc_info = root.minPcInfo(game.playfield) orelse
        return FindPcError.NoPcExists;
    var pieces_needed = pc_info.pieces_needed;
    var max_height = pc_info.height;

    const pieces = try getPieces(BagType, arena_allocator, game, placements.len + 1);

    if (save_hold) |hold| {
        // Requested hold piece is not in queue/hold
        for (pieces) |p| {
            if (p == hold) {
                break;
            }
        } else {
            return FindPcError.ImpossibleSaveHold;
        }
    }

    var cache: NodeSet = .init(arena_allocator);

    // Pre-allocate a queue for each placement
    const queues = try arena_allocator.alloc(movegen.MoveQueue, placements.len);
    for (0..queues.len) |i| {
        queues[i] = .init(arena_allocator, {});
    }
    const do_o_rotations = hasOKicks(game.kicks);

    // 20 is the lowest common multiple of the width of the playfield (10) and
    // the number of cells in a piece (4). 20 / 4 = 5 extra pieces for each
    // bigger perfect clear.
    while (pieces_needed <= placements.len and max_height <= 6) {
        defer pieces_needed += 5;
        defer max_height += 2;

        if (max_height < min_height) {
            continue;
        }

        if (try findPcInner(
            playfield,
            pieces[0 .. pieces_needed + 1],
            queues[0..pieces_needed],
            placements[0..pieces_needed],
            do_o_rotations,
            game.kicks,
            &cache,
            nn,
            @intCast(max_height),
            save_hold,
        )) {
            return placements[0..pieces_needed];
        }

        // Clear cache and queues
        cache.clearRetainingCapacity();
        for (queues) |*queue| {
            queue.items.len = 0;
        }
    }

    return FindPcError.SolutionTooLong;
}

/// Extracts `pieces_count` pieces from the game state, in the format
/// [current, hold, next...].
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
    pieces[0] = game.current.kind;
    if (pieces_count == 1) {
        return pieces;
    }

    const start: usize = if (game.hold_kind) |hold| blk: {
        pieces[1] = hold;
        break :blk 2;
    } else 1;

    for (game.next_pieces, start..) |piece, i| {
        if (i >= pieces.len) {
            break;
        }
        pieces[i] = piece;
    }

    // If next pieces are not enough, fill the rest from the bag
    var bag_copy = game.bag;
    for (@min(pieces.len, start + game.next_pieces.len)..pieces.len) |i| {
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

fn findPcInner(
    playfield: BoardMask,
    pieces: []PieceKind,
    queues: []movegen.MoveQueue,
    placements: []Placement,
    do_o_rotations: bool,
    kicks: *const KickFn,
    cache: *NodeSet,
    nn: NN,
    max_height: u3,
    save_hold: ?PieceKind,
) !bool {
    // Base case; check for perfect clear
    if (placements.len == 0) {
        return max_height == 0;
    }

    const node: SearchNode = .{
        // Shift mask up to encode `max_height`
        .playfield = @truncate(playfield.mask <<
            (@as(u6, BoardMask.HEIGHT - max_height) * BoardMask.WIDTH)),
        .held = pieces[0],
    };
    if ((try cache.getOrPut(node)).found_existing) {
        return false;
    }

    // Check if requested hold piece is in queue/hold
    const can_hold = if (save_hold) |hold| blk: {
        const idx = std.mem.lastIndexOfScalar(PieceKind, pieces, hold) orelse
            return false;
        break :blk idx >= 2 or (pieces.len > 1 and pieces[0] == pieces[1]);
    } else true;

    // Check for forced hold
    var held_odd_times = false;
    if (!can_hold and pieces[1] != save_hold.?) {
        std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
        held_odd_times = !held_odd_times;
    }

    // Add moves to queue
    queues[0].items.len = 0;
    const m1 = movegen.allPlacements(
        playfield,
        do_o_rotations,
        kicks,
        pieces[0],
        max_height,
    );
    // TODO: Prune based on piece dependencies?
    try movegen.orderMoves(
        &queues[0],
        playfield,
        pieces[0],
        m1,
        max_height,
        isPcPossible,
        nn,
        orderScore,
    );
    // Check for unique hold
    if (can_hold and pieces.len > 1 and pieces[0] != pieces[1]) {
        const m2 = movegen.allPlacements(
            playfield,
            do_o_rotations,
            kicks,
            pieces[1],
            max_height,
        );
        try movegen.orderMoves(
            &queues[0],
            playfield,
            pieces[1],
            m2,
            max_height,
            isPcPossible,
            nn,
            orderScore,
        );
    }

    while (queues[0].removeOrNull()) |move| {
        const placement = move.placement;
        // Hold if needed
        if (placement.piece.kind != pieces[0]) {
            std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
            held_odd_times = !held_odd_times;
        }
        assert(pieces[0] == placement.piece.kind);

        var board = playfield;
        board.place(.from(placement.piece), placement.pos);
        const cleared = board.clearLines(placement.pos.y);

        const new_height = max_height - cleared;
        if (try findPcInner(
            board,
            pieces[1..],
            queues[1..],
            placements[1..],
            do_o_rotations,
            kicks,
            cache,
            nn,
            new_height,
            save_hold,
        )) {
            @branchHint(.unlikely);
            placements[0] = placement;
            return true;
        }
    }

    // Unhold if held an odd number of times so that pieces are in the same order
    if (held_odd_times) {
        std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
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
    // to check the leftmost segment
    return true;
}

pub fn getFeatures(
    playfield: BoardMask,
    max_height: u3,
    inputs_used: [NN.INPUT_COUNT]bool,
) [NN.INPUT_COUNT]f32 {
    // Find highest block in each column. Heights start from 0
    var column = comptime blk: {
        var column = @as(u64, 1);
        column |= column << BoardMask.WIDTH;
        column |= column << BoardMask.WIDTH;
        column |= column << (3 * BoardMask.WIDTH);
        break :blk column;
    };

    var heights: [10]i32 = undefined;
    var highest: u3 = 0;
    for (0..10) |x| {
        const height: u3 = @intCast(6 - ((@clz(playfield.mask & column) - 4) / 10));
        heights[x] = height;
        highest = @max(highest, height);
        column <<= 1;
    }

    // Standard height (sqrt of sum of squares of heights)
    const std_h = if (inputs_used[0]) blk: {
        var sqr_sum: i32 = 0;
        for (heights) |h| {
            sqr_sum += h * h;
        }
        break :blk @sqrt(@as(f32, @floatFromInt(sqr_sum)));
    } else undefined;

    // Caves (empty cells with an overhang)
    const caves: f32 = if (inputs_used[1]) blk: {
        const aug_heights = inner: {
            var aug_h: [10]i32 = undefined;
            aug_h[0] = @min(heights[0] - 2, heights[1]);
            for (1..9) |x| {
                aug_h[x] = @min(
                    heights[x] - 2,
                    @max(heights[x - 1], heights[x + 1]),
                );
            }
            aug_h[9] = @min(heights[9] - 2, heights[8]);
            break :inner aug_h;
        };

        var caves: i32 = 0;
        for (0..@max(1, highest) - 1) |y| {
            var covered = ~playfield.row(@intCast(y)) &
                playfield.row(@intCast(y + 1));
            // Iterate through set bits
            while (covered != 0) : (covered &= covered - 1) {
                const x = @ctz(covered);
                if (y <= aug_heights[x]) {
                    // Caves deeper down get larger values
                    caves += heights[x] - @as(i32, @intCast(y));
                }
            }
        }

        break :blk @floatFromInt(caves);
    } else undefined;

    // Pillars (sum of min differences in heights)
    const pillars: f32 = if (inputs_used[2]) blk: {
        var pillars: i32 = 0;
        for (0..10) |x| {
            // Columns at the sides map to 0 if they are taller
            var diff: i32 = switch (x) {
                0 => @max(0, heights[1] - heights[0]),
                1...8 => @intCast(@min(
                    @abs(heights[x - 1] - heights[x]),
                    @abs(heights[x + 1] - heights[x]),
                )),
                9 => @max(0, heights[8] - heights[9]),
                else => unreachable,
            };
            // Exaggerate large differences
            if (diff > 2) {
                diff *= diff;
            }
            pillars += diff;
        }
        break :blk @floatFromInt(pillars);
    } else undefined;

    // Row trasitions
    const row_mask = comptime blk: {
        var row_mask: u64 = 0b1111111110;
        row_mask |= row_mask << BoardMask.WIDTH;
        row_mask |= row_mask << BoardMask.WIDTH;
        row_mask |= row_mask << (3 * BoardMask.WIDTH);
        break :blk row_mask;
    };
    const row_trans: f32 = if (inputs_used[3])
        @floatFromInt(@popCount(
            (playfield.mask ^ (playfield.mask << 1)) & row_mask,
        ))
    else
        undefined;

    // Column trasitions
    const col_trans: f32 = if (inputs_used[4]) blk: {
        var col_trans: u32 = @popCount(playfield.row(@max(1, highest) - 1));
        for (0..@max(1, highest) - 1) |y| {
            col_trans += @popCount(playfield.row(@intCast(y)) ^
                playfield.row(@intCast(y + 1)));
        }
        break :blk @floatFromInt(col_trans);
    } else undefined;

    return .{
        std_h,
        caves,
        pillars,
        row_trans,
        col_trans,
        // Max height
        @floatFromInt(max_height),
        // Empty cells
        @floatFromInt(@as(u6, max_height) * 10 - @popCount(playfield.mask)),
        @floatFromInt(playfield.checkerboardParity()),
        @floatFromInt(playfield.columnParity()),
    };
}

fn orderScore(playfield: BoardMask, max_height: u3, nn: NN) f32 {
    const features = getFeatures(playfield, max_height, nn.inputs_used);
    return nn.predict(features);
}

test "4-line PC" {
    const allocator = std.testing.allocator;

    var gamestate: GameState(SevenBag) = .init(
        SevenBag.init(0),
        &engine.kicks.srsPlus,
    );

    const nn: NN = try .load(allocator, "NNs/Fast3.json");
    defer nn.deinit(allocator);

    const placements = try allocator.alloc(Placement, 10);
    defer allocator.free(placements);

    const solution = try findPc(
        SevenBag,
        allocator,
        gamestate,
        nn,
        4,
        placements,
        .s,
    );
    try expect(solution.len == 10);
    for (solution, 0..) |placement, i| {
        if (gamestate.current.kind != placement.piece.kind) {
            gamestate.hold();
        }
        try expect(gamestate.current.kind == placement.piece.kind);
        gamestate.current.facing = placement.piece.facing;

        gamestate.pos = placement.pos;
        try expect(gamestate.lockCurrent(-1).pc == (i + 1 == solution.len));
        gamestate.nextPiece();
    }

    try expect(gamestate.hold_kind == .s);
}

test "6-line PC" {
    const allocator = std.testing.allocator;

    var gamestate: GameState(SevenBag) = .init(
        SevenBag.init(0),
        &engine.kicks.srsPlus,
    );

    const nn: NN = try .load(allocator, "NNs/Fast3.json");
    defer nn.deinit(allocator);

    const placements = try allocator.alloc(Placement, 15);
    defer allocator.free(placements);

    const solution = try findPc(
        SevenBag,
        allocator,
        gamestate,
        nn,
        6,
        placements,
        .s,
    );
    try expect(solution.len == 15);
    for (solution, 0..) |placement, i| {
        if (gamestate.current.kind != placement.piece.kind) {
            gamestate.hold();
        }
        try expect(gamestate.current.kind == placement.piece.kind);
        gamestate.current.facing = placement.piece.facing;

        gamestate.pos = placement.pos;
        try expect(gamestate.lockCurrent(-1).pc == (i + 1 == solution.len));
        gamestate.nextPiece();
    }

    try expect(gamestate.hold_kind == .s);
}

test isPcPossible {
    var playfield: BoardMask = .{};
    playfield.mask |= @as(u64, 0b0111111110) << 30;
    playfield.mask |= @as(u64, 0b0010000000) << 20;
    playfield.mask |= @as(u64, 0b0000001000) << 10;
    playfield.mask |= @as(u64, 0b0000001001);
    try expect(isPcPossible(playfield, 4));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0000000000) << 30;
    playfield.mask |= @as(u64, 0b0010011000) << 20;
    playfield.mask |= @as(u64, 0b0000011000) << 10;
    playfield.mask |= @as(u64, 0b0000011001);
    try expect(isPcPossible(playfield, 4));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0010011100) << 20;
    playfield.mask |= @as(u64, 0b0000011000) << 10;
    playfield.mask |= @as(u64, 0b0000011011);
    try expect(!isPcPossible(playfield, 3));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0010010000) << 20;
    playfield.mask |= @as(u64, 0b0000001000) << 10;
    playfield.mask |= @as(u64, 0b0000001011);
    try expect(isPcPossible(playfield, 3));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0100011100) << 20;
    playfield.mask |= @as(u64, 0b0010001000) << 10;
    playfield.mask |= @as(u64, 0b0111111011);
    try expect(!isPcPossible(playfield, 3));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0100010000) << 20;
    playfield.mask |= @as(u64, 0b0010011000) << 10;
    playfield.mask |= @as(u64, 0b0100011011);
    try expect(isPcPossible(playfield, 3));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0100111000) << 20;
    playfield.mask |= @as(u64, 0b0011011100) << 10;
    playfield.mask |= @as(u64, 0b1100111000);
    try expect(!isPcPossible(playfield, 3));

    playfield = .{};
    playfield.mask |= @as(u64, 0b0100111000) << 20;
    playfield.mask |= @as(u64, 0b0011111100) << 10;
    playfield.mask |= @as(u64, 0b0100111000);
    try expect(isPcPossible(playfield, 3));
}

test getFeatures {
    const features = getFeatures(
        .{
            .mask = 0b0000000100 << (5 * BoardMask.WIDTH) |
                0b0000100000 << (4 * BoardMask.WIDTH) |
                0b0011010001 << (3 * BoardMask.WIDTH) |
                0b1000000001 << (2 * BoardMask.WIDTH) |
                0b0010000001 << (1 * BoardMask.WIDTH) |
                0b1111111111 << (0 * BoardMask.WIDTH),
        },
        6,
        @splat(true),
    );
    try expect(features.len == NN.INPUT_COUNT);
    try expect(features[0] == 11.7046995);
    try expect(features[1] == 10);
    try expect(features[2] == 47);
    try expect(features[3] == 14);
    try expect(features[4] == 22);
    try expect(features[5] == 6);
    try expect(features[6] == 40);
    try expect(features[7] == 4);
    try expect(features[8] == 2);
}
