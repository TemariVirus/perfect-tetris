const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const expect = std.testing.expect;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const GameState = engine.GameState;
const KickFn = engine.kicks.KickFn;
const PieceKind = engine.pieces.PieceKind;
const SevenBag = engine.bags.SevenBag;

const root = @import("../root.zig");
const movegen = @import("movegen.zig");
const NN = root.NN;
const Placement = root.Placement;

const SearchNode = struct {
    rows: [23]u16,
    height: u7,
    held: PieceKind,
};
const NodeSet = std.AutoHashMap(SearchNode, void);

const FindPcError = root.FindPcError;

pub fn findPc(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    nn: NN,
    min_height: u7,
    placements: []Placement,
    save_hold: ?PieceKind,
) FindPcError![]Placement {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const pc_info = root.minPcInfo(game.playfield) orelse
        return FindPcError.NoPcExists;
    var pieces_needed: u16 = pc_info.pieces_needed;
    var max_height: u7 = pc_info.height;

    const pieces = try root.pc.getPieces(
        BagType,
        arena_allocator,
        game,
        placements.len + 1,
    );

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

    const do_o_rotations = root.pc.hasOKicks(game.kicks);

    // 20 is the lowest common multiple of the width of the playfield (10) and
    // the number of cells in a piece (4). 20 / 4 = 5 extra pieces for each
    // bigger perfect clear.
    while (pieces_needed <= placements.len) {
        defer pieces_needed += 5;
        defer max_height += 2;

        if (max_height < min_height) {
            continue;
        }

        if (try findPcInner(
            game.playfield,
            pieces[0 .. pieces_needed + 1],
            queues[0..pieces_needed],
            placements[0..pieces_needed],
            do_o_rotations,
            game.kicks,
            &cache,
            nn,
            max_height,
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

fn findPcInner(
    playfield: BoardMask,
    pieces: []PieceKind,
    queues: []movegen.MoveQueue,
    placements: []Placement,
    do_o_rotations: bool,
    kicks: *const KickFn,
    cache: *NodeSet,
    nn: NN,
    max_height: u7,
    save_hold: ?PieceKind,
) !bool {
    // Base case; check for perfect clear
    if (placements.len == 0) {
        return max_height == 0;
    }

    const node: SearchNode = .{
        .rows = playfield.rows[0..23].*,
        .height = max_height,
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
    const m1 = try movegen.allPlacements(
        playfield,
        do_o_rotations,
        kicks,
        pieces[0],
        max_height,
    );
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
        const m2 = try movegen.allPlacements(
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
        board.place(placement.piece.mask(), placement.pos);
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

fn isPcPossible(rows: []const u16) bool {
    var walls = ~BoardMask.EMPTY_ROW;
    for (rows) |row| {
        walls &= row | (row << 1);
    }
    walls &= walls ^ (walls >> 1); // Reduce consecutive walls to 1 wide walls

    while (walls != 0) {
        const old_walls = walls;
        walls &= walls - 1; // Clear lowest bit
        // A mask of all the bits before the removed wall
        const right_of_wall = (walls ^ old_walls) - 1;

        // Each "segment" separated by a wall must have a multiple of 4 empty cells,
        // as pieces can only be placed in one segment (each piece occupies 4 cells).
        var empty_count: u16 = 0;
        for (rows) |row| {
            // All of the other segments to the right are confirmed to have a
            // multiple of 4 empty cells, so it doesn't matter if we count them again.
            const segment = ~row & right_of_wall;
            empty_count += @popCount(segment);
        }
        if (empty_count % 4 != 0) {
            return false;
        }
    }

    return true;
}

fn orderScore(rows: []const u16, nn: NN) f32 {
    const features = getFeatures(rows, nn.inputs_used);
    return nn.predict(features);
}

pub fn getFeatures(
    rows: []const u16,
    inputs_used: [NN.INPUT_COUNT]bool,
) [NN.INPUT_COUNT]f32 {
    // Find highest block in each column. Heights start from 0
    var heights: [10]i32 = undefined;
    var highest: usize = 0;
    for (0..10) |x| {
        var height = rows.len;
        const col_mask = @as(u16, 1) << @intCast(10 - x);
        while (height > 0) {
            height -= 1;
            if ((rows[height] & col_mask) != 0) {
                height += 1;
                break;
            }
        }
        heights[9 - x] = @intCast(height);
        highest = @max(highest, height);
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
            var covered = ~rows[y] & rows[y + 1];
            covered >>= 1; // Remove padding
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
    const row_trans: f32 = if (inputs_used[3]) blk: {
        var row_trans: u32 = 0;
        for (0..highest) |y| {
            const row = rows[y];
            const trasitions = (row ^ (row << 1)) & 0b00000_111111111_00;
            row_trans += @popCount(trasitions);
        }
        break :blk @floatFromInt(row_trans);
    } else undefined;

    // Column trasitions
    const col_trans: f32 = if (inputs_used[4]) blk: {
        var col_trans: u32 = if (rows.len > 0)
            @popCount(rows[@max(1, highest) - 1] & ~BoardMask.EMPTY_ROW)
        else
            0;
        for (0..@max(1, highest) - 1) |y| {
            col_trans += @popCount(rows[y] ^ rows[y + 1]);
        }
        break :blk @floatFromInt(col_trans);
    } else undefined;

    // Empty cells
    const empty_cells: f32 = if (inputs_used[6]) blk: {
        var pop_count: u32 = 0;
        for (0..highest) |y| {
            pop_count += @popCount(rows[y] & ~BoardMask.EMPTY_ROW);
        }
        break :blk @floatFromInt(rows.len * 10 - pop_count);
    } else undefined;

    // Checkerboard parity
    const checkerboard_parity: f32 = if (inputs_used[7]) blk: {
        const mask1 = 0b1010101010 << 1;
        const mask2 = 0b0101010101 << 1;

        var count1: u32 = 0;
        var count2: u32 = 0;
        for (0..highest) |y| {
            if (y % 2 == 0) {
                count1 += @popCount(rows[y] & mask1);
                count2 += @popCount(rows[y] & mask2);
            } else {
                count1 += @popCount(rows[y] & mask2);
                count2 += @popCount(rows[y] & mask1);
            }
        }
        break :blk @floatFromInt(@max(count1, count2) - @min(count1, count2));
    } else undefined;

    // Column parity
    const column_parity: f32 = if (inputs_used[8]) blk: {
        const mask1 = 0b1010101010 << 1;
        const mask2 = 0b0101010101 << 1;

        var count1: u32 = 0;
        var count2: u32 = 0;
        for (0..highest) |y| {
            count1 += @popCount(rows[y] & mask1);
            count2 += @popCount(rows[y] & mask2);
        }
        break :blk @floatFromInt(@max(count1, count2) - @min(count1, count2));
    } else undefined;

    return .{
        std_h,
        caves,
        pillars,
        row_trans,
        col_trans,
        // Max height
        @floatFromInt(rows.len),
        empty_cells,
        checkerboard_parity,
        column_parity,
    };
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

test isPcPossible {
    var playfield: BoardMask = .{};
    playfield.rows[3] = (0b0111111110 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[2] = (0b0010000000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0000001000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0000001001 << 1) | BoardMask.EMPTY_ROW;
    try expect(isPcPossible(playfield.rows[0..4]));

    playfield = .{};
    playfield.rows[3] = (0b0000000000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[2] = (0b0010011000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0000011000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0000011001 << 1) | BoardMask.EMPTY_ROW;
    try expect(isPcPossible(playfield.rows[0..4]));

    playfield = .{};
    playfield.rows[2] = (0b0010011100 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0000011000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0000011011 << 1) | BoardMask.EMPTY_ROW;
    try expect(!isPcPossible(playfield.rows[0..3]));

    playfield = .{};
    playfield.rows[2] = (0b0010010000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0000001000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0000001011 << 1) | BoardMask.EMPTY_ROW;
    try expect(isPcPossible(playfield.rows[0..3]));

    playfield = .{};
    playfield.rows[2] = (0b0100011100 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0010001000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0111111011 << 1) | BoardMask.EMPTY_ROW;
    try expect(!isPcPossible(playfield.rows[0..3]));

    playfield = .{};
    playfield.rows[2] = (0b0100010000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0010011000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0100011011 << 1) | BoardMask.EMPTY_ROW;
    try expect(isPcPossible(playfield.rows[0..3]));

    playfield = .{};
    playfield.rows[2] = (0b0100111000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0011011100 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b1100111000 << 1) | BoardMask.EMPTY_ROW;
    try expect(!isPcPossible(playfield.rows[0..3]));

    playfield = .{};
    playfield.rows[2] = (0b0100111000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0011111100 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b0100111000 << 1) | BoardMask.EMPTY_ROW;
    try expect(isPcPossible(playfield.rows[0..3]));
}

test getFeatures {
    var playfield: BoardMask = .{};
    playfield.rows[5] = (0b0000000100 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[4] = (0b0000100000 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[3] = (0b0011010001 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[2] = (0b1000000001 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[1] = (0b0010000001 << 1) | BoardMask.EMPTY_ROW;
    playfield.rows[0] = (0b1111111111 << 1) | BoardMask.EMPTY_ROW;
    const features = getFeatures(
        playfield.rows[0..6],
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
