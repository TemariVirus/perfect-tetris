const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const engine = @import("engine");
const Facing = engine.pieces.Facing;
const KickFn = engine.kicks.KickFn;
const PieceKind = engine.pieces.PieceKind;
const Rotation = engine.kicks.Rotation;

const root = @import("root.zig");
const BoardMask = root.bit_masks.BoardMask;
const NN = root.NN;
const Placement = root.Placement;
const FindPcError = root.FindPcError;

const SearchNode = packed struct {
    playfield: u60,
    held: PieceKind,
};

pub const Options = struct {
    /// Allocator to use for scratch allocations.
    allocator: Allocator,
    /// The current state of the playfield.
    playfield: engine.bit_masks.BoardMask,
    /// The current piece queue. This is mutable for performance reasons, but
    /// the original state of the slice is restored upon function exit.
    ///
    /// The `getPieces` helper function extracts the queue from the game state
    /// into the correct format `[current, hold, next...]`.
    pieces: []PieceKind,
    /// Kick system to use.
    kicks: *const KickFn,
    /// Minimum height of the perfect clear to find.
    min_height: u7 = 0,
    /// NN to use as heuristic.
    nn: NN,
    /// That piece that must be in the hold slot at the end of the solution.
    /// Ignored if `null`.
    save_hold: ?PieceKind = null,
};

pub fn Generic(
    comptime movegen: anytype,
    comptime TBoardMask: type,
    comptime TSearchNode: type,
    comptime makeSearchNode: fn (TBoardMask, u7, PieceKind) TSearchNode,
    comptime validFn: anytype,
    comptime scoreFn: anytype,
    comptime height_limit: u7,
) type {
    return struct {
        pub fn findPc(
            board_mask: TBoardMask,
            options: Options,
            placements: []Placement,
        ) FindPcError![]Placement {
            const pc_info = root.minPcInfo(options.playfield) orelse
                return FindPcError.NoPcExists;
            var pieces_needed: u16 = pc_info.pieces_needed;
            var max_height: u7 = pc_info.height;
            const max_pieces = @min(placements.len, options.pieces.len);

            if (options.save_hold) |hold| {
                // Jump to until the first piece that matches the save hold
                const min_pieces: u64 = std.mem.indexOfScalar(PieceKind, options.pieces, hold) orelse
                    return FindPcError.ImpossibleSaveHold;
                if (pieces_needed < min_pieces) {
                    const extra = (min_pieces - pieces_needed + 4) / 5;
                    if (2 * extra + max_height > std.math.maxInt(u7)) {
                        return FindPcError.SolutionTooLong;
                    }
                    pieces_needed += @intCast(5 * extra);
                    max_height += @intCast(2 * extra);
                }
            }

            var cache: std.AutoHashMap(TSearchNode, void) = .init(options.allocator);
            defer cache.deinit();

            // Pre-allocate a queue for each placement
            const queues = try options.allocator.alloc(movegen.MoveQueue, max_pieces);
            defer options.allocator.free(queues);
            for (queues) |*q| {
                q.* = .init(options.allocator, {});
            }
            defer for (queues) |*q| q.deinit();
            const do_o_rotations = hasOKicks(options.kicks);

            // 20 is the lowest common multiple of the width of the playfield (10) and
            // the number of cells in a piece (4). 20 / 4 = 5 extra pieces for each
            // bigger perfect clear.
            while (pieces_needed <= max_pieces and max_height <= height_limit) : ({
                pieces_needed += 5;
                max_height = std.math.add(u7, max_height, 2) catch break;
            }) {
                if (max_height < options.min_height) {
                    continue;
                }

                if (try findPcInner(
                    board_mask,
                    options.pieces[0..@min(options.pieces.len, pieces_needed + 1)],
                    queues[0..pieces_needed],
                    placements[0..pieces_needed],
                    do_o_rotations,
                    options.kicks,
                    &cache,
                    options.nn,
                    max_height,
                    options.save_hold,
                )) {
                    return placements[0..pieces_needed];
                }

                cache.clearRetainingCapacity();
                for (queues) |*queue| {
                    queue.clearRetainingCapacity();
                }
            }

            return FindPcError.SolutionTooLong;
        }

        fn findPcInner(
            playfield: TBoardMask,
            pieces: []PieceKind,
            queues: []movegen.MoveQueue,
            placements: []Placement,
            do_o_rotations: bool,
            kicks: *const KickFn,
            cache: *std.AutoHashMap(TSearchNode, void),
            nn: NN,
            max_height: u7,
            save_hold: ?PieceKind,
        ) Allocator.Error!bool {
            assert(max_height <= height_limit);
            // Base case; check for perfect clear
            if (placements.len == 0) {
                return max_height == 0;
            }

            const node: TSearchNode = makeSearchNode(playfield, max_height, pieces[0]);
            if ((try cache.getOrPut(node)).found_existing) {
                return false;
            }

            // Check if requested hold piece is in queue/hold
            const can_hold = if (save_hold) |hold| blk: {
                const idx = std.mem.lastIndexOfScalar(PieceKind, pieces, hold) orelse
                    return false;
                break :blk idx >= 2 or (pieces.len > 1 and pieces[0] == pieces[1]);
            } else true;

            var held_odd_times = false;
            // Unhold if held an odd number of times so that pieces are in the same order
            defer if (held_odd_times) {
                std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
            };

            // Check for forced hold
            if (!can_hold and pieces.len > 1 and pieces[1] != save_hold.?) {
                std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
                held_odd_times = !held_odd_times;
            }

            // Add moves to queue
            queues[0].clearRetainingCapacity();
            const m1 = movegen.allPlacements(
                playfield,
                do_o_rotations,
                kicks,
                pieces[0],
                @intCast(max_height),
            );
            // TODO: Prune based on piece dependencies?
            try movegen.orderMoves(
                &queues[0],
                playfield,
                pieces[0],
                m1,
                @intCast(max_height),
                validFn,
                nn,
                scoreFn,
            );
            // Check for unique hold
            if (can_hold and pieces.len > 1 and pieces[0] != pieces[1]) {
                const m2 = movegen.allPlacements(
                    playfield,
                    do_o_rotations,
                    kicks,
                    pieces[1],
                    @intCast(max_height),
                );
                try movegen.orderMoves(
                    &queues[0],
                    playfield,
                    pieces[1],
                    m2,
                    @intCast(max_height),
                    validFn,
                    nn,
                    scoreFn,
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
                board.place(
                    switch (TBoardMask) {
                        engine.bit_masks.BoardMask => placement.piece.mask(),
                        else => .from(placement.piece),
                    },
                    placement.pos,
                );
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

            return false;
        }
    };
}

fn makeNode(playfield: BoardMask, max_height: u7, held: PieceKind) SearchNode {
    assert(max_height <= 6);
    return .{
        // Shift mask up to encode `max_height`
        .playfield = @truncate(playfield.mask <<
            @intCast((BoardMask.HEIGHT - max_height) * BoardMask.WIDTH)),
        .held = held,
    };
}

/// Finds a perfect clear with the least number of placements possible, and
/// returns the sequence of placements required to achieve it. The placements
/// are stored in `placements`.
///
/// Returns an error if a perfect clear is impossible for the given options, or
/// if the number of placements needed exceeds `placements.len`.
pub fn findPc(
    options: Options,
    placements: []Placement,
) FindPcError![]Placement {
    return Generic(
        root.movegen,
        BoardMask,
        SearchNode,
        makeNode,
        isPcPossible,
        orderScore,
        6,
    ).findPc(
        BoardMask.from(options.playfield),
        options,
        placements,
    );
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

const expect = std.testing.expect;
const GameState = engine.GameState;
const SevenBag = engine.bags.SevenBag;

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

    const pieces = try root.getPieces(SevenBag, allocator, gamestate, placements.len + 1);
    defer allocator.free(pieces);

    const solution = try findPc(.{
        .allocator = allocator,
        .playfield = gamestate.playfield,
        .pieces = pieces,
        .kicks = gamestate.kicks,
        .min_height = 4,
        .nn = nn,
        .save_hold = .s,
    }, placements);
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

    const pieces = try root.getPieces(SevenBag, allocator, gamestate, placements.len + 1);
    defer allocator.free(pieces);

    const solution = try findPc(.{
        .allocator = allocator,
        .playfield = gamestate.playfield,
        .pieces = pieces,
        .kicks = gamestate.kicks,
        .min_height = 6,
        .nn = nn,
        .save_hold = .s,
    }, placements);
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
