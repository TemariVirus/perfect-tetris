pub const bit_masks = @import("bit_masks.zig");
pub const movegen = @import("movegen.zig");
pub const movegen_slow = @import("slow/movegen.zig");
pub const next = @import("next.zig");
pub const pathfind = @import("pathfind.zig");
pub const pc = @import("pc.zig");
pub const pc_slow = @import("slow/pc.zig");

const std = @import("std");
const Allocator = std.mem.Allocator;
const BoundedArray = @import("bounded_array").BoundedArray;
const assert = std.debug.assert;
const json = std.json;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const Facing = engine.pieces.Facing;
const GameState = engine.GameState;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;

const NNInner = @import("zmai").genetic.neat.NN;

var default_nn: ?NN = null;
const load_default_nn = (struct {
    var buffer: [1152]u8 = undefined;

    pub fn load() void {
        if (default_nn == null) {
            @branchHint(.cold);
            var fba: std.heap.FixedBufferAllocator = .init(&buffer);
            default_nn = loadNNFromStr(
                fba.allocator(),
                @embedFile("nn_4l_json"),
            ) catch unreachable;
        }
    }
}).load;
pub fn defaultNN(allocator: Allocator) Allocator.Error!NN {
    load_default_nn();
    return try default_nn.?.clone(allocator);
}

pub const Placement = struct {
    piece: Piece,
    pos: Position,
};

pub const NN = struct {
    pub const INPUT_COUNT = 9;

    net: NNInner,
    inputs_used: [INPUT_COUNT]bool,

    pub fn load(io: std.Io, allocator: Allocator, path: []const u8) !NN {
        var inputs_used: [INPUT_COUNT]bool = undefined;
        const nn: NNInner = try .load(io, allocator, path, &inputs_used);
        return .{
            .net = nn,
            .inputs_used = inputs_used,
        };
    }

    pub fn deinit(self: NN, allocator: Allocator) void {
        self.net.deinit(allocator);
    }

    pub fn predict(self: NN, input: [INPUT_COUNT]f32) f32 {
        var output: [1]f32 = undefined;
        self.net.predict(&input, &output);
        return output[0];
    }

    /// Creates a deep copy of the neural network.
    pub fn clone(self: NN, allocator: Allocator) Allocator.Error!NN {
        const nodes = try allocator.dupe(NNInner.Node, self.net.nodes);
        errdefer allocator.free(nodes);

        const connections = try allocator.dupe(
            NNInner.Connection,
            self.net.connections.items[0..self.net.connections.flatLen()],
        );
        errdefer allocator.free(connections);
        const connection_splits = try allocator.dupe(u32, self.net.connections.splits);
        errdefer allocator.free(connection_splits);

        return .{
            .net = .{
                .nodes = nodes,
                .connections = .{
                    .items = connections.ptr,
                    .splits = connection_splits,
                },
                .hidden_activation = self.net.hidden_activation,
                .output_activation = self.net.output_activation,
            },
            .inputs_used = self.inputs_used,
        };
    }
};

pub const FixedBag = struct {
    pieces: []const PieceKind,
    index: usize = 0,

    pub fn next(self: *FixedBag) PieceKind {
        defer self.index += 1;
        return self.pieces[self.index];
    }
};

pub const PCSolution = struct {
    /// The maximum length of a next sequence in a `.pc` file.
    pub const MAX_SEQ_LEN = 16;
    pub const NextArray = BoundedArray(PieceKind, MAX_SEQ_LEN);
    pub const PlacementArray = BoundedArray(Placement, MAX_SEQ_LEN - 1);

    next: NextArray = .{},
    placements: PlacementArray = .{},

    /// Reads the next perfect clear solution from the reader, assuming the
    /// reader is reading a `.pc` file. If the reader is at the end of the
    /// file, this function returns `null`.
    pub fn readOne(reader: *std.Io.Reader) !?PCSolution {
        const seq = reader.takeInt(u48, .little) catch |e| {
            if (e == error.EndOfStream) {
                return null;
            }
            return e;
        };
        const holds = try reader.takeInt(u16, .little);

        var solution: PCSolution = .{};
        solution.next = unpackNext(seq);
        if (solution.next.len == 0) {
            return solution;
        }

        var hold: PieceKind = solution.next.buffer[0];
        for (1..solution.next.len) |i| {
            // Check if the piece is held
            var current = solution.next.buffer[i];
            if ((holds >> @intCast(i - 1)) & 1 == 1) {
                const temp = hold;
                hold = current;
                current = temp;
            }

            const placement = try reader.takeByte();
            const facing: Facing = @enumFromInt(@as(u2, @truncate(placement)));
            const piece: Piece = .{ .facing = facing, .kind = current };
            const canon_pos = placement >> 2;
            const pos = piece.fromCanonicalPosition(.{
                .x = @intCast(canon_pos % 10),
                .y = @intCast(canon_pos / 10),
            });

            solution.placements.appendAssumeCapacity(.{
                .piece = piece,
                .pos = pos,
            });
        }

        return solution;
    }

    /// Packs a next sequence into a u48.
    pub fn packNext(pieces: NextArray) u48 {
        var seq: u48 = 0;
        for (0..pieces.len) |i| {
            const p: u48 = @intFromEnum(pieces.buffer[i]);
            seq |= p << @intCast(3 * i);
        }
        // Fill remaining bits with 1s
        const shift = 3 * @as(u6, @intCast(pieces.len));
        seq |= @truncate(~@as(u64, 0) << shift);
        return seq;
    }

    /// Unpacks a u48 into a next sequence.
    pub fn unpackNext(seq: u48) NextArray {
        var pieces: NextArray = .{};
        while (pieces.len < MAX_SEQ_LEN) {
            const shift = 3 * @as(u6, @intCast(pieces.len));
            const p: u3 = @truncate(seq >> @intCast(shift));
            // 0b111 is the sentinel value for the end of the sequence
            if (p == 0b111) {
                break;
            }
            pieces.appendAssumeCapacity(@enumFromInt(p));
        }
        return pieces;
    }
};

pub const FindPcError = error{
    ImpossibleSaveHold,
    NoPcExists,
    OutOfMemory,
    SolutionTooLong,
};

/// Returns the minimum possible height and number of pieces needed for a
/// perfect clear given the playfield. There is no guarantee that a perfect
/// clear exists at the returned height. Returns `null` if a perfect clear is
/// impossible.
pub fn minPcInfo(playfield: BoardMask) ?struct {
    height: u6,
    pieces_needed: u8,
} {
    var height: u32 = playfield.height();
    const bits_set = blk: {
        var set: usize = 0;
        for (0..height) |i| {
            set += @popCount(playfield.rows[i] & ~BoardMask.EMPTY_ROW);
        }
        break :blk set;
    };
    var empty_cells = BoardMask.WIDTH * height - bits_set;

    // Assumes that all pieces have 4 cells and that the playfield is 10 cells wide.
    // Thus, an odd number of empty cells means that a perfect clear is impossible.
    if (empty_cells % 2 == 1) {
        return null;
    }
    // If the number of empty cells is not a multiple of 4, we need to fill
    // an extra row so that it becomes a multiple of 4
    if (empty_cells % 4 == 2) {
        empty_cells += 10;
        height += 1;
    }

    var pieces_needed = empty_cells / 4;
    // Don't return an empty solution
    if (pieces_needed == 0) {
        pieces_needed = 5;
        height += 2;
    }

    return .{
        .height = @intCast(height),
        .pieces_needed = @intCast(pieces_needed),
    };
}

/// Extracts `pieces_count` pieces from the game state, in the format
/// [current, hold, next...].
pub fn getPieces(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    pieces_count: usize,
) Allocator.Error![]PieceKind {
    if (pieces_count == 0) {
        return &.{};
    }

    var pieces = try allocator.alloc(PieceKind, pieces_count);
    errdefer allocator.free(pieces);
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

/// Finds a perfect clear with the least number of placements possible for the
/// given game state, and returns the sequence of placements required to
/// achieve it.
///
/// `min_height` determines the minimum height of the perfect clear.
///
/// `max_lookahead` determines the maximum number of pieces to look ahead.
///
/// That piece that must be in the hold slot at the end of the solution.
/// Ignored if `null`.
///
/// Returns an error if a perfect clear is impossible for the given options, or
/// if the number of placements needed exceeds `max_len`.
pub fn findPcAuto(
    comptime BagType: type,
    allocator: Allocator,
    game: GameState(BagType),
    nn: ?NN,
    min_height: u7,
    max_lookahead: u9,
    save_hold: ?PieceKind,
) FindPcError![]Placement {
    const placements = try allocator.alloc(Placement, max_lookahead);
    errdefer allocator.free(placements);
    const height = @max(
        min_height,
        (minPcInfo(game.playfield) orelse return FindPcError.NoPcExists).height,
    );

    const resolved_nn = nn orelse try defaultNN(allocator);
    defer if (nn == null) resolved_nn.deinit(allocator);

    const pieces = try getPieces(BagType, allocator, game, max_lookahead);
    defer allocator.free(pieces);

    // Use fast pc if possible
    if (height <= 6) {
        if (pc.findPc(
            .{
                .allocator = allocator,
                .playfield = game.playfield,
                .pieces = pieces,
                .kicks = game.kicks,
                .min_height = height,
                .nn = resolved_nn,
                .save_hold = save_hold,
            },
            placements,
        )) |solution| {
            return try allocator.realloc(placements, solution.len);
        } else |err| if (err != FindPcError.SolutionTooLong) {
            return err;
        }
    }

    const solution = try pc_slow.findPc(
        .{
            .allocator = allocator,
            .playfield = game.playfield,
            .pieces = pieces,
            .kicks = game.kicks,
            .min_height = @max(7, height),
            .nn = resolved_nn,
            .save_hold = save_hold,
        },
        placements,
    );
    return try allocator.realloc(placements, solution.len);
}

fn loadNNFromStr(allocator: Allocator, json_str: []const u8) !NN {
    const obj = try json.parseFromSlice(
        NNInner.NNJson,
        allocator,
        json_str,
        .{ .allocate = .alloc_if_needed },
    );
    defer obj.deinit();

    var inputs_used: [NN.INPUT_COUNT]bool = undefined;
    const _nn = try NNInner.fromJson(allocator, obj.value, &inputs_used);
    return .{
        .net = _nn,
        .inputs_used = inputs_used,
    };
}

// Needed for tests to be reachable
const PiecePosSet = @import("PiecePosSet.zig");
const ring_queue = @import("ring_queue.zig");
const testing = std.testing;

test {
    std.testing.refAllDecls(@This());
}

test minPcInfo {
    var playfield: BoardMask = .{};
    var info = minPcInfo(playfield);
    try testing.expect(info != null);
    try testing.expectEqual(2, info.?.height);
    try testing.expectEqual(5, info.?.pieces_needed);

    playfield = .{};
    playfield.rows[0] |= 0b0000001000 << 1;
    info = minPcInfo(playfield);
    try testing.expect(info == null);

    playfield = .{};
    playfield.rows[1] |= 0b0110000010 << 1;
    playfield.rows[0] |= 0b1100000001 << 1;
    info = minPcInfo(playfield);
    try testing.expect(info != null);
    try testing.expectEqual(3, info.?.height);
    try testing.expectEqual(6, info.?.pieces_needed);
}
