pub const bit_masks = @import("bit_masks.zig");
pub const movegen = @import("movegen.zig");
pub const next = @import("next.zig");
pub const pc = @import("pc.zig");

const std = @import("std");
const AnyReader = std.io.AnyReader;
const AnyWriter = std.io.AnyWriter;
const assert = std.debug.assert;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const Facing = engine.pieces.Facing;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;

pub const FindPcError = error{
    ImpossibleSaveHold,
    NoPcExists,
    SolutionTooLong,
};

pub const Placement = struct {
    piece: Piece,
    pos: Position,
};

pub const FixedBag = struct {
    pieces: []const PieceKind,
    index: usize = 0,

    pub fn init(seed: u64) FixedBag {
        _ = seed; // autofix
        return .{ .pieces = &.{} };
    }

    pub fn next(self: *FixedBag) PieceKind {
        if (self.index >= self.pieces.len) {
            return undefined;
        }

        defer self.index += 1;
        return self.pieces[self.index];
    }

    pub fn setSeed(self: *FixedBag, seed: u64) void {
        _ = seed; // autofix
        self.index = 0;
    }
};

pub const PCSolution = struct {
    /// The maximum length of a next sequence in a `.pc` file.
    pub const MAX_SEQ_LEN = 16;
    pub const NextArray = std.BoundedArray(PieceKind, MAX_SEQ_LEN);
    pub const PlacementArray = std.BoundedArray(Placement, MAX_SEQ_LEN - 1);

    next: NextArray = NextArray{},
    placements: PlacementArray = PlacementArray{},

    /// Reads the next perfect clear solution from the reader, assuming the
    /// reader is reading a `.pc` file. If the reader is at the end of the
    /// file, this function returns `null`.
    pub fn readOne(reader: AnyReader) !?PCSolution {
        const seq = reader.readInt(u48, .little) catch |e| {
            if (e == error.EndOfStream) {
                return null;
            }
            return e;
        };
        const holds = try reader.readInt(u16, .little);

        var solution = PCSolution{};
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

            const placement = try reader.readByte();
            const facing: Facing = @enumFromInt(@as(u2, @truncate(placement)));
            const piece = Piece{ .facing = facing, .kind = current };
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

    pub fn write(self: PCSolution, writer: AnyWriter) !void {
        var holds: u16 = 0;
        var hold: PieceKind = self.next.buffer[0];
        for (self.placements.slice(), 1..) |placement, i| {
            if (placement.piece.kind != self.next.buffer[i]) {
                holds |= @as(u16, 1) << @intCast(i - 1);
                hold = self.next.buffer[i];
            }
        }

        try writer.writeInt(u48, packNext(self.next), .little);
        try writer.writeInt(u16, holds, .little);

        for (self.placements.slice()) |placement| {
            const facing: u8 = @intFromEnum(placement.piece.facing);
            const canon_pos = placement.piece.canonicalPosition(placement.pos);
            const pos: u8 = @intCast(canon_pos.y * 10 + canon_pos.x);
            assert(pos < 60);
            const byte = facing | (pos << 2);
            try writer.writeByte(byte);
        }
    }

    /// Packs a next sequence into a u48.
    pub fn packNext(pieces: NextArray) u48 {
        var seq: u48 = 0;
        for (0..pieces.len) |i| {
            const p: u48 = @intFromEnum(pieces.buffer[i]);
            seq |= p << @intCast(3 * i);
        }
        // Fill remaining bits with 1s
        const shift = 3 * @as(u6, pieces.len);
        seq |= @truncate(~@as(u64, 0) << shift);
        return seq;
    }

    /// Unpacks a u48 into a next sequence.
    pub fn unpackNext(seq: u48) NextArray {
        var pieces = NextArray{};
        while (pieces.len < MAX_SEQ_LEN) {
            const shift = 3 * @as(u6, pieces.len);
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
    const empty_cells = BoardMask.WIDTH * height - bits_set;

    // Assumes that all pieces have 4 cells and that the playfield is 10 cells wide.
    // Thus, an odd number of empty cells means that a perfect clear is impossible.
    if (empty_cells % 2 == 1) {
        return null;
    }

    var pieces_needed = if (empty_cells % 4 == 2)
        // If the number of empty cells is not a multiple of 4, we need to fill
        // an extra so that it becomes a multiple of 4
        // 2 + 10 = 12 which is a multiple of 4
        (empty_cells + 10) / 4
    else
        empty_cells / 4;
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

test {
    std.testing.refAllDecls(@This());
}
