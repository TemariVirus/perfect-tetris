const std = @import("std");
const assert = std.debug.assert;

const engine = @import("engine");
const BoardMaskEngine = engine.bit_masks.BoardMask;
const Facing = engine.pieces.Facing;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const PieceMaskEngine = engine.bit_masks.PieceMask;
const Position = engine.pieces.Position;

/// A 10 x 6 bit mask.
/// Coordinates start at (0, 0) at the bottom left corner.
/// X increases rightwards.
/// Y increases upwards.
pub const BoardMask = struct {
    pub const WIDTH = 10;
    pub const HEIGHT = 6;

    mask: u64 = 0,

    /// Create a new BoardMask from the engine representation.
    pub fn from(board_mask: BoardMaskEngine) BoardMask {
        var mask: u64 = 0;
        for (0..6) |y| {
            mask |= @as(
                u64,
                board_mask.rows[y] & ~BoardMaskEngine.EMPTY_ROW,
            ) << @intCast(y * 10) >> 1;
        }
        // Make sure the space outside the board mask is empty
        for (6..BoardMaskEngine.HEIGHT) |y| {
            assert(board_mask.rows[y] == BoardMaskEngine.EMPTY_ROW);
        }
        return .{ .mask = mask };
    }

    /// Get the bits of a row.
    pub fn row(self: BoardMask, y: u3) u10 {
        assert(y >= 0 and y < HEIGHT);
        return @truncate(self.mask >> (@as(u6, y) * 10));
    }

    /// Convert a x, y coordinate to a bit position in the mask.
    pub fn getShift(pos: Position) i8 {
        return @min(63, pos.y * WIDTH - pos.x);
    }

    /// Check if a piece would collide with the board mask at a given position.
    pub fn collides(self: BoardMask, piece: Piece, pos: Position) bool {
        assert(pos.x >= piece.minX());
        assert(pos.x <= piece.maxX());
        assert(pos.y >= piece.minY());

        const shift = getShift(pos);
        if (shift > 0) {
            return self.mask &
                (PieceMask.from(piece).mask << @intCast(shift)) != 0;
        }
        return self.mask &
            (PieceMask.from(piece).mask >> @intCast(-shift)) != 0;
    }

    /// Place a piece on the board mask at a given position.
    pub fn place(self: *BoardMask, piece: PieceMask, pos: Position) void {
        const shift = getShift(pos);
        self.mask |= if (shift > 0)
            piece.mask << @intCast(shift)
        else
            piece.mask >> @intCast(-shift);
    }

    /// Clears all filled lines in the playfield.
    /// Returns the number of lines cleared.
    pub fn clearLines(self: *BoardMask, y: i8) u3 {
        var cleared: u3 = 0;
        var i: usize = @max(0, y);
        var full_row = comptime (@as(u64, 1) << BoardMask.WIDTH) - 1;
        full_row = full_row << @intCast(i * WIDTH);
        while (i + cleared < HEIGHT) {
            if (self.mask & full_row == full_row) {
                cleared += 1;
                const bottom_mask = lsb(full_row) - 1;
                const bottom = self.mask & bottom_mask;
                const top = (self.mask >> WIDTH) & ~bottom_mask;
                self.mask = top | bottom;
            } else {
                i += 1;
                full_row <<= 10;
            }
        }
        return cleared;
    }

    /// Returns the checkerboard parity of playfield.
    /// https://harddrop.com/wiki/Parity#Perfect_Clears
    pub fn checkerboardParity(self: BoardMask) u5 {
        const mask1 = comptime blk: {
            var mask: u64 = 0;
            for (0..HEIGHT) |i| {
                const row_mask = if (i % 2 == 0)
                    0b1010101010
                else
                    0b0101010101;
                mask |= row_mask << (i * 10);
            }
            break :blk mask;
        };
        const mask2 = comptime (~mask1 << 4 >> 4); // Ignore partial 7th row

        const count1 = @popCount(self.mask & mask1);
        const count2 = @popCount(self.mask & mask2);
        return @intCast(@max(count1, count2) - @min(count1, count2));
    }

    /// Returns the column parity of playfield.
    pub fn columnParity(self: BoardMask) u5 {
        const mask1 = comptime blk: {
            var mask: u64 = 0;
            for (0..HEIGHT) |i| {
                mask |= 0b0101010101 << (i * 10);
            }
            break :blk mask;
        };
        const mask2 = comptime (~mask1 << 4 >> 4); // Ignore partial 7th row

        const count1 = @popCount(self.mask & mask1);
        const count2 = @popCount(self.mask & mask2);
        return @intCast(@max(count1, count2) - @min(count1, count2));
    }
};

/// A 10 x 4 bit mask.
/// Coordinates start at (0, 0) at the bottom left corner.
/// X increases rightwards.
/// Y increases upwards.
pub const PieceMask = struct {
    pub const WIDTH = 10;
    pub const HEIGHT = 4;

    mask: u64,

    /// Gets the PieceMask for a given piece.
    pub fn from(piece: Piece) PieceMask {
        const table = comptime makeAttributeTable(PieceMask, fromRaw);
        return table[@as(u5, @bitCast(piece))];
    }

    fn fromRaw(piece: Piece) PieceMask {
        return fromMask(piece.mask());
    }

    /// Create a new PieceMask from the engine representation.
    pub fn fromMask(piece_mask: PieceMaskEngine) PieceMask {
        var mask: u64 = 0;
        for (0..4) |i| {
            mask |= @as(u64, (piece_mask.rows[i] >> 1)) << (i * 10);
        }
        return .{ .mask = mask };
    }

    /// Returns `true` if the 4 minos of `self` and `other` are in the same positions;
    /// otherwise, `false`.
    pub fn minosEqual(
        self: PieceMask,
        self_pos: Position,
        other: PieceMask,
        other_pos: Position,
    ) bool {
        assert(@popCount(self.mask) == 4 and @popCount(other.mask) == 4);

        const self_offset = self_pos.y * WIDTH - self_pos.x;
        const other_offset = other_pos.y * WIDTH - other_pos.x;
        const offset = self_offset - other_offset;
        // 64 bits - 4 minos = 60 is the maximum offset that can return true
        if (@abs(offset) > 60) {
            return false;
        }

        return std.math.shr(u64, self.mask, offset) == other.mask;
    }

    fn makeAttributeTable(
        comptime T: type,
        comptime attribute: fn (Piece) T,
    ) [28]T {
        var table: [28]T = undefined;
        for (@typeInfo(PieceKind).@"enum".fields) |p| {
            for (@typeInfo(Facing).@"enum".fields) |f| {
                const piece = Piece{
                    .facing = @enumFromInt(f.value),
                    .kind = @enumFromInt(p.value),
                };
                table[@as(u5, @bitCast(piece))] = attribute(piece);
            }
        }
        return table;
    }
};

/// Returns a mask containing only the least significant bit of x.
pub fn lsb(x: u64) u64 {
    return @as(u64, @bitCast(@as(i64, @bitCast(x)) & -@as(i64, @bitCast(x))));
}
