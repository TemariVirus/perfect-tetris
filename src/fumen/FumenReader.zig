const std = @import("std");
const Allocator = mem.Allocator;
const AnyWriter = std.io.AnyWriter;
const assert = std.debug.assert;
const mem = std.mem;
const unicode = std.unicode;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const Color = ColorArray.PackedColor;
const ColorArray = engine.player.ColorArray;
const Facing = engine.pieces.Facing;
const GameState = engine.GameState(FixedBag);
const KickFn = engine.kicks.KickFn;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;

const FumenArgs = @import("root.zig").FumenArgs;
const Placement = @import("perfect-tetris").Placement;

const FumenReader = @This();
const NextArray = std.ArrayList(PieceKind);

const FIELD_WIDTH = 10;
const FIELD_HEIGHT = 24;
const FIELD_LEN = FIELD_WIDTH * FIELD_HEIGHT;

const b64_encode = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const b64_decode = blk: {
    var table: [256]?u6 = @splat(null);
    for (b64_encode, 0..) |char, i| {
        table[char] = i;
    }
    break :blk table;
};

const caption_encode = blk: {
    var table: [256]?u7 = @splat(null);
    for (caption_decode, 0..) |char, i| {
        table[char] = i;
    }
    break :blk table;
};
const caption_decode =
    \\ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
;

allocator: Allocator,
data: []const u8,
pos: usize = 0,
// Extra data kept accross pages for bookkeeping
field: [FIELD_LEN]FumenBlock = @splat(.empty),
field_repeat: u6 = 0,
hold: ?PieceKind = null,
current: ?PieceKind = null,
// Next pieces are stored in reverse order
next: NextArray,

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

const FumenBlock = enum(u8) {
    empty = 0,
    i = 1,
    l = 2,
    o = 3,
    z = 4,
    t = 5,
    j = 6,
    s = 7,
    garbage = 8,

    pub fn toEngine(self: FumenBlock) Color {
        return switch (self) {
            .empty => .empty,
            .i => PieceKind.color(.i),
            .l => PieceKind.color(.l),
            .o => PieceKind.color(.o),
            .z => PieceKind.color(.z),
            .t => PieceKind.color(.t),
            .j => PieceKind.color(.j),
            .s => PieceKind.color(.s),
            .garbage => .garbage,
        };
    }

    pub fn toPieceKind(self: FumenBlock) error{Invalid}!PieceKind {
        return switch (self) {
            .i => .i,
            .l => .l,
            .o => .o,
            .z => .z,
            .t => .t,
            .j => .j,
            .s => .s,
            else => return error.Invalid,
        };
    }

    pub fn fromPieceKind(self: PieceKind) FumenBlock {
        return switch (self) {
            .i => .i,
            .l => .l,
            .o => .o,
            .z => .z,
            .t => .t,
            .j => .j,
            .s => .s,
        };
    }
};

const FumenRotation = enum(u2) {
    south = 0,
    east = 1,
    north = 2,
    west = 3,

    pub fn toEngine(self: FumenRotation) Facing {
        return switch (self) {
            .south => .down,
            .east => .right,
            .north => .up,
            .west => .left,
        };
    }

    pub fn fromEngine(self: Facing) FumenRotation {
        return switch (self) {
            .down => .south,
            .right => .east,
            .up => .north,
            .left => .west,
        };
    }
};

pub const AllocOrFumenError = Allocator.Error || FumenError;
pub const FumenError = error{
    EndOfData,
    InvalidBlock,
    InvalidCaption,
    InvalidFieldLength,
    InvalidFieldRepeat,
    InvalidPieceLocation,
    InvalidPieceLetter,
    InvalidQuizCaption,
    InvalidQuizPiece,
    NotQuiz,
    UnsupportedFumenVersion,
};

pub const ParsedFumen = struct {
    reader: FumenReader,
    playfield: BoardMask,
    hold: ?PieceKind,
    next: []PieceKind,

    pub fn init(reader: *FumenReader) AllocOrFumenError!ParsedFumen {
        var playfield = BoardMask{};
        for (0..FIELD_HEIGHT - 1) |y| {
            for (0..FIELD_WIDTH) |x| {
                playfield.set(
                    x,
                    FIELD_HEIGHT - 2 - y,
                    reader.field[y * FIELD_WIDTH + x] != .empty,
                );
            }
        }

        if (reader.current) |current| {
            try reader.next.append(current);
        }
        mem.reverse(PieceKind, reader.next.items);
        if (reader.next.items.len == 0) {
            return FumenError.NotQuiz;
        }
        const next = try reader.next.toOwnedSlice();

        return .{
            .reader = reader.*,
            .playfield = playfield,
            .hold = reader.hold,
            .next = next,
        };
    }

    pub fn deinit(self: ParsedFumen, allocator: Allocator) void {
        self.reader.deinit();
        allocator.free(self.next);
    }

    pub fn toGameState(self: ParsedFumen, kicks: *const KickFn) GameState {
        var gamestate = GameState.init(
            FixedBag{ .pieces = self.next },
            kicks,
        );
        gamestate.playfield = self.playfield;
        gamestate.hold_kind = self.hold;
        return gamestate;
    }
};

const QuizWriter = struct {
    buf: [4]u8 = undefined,
    pos: usize = 0,

    fn writeRaw(self: *QuizWriter, writer: AnyWriter, char: u8) !void {
        assert(std.mem.indexOfScalar(u8, caption_decode, char) != null);
        if (self.pos == 4) {
            var v: u32 = 0;
            for (0..4) |i| {
                v *= 96;
                v += caption_encode[self.buf[3 - i]].?;
            }
            try writer.writeAll(&unpoll(5, v));
            self.pos = 0;
        }

        self.buf[self.pos] = char;
        self.pos += 1;
    }

    pub fn write(self: *QuizWriter, writer: AnyWriter, char: u8) !void {
        // Assumes char is part of a quiz
        assert(std.mem.indexOfScalar(u8, "#Q=[]()IOTLJSZ", char) != null);

        if (std.mem.indexOfScalar(u8, "#=[]()", char) == null) {
            try self.writeRaw(writer, char);
            return;
        }

        // Escape characters
        try self.writeRaw(writer, '%');
        for (std.fmt.bytesToHex([1]u8{char}, .upper)) |h| {
            try self.writeRaw(writer, h);
        }
    }

    pub fn writeAll(
        self: *QuizWriter,
        writer: AnyWriter,
        char: []const u8,
    ) !void {
        for (char) |c| {
            try self.write(writer, c);
        }
    }

    pub fn flush(self: *QuizWriter, writer: AnyWriter) !void {
        if (self.pos == 0) {
            return;
        }

        @memset(self.buf[self.pos..], caption_decode[0]);
        self.pos = 4;
        try self.writeRaw(writer, caption_decode[0]);
        self.pos = 0;
    }
};

pub fn parse(
    allocator: Allocator,
    fumen: []const u8,
) AllocOrFumenError!ParsedFumen {
    var reader: FumenReader = try .init(allocator, fumen);
    errdefer reader.deinit();

    while (!reader.done()) {
        try reader.readPage();
    }
    if (reader.field_repeat != 0) {
        return FumenError.InvalidFieldRepeat;
    }

    var parsed: ParsedFumen = try .init(&reader);
    parsed.reader.data = fumen;
    return parsed;
}

pub fn outputFumen(
    args: FumenArgs,
    parsed: ParsedFumen,
    solution: []const Placement,
    writer: AnyWriter,
) !void {
    // Initialise fumen
    const input = parsed.reader.data;
    const start = std.mem.lastIndexOf(u8, input, "115@") orelse
        unreachable; // Already checked in FumenReader.init
    const end = if (std.mem.indexOfScalar(u8, input[start..], '#')) |i|
        start + i
    else
        input.len;

    // Always write preceding url (if any)
    try writer.writeAll(input[0..start -| 1]);
    try writer.writeByte(args.@"output-type".toChr());
    if (args.append) {
        try writer.writeAll(input[start..end]);

        // Write first page
        try writer.writeAll(&unpoll(2, 9 * FIELD_LEN - 1));
        try writer.writeAll(&unpoll(1, 0));
        try writePieceAndFlags(writer, solution[0], false, false);
    } else {
        try writer.writeAll("115@");

        // Write field
        var block = parsed.reader.field[0];
        var len: u32 = 0;
        for (1..FIELD_LEN) |i| {
            if (parsed.reader.field[i] == block) {
                len += 1;
                continue;
            }

            try writer.writeAll(&unpoll(
                2,
                (@as(u32, @intFromEnum(block)) + 8) * FIELD_LEN + len,
            ));
            block = parsed.reader.field[i];
            len = 0;
        }
        try writer.writeAll(&unpoll(
            2,
            (@as(u32, @intFromEnum(block)) + 8) * FIELD_LEN + len,
        ));
        // Write field repeat if field is empty
        if (block == .empty and len == FIELD_LEN - 1) {
            try writer.writeAll(&unpoll(1, 0));
        }

        // Write first piece
        try writePieceAndFlags(writer, solution[0], true, true);
        // Write caption
        try writeQuiz(writer, parsed.reader.hold, parsed.next);
    }

    // Write solution
    for (solution[1..], 0..) |p, i| {
        // Assumes that the current fumen is a quiz, so field setting is
        // completely empty
        if (i % 64 == 0) {
            try writer.writeAll(&unpoll(2, 9 * FIELD_LEN - 1));
            try writer.writeAll(&unpoll(1, @min(64, solution.len - i)));
        }

        try writePieceAndFlags(writer, p, false, false);
    }

    if (args.append) {
        try writer.writeAll(input[end..]);
    }

    try writer.writeByte('\n');
}

fn init(allocator: Allocator, fumen: []const u8) FumenError!FumenReader {
    // Only version 1.15 is supported
    const start = mem.lastIndexOf(u8, fumen, "115@") orelse
        return FumenError.UnsupportedFumenVersion;
    // Fumen may be a url, remove addons (which come after a '#')
    const end = if (mem.indexOfScalar(u8, fumen[start..], '#')) |i|
        start + i
    else
        fumen.len;
    return .{
        .allocator = allocator,
        // + 4 to skip the "115@" prefix
        .data = fumen[start + 4 .. end],
        .next = .init(allocator),
    };
}

fn deinit(self: FumenReader) void {
    self.next.deinit();
}

fn done(self: *FumenReader) bool {
    const old_pos = self.pos;
    _ = self.pollOne() catch |e| if (e == FumenError.EndOfData) return true;
    self.pos = old_pos;
    return false;
}

fn pollOne(self: *FumenReader) FumenError!u6 {
    // Read until next valid character (fumens sometimes contain '?'s that
    // should be ignored)
    while (self.pos < self.data.len) : (self.pos += 1) {
        if (b64_decode[self.data[self.pos]]) |v| {
            self.pos += 1;
            return v;
        }
    }
    return FumenError.EndOfData;
}

fn poll(self: *FumenReader, n: usize) FumenError!u32 {
    assert(n <= 5);

    var result: u32 = 0;
    for (0..n) |i| {
        result += @as(u32, try self.pollOne()) << @intCast(6 * i);
    }
    return result;
}

fn unpoll(comptime n: usize, v: u32) [n]u8 {
    var result: [n]u8 = undefined;
    var _v = v;
    for (0..n) |i| {
        result[i] = b64_encode[_v % 64];
        _v /= 64;
    }

    assert(_v == 0);
    return result;
}

fn readField(self: *FumenReader) FumenError!void {
    if (self.field_repeat > 0) {
        self.field_repeat -= 1;
        return;
    }

    // If diff is empty, read the repeat count
    if (try self.poll(2) == 9 * self.field.len - 1) {
        self.field_repeat = try self.pollOne();
        return;
    }
    self.pos -= 2;

    // Read run-length encoded playfield
    var total: u32 = 0;
    while (total < self.field.len) {
        const run = try self.poll(2);

        const len = (run % FIELD_LEN) + 1;
        for (total..total + len) |i| {
            // Don't substract 8 yet to prevent underflow
            const block_temp = @intFromEnum(self.field[i]) + (run / FIELD_LEN);
            if (block_temp < 8 or block_temp > 16) {
                return FumenError.InvalidBlock;
            }
            self.field[i] = @enumFromInt(block_temp - 8);
        }
        total += len;
    }

    if (total != self.field.len) {
        return FumenError.InvalidFieldLength;
    }
}

fn boolFromInt(x: u1) bool {
    return switch (x) {
        0 => false,
        1 => true,
    };
}

fn readPieceAndFlags(self: *FumenReader) FumenError!struct {
    ?Piece,
    ?Position,
    struct {
        raise: bool,
        mirror: bool,
        has_caption: bool,
        lock: bool,
    },
} {
    var v = try self.poll(3);

    const piece: ?Piece = blk: {
        const block: FumenBlock = @enumFromInt(v % 8);
        v /= 8;
        const rotation: FumenRotation = @enumFromInt(v % 4);
        v /= 4;
        break :blk if (block == .empty)
            null
        else
            Piece{
                .facing = rotation.toEngine(),
                .kind = block.toPieceKind() catch @panic("Invalid empty piece"),
            };
    };

    const pos: ?Position = blk: {
        const location = v % FIELD_LEN;
        break :blk if (piece) |p| (Position{
            .x = @intCast(location % FIELD_WIDTH),
            .y = @intCast(location / FIELD_WIDTH),
        }).add(getOffset(p)) else null;
    };
    v /= FIELD_LEN;

    const raise = boolFromInt(@intCast(v % 2));
    v /= 2;
    const mirror = boolFromInt(@intCast(v % 2));
    v /= 2;
    // NOTE: The color flag is only used for the first page. From the
    // second page onwards, `color` is always `false`.
    _ = boolFromInt(@intCast(v % 2)); // color
    v /= 2;
    const has_caption = boolFromInt(@intCast(v % 2));
    v /= 2;
    const lock = !boolFromInt(@intCast(v % 2));

    return .{
        piece,
        pos,
        .{
            .raise = raise,
            .mirror = mirror,
            .has_caption = has_caption,
            .lock = lock,
        },
    };
}

fn writePieceAndFlags(
    writer: AnyWriter,
    p: Placement,
    has_caption: bool,
    color: bool,
) !void {
    // flag_lock = true
    var v: u32 = 0;
    // flag_comment
    v *= 2;
    v += if (has_caption) 1 else 0;
    // flag_color
    v *= 2;
    v += if (color) 1 else 0;
    // flag_mirror = false
    v *= 2;
    v += 0;
    // flag_raise = false
    v *= 2;
    v += 0;
    // location
    v *= 240;
    const offset = getOffset(p.piece);
    const x: u32 = @intCast(p.pos.x - offset.x);
    const y: u32 = @intCast(FIELD_HEIGHT - 2 - (p.pos.y + offset.y));
    v += y * FIELD_WIDTH + x;
    // rotation
    v *= 4;
    v += @intFromEnum(FumenRotation.fromEngine(p.piece.facing));
    // Piece
    v *= 8;
    v += @intFromEnum(FumenBlock.fromPieceKind(p.piece.kind));

    try writer.writeAll(&unpoll(3, v));
}

fn readCaption(self: *FumenReader) AllocOrFumenError![]u8 {
    const len = try self.poll(2);
    var caption: std.ArrayList(u8) = try .initCapacity(self.allocator, len);
    errdefer caption.deinit();

    var i: u32 = 0;
    while (i < len) : (i += 4) {
        var v = try self.poll(5);
        for (0..@min(4, len - i)) |_| {
            if (v % 96 >= caption_decode.len) {
                return FumenError.InvalidCaption;
            }
            caption.appendAssumeCapacity(caption_decode[v % 96]);
            v /= 96;
        }
    }

    // Unescape characters
    var read: usize = 0;
    var write: usize = 0;
    while (read < caption.items.len) {
        if (caption.items[read] != '%') {
            caption.items[write] = caption.items[read];
            read += 1;
            write += 1;
            continue;
        }

        if (read + 2 >= caption.items.len) {
            return FumenError.InvalidCaption;
        }
        const codepoint = if (caption.items[read + 1] != 'u') blk: {
            defer read += 3;
            break :blk std.fmt.parseInt(
                u21,
                caption.items[read + 1 .. read + 3],
                16,
            ) catch return FumenError.InvalidCaption;
        } else blk: {
            if (read + 5 >= caption.items.len) {
                return FumenError.InvalidCaption;
            }
            defer read += 6;
            break :blk std.fmt.parseInt(
                u21,
                caption.items[read + 2 .. read + 6],
                16,
            ) catch return FumenError.InvalidCaption;
        };
        write +=
            unicode.utf8Encode(codepoint, caption.items[write..]) catch
                return FumenError.InvalidCaption;
        continue;
    }

    caption.items.len = write;
    return try caption.toOwnedSlice();
}

fn readPieceKind(char: u8) ?PieceKind {
    return switch (char) {
        'I' => .i,
        'O' => .o,
        'T' => .t,
        'L' => .l,
        'J' => .j,
        'S' => .s,
        'Z' => .z,
        else => null,
    };
}

fn readQuiz(self: *FumenReader, caption: []const u8) AllocOrFumenError!void {
    // Hold piece
    var i: usize = 0;
    if (caption[i] != '[') {
        return FumenError.InvalidQuizCaption;
    }
    i += 1;
    if (caption[i] == ']') {
        self.hold = null;
    } else {
        self.hold = readPieceKind(caption[i]) orelse
            return FumenError.InvalidPieceLetter;
        i += 1;
        if (caption[i] != ']') {
            return FumenError.InvalidQuizCaption;
        }
    }
    i += 1;

    // Current piece
    if (caption[i] != '(') {
        return FumenError.InvalidQuizCaption;
    }
    i += 1;
    if (caption[i] == ')') {
        self.current = null;
    } else {
        self.current = readPieceKind(caption[i]) orelse
            return FumenError.InvalidPieceLetter;
        i += 1;
        if (caption[i] != ')') {
            return FumenError.InvalidQuizCaption;
        }
    }
    i += 1;

    self.next.clearRetainingCapacity();
    // Leave an empty slot for the current piece
    try self.next.ensureTotalCapacityPrecise(caption.len - i + 1);
    for (i..caption.len) |j| {
        const piece_kind = readPieceKind(caption[j]) orelse
            return FumenError.InvalidPieceLetter;
        self.next.appendAssumeCapacity(piece_kind);
    }
    mem.reverse(PieceKind, self.next.items);
}

fn pieceKindStr(piece: PieceKind) u8 {
    return switch (piece) {
        .i => 'I',
        .o => 'O',
        .t => 'T',
        .l => 'L',
        .j => 'J',
        .s => 'S',
        .z => 'Z',
    };
}

fn writeQuiz(
    writer: AnyWriter,
    hold: ?PieceKind,
    next: []const PieceKind,
) !void {
    assert(next.len > 0);
    // Write length
    try writer.writeAll(&unpoll(
        2,
        @intCast(19 + next.len + @as(u32, if (hold == null) 0 else 1)),
    ));

    // Quiz prefix and hold piece
    var qw = QuizWriter{};
    try qw.writeAll(writer, "#Q=[");
    if (hold) |h| {
        try qw.write(writer, pieceKindStr(h));
    }
    try qw.write(writer, ']');

    // Current piece
    try qw.write(writer, '(');
    try qw.write(writer, pieceKindStr(next[0]));
    try qw.write(writer, ')');

    // Next pieces
    for (next[1..]) |p| {
        try qw.write(writer, pieceKindStr(p));
    }

    try qw.flush(writer);
}

fn getOffset(piece: Piece) Position {
    return switch (piece.kind) {
        .i => switch (piece.facing) {
            .up => .{ .x = -1, .y = 2 },
            .right => .{ .x = -2, .y = 2 },
            .down => .{ .x = -1, .y = 1 },
            .left => .{ .x = -1, .y = 2 },
        },
        .o => .{ .x = -1, .y = 2 },
        .t, .j, .l => .{ .x = -1, .y = 1 },
        .s => switch (piece.facing) {
            .up => .{ .x = -1, .y = 2 },
            .right => .{ .x = -2, .y = 1 },
            .down => .{ .x = -1, .y = 1 },
            .left => .{ .x = -1, .y = 1 },
        },
        .z => switch (piece.facing) {
            .up => .{ .x = -1, .y = 2 },
            .right => .{ .x = -1, .y = 1 },
            .down => .{ .x = -1, .y = 1 },
            .left => .{ .x = 0, .y = 1 },
        },
    };
}

fn clearLines(self: *FumenReader) void {
    // Bottom row should not be cleared, start from second bottom row
    var y: usize = FIELD_HEIGHT - 1;
    var cleared: usize = 0;
    while (y > 0) {
        y -= 1;
        var full = true;
        for (0..FIELD_WIDTH) |x| {
            self.field[(y + cleared) * FIELD_WIDTH + x] =
                self.field[y * FIELD_WIDTH + x];
            if (self.field[y * FIELD_WIDTH + x] == .empty) {
                full = false;
            }
        }
        if (full) {
            cleared += 1;
        }
    }

    // Add empty rows at the top
    for (0..cleared) |i| {
        for (0..FIELD_WIDTH) |x| {
            self.field[i * FIELD_WIDTH + x] = .empty;
        }
    }
}

fn riseField(self: *FumenReader) void {
    for (0..self.field.len - FIELD_WIDTH) |i| {
        self.field[i] = self.field[i + FIELD_WIDTH];
    }
    // Empty botom row
    for (self.field.len - FIELD_WIDTH..self.field.len) |i| {
        self.field[i] = .empty;
    }
}

fn mirrorField(self: *FumenReader) void {
    // Don't mirror bottom row
    for (0..FIELD_HEIGHT - 1) |y| {
        mem.reverse(
            FumenBlock,
            self.field[y * FIELD_WIDTH .. (y + 1) * FIELD_WIDTH],
        );
    }
}

fn readPage(self: *FumenReader) AllocOrFumenError!void {
    try self.readField();
    const piece, const pos, const flags = try self.readPieceAndFlags();

    if (flags.has_caption) {
        const caption = try self.readCaption();
        defer self.allocator.free(caption);
        // Check for quiz prefix
        if (mem.startsWith(u8, caption, "#Q=")) {
            try self.readQuiz(caption[3..]);
            if (self.current == null) {
                self.current = self.next.pop();
            }
        } else {
            // Else empty quiz state
            self.hold = null;
            self.current = null;
            self.next.clearRetainingCapacity();
        }
    }

    if (!flags.lock) {
        return;
    }

    // Place piece
    if (piece) |p| outer: {
        for (0..4) |x| {
            for (0..4) |y| {
                if (!p.mask().get(x, y)) {
                    continue;
                }

                const mino_pos = pos.?.add(.{
                    .x = @intCast(x),
                    .y = -@as(i8, @intCast(y)),
                });
                // Pieces cannot be placed in bottom row, but can extend out of
                // the top
                if (mino_pos.x < 0 or
                    mino_pos.x >= FIELD_WIDTH or
                    mino_pos.y >= FIELD_HEIGHT - 1)
                {
                    return FumenError.InvalidPieceLocation;
                }
                const index: usize = blk: {
                    const index: i32 = @as(i32, mino_pos.y) * FIELD_WIDTH + mino_pos.x;
                    if (index < 0) {
                        continue;
                    }
                    break :blk @intCast(index);
                };
                // Pieces should not overlap with existing blocks
                if (self.field[index] != .empty) {
                    return FumenError.InvalidPieceLocation;
                }
                self.field[index] = FumenBlock.fromPieceKind(p.kind);
            }
        }

        // Don't update quiz state if quiz is over
        if (self.current == null and
            self.hold == null and
            self.next.items.len == 0)
        {
            break :outer;
        }

        assert(self.current != null);
        if (p.kind != self.current) {
            mem.swap(?PieceKind, &self.current, &self.hold);
            if (self.current == null) {
                self.current = self.next.pop();
            }
        }
        if (p.kind != self.current) {
            return FumenError.InvalidQuizPiece;
        }
        self.current = self.next.pop();
    }
    self.clearLines();

    if (flags.raise) {
        self.riseField();
    }
    if (flags.mirror) {
        self.mirrorField();
    }
}
