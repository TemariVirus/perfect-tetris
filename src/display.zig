const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const File = std.fs.File;
const io = std.io;
const SolutionIndex = std.ArrayListUnmanaged(u64);

const engine = @import("engine");
const Facing = engine.pieces.Facing;
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;

const nterm = @import("nterm");
const Colors = nterm.Colors;
const View = nterm.View;

const PCSolution = @import("perfect-tetris").PCSolution;

// Set max sequence length to 16 to handle up to 6-line PCs.
const MAX_SEQ_LEN = 16;
const INDEX_INTERVAL = 1 << 18;

pub const DisplayArgs = struct {
    help: bool = false,

    pub const wrap_len: u32 = 50;

    pub const shorthands = .{
        .h = "help",
    };

    pub const meta = .{
        .usage_summary = "display [options] PATH",
        .full_text =
        \\Displays the perfect clear solutions saved at PATH. Press `enter` to
        \\display the next solution. To seek to a specific solution, type the
        \\solution number and press `enter`. Only supports `.pc` files.
        ,
        .option_docs = .{
            .help = "Print this help message.",
        },
    };
};

pub fn main(allocator: Allocator, args: DisplayArgs, path: []const u8) !void {
    _ = args; // autofix

    try nterm.init(
        allocator,
        io.getStdOut(),
        1,
        0,
        0,
        null,
        null,
    );
    defer nterm.deinit();

    const pc_file = try std.fs.cwd().openFile(path, .{});
    defer pc_file.close();
    const reader = pc_file.reader();

    var solution_count: ?u64 = null;
    const SOLUTION_MIN_SIZE = 8;
    var solution_index = try SolutionIndex.initCapacity(
        allocator,
        (try pc_file.getEndPos()) / SOLUTION_MIN_SIZE / INDEX_INTERVAL + 1,
    );
    defer solution_index.deinit(allocator);

    solution_index.appendAssumeCapacity(0);
    const index_thread = try std.Thread.spawn(
        .{ .allocator = allocator },
        indexThread,
        .{ path, &solution_count, &solution_index },
    );
    index_thread.detach();

    var i: u64 = 0;
    const stdin = io.getStdIn().reader();
    while (try PCSolution.readOne(reader)) |s| {
        const next_len = @as(usize, s.next.len);
        try nterm.setCanvasSize(
            (11 + 5) * 2 + 1,
            @intCast(@max(22, next_len * 3 + 2)),
        );
        drawSequence(s.next.buffer[0..next_len]);

        const matrix_box = View{
            .left = 0,
            .top = nterm.canvasSize().height - 22,
            .width = 10 * 2 + 2,
            .height = 22,
        };
        matrix_box.drawBox(
            0,
            0,
            matrix_box.height,
            matrix_box.width,
            Colors.WHITE,
            null,
        );

        var row_occupancy = [_]u8{0} ** 20;
        const matrix_view = matrix_box.sub(
            1,
            1,
            matrix_box.width - 2,
            matrix_box.height - 2,
        );
        for (s.placements.buffer[0..s.placements.len]) |p| {
            drawMatrixPiece(matrix_view, &row_occupancy, p.piece, p.pos);
        }

        printFooter(i, solution_count);
        nterm.render() catch |err| {
            // Trying to render after the terminal has been closed results
            // in an error, in which case stop the program gracefully.
            if (err == error.NotInitialized) {
                return;
            }
            return err;
        };

        // Read until enter is pressed
        const bytes = try stdin.readUntilDelimiterAlloc(
            allocator,
            '\n',
            std.math.maxInt(usize),
        );
        defer allocator.free(bytes);

        if (std.fmt.parseInt(u64, bytes[0 .. bytes.len - 1], 10)) |n| {
            if (n == 0) {
                i = 0;
                try pc_file.seekTo(0);
            } else if (try seekToSolution(pc_file, n - 1, solution_index)) {
                i = n - 1;
            } else {
                // Go back to start of current solution
                try pc_file.seekBy(-@as(i64, @intCast(next_len)) - 7);
            }
        } else |_| {
            // Only go to next solution if the input is empty.
            if (bytes.len == 1) {
                i += 1;
            } else {
                // Go back to start of current solution
                try pc_file.seekBy(-@as(i64, @intCast(next_len)) - 7);
            }
        }
    }
}

fn printFooter(pos: u64, end: ?u64) void {
    if (end) |e| {
        nterm.view().printAt(
            0,
            nterm.canvasSize().height - 1,
            Colors.WHITE,
            null,
            "Solution {} of {}",
            .{ pos + 1, e },
        );
    } else {
        nterm.view().printAt(
            0,
            nterm.canvasSize().height - 1,
            Colors.WHITE,
            null,
            "Solution {} of ?",
            .{pos + 1},
        );
    }
}

fn nextSolution(reader: anytype) u64 {
    const solution = (PCSolution.readOne(reader) catch return 0) orelse
        return 0;
    return 8 + @as(u64, solution.next.len) - 1;
}

// Get the index to the start of a solution at regular intervals.
// This greatly improves the performance of seeking to a solution.
// Due to the time needed to index the file, this is done in a separate thread.
fn indexThread(
    path: []const u8,
    solution_count: *?u64,
    solution_index: *SolutionIndex,
) !void {
    const pc_file = try std.fs.cwd().openFile(path, .{});
    defer pc_file.close();
    var bf = io.bufferedReader(pc_file.reader());
    const reader = bf.reader();

    var pos: u64 = 0;
    var count: u64 = 0;
    while (true) : (count += 1) {
        const len = nextSolution(reader);
        if (len == 0) {
            break;
        }

        if (count != 0 and count % INDEX_INTERVAL == 0) {
            solution_index.appendAssumeCapacity(pos);
        }

        pos += len;
    }

    solution_count.* = count;
}

fn seekToSolution(file: File, n: u64, solution_index: SolutionIndex) !bool {
    const old_pos = try file.getPos();

    // Get closest index before n
    const index = @min(solution_index.items.len - 1, n / INDEX_INTERVAL);
    var pos = solution_index.items[index];
    try file.seekTo(pos);

    var bf = io.bufferedReader(file.reader());
    const reader = bf.reader();

    for (index * INDEX_INTERVAL..n) |_| {
        const len = nextSolution(reader);
        if (len == 0) {
            try file.seekTo(old_pos);
            return false;
        }
        pos += len;
    }

    // Don't seek if we just passed the last solution (i.e., n == solution count)
    if (nextSolution(reader) == 0) {
        try file.seekTo(old_pos);
        return false;
    }

    try file.seekTo(pos);
    return true;
}

/// Get the positions of the minos of a piece relative to the bottom left corner.
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

fn drawSequence(pieces: []const PieceKind) void {
    assert(pieces.len <= MAX_SEQ_LEN);

    const WIDTH = 2 * 4 + 2;
    const box_view = View{
        .left = nterm.canvasSize().width - WIDTH,
        .top = 0,
        .width = WIDTH,
        .height = @intCast(pieces.len * 3 + 2),
    };
    box_view.drawBox(
        0,
        0,
        box_view.width,
        box_view.height,
        Colors.WHITE,
        null,
    );

    const box = box_view.sub(1, 1, box_view.width - 2, box_view.height - 2);
    for (pieces, 0..) |p, i| {
        const piece = Piece{ .facing = .up, .kind = p };
        drawPiece(box, piece, 0, @intCast(i * 3));
    }
}

fn drawPiece(view: View, piece: Piece, x: i8, y: i8) void {
    const minos = getMinos(piece);
    const color = piece.kind.color();
    for (minos) |mino| {
        const mino_x = x + mino.x;
        // The y coordinate is flipped when converting to nterm coordinates.
        const mino_y = y + (3 - mino.y);
        _ = view.writeText(
            @intCast(mino_x * 2),
            @intCast(mino_y),
            color,
            color,
            "  ",
        );
    }
}

/// Draw a piece in the matrix view, and update the row occupancy.
fn drawMatrixPiece(
    view: View,
    row_occupancy: []u8,
    piece: Piece,
    pos: Position,
) void {
    const minos = getMinos(piece);
    const color = piece.kind.color();
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

        const mino_x = pos.x + mino.x;
        // The y coordinate is flipped when converting to nterm coordinates.
        const mino_y = 19 - pos.y - mino.y - cleared;
        _ = view.writeText(
            @intCast(mino_x * 2),
            @intCast(mino_y),
            color,
            color,
            "  ",
        );

        row_occupancy[@intCast(19 - mino_y)] += 1;
    }
}
