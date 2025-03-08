const std = @import("std");
const Allocator = std.mem.Allocator;
const AnyReader = std.io.AnyReader;
const assert = std.debug.assert;
const File = std.fs.File;
const io = std.io;
const SolutionIndex = std.ArrayListUnmanaged(u64);

const engine = @import("engine");
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;
const Position = engine.pieces.Position;

const vaxis = @import("vaxis");
const BorderOptions = Window.BorderOptions;
const Key = vaxis.Key;
const Loop = vaxis.Loop(Event);
const TextInput = vaxis.widgets.TextInput;
const Tty = vaxis.Tty;
const Window = vaxis.Window;

const root = @import("perfect-tetris");
const PCSolution = root.PCSolution;
const Placement = root.Placement;

pub const panic = vaxis.panic_handler;
pub const std_options: std.Options = .{
    .log_scope_levels = &.{
        .{ .scope = .vaxis, .level = .err },
        .{ .scope = .vaxis_parser, .level = .err },
    },
};

const INDEX_INTERVAL = 1 << 17;
const BORDER_GLYPHS: BorderOptions.Glyphs = .{
    .custom = .{ "╔", "═", "╗", "║", "╝", "╚" },
};

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

const Event = union(enum) {
    key_press: Key,
    solution_count: u64,
    winsize: vaxis.Winsize,
};

pub fn main(allocator: Allocator, args: DisplayArgs, path: []const u8) !void {
    _ = args; // autofix

    // Open PC file
    const pc_file = try std.fs.cwd().openFile(path, .{});
    defer pc_file.close();
    const reader = pc_file.reader().any();

    // Set up terminal and vaxis
    var tty: Tty = try .init();
    defer tty.deinit();
    var bf = tty.bufferedWriter();
    const stdout = bf.writer().any();

    var vx = try vaxis.init(allocator, .{});
    defer vx.deinit(allocator, tty.anyWriter());
    try vx.enterAltScreen(tty.anyWriter());
    try vx.enableDetectedFeatures(tty.anyWriter());

    // Vaxis event loop
    var loop: Loop = .{
        .tty = &tty,
        .vaxis = &vx,
    };
    try loop.init();
    try loop.start();

    // Start indexing thread
    const SOLUTION_MIN_SIZE = 8;
    var solution_index: SolutionIndex = try .initCapacity(
        allocator,
        (try pc_file.getEndPos()) / SOLUTION_MIN_SIZE / INDEX_INTERVAL + 1,
    );
    defer solution_index.deinit(allocator);

    solution_index.appendAssumeCapacity(0);
    var stop_index_thread = false;
    const index_thread: std.Thread = try .spawn(
        .{ .allocator = allocator },
        indexThread,
        .{ &stop_index_thread, path, &solution_index, &loop },
    );
    defer stop_index_thread = true;
    index_thread.detach();

    // Widgets
    var text_input: TextInput = .init(allocator, &vx.unicode);
    defer text_input.deinit();

    // Run main loop
    var pos: u64 = 0;
    var solution_count: u64 = 0;
    var sol = (try PCSolution.readOne(reader)) orelse return;
    while (true) {
        const event = loop.nextEvent();
        switch (event) {
            .key_press => |key| blk: {
                // Exit on ctrl+c or ctrl+d
                if (key.matchExact('c', .{ .ctrl = true }) or
                    key.matchExact('d', .{ .ctrl = true }))
                {
                    return;
                }

                if (!key.matches(Key.enter, .{})) {
                    try text_input.update(.{ .key_press = key });
                    break :blk;
                }

                // Process text in input when enter is pressed
                const text = try text_input.buf.toOwnedSlice();
                defer allocator.free(text);

                const trimmed = std.mem.trim(u8, text, " ");
                if (std.fmt.parseInt(u64, trimmed, 10)) |n| {
                    if (try seekToSolution(pc_file, n -| 1, solution_index)) {
                        sol = (try PCSolution.readOne(reader)) orelse return;
                        pos = n -| 1;
                    }
                } else |_| if (text.len == 0) {
                    // Go to next solution if the input is empty.
                    sol = (try PCSolution.readOne(reader)) orelse return;
                    pos += 1;
                }
            },
            .solution_count => |count| solution_count = count,
            .winsize => |ws| try vx.resize(allocator, stdout, ws),
        }

        const win = vx.window();
        win.clear();

        const matrix_width = 22;
        const matrix_height = 22;
        const footer_height = 1;
        const next_len: u5 = @intCast(sol.next.len);
        const next_width = 10;
        const next_height = @as(u16, next_len) * 3 + 2;

        const main_width: u16 = matrix_width + 1 + next_width;
        const main_height: u16 = @max(matrix_height, next_height) + footer_height;
        const main_win = win.child(.{
            .x_off = (win.width -| main_width) / 2,
            .y_off = (win.height -| main_height) / 2,
            .width = main_width,
            .height = main_height,
            .border = .{},
        });

        const next_win = main_win.child(.{
            .x_off = matrix_width + 1,
            .y_off = 0,
            .width = next_width,
            .height = next_height,
            .border = .{
                .where = .{ .other = .{
                    .left = true,
                    .top = true,
                    .right = true,
                    .bottom = main_win.height >= next_height,
                } },
                .glyphs = BORDER_GLYPHS,
            },
        });
        drawSequence(next_win, sol.next.slice());

        const matrix_win = main_win.child(.{
            .x_off = 0,
            .y_off = next_height -| matrix_height,
            .width = matrix_width,
            .height = matrix_height,
            .border = .{
                .where = .{
                    .other = .{
                        .left = true,
                        .top = true,
                        .right = true,
                        .bottom = main_win.height >= @max(next_height, matrix_height),
                    },
                },
                .glyphs = BORDER_GLYPHS,
            },
        });
        drawMatrix(matrix_win, sol.placements.slice());

        const footer_win = main_win.child(.{
            .x_off = 0,
            .y_off = main_height - 2,
            .width = main_width,
            .height = 2,
        });

        var buf: [53]u8 = undefined;
        const text = std.fmt.bufPrint(
            &buf,
            "Solution {} of {}",
            .{ pos + 1, solution_count },
        ) catch unreachable;
        _ = footer_win.printSegment(
            .{ .text = text },
            .{ .row_offset = 0, .wrap = .none },
        );
        _ = footer_win.printSegment(
            .{ .text = "Skip to: " },
            .{ .row_offset = 1 },
        );

        text_input.draw(footer_win.child(.{
            .x_off = 9,
            .y_off = 1,
            .width = footer_win.width - 9,
            .height = 1,
        }));

        try vx.render(stdout);
        try bf.flush();
    }
}

fn nextSolution(reader: AnyReader) u64 {
    const solution = (PCSolution.readOne(reader) catch return 0) orelse
        return 0;
    return 8 + @as(u64, solution.next.len) - 1;
}

// Get the index to the start of a solution at regular intervals.
// This greatly improves the performance of seeking to a solution.
// Due to the time needed to index the file, this is done in a separate thread.
fn indexThread(
    should_stop: *const bool,
    path: []const u8,
    solution_index: *SolutionIndex,
    loop: *Loop,
) !void {
    const pc_file = try std.fs.cwd().openFile(path, .{});
    defer pc_file.close();
    var bf = io.bufferedReader(pc_file.reader());
    const reader = bf.reader().any();

    var pos: u64 = 0;
    var count: u64 = 0;
    while (!should_stop.*) : (count += 1) {
        const len = nextSolution(reader);
        if (len == 0) {
            break;
        }

        if (count != 0 and count % INDEX_INTERVAL == 0) {
            solution_index.appendAssumeCapacity(pos);
            loop.postEvent(.{ .solution_count = count });
        }

        pos += len;
    }

    loop.postEvent(.{ .solution_count = count });
}

fn seekToSolution(file: File, n: u64, solution_index: SolutionIndex) !bool {
    const old_pos = try file.getPos();

    // Get closest index before n
    const index = @min(solution_index.items.len - 1, n / INDEX_INTERVAL);
    var pos = solution_index.items[index];
    try file.seekTo(pos);

    var bf = io.bufferedReader(file.reader());
    const reader = bf.reader().any();

    for (index * INDEX_INTERVAL..n) |_| {
        const len = nextSolution(reader);
        if (len == 0) {
            try file.seekTo(old_pos);
            return false;
        }
        pos += len;
    }

    // Don't seek if we just passed the end of the file
    if (nextSolution(reader) == 0) {
        try file.seekTo(old_pos);
        return false;
    }

    try file.seekTo(pos);
    return true;
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

fn drawPiece(win: Window, piece: Piece, x: i8, y: i8) void {
    const minos = getMinos(piece);
    const color = piece.kind.color();
    for (minos) |mino| {
        const mino_x = x + mino.x;
        // The y coordinate is flipped when converting to nterm coordinates.
        const mino_y = y + (3 - mino.y);
        _ = win.printSegment(.{
            .text = "  ",
            .style = .{
                .fg = .{ .index = color },
                .bg = .{ .index = color },
            },
        }, .{
            .col_offset = @intCast(mino_x * 2),
            .row_offset = @intCast(mino_y),
        });
    }
}

fn drawSequence(win: Window, pieces: []const PieceKind) void {
    for (pieces, 0..) |p, i| {
        const piece: Piece = .{ .facing = .up, .kind = p };
        drawPiece(win, piece, 0, @intCast(i * 3));
    }
}

/// Draw a piece in the matrix view, and update the row occupancy.
fn drawMatrixPiece(
    win: Window,
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
        _ = win.printSegment(.{
            .text = "  ",
            .style = .{
                .fg = .{ .index = color },
                .bg = .{ .index = color },
            },
        }, .{
            .col_offset = @intCast(mino_x * 2),
            .row_offset = @intCast(mino_y),
        });

        row_occupancy[@intCast(19 - mino_y)] += 1;
    }
}

fn drawMatrix(win: Window, placements: []const Placement) void {
    var row_occupancy: [20]u8 = @splat(0);
    for (placements) |p| {
        drawMatrixPiece(win, &row_occupancy, p.piece, p.pos);
    }
}
