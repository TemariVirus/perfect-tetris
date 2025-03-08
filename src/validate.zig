const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const engine = @import("engine");
const Bag = engine.bags.NoBag;
const Facing = engine.pieces.Facing;
const GameState = engine.GameState(Bag);
const Piece = engine.pieces.Piece;
const PieceKind = engine.pieces.PieceKind;

const PCSolution = @import("perfect-tetris").PCSolution;

pub const ValidateArgs = struct {
    help: bool = false,

    pub const wrap_len: u32 = 50;

    pub const shorthands = .{
        .h = "help",
    };

    pub const meta = .{
        .usage_summary = "validate [options] PATHS...",
        .full_text =
        \\Validates the perfect clear solutions saved at PATHS. This will validate
        \\that PATHS are valid .pc files and that all solutions are valid perfect
        \\clear solutions.
        ,
        .option_docs = .{
            .help = "Print this help message.",
        },
    };
};

pub fn main(args: ValidateArgs, path: []const u8) !void {
    _ = args; // autofix

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    try std.io.getStdOut().writer().print("Validating {s}\n", .{path});

    var solution_count: u64 = 0;
    var buf_reader = std.io.bufferedReader(file.reader());
    const reader = buf_reader.reader();
    while (try PCSolution.readOne(reader.any())) |solution| {
        if (solution.next.len == 0) {
            try printValidationError(file, buf_reader, solution_count);
            return;
        }

        var state: GameState = .init(Bag.init(0), engine.kicks.none);
        for (0..solution.placements.len) |i| {
            state.current = solution.placements.buffer[i].piece;
            state.pos = solution.placements.buffer[i].pos;

            const info = state.lockCurrent(-1);
            // Last move must be a PC
            if (i == solution.placements.len - 1 and !info.pc) {
                try printValidationError(file, buf_reader, solution_count);
                return;
            }
        }

        solution_count += 1;
    }

    try std.io.getStdOut().writer().print(
        "Validated {} solutions. All solutions ok.\n",
        .{solution_count},
    );
}

fn printValidationError(
    file: std.fs.File,
    buf_reader: anytype,
    solution_count: u64,
) !void {
    const bytes = try file.getPos() -
        @as(u64, @intCast(buf_reader.end)) +
        @as(u64, @intCast(buf_reader.start));
    try std.io.getStdOut().writer().print(
        "Error at solution {} (byte {})\n",
        .{ solution_count, bytes },
    );
}
