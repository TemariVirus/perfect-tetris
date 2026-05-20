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

pub fn main(io: std.Io, args: ValidateArgs, path: []const u8) !void {
    _ = args; // autofix

    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    try stdout.interface.print("Validating {s}\n", .{path});

    var solution_count: u64 = 0;
    var buf: [16 * 1024]u8 = undefined;
    var reader = file.reader(io, &buf);
    while (try PCSolution.readOne(&reader.interface)) |solution| {
        if (solution.next.len == 0) {
            try printValidationError(&stdout.interface, reader, solution_count);
            return;
        }

        var state: GameState = .init(Bag.init(0), engine.kicks.none);
        for (0..solution.placements.len) |i| {
            state.current = solution.placements.buffer[i].piece;
            state.pos = solution.placements.buffer[i].pos;

            const info = state.lockCurrent(-1);
            // Last move must be a PC
            if (i == solution.placements.len - 1 and !info.pc) {
                try printValidationError(&stdout.interface, reader, solution_count);
                return;
            }
        }

        solution_count += 1;
    }

    try stdout.interface.print(
        "Validated {} solutions. All solutions ok.\n",
        .{solution_count},
    );
}

fn printValidationError(
    stdout: *std.Io.Writer,
    file: std.Io.File.Reader,
    solution_count: u64,
) !void {
    const bytes = file.logicalPos();
    try stdout.print(
        "Error at solution {d} (byte {d})\n",
        .{ solution_count, bytes },
    );
}
