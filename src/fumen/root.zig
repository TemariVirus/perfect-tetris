const std = @import("std");
const Allocator = std.mem.Allocator;
const AnyWriter = std.io.AnyWriter;

const engine = @import("engine");
const kicks = engine.kicks;
const PieceKind = engine.pieces.PieceKind;

const root = @import("perfect-tetris");
const NN = root.NN;
const FindPcError = root.FindPcError;

const enumValuesHelp = @import("../main.zig").enumValuesHelp;
const KicksOption = @import("../main.zig").KicksOption;
pub const FumenReader = @import("FumenReader.zig");

pub const PieceKindArg = enum {
    i,
    o,
    t,
    s,
    z,
    l,
    j,
    I,
    O,
    T,
    S,
    Z,
    L,
    J,

    pub fn toEngine(self: ?PieceKindArg) ?PieceKind {
        if (self == null) {
            return null;
        }
        return switch (self.?) {
            .i, .I => .i,
            .o, .O => .o,
            .t, .T => .t,
            .s, .S => .s,
            .z, .Z => .z,
            .l, .L => .l,
            .j, .J => .j,
        };
    }
};

pub const OutputMode = enum {
    edit,
    list,
    view,

    pub fn toChr(self: OutputMode) u8 {
        return switch (self) {
            .edit => 'v',
            .list => 'D',
            .view => 'm',
        };
    }
};

pub const FumenArgs = struct {
    append: bool = false,
    help: bool = false,
    kicks: KicksOption = .srs,
    @"min-height": u7 = 1,
    nn: ?[]const u8 = null,
    save: ?PieceKindArg = null,
    @"output-type": OutputMode = .view,
    verbose: bool = false,

    pub const wrap_len: u32 = 35;

    pub const shorthands = .{
        .a = "append",
        .h = "help",
        .k = "kicks",
        .m = "min-height",
        .n = "nn",
        .s = "save",
        .t = "output-type",
        .v = "verbose",
    };

    pub const meta = .{
        .usage_summary = "fumen [options] INPUTS...",
        .full_text =
        \\Finds the shortest perfect clear solution of each input fumen.
        \\The queue is encoded in the comment of the fumen, in the format:
        \\
        \\#Q=[<HOLD>](<CURRENT>)<NEXT1><NEXT2>...<NEXTn>
        \\
        \\Outputs each solution to stdout as a new fumen, separated by newlines.
        \\If the fumen is inside a url, the url will be preserved in the output.
        \\Fumen editor: https://fumen.zui.jp/#english.js
        ,
        .option_docs = .{
            .append = "Append solution frames to input fumen instead of making a new fumen from scratch.",
            .help = "Print this help message.",
            .kicks = std.fmt.comptimePrint(
                "Kick/rotation system to use. For kick systems that have a 180-less and 180 variant, the 180-less variant has no 180 rotations. The 180 variant has 180 rotations but no 180 kicks. " ++
                    enumValuesHelp(FumenArgs, KicksOption) ++
                    " (default: {s})",
                .{@tagName((FumenArgs{}).kicks)},
            ),
            .@"min-height" = "Overrides the minimum height of the PC to find.",
            .nn = "The path to the neural network to use for the solver. The path may be absolute, relative to the current working directory, or relative to the executable's directory. If not provided, a default built-in NN will be used.",
            .save = "The piece type to save in the hold slot by the end of the perfect clear. This option is not case-sensitive. If not specified, any piece may go into the hold slot. " ++
                enumValuesHelp(FumenArgs, PieceKind),
            .@"output-type" = std.fmt.comptimePrint(
                "The type of fumen to output. If append is true, this option is ignored. " ++
                    enumValuesHelp(FumenArgs, OutputMode) ++
                    " (default: {s})",
                .{@tagName((FumenArgs{}).@"output-type")},
            ),
            .verbose = "Print solve time and solution length to stderr.",
        },
    };
};

pub fn main(
    allocator: Allocator,
    args: FumenArgs,
    fumen: []const u8,
    nn: ?NN,
    writer: AnyWriter,
) !void {
    const start_t = std.time.nanoTimestamp();

    const parsed = try FumenReader.parse(allocator, fumen);
    defer parsed.deinit(allocator);

    const gamestate = parsed.toGameState(args.kicks.toEngine());
    const solution = root.findPcAuto(
        root.FixedBag,
        allocator,
        gamestate,
        nn,
        args.@"min-height",
        parsed.next.len,
        PieceKindArg.toEngine(args.save),
    ) catch |err| blk: {
        if (err != FindPcError.ImpossibleSaveHold and
            err != FindPcError.NoPcExists and
            err != FindPcError.SolutionTooLong)
        {
            return err;
        }
        std.debug.print("Unable to solve: {}\n", .{err});
        break :blk null;
    };
    defer if (solution) |sol| {
        allocator.free(sol);
    };
    const time_taken = std.time.nanoTimestamp() - start_t;

    // Print output fumen
    if (solution) |sol| {
        try FumenReader.outputFumen(args, parsed, sol, writer);
    }

    if (args.verbose) {
        if (solution) |sol| {
            std.debug.print(
                "Found solution of length {} in {}\n",
                .{ sol.len, std.fmt.fmtDuration(@intCast(time_taken)) },
            );
        } else {
            std.debug.print(
                "No solution found in {}\n",
                .{std.fmt.fmtDuration(@intCast(time_taken))},
            );
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
