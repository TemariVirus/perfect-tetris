const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;

const build = @import("build");
const demo = @import("demo.zig");
const DemoArgs = demo.DemoArgs;
const display = @import("display.zig");
const DisplayArgs = display.DisplayArgs;
const fumen = @import("fumen/root.zig");
const FumenArgs = fumen.FumenArgs;
const validate = @import("validate.zig");
const ValidateArgs = validate.ValidateArgs;

const NN = @import("perfect-tetris").NN;
const NNInner = @import("zmai").genetic.neat.NN;
const kicks = @import("engine").kicks;

const zig_args = @import("zig-args");
const Error = zig_args.Error;

const runtime_safety = switch (@import("builtin").mode) {
    .Debug, .ReleaseSafe => true,
    .ReleaseFast, .ReleaseSmall => false,
};
var gpa: std.heap.DebugAllocator(.{}) = .init;

const Args = struct {
    help: bool = false,

    pub const shorthands = .{
        .h = "help",
    };

    pub const meta = .{
        .usage_summary = "COMMAND [options] [INPUT]",
        .full_text =
        \\Blazingly fast Tetris perfect clear solver. Run `pc COMMAND --help` for
        \\command-specific help.
        \\
        \\Report issues at https://github.com/TemariVirus/perfect-tetris/issues
        \\
        \\Commands:
        \\  demo         Demostrates the solver's speed with a tetris playing bot.
        \\  display      Displays the perfect clear solutions saved at PATH.
        \\  fumen        Produces a perfect clear solution for each input fumen.
        \\  validate     Validates the perfect clear solutions saved at PATHS.
        \\  version      Prints the program's version.
        ,
        .option_docs = .{
            .help = "Print this help message.",
        },
    };
};

const VerbType = enum {
    demo,
    display,
    fumen,
    validate,
    version,
};

const Verb = union(VerbType) {
    demo: DemoArgs,
    display: DisplayArgs,
    fumen: FumenArgs,
    validate: ValidateArgs,
    version: void,
};

pub const KicksOption = enum {
    none,
    none180,
    srs,
    srs180,
    srsPlus,
    srsTetrio,

    pub fn toEngine(self: KicksOption) *const kicks.KickFn {
        return &switch (self) {
            .none => kicks.none,
            .none180 => kicks.none180,
            .srs => kicks.srs,
            .srs180 => kicks.srs180,
            .srsPlus => kicks.srsPlus,
            .srsTetrio => kicks.srsTetrio,
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = gpa.allocator();
    defer if (runtime_safety) {
        _ = gpa.deinit();
    };

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var stdout = std.Io.File.stdout().writer(io, &.{});
    var stderr = std.Io.File.stderr().writer(io, &.{});

    const exe_args = try zig_args.parseWithVerbForCurrentProcess(
        Args,
        Verb,
        init,
        .{ .forward = handleArgsError },
    );
    defer exe_args.deinit();

    const exe_name = std.fs.path.stem(exe_args.executable_name.?);
    const verb = exe_args.verb orelse {
        try zig_args.printHelp(Args, exe_name, &stdout.interface);
        return;
    };

    // Help flag gets consumed by global options, so use that instead.
    // Verb-specific help flags only exist for the help message.
    switch (verb) {
        .demo => |args| {
            if (exe_args.options.help) {
                try zig_args.printHelp(DemoArgs, exe_name, &stdout.interface);
                return;
            }

            if (args.pps <= 0) {
                try stderr.interface.print("PPS option must be greater than 0\n", .{});
                return;
            }

            const nn = try loadNN(io, allocator, args.nn);
            defer if (nn) |_nn| _nn.deinit(allocator);
            try demo.main(io, allocator, args, nn);
        },
        .display => |args| {
            if (exe_args.options.help or exe_args.positionals.len == 0) {
                try zig_args.printHelp(DisplayArgs, exe_name, &stdout.interface);
                return;
            }

            try display.main(io, allocator, init.environ_map, args, exe_args.positionals[0]);
        },
        .fumen => |args| {
            if (exe_args.options.help or exe_args.positionals.len == 0) {
                try zig_args.printHelp(FumenArgs, exe_name, &stdout.interface);
                return;
            }

            const nn = try loadNN(io, allocator, args.nn);
            defer if (nn) |_nn| _nn.deinit(allocator);

            for (exe_args.positionals) |fumen_str| {
                try fumen.main(io, allocator, args, fumen_str, nn);
            }
        },
        .validate => |args| {
            if (exe_args.options.help or exe_args.positionals.len == 0) {
                try zig_args.printHelp(ValidateArgs, exe_name, &stdout.interface);
                return;
            }

            for (exe_args.positionals) |path| {
                try validate.main(io, args, path);
            }
        },
        .version => {
            try stdout.interface.print("{s}\n", .{build.version});
        },
    }
}

fn handleArgsError(err: Error) error{}!void {
    std.log.err("{f}", .{err});
    std.process.exit(1);
}

pub fn enumValuesHelp(Enum: type) []const u8 {
    if (!@inComptime()) @panic("Must be called in comptime");

    const total_len = blk: {
        var counter: std.Io.Writer.Discarding = .init(&.{});
        writeEnumValuesHelp(Enum, &counter.writer) catch unreachable;
        break :blk counter.fullCount();
    };

    var buf: [total_len]u8 = undefined;
    var str: std.Io.Writer = .fixed(&buf);
    writeEnumValuesHelp(Enum, &str) catch unreachable;
    return str.buffered();
}

fn writeEnumValuesHelp(Enum: type, writer: *std.Io.Writer) !void {
    try writer.writeAll("Supported Values: [");
    for (@typeInfo(Enum).@"enum".fields, 0..) |field, i| {
        try writer.writeAll(field.name);
        if (i < @typeInfo(Enum).@"enum".fields.len - 1) {
            try writer.writeByte(',');
        }
    }
    try writer.writeByte(']');
}

/// Returns the first path that exists, relative to different locations in the
/// following order:
///
/// - Absolute path (no allocation)
/// - The current working directory (no allocation)
/// - The directory containing the executable
///
/// If no match is found, returns `AccessError.FileNotFound`.
pub fn resolvePath(io: std.Io, allocator: Allocator, path: []const u8) ![]const u8 {
    const AccessError = std.Io.Dir.AccessError;

    // Absolute path
    if (std.fs.path.isAbsolute(path)) {
        try std.Io.Dir.accessAbsolute(io, path, .{});
        return path;
    }

    // Current working directory
    if (std.Io.Dir.cwd().access(io, path, .{})) |_| {
        return path;
    } else |e| {
        if (e != AccessError.FileNotFound) {
            return e;
        }
    }

    // Relative to executable
    const exe_path = try std.process.executableDirPathAlloc(io, allocator);
    defer allocator.free(exe_path);

    const exe_rel_path = try std.fs.path.join(allocator, &.{
        exe_path,
        path,
    });
    if (std.Io.Dir.accessAbsolute(io, exe_rel_path, .{})) |_| {
        return exe_rel_path;
    } else |e| {
        allocator.free(exe_rel_path);
        if (e != AccessError.FileNotFound) {
            return e;
        }
    }

    return AccessError.FileNotFound;
}

pub fn loadNN(io: std.Io, allocator: Allocator, path: ?[]const u8) !?NN {
    if (path) |p| {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        const nn_path = try resolvePath(io, arena.allocator(), p);
        return try NN.load(io, allocator, nn_path);
    } else {
        return null;
    }
}
