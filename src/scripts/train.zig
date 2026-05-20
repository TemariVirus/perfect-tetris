const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const AtomicInt = std.atomic.Value(i32);
const Dir = std.Io.Dir;
const json = std.json;
const Mutex = std.Io.Mutex;
const os = std.os;
const SIG = os.linux.SIG;
const time = std.time;

const engine = @import("engine");
const GameState = engine.GameState(SevenBag);
const SevenBag = engine.bags.SevenBag;

const PeriodicTrigger = @import("nterm").PeriodicTrigger;

const zmai = @import("zmai");
const neat = zmai.genetic.neat;
const Trainer = neat.Trainer;

const root = @import("perfect-tetris");
const NN = root.NN;
const pc = root.pc;

// Height of perfect clears to find
const HEIGHT = 4;
comptime {
    assert(HEIGHT % 2 == 0);
}
// Number of threads to use
const THREADS = 4;

const SAVE_INTERVAL: std.Io.Duration = .fromSeconds(10);
const SAVE_DIR = "pops/relu-identity/";

const GENERATIONS = 50;
const POPULATION_SIZE = 500;
const OPTIONS: Trainer.Options = .{
    .species_target = 15,
    .mutate_options = .{
        .node_add_prob = 0.04,
    },
    .nn_options = .{
        .input_count = NN.INPUT_COUNT,
        .output_count = 1,
        .hidden_activation = .relu,
        .output_activation = .identity,
    },
};

var saving_threads: AtomicInt = .init(0);
var generation_done: WideBool = .false;

const WideBool = enum(u32) {
    false,
    true,
};

const SaveJson = struct {
    seed: u64,
    trainer: Trainer.TrainerJson,
    fitnesses: []const ?f64,
};

pub fn main(init: std.process.Init) !void {
    setupExitHandler();

    const allocator = init.gpa;
    const io = init.io;
    zmai.setRandomSeed(blk: {
        var seed: u64 = undefined;
        io.random(std.mem.asBytes(&seed));
        break :blk seed;
    });

    var trainer, const fitnesses, var seed = try loadOrInit(
        io,
        allocator,
        SAVE_DIR ++ "current.json",
        POPULATION_SIZE,
        OPTIONS,
    );
    defer trainer.deinit();
    defer allocator.free(fitnesses);

    var fitnesses_lock: Mutex = .init;
    var threads: [THREADS]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try .spawn(
            .{ .allocator = allocator },
            doWork,
            .{ io, allocator, &seed, &trainer, fitnesses, &fitnesses_lock },
        );
    }
    defer for (threads) |thread| {
        thread.join();
    };

    var save_timer: PeriodicTrigger = .init(io, SAVE_INTERVAL, true);
    outer: while (trainer.generation < GENERATIONS) {
        try io.sleep(.fromMilliseconds(10), .real);
        if (save_timer.trigger()) |_| {
            try saveSafe(
                io,
                allocator,
                SAVE_DIR ++ "current.json",
                seed,
                trainer,
                fitnesses,
            );
        }

        // Wait for all fitnesses to be calculated
        for (fitnesses) |fit| {
            if (std.math.isNan(fit orelse continue :outer)) {
                continue :outer;
            }
        }
        generation_done = .true;

        const path = try std.fmt.allocPrint(
            allocator,
            "{s}{}.json",
            .{ SAVE_DIR, trainer.generation },
        );
        defer allocator.free(path);
        try saveSafe(io, allocator, path, seed, trainer, fitnesses);

        const final_fitnesses = try allocator.alloc(f64, fitnesses.len);
        defer allocator.free(final_fitnesses);
        for (0..fitnesses.len) |i| {
            final_fitnesses[i] = fitnesses[i] orelse
                @panic("Fitness set to null after being non-null");
        }

        printGenerationStats(trainer, final_fitnesses);
        io.random(std.mem.asBytes(&seed));
        try trainer.nextGeneration(final_fitnesses);

        try fitnesses_lock.lock(io);
        @memset(fitnesses, null);
        fitnesses_lock.unlock(io);

        generation_done = .false;
        io.futexWake(WideBool, &generation_done, THREADS);
    }
}

const handle_signals = [_]SIG{
    SIG.ABRT,
    SIG.INT,
    SIG.QUIT,
    SIG.TERM,
};
fn setupExitHandler() void {
    if (@import("builtin").os.tag == .windows) {
        const signal = struct {
            extern "c" fn signal(
                sig: c_int,
                func: *const fn (c_int, c_int) callconv(os.windows.WINAPI) void,
            ) callconv(.C) *anyopaque;
        }.signal;
        for (handle_signals) |sig| {
            _ = signal(sig, handleExitWindows);
        }
    } else {
        const action: os.linux.Sigaction = .{
            .handler = .{ .handler = handleExit },
            .mask = os.linux.sigemptyset(),
            .flags = 0,
        };
        for (handle_signals) |sig| {
            _ = os.linux.sigaction(sig, &action, null);
        }
    }
}

fn handleExit(sig: SIG) callconv(.c) void {
    if (std.mem.containsAtLeast(SIG, &handle_signals, 1, &.{sig})) {
        // Set to -1 to signal saves to stop and then wait for saves to finish
        const saving_count = saving_threads.swap(-1, .monotonic);
        while (saving_threads.load(.monotonic) >= -saving_count) {
            std.Thread.yield() catch {};
        }
        std.process.exit(0);
    }
}

fn handleExitWindows(sig: c_int, _: c_int) callconv(.C) void {
    handleExit(sig);
}

fn loadOrInit(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    population_size: usize,
    options: Trainer.Options,
) !struct { Trainer, []?f64, u64 } {
    // Init if file does not exist
    Dir.cwd().access(io, path, .{}) catch |e|
        if (e == Dir.AccessError.FileNotFound) {
            var trainer = try Trainer.init(
                allocator,
                population_size,
                0.6,
                options,
            );
            errdefer trainer.deinit();
            const fitnesses = try allocator.alloc(?f64, population_size);
            @memset(fitnesses, null);
            var seed: u64 = undefined;
            io.random(std.mem.asBytes(&seed));
            return .{
                trainer,
                fitnesses,
                seed,
            };
        } else return e;

    const file = try Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var file_reader = file.reader(io, &buf);

    var reader: json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    const parsed = try json.parseFromTokenSource(
        SaveJson,
        allocator,
        &reader,
        .{},
    );
    defer parsed.deinit();

    var trainer: Trainer = try .from(allocator, parsed.value.trainer);
    trainer.options = options;
    errdefer trainer.deinit();

    const fitnesses = try allocator.alloc(?f64, trainer.population.len);
    @memcpy(fitnesses[0..parsed.value.fitnesses.len], parsed.value.fitnesses);
    @memset(fitnesses[parsed.value.fitnesses.len..], null);
    // Unset any NaNs from previous runs
    for (fitnesses) |*fit| {
        if (fit.* != null and std.math.isNan(fit.*)) {
            fit.* = null;
        }
    }

    return .{
        trainer,
        fitnesses,
        parsed.value.seed,
    };
}

fn save(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    seed: u64,
    trainer: Trainer,
    fitnesses: []const ?f64,
) !void {
    // Multiple threads can save at the same time, but no threads should start
    // saving when we are exiting
    if (saving_threads.load(.monotonic) < 0) {
        return;
    }

    _ = saving_threads.fetchAdd(1, .monotonic);
    defer _ = saving_threads.fetchSub(1, .monotonic);

    const trainer_json: Trainer.TrainerJson = try .init(allocator, trainer);
    defer trainer_json.deinit(allocator);

    try Dir
        .cwd()
        .createDirPath(io, std.fs.path.dirname(path) orelse return error.InvalidPath);
    const file = try Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writer(io, &buf);

    try writer.interface.print("{f}", .{
        json.fmt(SaveJson{
            .seed = seed,
            .trainer = trainer_json,
            .fitnesses = fitnesses,
        }, .{}),
    });
    try writer.flush();
}

/// Performs saving on a new thread so that it is not interrupted by the exit
/// handler.
fn saveSafe(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    seed: u64,
    trainer: Trainer,
    fitnesses: []?f64,
) !void {
    const save_thread: std.Thread = try .spawn(
        .{ .allocator = allocator },
        save,
        .{ io, allocator, path, seed, trainer, fitnesses },
    );
    save_thread.join();
}

fn printGenerationStats(trainer: Trainer, fitnesses: []const f64) void {
    var avg_fitness: f64 = 0;
    for (fitnesses) |f| {
        avg_fitness += f;
    }
    avg_fitness /= @floatFromInt(fitnesses.len);

    std.debug.print("\nGen {}, Species: {}, Avg fit: {d:.3}, Max Fit: {d:.3}\n\n", .{
        trainer.generation,
        trainer.species.len,
        avg_fitness,
        std.mem.max(f64, fitnesses),
    });
}

fn doWork(
    io: std.Io,
    allocator: Allocator,
    seed: *const u64,
    trainer: *const Trainer,
    fitnesses: []?f64,
    fitnesses_lock: *Mutex,
) !void {
    while (true) {
        const i = blk: {
            const old_cancel_protect = io.swapCancelProtection(.blocked);
            defer _ = io.swapCancelProtection(old_cancel_protect);
            fitnesses_lock.lock(io) catch unreachable;
            defer fitnesses_lock.unlock(io);

            for (fitnesses, 0..) |fit, i| {
                if (fit == null) {
                    // Reserve this index by setting it to NaN
                    fitnesses[i] = std.math.nan(f64);
                    break :blk i;
                }
            }
            break :blk trainer.population.len;
        };
        // Wait for work if all fitnesses have been calculated
        if (i == trainer.population.len) {
            io.futexWaitUncancelable(WideBool, &generation_done, .true);
            continue;
        }

        // Update fitness of genome i
        var inputs_used: [OPTIONS.nn_options.input_count]bool = undefined;
        var nn: neat.NN = try .init(
            allocator,
            trainer.population[i],
            OPTIONS.nn_options,
            &inputs_used,
        );
        defer nn.deinit(allocator);

        const fitness = try getFitness(io, allocator, seed.*, .{
            .inputs_used = inputs_used,
            .net = nn,
        });
        fitnesses[i] = fitness;
        std.debug.print("NN {}: {d:.3}\n", .{ i, fitness });
    }
}

fn getFitness(io: std.Io, allocator: Allocator, seed: u64, nn: NN) !f64 {
    const RUN_COUNT = 100;

    const placements = try allocator.alloc(root.Placement, HEIGHT * 10 / 4);
    defer allocator.free(placements);
    const pieces = try allocator.alloc(engine.pieces.PieceKind, placements.len + 1);
    defer allocator.free(pieces);

    var rand: std.Random.DefaultPrng = .init(seed);
    const start: std.Io.Timestamp = .now(io, .real);
    for (0..RUN_COUNT) |i| {
        // Start at random position in bag
        var bag: SevenBag = .init(rand.next());
        bag.random.random().shuffle(engine.pieces.PieceKind, &bag.pieces);
        bag.index = bag.random.random().uintLessThan(u8, bag.pieces.len);
        for (pieces) |*p| {
            p.* = bag.next();
        }

        // Optimize for 4 line PCs
        const solution = pc.findPc(
            .{
                .allocator = allocator,
                .playfield = .{},
                .pieces = pieces,
                .kicks = engine.kicks.srs,
                .min_height = HEIGHT,
                .nn = nn,
            },
            placements,
        ) catch |e| {
            if (e != root.FindPcError.SolutionTooLong) {
                return e;
            } else {
                continue;
            }
        };
        std.mem.doNotOptimizeAway(solution);

        // Return fitness early if too low every 10 runs
        if (i == 0 or i % 10 != 0) {
            continue;
        }
        const time_taken = @as(f64, @floatFromInt(start.untilNow(io, .real).toNanoseconds())) /
            @as(f64, @floatFromInt(i));
        const fitness = time.ns_per_s / (time_taken + 1);
        if (fitness < 15) {
            return fitness;
        }
    }

    // Return solutions/s as fitness
    const time_taken = @as(f64, @floatFromInt(start.untilNow(io, .real).toNanoseconds())) / RUN_COUNT;
    // Add 1 to avoid division by zero
    return time.ns_per_s / (time_taken + 1);
}
