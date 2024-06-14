const std = @import("std");
const Allocator = std.mem.Allocator;
const AtomicBool = std.atomic.Value(i32);
const fs = std.fs;
const json = std.json;
const Mutex = std.Thread.Mutex;
const os = std.os;
const SIG = os.linux.SIG;
const time = std.time;

const engine = @import("engine");
const GameState = engine.GameState(SevenBag);
const SevenBag = engine.bags.SevenBag;

const zmai = @import("zmai");
const neat = zmai.genetic.neat;
const Trainer = neat.Trainer;

const root = @import("root.zig");
const NN = root.NN;
const pc = root.pc;

const THREADS = 4;
const SAVE_INTERVAL = 10 * time.ns_per_s;
const SAVE_DIR = "pops/";

const GENERATIONS = 20;
const POPULATION_SIZE = 150;
const OPTIONS = Trainer.Options{
    .species_target = 10,
    .nn_options = .{
        .input_count = 5,
        .output_count = 1,
        .hidden_activation = .swish,
        .output_activation = .gaussian,
    },
};

var saving_threads = AtomicBool.init(0);

const SaveJson = struct {
    seed: u64,
    trainer: Trainer.TrainerJson,
    fitnesses: []const ?f64,
};

pub fn main() !void {
    setupExitHandler();
    zmai.setRandomSeed(std.crypto.random.int(u64));

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var trainer, const fitnesses, var seed = try loadOrInit(
        allocator,
        SAVE_DIR ++ "current.json",
        POPULATION_SIZE,
        OPTIONS,
    );
    defer trainer.deinit();
    defer allocator.free(fitnesses);

    var fitnesses_lock = Mutex{};
    var threads: [THREADS]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(
            .{ .allocator = allocator },
            doWork,
            .{ &seed, &trainer, fitnesses, &fitnesses_lock },
        );
    }
    defer for (threads) |thread| {
        thread.join();
    };

    var save_timer = engine.PeriodicTrigger.init(SAVE_INTERVAL);
    outer: while (trainer.generation < GENERATIONS) {
        time.sleep(time.ns_per_ms);
        if (save_timer.trigger()) |_| {
            try saveSafe(allocator, SAVE_DIR ++ "current.json", seed, trainer, fitnesses);
        }

        // Wait for all fitnesses to be calculated
        for (fitnesses) |fit| {
            if (std.math.isNan(fit orelse continue :outer)) {
                continue :outer;
            }
        }

        const path = try std.fmt.allocPrint(allocator, "{s}{}.json", .{ SAVE_DIR, trainer.generation });
        defer allocator.free(path);
        try saveSafe(allocator, path, seed, trainer, fitnesses);

        const final_fitnesses = try allocator.alloc(f64, fitnesses.len);
        defer allocator.free(final_fitnesses);
        for (0..fitnesses.len) |i| {
            final_fitnesses[i] = fitnesses[i] orelse unreachable;
        }

        printGenerationStats(trainer, final_fitnesses);
        seed = std.crypto.random.int(u64);
        if (trainer.generation != GENERATIONS - 1) {
            try trainer.nextGeneration(final_fitnesses);
        }

        fitnesses_lock.lock();
        @memset(fitnesses, null);
        fitnesses_lock.unlock();
    }
}

fn setupExitHandler() void {
    if (@import("builtin").os.tag == .windows) {
        const signal = struct {
            extern "c" fn signal(
                sig: c_int,
                func: *const fn (c_int, c_int) callconv(os.windows.WINAPI) void,
            ) callconv(.C) *anyopaque;
        }.signal;
        _ = signal(SIG.INT, handleExitWindows);
    } else {
        const action = os.linux.Sigaction{
            .handler = .{ .handler = handleExit },
            .mask = os.linux.empty_sigset,
            .flags = 0,
        };
        _ = os.linux.sigaction(SIG.INT, &action, null);
    }
}

fn handleExit(sig: c_int) callconv(.C) void {
    switch (sig) {
        SIG.INT => {
            const TIMEOUT = 10 * time.ns_per_s;
            const start = time.nanoTimestamp();

            // Set to -1 to signal saves to stop and wait for saves to finish
            const saving_count = saving_threads.swap(-1, .monotonic);
            while (saving_threads.raw >= -saving_count) {
                // Force stop if saves take too long to finish
                if (time.nanoTimestamp() - start > TIMEOUT) {
                    break;
                }
                time.sleep(time.ns_per_ms);
            }
            std.process.exit(0);
        },
        // This handler is only registered for SIG.INT
        else => unreachable,
    }
}

fn handleExitWindows(sig: c_int, _: c_int) callconv(.C) void {
    handleExit(sig);
}

fn loadOrInit(
    allocator: Allocator,
    path: []const u8,
    population_size: usize,
    options: Trainer.Options,
) !struct { Trainer, []?f64, u64 } {
    // Init if file does not exist
    fs.cwd().access(path, .{}) catch |e| if (e == fs.Dir.AccessError.FileNotFound) {
        var trainer = try Trainer.init(allocator, population_size, 1.0, options);
        errdefer trainer.deinit();
        const fitnesses = try allocator.alloc(?f64, population_size);
        @memset(fitnesses, null);
        return .{
            trainer,
            fitnesses,
            std.crypto.random.int(u64),
        };
    } else return e;

    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    var reader = json.reader(allocator, file.reader());
    defer reader.deinit();

    const parsed = try json.parseFromTokenSource(
        SaveJson,
        allocator,
        &reader,
        .{},
    );
    defer parsed.deinit();

    var trainer = try Trainer.from(allocator, parsed.value.trainer);
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
    allocator: Allocator,
    path: []const u8,
    seed: u64,
    trainer: Trainer,
    fitnesses: []const ?f64,
) !void {
    // Multiple threads can save at the same time, but no threads should start
    // saving when we are exiting
    if (saving_threads.raw < 0) {
        return;
    }

    _ = saving_threads.fetchAdd(1, .monotonic);
    defer _ = saving_threads.fetchSub(1, .monotonic);

    const trainer_json = try Trainer.TrainerJson.init(allocator, trainer);
    defer trainer_json.deinit(allocator);

    try fs.cwd().makePath(std.fs.path.dirname(path) orelse return error.InvalidPath);
    const file = try fs.cwd().createFile(path, .{});
    defer file.close();

    try json.stringify(SaveJson{
        .seed = seed,
        .trainer = trainer_json,
        .fitnesses = fitnesses,
    }, .{}, file.writer());
}

/// Performs saving on a new thread so that it is not interrupted by the exit
/// handler.
fn saveSafe(
    allocator: Allocator,
    path: []const u8,
    seed: u64,
    trainer: Trainer,
    fitnesses: []?f64,
) !void {
    const saveOnceThread = struct {
        fn saveOnceThread(
            _path: []const u8,
            _seed: u64,
            _trainer: Trainer,
            _fitnesses: []?f64,
        ) !void {
            var gpa = std.heap.GeneralPurposeAllocator(.{}){};
            const _allocator = gpa.allocator();
            defer _ = gpa.deinit();

            try save(_allocator, _path, _seed, _trainer, _fitnesses);
        }
    }.saveOnceThread;

    const save_thread = try std.Thread.spawn(
        .{ .allocator = allocator },
        saveOnceThread,
        .{ path, seed, trainer, fitnesses },
    );
    save_thread.detach();
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
    seed: *const u64,
    trainer: *const Trainer,
    fitnesses: []?f64,
    fitnesses_lock: *Mutex,
) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    while (true) {
        const i = blk: {
            fitnesses_lock.lock();
            defer fitnesses_lock.unlock();

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
            time.sleep(time.ns_per_ms);
            continue;
        }

        // Update fitness of genome i
        var inputs_used: [OPTIONS.nn_options.input_count]bool = undefined;
        var nn = try neat.NN.init(
            allocator,
            trainer.population[i],
            OPTIONS.nn_options,
            &inputs_used,
        );
        defer nn.deinit(allocator);

        const fitness = try getFitness(allocator, seed.*, .{
            .inputs_used = inputs_used,
            .net = nn,
        });
        fitnesses[i] = fitness;
        std.debug.print("NN {}: {d:.3}\n", .{ i, fitness });
    }
}

fn getFitness(allocator: Allocator, seed: u64, nn: NN) !f64 {
    const RUN_COUNT = 100;

    var rand = std.Random.DefaultPrng.init(seed);
    var timer = try time.Timer.start();
    for (0..RUN_COUNT) |i| {
        // Start at random position in bag
        var bag = SevenBag.init(rand.next());
        bag.random.random().shuffle(engine.pieces.PieceKind, &bag.pieces);
        bag.index = bag.random.random().uintLessThan(u8, bag.pieces.len);
        const gamestate = GameState.init(
            bag,
            engine.kicks.srsPlus,
        );

        // Optimize for 4 line PCs (but not all states have a 4 line PC so
        // provide extra pieces for a 6 line PC)
        const solution = try pc.findPc(allocator, gamestate, nn, 4, 16);
        std.mem.doNotOptimizeAway(solution);
        allocator.free(solution);

        if (i % 10 != 0) {
            continue;
        }
        // Return fitness early if too low
        const time_taken = @as(f64, @floatFromInt(timer.read())) / RUN_COUNT;
        const fitness = time.ns_per_s / (time_taken + time.ns_per_ms);
        if (fitness < 5) {
            return fitness;
        }
    }

    // Return solutions/s as fitness
    const time_taken = @as(f64, @floatFromInt(timer.read())) / RUN_COUNT;
    // Add 1ms to avoid division by zero
    return time.ns_per_s / (time_taken + time.ns_per_ms);
}
