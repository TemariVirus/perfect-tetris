// TODO: Use pathfinding to better animate the PCs

const std = @import("std");
const Allocator = std.mem.Allocator;
const SolutionList = std.ArrayList([]Placement);
const time = std.time;

const engine = @import("engine");
const GameState = Player.GameState;
const kicks = engine.kicks;
const Player = engine.Player(SevenBag);
const SevenBag = engine.bags.SevenBag;

const nterm = @import("nterm");
const Colors = nterm.Colors;
const PeriodicTrigger = nterm.PeriodicTrigger;
const View = nterm.View;

const root = @import("perfect-tetris");
const NN = root.NN;
const Placement = root.Placement;

const enumValuesHelp = @import("main.zig").enumValuesHelp;
const KicksOption = @import("main.zig").KicksOption;

const FRAMERATE = 60;
const FPS_TIMING_WINDOW = FRAMERATE * 2;
/// The maximum number of perfect clears to calculate in advance.
const MAX_PC_QUEUE = 128;

pub const DemoArgs = struct {
    help: bool = false,
    kicks: KicksOption = .srsPlus,
    @"min-height": u7 = 4,
    nn: ?[]const u8 = null,
    pps: u32 = 10,
    seed: ?u64 = null,

    pub const wrap_len: u32 = 48;

    pub const shorthands = .{
        .h = "help",
        .k = "kicks",
        .m = "min-height",
        .n = "nn",
        .p = "pps",
        .s = "seed",
    };

    pub const meta = .{
        .usage_summary = "demo [options]",
        .full_text = "Demostrates the perfect clear solver's speed with a tetris playing bot.",
        .option_docs = .{
            .help = "Print this help message.",
            .kicks = std.fmt.comptimePrint(
                "Kick/rotation system to use. For kick systems that have a 180-less and 180 variant, the 180-less variant has no 180 rotations. The 180 variant has 180 rotations but no 180 kicks. " ++
                    enumValuesHelp(KicksOption) ++
                    " (default: {s})",
                .{@tagName((DemoArgs{}).kicks)},
            ),
            .@"min-height" = std.fmt.comptimePrint(
                "The minimum height of PCs to find. (default: {d})",
                .{(DemoArgs{}).@"min-height"},
            ),
            .nn = "The path to the neural network to use for the solver. The path may be absolute, relative to the current working directory, or relative to the executable's directory. If not provided, a default built-in NN will be used.",
            .pps = std.fmt.comptimePrint(
                "The target pieces per second of the bot. (default: {d})",
                .{(DemoArgs{}).pps},
            ),
            .seed = "The seed to use for the 7-bag randomizer. If not provided, a random value will be used.",
        },
    };
};

pub fn main(allocator: Allocator, args: DemoArgs, nn: ?NN) !void {
    // Add 2 to create a 1-wide empty boarder on the left and right.
    try nterm.init(
        allocator,
        std.io.getStdOut(),
        Player.DISPLAY_W + 2,
        Player.DISPLAY_H,
        null,
        null,
    );
    defer nterm.deinit();

    const seed = args.seed orelse std.crypto.random.int(u64);
    const settings: engine.GameSettings = .{
        .g = 0,
        .display_stats = .{
            .pps,
            .app,
            .sent,
        },
        .target_mode = .none,
    };
    var player: Player = .init(
        "PC Solver",
        SevenBag.init(seed),
        args.kicks.toEngine(),
        settings,
        .{
            .left = 1,
            .top = 0,
            .width = Player.DISPLAY_W,
            .height = Player.DISPLAY_H,
        },
        playSfxDummy,
    );

    var placement_i: usize = 0;
    var pc_queue: SolutionList = try .initCapacity(allocator, MAX_PC_QUEUE);
    defer pc_queue.deinit();
    defer for (pc_queue.items) |solution| {
        allocator.free(solution);
    };
    var pc_queue_lock: std.Thread.Mutex = .{};

    const pc_thread: std.Thread = try .spawn(.{
        .allocator = allocator,
    }, pcThread, .{
        allocator,
        nn,
        args.@"min-height",
        player.state,
        &pc_queue,
        &pc_queue_lock,
    });
    pc_thread.detach();

    var render_timer: PeriodicTrigger = .init(time.ns_per_s / FRAMERATE, true);
    var place_timer: PeriodicTrigger = .init(time.ns_per_s / args.pps, false);
    while (true) {
        var triggered = false;

        if (render_timer.trigger()) |dt| {
            triggered = true;
            player.tick(dt, 0, &.{});
            player.draw();
            nterm.render() catch |err| {
                // Trying to render after the terminal has been closed results
                // in an error, in which case stop the program gracefully.
                if (err == error.NotInitialized) {
                    return;
                }
                return err;
            };
        }
        while (place_timer.trigger()) |_| {
            triggered = true;
            placePcPiece(
                allocator,
                &player,
                &pc_queue,
                &pc_queue_lock,
                &placement_i,
            );
        }

        if (!triggered) {
            time.sleep(1 * time.ns_per_ms);
        }
    }
}

fn placePcPiece(
    allocator: Allocator,
    game: *Player,
    queue: *SolutionList,
    queue_lock: *std.Thread.Mutex,
    placement_i: *usize,
) void {
    if (queue.items.len == 0) {
        return;
    }
    const placements = queue.items[0];

    const placement = placements[placement_i.*];
    if (placement.piece.kind != game.state.current.kind) {
        game.hold();
    }
    game.state.pos = placement.pos;
    game.state.current = placement.piece;
    game.hardDrop(0, &.{});
    placement_i.* += 1;

    // Start next perfect clear
    if (placement_i.* == placements.len) {
        queue_lock.lock();
        defer queue_lock.unlock();
        allocator.free(queue.orderedRemove(0));
        placement_i.* = 0;
    }
}

fn pcThread(
    allocator: Allocator,
    nn: ?NN,
    min_height: u7,
    state: GameState,
    queue: *SolutionList,
    queue_lock: *std.Thread.Mutex,
) !void {
    var game = state;
    const max_len = @max(15, ((@as(usize, min_height) + 1) * 10 / 4));

    while (true) {
        while (queue.items.len >= MAX_PC_QUEUE) {
            time.sleep(time.ns_per_ms);
        }

        // A 2- or 4-line PC is not always possible. 15 placements is enough
        // for a 6-line PC.
        const solution = try root.findPcAuto(
            SevenBag,
            allocator,
            game,
            nn,
            min_height,
            max_len,
            null,
        );
        for (solution) |placement| {
            if (game.current.kind != placement.piece.kind) {
                game.hold();
            }
            game.current = placement.piece;
            game.pos = placement.pos;
            _ = game.lockCurrent(-1);
            game.nextPiece();
        }

        {
            queue_lock.lock();
            defer queue_lock.unlock();
            queue.appendAssumeCapacity(solution);
        }
    }
}

/// Dummy function to satisfy the Player struct.
fn playSfxDummy(_: engine.player.Sfx) void {}
