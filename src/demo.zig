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

const FRAMERATE = 60;
const FPS_TIMING_WINDOW = FRAMERATE * 2;
/// The maximum number of perfect clears to calculate in advance.
const MAX_PC_QUEUE = 128;

pub const DemoArgs = struct {
    help: bool = false,
    nn: ?[]const u8 = null,
    pps: u32 = 10,

    pub const wrap_len: u32 = 45;

    pub const shorthands = .{
        .h = "help",
        .n = "nn",
        .p = "pps",
    };

    pub const meta = .{
        .usage_summary = "demo [options]",
        .full_text = "Demostrates the perfect clear solver's speed with a tetris playing bot.",
        .option_docs = .{
            .help = "Print this help message.",
            .nn = "The path to the neural network to use for the solver. The path may be absolute, relative to the current working directory, or relative to the executable's directory. If not provided, a default built-in NN will be used.",
            .pps = std.fmt.comptimePrint("The target pieces per second of the bot. (default: {})", .{(DemoArgs{}).pps}),
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
        SevenBag.init(0),
        kicks.srsPlus,
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

    const pc_thread: std.Thread = try .spawn(.{
        .allocator = allocator,
    }, pcThread, .{ allocator, nn, player.state, &pc_queue });
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
            placePcPiece(allocator, &player, &pc_queue, &placement_i);
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
        allocator.free(queue.orderedRemove(0));
        placement_i.* = 0;
    }
}

fn pcThread(
    allocator: Allocator,
    nn: ?NN,
    state: GameState,
    queue: *SolutionList,
) !void {
    var game = state;

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
            2,
            15,
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

        try queue.append(solution);
    }
}

/// Dummy function to satisfy the Player struct.
fn playSfxDummy(_: engine.player.Sfx) void {}
