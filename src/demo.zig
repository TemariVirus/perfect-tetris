const std = @import("std");
const Allocator = std.mem.Allocator;
const SolutionList = std.ArrayListUnmanaged([]Placement);
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
const pathfind = root.pathfind;

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

const Move = union(enum) {
    hold: void,
    left: void,
    right: void,
    rotate_cw: void,
    rotate_double: void,
    rotate_ccw: void,
    softdrop: u6,
    harddrop: void,
};

const AutoPlayer = struct {
    allocator: Allocator,
    player: Player,

    solve_signal: std.Thread.Semaphore = .{ .permits = MAX_PC_QUEUE },
    solutions_mutex: std.Thread.Mutex = .{},
    solutions: SolutionList,

    pps: u32,
    countdown: u64 = 0,
    ns_per_move: u64 = 0,
    moves: std.ArrayListUnmanaged(Move) = .empty,

    pub fn init(allocator: Allocator, player: Player, pps: u32) !AutoPlayer {
        return AutoPlayer{
            .allocator = allocator,
            .player = player,
            .solutions = try .initCapacity(allocator, MAX_PC_QUEUE),
            .pps = pps,
        };
    }

    pub fn deinit(self: *AutoPlayer) void {
        // Stop all incoming solves before freeing `self.solutions`
        {
            self.solve_signal.mutex.lock();
            defer self.solve_signal.mutex.unlock();
            self.solve_signal.permits = 0;
        }
        {
            self.solutions_mutex.lock();
            defer self.solutions_mutex.unlock();
            for (self.solutions.items) |sol| {
                self.allocator.free(sol);
            }
            self.solutions.deinit(self.allocator);
        }
        self.moves.deinit(self.allocator);
    }

    pub fn tick(self: *AutoPlayer, nanoseconds: u64) Allocator.Error!void {
        self.tickPlayer(nanoseconds);

        var ns = nanoseconds;
        while (ns > 0) {
            while (self.moves.items.len == 0) {
                if (self.solutions.items.len == 0) {
                    return;
                }
                try self.nextMoves();
            }

            const advanced_ns = @min(self.countdown, ns);
            ns -= advanced_ns;
            self.makeMove(advanced_ns);
        }
    }

    fn tickPlayer(self: *AutoPlayer, nanoseconds: u64) void {
        // Prevent auto-locking
        self.player.move_count = 0;
        self.player.last_move_time = std.math.maxInt(u64);
        self.player.tick(nanoseconds, 0, &.{});
    }

    fn nextMoves(self: *AutoPlayer) Allocator.Error!void {
        self.moves.clearRetainingCapacity();

        const solution = self.nextSolution();
        defer self.allocator.free(solution);

        var gamestate = self.player.state;
        for (solution) |placement| {
            if (gamestate.current.kind != placement.piece.kind) {
                gamestate.hold();
                try self.moves.append(self.allocator, .hold);
            }

            const path = pathfind.pathfind(
                gamestate.playfield,
                gamestate.kicks,
                .{ .piece = gamestate.current, .pos = gamestate.pos },
                placement,
            ).?;

            var drop_count: u6 = 0;
            for (path.constSlice()) |move| {
                switch (move) {
                    .drop => drop_count += 1,
                    else => {
                        if (drop_count > 0) {
                            try self.moves.append(self.allocator, .{ .softdrop = drop_count });
                            drop_count = 0;
                        }
                        try self.moves.append(self.allocator, switch (move) {
                            .left => .left,
                            .right => .right,
                            .rotate_cw => .rotate_cw,
                            .rotate_double => .rotate_double,
                            .rotate_ccw => .rotate_ccw,
                            .drop => unreachable,
                        });
                    },
                }
            }

            gamestate.current = placement.piece;
            gamestate.pos = placement.pos;
            _ = gamestate.lockCurrent(-1);
            gamestate.nextPiece();
            try self.moves.append(self.allocator, .harddrop);
        }

        self.ns_per_move = time.ns_per_s * solution.len / (self.pps * self.moves.items.len);
        self.countdown = self.ns_per_move;
    }

    fn nextSolution(self: *AutoPlayer) []Placement {
        const solution = blk: {
            self.solutions_mutex.lock();
            defer self.solutions_mutex.unlock();
            break :blk self.solutions.orderedRemove(0);
        };
        self.solve_signal.post();
        return solution;
    }

    fn makeMove(self: *AutoPlayer, advanced_ns: u64) void {
        self.countdown -= advanced_ns;
        defer if (self.countdown == 0) {
            self.countdown = self.ns_per_move;
            _ = self.moves.orderedRemove(0);
        };
        const move = self.moves.items[0];

        if (move == .softdrop) {
            const dy = move.softdrop;
            const ns_per_drop = self.ns_per_move / dy;

            const old_y = (self.countdown + advanced_ns) / ns_per_drop;
            const new_y = self.countdown / ns_per_drop;
            const dropped = old_y - new_y;
            self.player.state.pos.y -= @intCast(dropped);
            self.player.score += dropped;
            return;
        }

        if (self.countdown > 0) {
            return;
        }

        switch (move) {
            .hold => self.player.hold(),
            .left => self.player.moveLeft(false),
            .right => self.player.moveRight(false),
            .rotate_cw => self.player.rotateCw(),
            .rotate_double => self.player.rotateDouble(),
            .rotate_ccw => self.player.rotateCcw(),
            .softdrop => unreachable,
            .harddrop => self.player.hardDrop(0, &.{}),
        }
    }

    pub fn appendSolution(self: *AutoPlayer, solution: []Placement) !void {
        self.solutions_mutex.lock();
        defer self.solutions_mutex.unlock();
        self.solutions.appendAssumeCapacity(try self.allocator.dupe(Placement, solution));
    }
};

pub fn main(allocator: Allocator, args: DemoArgs, nn: ?NN) !void {
    try nterm.init(
        allocator,
        std.io.getStdOut(),
        Player.DISPLAY_W,
        Player.DISPLAY_H,
        null,
        null,
    );
    defer nterm.deinit();

    const seed = args.seed orelse std.crypto.random.int(u64);
    const settings: engine.GameSettings = .{
        .g = 0,
        .autolock_grace = std.math.maxInt(u8),
        .lock_delay = std.math.maxInt(u32),
        .display_stats = .{
            .pps,
            .app,
            .sent,
        },
        .target_mode = .none,
    };
    var auto: AutoPlayer = try .init(allocator, .init(
        "PC Solver",
        SevenBag.init(seed),
        args.kicks.toEngine(),
        settings,
        .{
            .left = 0,
            .top = 0,
            .width = Player.DISPLAY_W,
            .height = Player.DISPLAY_H,
        },
        playSfxDummy,
    ), args.pps);

    const pc_thread: std.Thread = try .spawn(
        .{ .allocator = allocator },
        pcThread,
        .{ nn, args.@"min-height", auto.player.state, &auto },
    );
    pc_thread.detach();

    var render_timer: PeriodicTrigger = .init(time.ns_per_s / FRAMERATE, true);
    while (true) {
        if (render_timer.trigger()) |dt| {
            nterm.render() catch |err| {
                // Trying to render after the terminal has been closed results
                // in an error, in which case stop the program gracefully.
                if (err == error.NotInitialized) {
                    return;
                }
                return err;
            };
            auto.tick(dt) catch @panic("OOM");
            auto.player.draw();
        }
        time.sleep(1 * time.ns_per_ms);
    }
}

fn pcThread(nn: ?NN, min_height: u7, state: GameState, auto: *AutoPlayer) !void {
    const allocator = auto.allocator;
    var game = state;
    // A 2- or 4-line PC is not always possible. 15 placements is enough
    // for a 6-line PC.
    const max_len = @max(15, ((@as(usize, min_height) + 1) * 10 / 4));

    while (true) {
        auto.solve_signal.wait();

        const solution = try root.findPcAuto(
            SevenBag,
            allocator,
            game,
            nn,
            min_height,
            max_len,
            null,
        );
        defer allocator.free(solution);
        for (solution) |placement| {
            if (game.current.kind != placement.piece.kind) {
                game.hold();
            }
            game.current = placement.piece;
            game.pos = placement.pos;
            _ = game.lockCurrent(-1);
            game.nextPiece();
        }

        try auto.appendSolution(solution);
    }
}

/// Dummy function to satisfy the Player struct.
fn playSfxDummy(_: engine.player.Sfx) void {}
