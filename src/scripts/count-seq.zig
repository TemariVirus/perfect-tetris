const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const next = @import("perfect-tetris").next;
const PieceKind = next.PieceKind;

/// Multi-threaded version of `next.SequenceIterator` meant for counting the number of unique sequences.
/// Refer to `next.SequenceIterator` for implementation details.
/// This struct makes use of the fact that `lock_iter` splits the computation of sequences into independent chunks,
/// which makes counting embarrassingly parallel.
fn SequenceCounter(comptime len: usize) type {
    const unlocked = @min(6, len - 1);
    assert(len > 2);
    assert(unlocked <= len - 1);
    assert(len <= 64 / 3);
    return struct {
        pub const lock_len = @max(1, len - unlocked - 1);
        pub const LockIter = next.DigitsIterator(lock_len);
        pub const NextIter = next.NextIterator(len - 1, lock_len);
        const SequenceSet = std.AutoHashMap(u64, void);
        const NextArray = next.PieceArray(len);

        pub const Worker = struct {
            current: NextArray,
            next_iter: NextIter,
            seen: SequenceSet,

            pub fn init(allocator: Allocator) @This() {
                var self = @This(){
                    .current = undefined,
                    .next_iter = undefined,
                    .seen = .init(allocator),
                };
                self.next_iter.sizes[0] = 0; // Mark as done
                return self;
            }

            pub fn deinit(self: *@This()) void {
                self.seen.deinit();
            }
        };

        lock_iter: LockIter,

        pub fn init() @This() {
            return .{
                // By letting the first piece kind be a "wildcard", we only
                // need to take every 7th sequence, reducing the size of `seen`
                // by a factor of 7.
                .lock_iter = .init(0, 7),
            };
        }

        pub fn countChunk(self: *@This(), worker: *Worker) !?u64 {
            // We have exhausted all holds, advance the next iterator
            if (self.advanceNext(worker)) |pieces| {
                worker.current = .init(pieces.items);
                // No need to set the hold as it is already 0
                // worker.current = worker.current.set(len - 1, 0);
            } else {
                // Iterator is exhausted
                return null;
            }

            var count: u64 = 0;
            inline for (0..7) |_| {
                // NextIterator doesn't guarantee unique sequences, so we need
                // to check by ourselves
                const result = try worker.seen.getOrPut(canonical(worker.current).items);
                count += @intFromBool(!result.found_existing);
                // Increment hold to next piece
                worker.current = worker.current.increment(len - 1);
            }

            // Swap wildcard with each of the 7 pieces
            return count * 7;
        }

        /// Puts the array in cannonical order by ensuring that the hold is
        /// smaller than the current next piece.
        inline fn canonical(current: NextArray) NextArray {
            if (current.get(len - 1) <= current.get(len - 2)) {
                return current;
            }
            return current
                .set(len - 2, current.get(len - 1))
                .set(len - 1, current.get(len - 2));
        }

        inline fn advanceNext(self: *@This(), worker: *Worker) ?NextIter.NextArray {
            while (true) {
                if (worker.next_iter.next()) |pieces| {
                    return pieces;
                }

                // If next is exhausted, advance the locks to the next chunk
                if (self.advanceLocks()) |locks| {
                    worker.next_iter = .init(locks);
                } else {
                    return null;
                }
                if (worker.seen.count() > 0) {
                    worker.seen.clearRetainingCapacity();
                }
            }
        }

        fn advanceLocks(self: *@This()) ?[lock_len]u3 {
            if (@atomicLoad(u64, &self.lock_iter.value, .monotonic) >= LockIter.end) {
                return null;
            }

            var digits: [lock_len]u3 = undefined;
            var value = @atomicRmw(u64, &self.lock_iter.value, .Add, self.lock_iter.step, .monotonic);
            if (value >= LockIter.end) {
                return null;
            }

            for (0..lock_len) |i| {
                digits[i] = @intCast(value % LockIter.base);
                value /= LockIter.base;
            }
            return digits;
        }
    };
}

fn workerThread(comptime len: usize) fn (
    Allocator,
    *SequenceCounter(len),
    *std.atomic.Value(u64),
) void {
    const Counter = SequenceCounter(len);
    return (struct {
        fn func(allocator: Allocator, counter: *Counter, total_count: *std.atomic.Value(u64)) void {
            var worker: Counter.Worker = .init(allocator);
            defer worker.deinit();

            var count: u64 = 0;
            while (counter.countChunk(&worker) catch unreachable) |c| {
                count += c;
            }
            _ = total_count.fetchAdd(count, .monotonic);
        }
    }).func;
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{
        .stack_size = 256 * 1024,
    });
    defer threaded.deinit();
    const io = threaded.io();

    std.debug.print("| Length | Non-equivalent     | Duration   |\n", .{});
    std.debug.print("| ------ | ------------------ | ---------- |\n", .{});

    const cpu_count = std.Thread.getCpuCount() catch 4;
    inline for (3..22) |n| {
        var group: std.Io.Group = .init;
        defer group.cancel(io);

        const Counter = SequenceCounter(n);
        var counter: Counter = .init();
        var count: std.atomic.Value(u64) = .init(0);

        const start: std.Io.Timestamp = .now(io, .real);
        for (0..cpu_count) |_| {
            group.concurrent(io, workerThread(n), .{ allocator, &counter, &count }) catch {
                workerThread(n)(allocator, &counter, &count);
                break;
            };
        }
        try group.await(io);
        const time_taken = start.untilNow(io, .real);

        var time_buf: [34]u8 = undefined;
        const time_taken_str = std.fmt.bufPrint(&time_buf, "{f}", .{time_taken}) catch unreachable;
        std.debug.print("| {d:<6} | {d:<18} | {s:<10} |\n", .{
            n - 1,
            count.raw,
            time_taken_str,
        });
    }
}
