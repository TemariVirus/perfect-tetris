//! Iterators for generating all possible sequences of next pieces, assuming a
//! 7-bag randomiser is used.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const PieceKind = @import("engine").pieces.PieceKind;

const MAX_BAG_LEN = 7;
const BagLenInt = std.math.Log2IntCeil(@Type(.{ .int = .{
    .bits = MAX_BAG_LEN,
    .signedness = .unsigned,
} }));

/// Stores an immutable array of integers (std.PackedIntArray but with a single
/// backing integer).
pub fn PieceArray(comptime len: usize) type {
    assert(len > 0);
    assert(len <= 64 / 3);
    assert(@bitSizeOf(PieceKind) == 3);
    return struct {
        pub const item_size = @bitSizeOf(u3);
        pub const bits = item_size * len;
        pub const mask = @as(u64, (1 << bits) - 1);
        pub const item_mask = @as(u64, (1 << item_size) - 1);

        // There is one extra bit that may be used too indicate nullity but
        // doing so shows no visible performance improvement.
        items: u64,
        comptime len: usize = len,

        pub fn init(items: u64) @This() {
            assert(mask & items == items);
            return .{ .items = items };
        }

        pub fn get(self: @This(), index: usize) u3 {
            assert(index < self.len);
            return @truncate(item_mask &
                (self.items >> @intCast(index * item_size)));
        }

        pub fn set(self: @This(), index: usize, value: u3) @This() {
            assert(index < self.len);
            // Mask is not needed at the end as all out of bounds bits are already 0
            assert(~mask & self.items == 0);
            return .{
                .items = self.items &
                    // Write 0 at index
                    ~(item_mask << @intCast(index * item_size)) |
                    // Write value at index
                    (@as(u64, value) << @intCast(index * item_size)),
            };
        }

        pub fn shiftLeft(self: @This(), shift: usize) @This() {
            assert(shift <= self.len);
            return .{
                .items = mask & (self.items << @intCast(shift * item_size)),
            };
        }

        pub fn shiftRight(self: @This(), shift: usize) @This() {
            assert(shift <= self.len);
            // Mask is not needed at the end as all out of bounds bits are already 0
            assert(~mask & self.items == 0);
            return .{ .items = self.items >> @intCast(shift * item_size) };
        }
    };
}

/// Iterates through all possible sequences of next pieces of length `len`
/// assuming a 7-bag randomiser. Due to partial bags, this iterator often
/// produces multiple sequences that look identical.
pub fn NextIterator(comptime len: usize, comptime lock_len: usize) type {
    assert(len > 1);
    assert(len >= lock_len);
    return struct {
        pub const n_bags = (len - 2) / MAX_BAG_LEN + 2;
        pub const NextArray = PieceArray(len);

        /// Iterates through a truncated 7-bag of length `size`.
        pub const BagIterator = struct {
            /// Indicates which pieces have not been used in previous indices.
            frees: [MAX_BAG_LEN]u7,
            /// Indicates which pieces are left to iterate through.
            iters: [MAX_BAG_LEN]u7,

            pub fn init(size: BagLenInt, locks: []const u3) @This() {
                assert(size > 0 and size <= MAX_BAG_LEN);

                var frees: [MAX_BAG_LEN]u7 = undefined;
                // First position may be any piece
                frees[0] = std.math.maxInt(u7);
                var iters: [MAX_BAG_LEN]u7 = undefined;
                iters[0] = if (0 < locks.len) lockedIter(locks[0]) else frees[0];
                for (1..size) |i| {
                    const selected_piece = lsb(iters[i - 1]);
                    frees[i] = frees[i - 1] ^ selected_piece;
                    iters[i] = if (i < locks.len)
                        lockedIter(locks[i])
                    else
                        frees[i];
                }

                // If locks are impossible to satisfy, return an empty iterator
                if (@popCount(frees[size - 1] ^ lsb(iters[size - 1])) != 7 - size) {
                    iters[0] = 0;
                }
                return .{ .frees = frees, .iters = iters };
            }

            inline fn lockedIter(lock: u3) u7 {
                // Only the locked piece may be iterated through
                return @as(u7, 1) << lock;
            }

            /// Returns the least significant bit of `x`.
            inline fn lsb(x: u7) u7 {
                return x & (~x +% 1);
            }

            /// Add the bags pieces to `pieces` at `offset`. If the iterator is
            /// exhausted, returns null.
            pub fn next(
                self: *@This(),
                pieces: NextArray,
                offset: usize,
                size: BagLenInt,
            ) ?NextArray {
                if (self.iters[0] == 0) {
                    return null;
                }

                // Shift pieces to the right offset, the write the bag's pieces
                var p = pieces.shiftRight(offset);
                for (0..size) |i| {
                    p = p.set(i, @ctz(self.iters[i]));
                }
                // Reverse the shift and write the other pieces back
                p = p.shiftLeft(offset);
                const lower = pieces.items &
                    ~(NextArray.mask << @intCast(offset * 3));
                p = .init(p.items | lower);

                // Advance iters
                var i = size - 1;
                self.iters[i] &= self.iters[i] - 1;
                while (i > 0) : (i -= 1) {
                    // If this index is exhausted, advance the previous one
                    if (self.iters[i] != 0) {
                        break;
                    }
                    // Remove the least significant bit as it just got iterated through
                    self.iters[i - 1] &= self.iters[i - 1] - 1;
                } else if (self.iters[0] == 0) {
                    return p;
                }

                // Refill exhausted iters and update frees
                while (i < size - 1) : (i += 1) {
                    const selected_piece = lsb(self.iters[i]);
                    self.frees[i + 1] = self.frees[i] ^ selected_piece;
                    self.iters[i + 1] = self.frees[i + 1];
                }

                return p;
            }
        };

        // Sizes can be reduced to a single int but the logic becomes more complicated.
        /// The current sizes of the bags.
        sizes: [n_bags]BagLenInt,
        bags: [n_bags]BagIterator,
        locks: [lock_len]u3,
        pieces: NextArray,

        /// Initializes the iterator, with the first few pieces locked to the
        /// values of `locks`.
        pub fn init(locks: [lock_len]u3) @This() {
            var size_left = len;
            var sizes: [n_bags]BagLenInt = undefined;
            for (0..n_bags) |i| {
                sizes[i] = @min(size_left, MAX_BAG_LEN);
                size_left -= sizes[i];
            }

            var bags: [n_bags]BagIterator = undefined;
            var start: usize = len;
            var pieces: NextArray = .init(0);

            // Distribute locks to bags, advancing bag sizes as needed if bags
            // are unable to satisfy the locks
            outer: while (!done(sizes)) {
                for (0..n_bags) |i| {
                    if (sizes[i] == 0) {
                        break;
                    }
                    start -= sizes[i];
                    bags[i] = initBag(start, sizes[i], locks);
                    if (i == 0) {
                        continue;
                    }

                    if (bags[i].next(pieces, start, sizes[i])) |p| {
                        pieces = p;
                    } else {
                        nextSize(&sizes);
                        start = len;
                        continue :outer;
                    }
                }
                break;
            }

            return .{
                .sizes = sizes,
                .bags = bags,
                .locks = locks,
                .pieces = pieces,
            };
        }

        inline fn done(sizes: [n_bags]BagLenInt) bool {
            return sizes[0] == 0 or sizes[sizes.len - 1] == MAX_BAG_LEN;
        }

        inline fn initBag(
            start: usize,
            size: BagLenInt,
            locks: [lock_len]u3,
        ) BagIterator {
            return .init(
                size,
                locks[@min(locks.len, start)..@min(locks.len, start + size)],
            );
        }

        pub fn next(self: *@This()) ?NextArray {
            if (done(self.sizes)) {
                return null;
            }

            // Update pieces with pieces from the last non-exhausted bag
            var i: usize = 0;
            var start: usize = len;
            while (i < n_bags) : (i += 1) {
                if (self.sizes[i] == 0) {
                    continue;
                }
                start -= self.sizes[i];
                if (self.bags[i].next(self.pieces, start, self.sizes[i])) |p| {
                    self.pieces = p;
                    break;
                }
            } else {
                // If all bags are exhausted, advance to the next set of bags
                assert(start == 0);
                nextSize(&self.sizes);
                if (done(self.sizes)) {
                    return null;
                }
            }

            // Replace exhausted bags and update the remaining pieces
            while (i > 0) {
                i -= 1;
                if (self.sizes[i] == 0) {
                    continue;
                }
                if (i < n_bags - 1) {
                    start += self.sizes[i + 1];
                }

                self.bags[i] = initBag(start, self.sizes[i], self.locks);
                if (self.bags[i].next(self.pieces, start, self.sizes[i])) |p| {
                    self.pieces = p;
                } else {
                    // If bag was unable to satisfy the locks, advance to the
                    // next set of bags
                    nextSize(&self.sizes);
                    if (done(self.sizes)) {
                        return null;
                    }
                    i = n_bags;
                    start = 0;
                }
            }
            assert(start + self.sizes[0] == len);

            return self.pieces;
        }

        /// Advances to the next possible combination of bag sizes.
        inline fn nextSize(sizes: []BagLenInt) void {
            sizes[0] -= 1;
            if (sizes.len > 2 and sizes[sizes.len - 2] < MAX_BAG_LEN) {
                sizes[sizes.len - 2] += 1;
            } else {
                sizes[sizes.len - 1] += 1;
            }
        }
    };
}

/// Iterates through numbers up to 7^len, returning the digits (in base 7) each
/// time.
pub fn DigitsIterator(comptime len: usize) type {
    return struct {
        pub const base = 7;
        pub const end = std.math.powi(u64, base, len) catch
            @compileError("len too large");

        value: u64,
        step: u64,
        comptime len: usize = len,

        pub fn init(start: u64, step: u64) @This() {
            return .{ .value = start, .step = step };
        }

        pub fn next(self: *@This()) ?[len]u3 {
            if (self.value >= end) {
                return null;
            }

            var digits: [len]u3 = undefined;
            var value = self.value;
            for (0..len) |i| {
                digits[i] = @intCast(value % base);
                value /= base;
            }

            self.value +|= self.step;
            return digits;
        }
    };
}

/// Iterates through all non-equivalent next sequences (including the hold) of
/// length `len`. 'unlocked' is used to control the way iterations are chunked.
/// A higher value uses more memory but reduces redundant computations.
pub fn SequenceIterator(comptime len: usize, comptime unlocked: usize) type {
    assert(len > 2);
    assert(unlocked <= len - 1);
    assert(len <= 64 / 3);
    return struct {
        pub const lock_len = @max(1, len - unlocked - 1);
        pub const LockIter = DigitsIterator(lock_len);
        pub const NextIter = NextIterator(len - 1, lock_len);
        pub const SequenceSet = std.AutoHashMap(u64, void);

        swap: u3 = 6,
        current: PieceArray(len),
        // The lock iterator is used to iterate through all sequences with the
        // same starting pieces. Thus, we can clear the seen set every time the
        // lock iterator is advanced, capping the memory usage.
        lock_iter: LockIter,
        next_iter: NextIter,
        seen: SequenceSet,

        pub fn init(allocator: Allocator) @This() {
            var self = @This(){
                .current = .init(0),
                // By letting the first piece kind be a "wildcard", we only
                // need to take every 7th sequence, reducing the size of `seen`
                // by a factor of 7.
                .lock_iter = .init(0, 7),
                .next_iter = undefined,
                .seen = .init(allocator),
            };
            self.current = self.current.set(len - 1, 6);
            self.next_iter = .init(self.lock_iter.next().?);
            return self;
        }

        pub fn deinit(self: *@This()) void {
            self.seen.deinit();
        }

        /// The returned sequence is in this order: [hold, current, next...].
        /// There is always a hold piece. The sequence is guaranteed to be
        /// unique.
        pub fn next(self: *@This()) !?[len]PieceKind {
            if (self.done()) {
                return null;
            }

            // Swap wildcard with each of the 7 pieces
            self.swap += 1;
            if (self.swap < 7) {
                return unpack(self.current, self.swap);
            }

            while (true) {
                // Increment hold to next piece
                self.current = self.current.set(
                    len - 1,
                    self.current.get(len - 1) + 1,
                );
                // If we have exhausted all holds, advance the next iterator
                if (self.current.get(len - 1) == 7) {
                    if (self.advanceNext()) |pieces| {
                        self.current = .init(pieces.items);
                        // No need to set the hold as it is already 0
                        // self.current = self.current.set(len - 1, 0);
                    } else {
                        // Iterator is exhausted
                        return null;
                    }
                }

                // NextIterator doesn't guarantee unique sequences, so we need
                // to check by ourselves
                const result = try self.seen.getOrPut(
                    canonical(self.current).items,
                );
                if (result.found_existing) {
                    continue;
                }

                self.swap = 0;
                return unpack(self.current, self.swap);
            }
        }

        inline fn advanceNext(self: *@This()) ?NextIter.NextArray {
            while (true) {
                if (self.next_iter.next()) |pieces| {
                    return pieces;
                }

                // If mext is exhausted, advance the locks to the next chunk
                // and clear the seen set
                if (self.lock_iter.next()) |locks| {
                    self.seen.clearRetainingCapacity();
                    self.next_iter = .init(locks);
                } else {
                    self.seen.clearAndFree();
                    return null;
                }
            }
        }

        pub inline fn done(self: @This()) bool {
            return self.swap == 7;
        }

        /// Puts the array in cannonical order by ensuring that the hold is
        /// smaller than the current next piece.
        inline fn canonical(current: PieceArray(len)) PieceArray(len) {
            if (current.get(len - 1) <= current.get(len - 2)) {
                return current;
            }
            return current
                .set(len - 2, current.get(len - 1))
                .set(len - 1, current.get(len - 2));
        }

        /// Unpacks the array into a sequence of pieces, swapping the wildcard
        /// with the given piece.
        inline fn unpack(arr: PieceArray(len), swap: u3) [len]PieceKind {
            var pieces: [len]PieceKind = undefined;
            for (0..len) |i| {
                // Reverse order
                const rev_i = len - i - 1;
                const piece = arr.get(rev_i);
                // Swap wildcard (0) with `swap`
                const swapped = if (piece == 0)
                    swap
                else if (piece == swap)
                    0
                else
                    piece;
                pieces[i] = @enumFromInt(swapped);
            }
            // Ensure canonical order
            if (@as(u3, @intFromEnum(pieces[1])) >
                @as(u3, @intFromEnum(pieces[0])))
            {
                std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
            }
            return pieces;
        }
    };
}

test SequenceIterator {
    const COUNTS = [_]u64{
        0,
        7,
        28,
        196,
        1_365,
        9_198,
        57_750,
        326_340,
        1_615_320,
        6_849_360,
        24_857_280,
        // 79_516_080,
        // 247_474_080,
        // 880_180_560,
        // 3_683_700_720,
        // 15_528_492_000,
        // 57_596_696_640,
        // 189_672_855_120,
        // 549_973_786_320,
        // 1_554_871_505_040,
    };

    inline for (3..COUNTS.len) |i| {
        var iter: SequenceIterator(i, @min(6, i - 1)) =
            .init(std.testing.allocator);
        defer iter.deinit();

        var count: u64 = 0;
        while (try iter.next()) |_| {
            count += 1;
        }
        try expect(count == COUNTS[i]);
    }
}
