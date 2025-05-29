const Allocator = @import("std").mem.Allocator;
const assert = @import("std").debug.assert;

/// A ring queue. `std.RingBuffer` modified to allow a generic data type.
pub fn RingQueue(T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        read_index: usize = 0,
        write_index: usize = 0,

        pub const Error = error{Full};

        /// Allocate a new `RingQueue`; `deinit()` should be called to free the queue.
        pub fn init(allocator: Allocator, capacity: usize) Allocator.Error!Self {
            const values = try allocator.alloc(T, capacity);
            return Self{
                .data = values,
                .write_index = 0,
                .read_index = 0,
            };
        }

        /// Free the data backing a `RingQueue`; must be passed the same `Allocator` as
        /// `init()`.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            allocator.free(self.data);
            self.* = undefined;
        }

        /// Returns `index` modulo the length of the backing slice.
        pub fn mask(self: Self, index: usize) usize {
            return index % self.data.len;
        }

        /// Returns `index` modulo twice the length of the backing slice.
        pub fn mask2(self: Self, index: usize) usize {
            return index % (2 * self.data.len);
        }

        /// Write `value` into the ring queue. Returns `error.Full` if the ring
        /// queue is full.
        pub fn enqueue(self: *Self, value: T) Error!void {
            if (self.isFull()) return error.Full;
            self.enqueueAssumeCapacity(value);
        }

        /// Write `value` into the ring queue. If the ring queue is full, the
        /// oldest value is overwritten.
        pub fn enqueueAssumeCapacity(self: *Self, value: T) void {
            self.data[self.mask(self.write_index)] = value;
            self.write_index = self.mask2(self.write_index + 1);
        }

        /// Consume a value from the ring queue and return it. Returns `null` if the
        /// ring queue is empty.
        pub fn dequeue(self: *Self) ?T {
            if (self.isEmpty()) return null;
            return self.dequeueAssumeLength();
        }

        /// Consume a value from the ring queue and return it; asserts that the queue
        /// is not empty.
        pub fn dequeueAssumeLength(self: *Self) T {
            assert(!self.isEmpty());
            const value = self.data[self.mask(self.read_index)];
            self.read_index = self.mask2(self.read_index + 1);
            return value;
        }

        /// Returns `true` if the ring queue is empty and `false` otherwise.
        pub fn isEmpty(self: Self) bool {
            return self.write_index == self.read_index;
        }

        /// Returns `true` if the ring queue is full and `false` otherwise.
        pub fn isFull(self: Self) bool {
            return self.mask2(self.write_index + self.data.len) == self.read_index;
        }

        /// Returns the length of data available for reading
        pub fn len(self: Self) usize {
            const wrap_offset = 2 * self.data.len * @intFromBool(self.write_index < self.read_index);
            const adjusted_write_index = self.write_index + wrap_offset;
            return adjusted_write_index - self.read_index;
        }
    };
}
