const std = @import("std");

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();

    var args = try std.process.argsWithAllocator(debug_allocator.allocator());
    defer args.deinit();
    _ = args.skip(); // Executable path

    const stdout = std.io.getStdOut().writer();
    while (args.next()) |path| {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();

        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        var buf: [4096]u8 = undefined;
        while (true) {
            const n = try file.read(&buf);
            if (n == 0) break;
            hasher.update(buf[0..n]);
        }

        try stdout.print("{s}  {s}\n", .{
            std.fmt.fmtSliceHexLower(&hasher.finalResult()),
            std.fs.path.basename(path),
        });
    }
}
