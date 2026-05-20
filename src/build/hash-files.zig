const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(init.gpa);
    defer args.deinit();
    _ = args.skip(); // Executable path

    var stdout = std.Io.File.stdout().writer(init.io, &.{});
    while (args.next()) |path| {
        const file = try std.Io.Dir.cwd().openFile(init.io, path, .{});
        defer file.close(init.io);

        var hasher: std.crypto.hash.sha2.Sha256 = .init(.{});
        var buf: [4096]u8 = undefined;
        while (true) {
            const n = file.readStreaming(init.io, &.{&buf}) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            hasher.update(buf[0..n]);
        }

        try stdout.interface.print("{s}  {s}\n", .{
            std.fmt.bytesToHex(&hasher.finalResult(), .lower),
            std.fs.path.basename(path),
        });
    }
}
