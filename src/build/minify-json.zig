const std = @import("std");
const json = std.json;
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.next().?; // Executable path
    const path = args.next() orelse return error.MissingJsonPath;

    const json_str = try Io.Dir.readFileAlloc(.cwd(), io, path, allocator, .unlimited);
    defer allocator.free(json_str);

    const value = try json.parseFromSlice(json.Value, allocator, json_str, .{});
    defer value.deinit();

    var buf: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writer(io, &buf);
    try stdout.interface.print("{f}", .{json.fmt(value.value, .{ .whitespace = .minified })});
    try stdout.flush();
}
