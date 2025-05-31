const std = @import("std");
const json = std.json;

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next().?; // Executable path
    const path = args.next() orelse return error.MissingJsonPath;

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    var reader = json.reader(allocator, file.reader());
    defer reader.deinit();

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const value: json.Value = try .jsonParse(
        arena.allocator(),
        &reader,
        .{ .max_value_len = 1_000_000 },
    );

    const stdout = std.io.getStdOut().writer();
    var writer = json.writeStream(stdout, .{ .whitespace = .minified });
    try value.jsonStringify(&writer);
}
