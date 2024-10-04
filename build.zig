const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Engine dependency
    const engine_module = b.dependency("engine", .{
        .target = target,
        .optimize = optimize,
    }).module("engine");

    // vaxis dependency
    const vaxis_module = b.dependency("vaxis", .{
        .target = target,
        .optimize = optimize,
    }).module("vaxis");

    // zig-args dependency
    const args_module = b.dependency("zig-args", .{
        .target = target,
        .optimize = optimize,
    }).module("args");

    // Expose the library root
    const root_module = b.addModule("perfect-tetris", .{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{
            .{ .name = "engine", .module = engine_module },
            .{
                .name = "nterm",
                .module = engine_module.import_table.get("nterm").?,
            },
            .{ .name = "vaxis", .module = vaxis_module },
        },
    });

    buildExe(b, target, optimize, root_module, args_module);
    buildTests(b, root_module);
    buildBench(b, target, root_module);
}

fn stripModuleRecursive(module: *Build.Module) void {
    module.strip = true;
    var iter = module.import_table.iterator();
    while (iter.next()) |entry| {
        stripModuleRecursive(entry.value_ptr.*);
    }
}

fn buildExe(
    b: *Build,
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_module: *Build.Module,
    args_module: *Build.Module,
) void {
    const exe = b.addExecutable(.{
        .name = "pc",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("perfect-tetris", root_module);
    exe.root_module.addImport("zig-args", args_module);
    exe.root_module.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );
    exe.root_module.addImport(
        "vaxis",
        root_module.import_table.get("vaxis").?,
    );
    exe.root_module.addImport(
        "nterm",
        root_module.import_table.get("nterm").?,
    );

    if (b.option(bool, "strip", "Strip executable binary") orelse false) {
        stripModuleRecursive(&exe.root_module);
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

fn buildTests(b: *Build, root_module: *Build.Module) void {
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
    });
    lib_tests.root_module.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);
}

fn buildBench(
    b: *Build,
    target: Build.ResolvedTarget,
    root_module: *Build.Module,
) void {
    const bench_exe = b.addExecutable(.{
        .name = "benchmarks",
        .root_source_file = b.path("src/scripts/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_exe.root_module.addImport("perfect-tetris", root_module);
    bench_exe.root_module.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );

    const bench_cmd = b.addRunArtifact(bench_exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bench_cmd.addArgs(args);
    }
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_cmd.step);
}
