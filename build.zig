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

    // zmai dependency
    const zmai_module = b.dependency("zmai", .{
        .target = target,
        .optimize = optimize,
    }).module("zmai");

    // zig-args dependency
    const args_module = b.dependency("args", .{
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
            .{ .name = "zmai", .module = zmai_module },
        },
    });

    // Add NN files
    const install_NNs = b.addInstallDirectory(.{
        .source_dir = b.path("NNs"),
        .install_dir = .bin,
        .install_subdir = "NNs",
    });
    root_module.addAnonymousImport("nn_4l_json", .{
        .root_source_file = minifyJson(
            b,
            b.path("NNs/Fast3.json"),
            "nn_4l.json",
        ),
    });

    buildExe(b, target, optimize, root_module, args_module);
    buildSolve(b, target, optimize, root_module, install_NNs);
    buildTests(b, root_module);
    buildBench(b, target, root_module);
    buildTrain(b, target, optimize, root_module);
}

fn minifyJson(
    b: *Build,
    path: Build.LazyPath,
    name: []const u8,
) Build.LazyPath {
    const minify_exe = b.addExecutable(.{
        .name = "minify-json",
        .root_source_file = b.path("src/build/minify-json.zig"),
        .target = b.resolveTargetQuery(
            Build.parseTargetQuery(.{}) catch
                @panic("minifyJson: Failed to resolve target"),
        ),
    });

    const minify_cmd = b.addRunArtifact(minify_exe);
    minify_cmd.expectExitCode(0);
    minify_cmd.addFileArg(path);
    return minify_cmd.addOutputFileArg(name);
}

fn stripModuleRecursive(module: *Build.Module) void {
    module.strip = true;
    var iter = module.import_table.iterator();
    while (iter.next()) |entry| {
        stripModuleRecursive(entry.value_ptr.*);
    }
}

fn packageVersion(b: *Build) []const u8 {
    var ast = std.zig.Ast.parse(b.allocator, @embedFile("build.zig.zon"), .zon) catch
        @panic("Out of memory");
    defer ast.deinit(b.allocator);

    var buf: [2]std.zig.Ast.Node.Index = undefined;
    const zon = ast.fullStructInit(&buf, ast.nodes.items(.data)[0].lhs) orelse
        @panic("Failed to parse build.zig.zon");

    for (zon.ast.fields) |field| {
        const field_name = ast.tokenSlice(ast.firstToken(field) - 2);
        if (std.mem.eql(u8, field_name, "version")) {
            const version_string = ast.tokenSlice(ast.firstToken(field));
            // Remove surrounding quotes
            return version_string[1 .. version_string.len - 1];
        }
    }
    @panic("Field 'version' missing from build.zig.zon");
}

fn buildExe(
    b: *Build,
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_module: *Build.Module,
    args_module: *Build.Module,
) void {
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("perfect-tetris", root_module);
    exe_mod.addImport("zig-args", args_module);
    exe_mod.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );
    exe_mod.addImport(
        "vaxis",
        root_module.import_table.get("vaxis").?,
    );
    exe_mod.addImport(
        "nterm",
        root_module.import_table.get("nterm").?,
    );
    exe_mod.addImport(
        "zmai",
        root_module.import_table.get("zmai").?,
    );

    const options = b.addOptions();
    options.addOption([]const u8, "version", packageVersion(b));
    exe_mod.addImport("build", options.createModule());

    if (b.option(bool, "strip", "Strip executable binary") orelse false) {
        stripModuleRecursive(exe_mod);
    }

    const exe = b.addExecutable(.{
        .name = "pc",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

fn buildSolve(
    b: *Build,
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_module: *Build.Module,
    install_NNs: *Build.Step.InstallDir,
) void {
    const exe = b.addExecutable(.{
        .name = "solve",
        .root_source_file = b.path("src/scripts/solve.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("perfect-tetris", root_module);
    exe.root_module.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("solve", "Run the PC solver");
    run_step.dependOn(&run_cmd.step);

    const install = b.addInstallArtifact(exe, .{});
    run_step.dependOn(&install.step);
    install.step.dependOn(&install_NNs.step);
}

fn buildTests(b: *Build, root_module: *Build.Module) void {
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
    });
    lib_tests.root_module.addImport(
        "nn_4l_json",
        root_module.import_table.get("nn_4l_json").?,
    );
    lib_tests.root_module.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );
    lib_tests.root_module.addImport(
        "zmai",
        root_module.import_table.get("zmai").?,
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

fn buildTrain(
    b: *Build,
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_module: *Build.Module,
) void {
    const train_exe = b.addExecutable(.{
        .name = "nn-train",
        .root_source_file = b.path("src/scripts/train.zig"),
        .target = target,
        .optimize = optimize,
    });
    train_exe.root_module.addImport("perfect-tetris", root_module);
    train_exe.root_module.addImport(
        "engine",
        root_module.import_table.get("engine").?,
    );
    train_exe.root_module.addImport(
        "nterm",
        root_module.import_table.get("nterm").?,
    );
    train_exe.root_module.addImport(
        "zmai",
        root_module.import_table.get("zmai").?,
    );

    const train_cmd = b.addRunArtifact(train_exe);
    train_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        train_cmd.addArgs(args);
    }
    const train_step = b.step("train", "Train neural networks");
    train_step.dependOn(&train_cmd.step);

    const install = b.addInstallArtifact(train_exe, .{});
    train_step.dependOn(&install.step);
}
