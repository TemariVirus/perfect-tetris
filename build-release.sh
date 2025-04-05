zig build -p release -Doptimize=ReleaseFast -Dtarget=x86_64-windows-gnu -Dcpu=x86_64 -Dstrip
mv release/bin/pc.exe release/bin/pc-oldie.exe
zig build -p release -Doptimize=ReleaseFast -Dtarget=x86_64-windows-gnu -Dcpu=x86_64_v3 -Dstrip
zig build -p release -Doptimize=ReleaseFast -Dtarget=x86_64-linux-gnu -Dcpu=x86_64 -Dstrip
mv release/bin/pc release/bin/pc-oldie
zig build -p release -Doptimize=ReleaseFast -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v3 -Dstrip
