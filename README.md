# perfect-tetris

Blazingly fast Tetris perfect clear solver. Accepts input in fumen format and
outputs solutions in fumen format. For performance metrics, check out
[benchmarks](#benchmarks).

## Usage

```bash
pc COMMAND [options] [INPUT]
```

Run `pc COMMAND --help` for command-specific help.

## Commands

### Demo

```bash
pc demo [options]
```

Demostrates the perfect clear solver's speed with a tetris playing bot. The
solver runs on a single thread in this demo.

#### Options

`-h`, `--help` Print the help message.

`-k`, `--kicks` Kick/rotation system to use. For kick systems that have a
180-less and 180 variant, the 180-less variant has no 180 rotations. The 180
variant has 180 rotations but no 180 kicks. Supported Values:
[`none`,`none180`,`srs`,`srs180`,`srsPlus`,`srsTetrio`] (default: `srsPlus`)

`-m`, `--min-height` The minimum height of PCs to find. (default: `4`)

`-n`, `--nn` The path to the neural network to use for the solver. The path may
be absolute, relative to the current working directory, or relative to the
executable's directory. If not provided, a default built-in NN will be used.

`-p`, `--pps` The target pieces per second of the bot. (default: `10`)

`-s`, `--seed` The seed to use for the 7-bag randomizer. If not provided, a
random value will be used.

### Display

```bash
pc display [options] PATH
```

Displays the perfect clear solutions saved at `PATH`. Press `enter` to display
the next solution. To seek to a specific solution, type the solution number and
press `enter`. Only supports `.pc` files.

#### Options

`-h`, `--help` Print the help message.

### Fumen

```bash
pc fumen [options] INPUTS...
```

Produces a perfect clear solution for each input fumen. Outputs each solution
to stdout as a new fumen, separated by newlines. If the fumen is inside a url,
the url will be preserved in the output. Fumen editor:
[https://fumen.zui.jp/#english.js](https://fumen.zui.jp/#english.js)

Finds the shortest perfect clear solution of each input fumen. The queue is
encoded in the comment of the fumen, in the format:

`#Q=[<HOLD>](<CURRENT>)<NEXT1><NEXT2>...<NEXTn>`

Outputs each solution to stdout as a new fumen, separated by newlines. If the
fumen is inside a url, the url will be preserved in the output. Fumen editor:
[https://fumen.zui.jp/#english.js](https://fumen.zui.jp/#english.js)

#### Options

`-a`, `--append` Append solution frames to input fumen instead of making a new
fumen from scratch.

`-h`, `--help` Print the help message.

`-k`, `--kicks` Kick/rotation system to use. For kick systems that have a
180-less and 180 variant, the 180-less variant has no 180 rotations. The 180
variant has 180 rotations but no 180 kicks. Supported Values:
[`none`,`none180`,`srs`,`srs180`,`srsPlus`,`srsTetrio`] (default: `srs`)

`-m`, `--min-height` Overrides the minimum height of the PC to find.

`-n`, `--nn` The path to the neural network to use for the solver. The path may
be absolute, relative to the current working directory, or relative to the
executable's directory. If not provided, a default built-in NN will be used.

`-s`, `--save` The piece type to save in the hold slot by the end of the
perfect clear. This option is not case-sensitive. If not specified, any piece
may go into the hold slot. Supported Values: [`i`,`o`,`t`,`s`,`z`,`l`,`j`]

`-t`, `--output-type` The type of fumen to output. Overwrites the input fumen's
type even when append is true. Supported Values: [edit,list,view]
(default: view)

`-v`, `--verbose` Print solve time and solution length to stderr.

### Validate

```bash
pc validate [options] PATHS...
```

Validates the perfect clear solutions saved at `PATHS`. This will validate that
`PATHS` are valid .pc files and that all solutions are valid perfect clear
solutions.

#### Options

`-h`, `--help` Print the help message.

## Build commands

You will need to have Zig installed to build the project. You can download Zig
from [here](https://ziglang.org/download/).

This project currently uses Zig 0.13.0.

### Run

```bash
zig build run -- COMMAND [ARGS] [INPUT]
```

Runs the main program. Refer to [usage](#usage) for more information. The `-Dstrip`
option may be passed to the compiler to produce a smaller binary.

### Solve

```bash
zig build solve
```

Finds all perfect clear solutions of a given height from an empty playfield,
and saves the solutions to disk in [`.pc`](#pc-file-format) format.

Settings may be adjusted in the top-level declarations in `src/scripts/solve.zig`.

### Test

```bash
zig build test
```

Runs all tests.

### Benchmark

```bash
zig build bench
```

Runs the benchmarks.

### Train

```bash
zig build train
```

Trains a population of neural networks to solve perfect clears as fast as
possible. The population is saved at the end of every generation.

Settings may be adjusted in the top-level declarations in `src/scripts/train.zig`.

## Benchmarks

```txt
CPU: Intel Core i7-8700
OS: Windows 11
Run command: zig build bench -Dcpu=x86_64_v3
```

The results below are agregated over 200 random 1st PC sequences from a 7-bag randomiser.

| PC height | Kick system | Mean time | Max time  | Mean memory | Max memory |
| --------- | ----------- | --------- | --------- | ----------- | ---------- |
| 4         | SRS+        | 10.697ms  | 237.634ms | 245.380KiB  | 9.455MiB   |
| 6         | SRS+        | 9.253ms   | 217.802ms | 173.771KiB  | 2.430MiB   |

## .pc file format

The `.pc` file format is a binary format that stores perfect clear solutions.

The `.pc` format can only store solutions with up to 15 placements, and that
start with an empty playfield. The format consists of only a list of solutions
with no padding or metadata. The format of a solution is as follows:

| Bytes | Name       | Description                                      |
| ----- | ---------- | ------------------------------------------------ |
| 0-5   | sequence   | The sequence of hold and next pieces             |
| 6-7   | holds      | 15 binary flags indicating where holds are used  |
| 8+    | placements | The list of placements that make up the solution |

### Sequence

The sequence is a 6-byte integer that stores the sequence of hold and next
pieces (a total of 16 3-bit values). The sequence is stored in little-endian
order. Bits 0-2 indicate the hold piece, bits 3-5 indicate the current piece,
bits 6-8 indicate the first piece in the next queue, bits 9-11 indicate the
second piece in the next queue, etc. Each 3-bit value maps to a different
piece:

| Value | Piece    |
| ----- | -------- |
| 000   | I        |
| 001   | O        |
| 010   | T        |
| 011   | S        |
| 100   | Z        |
| 101   | L        |
| 110   | J        |
| 111   | sentinel |

Once a sentinel value is reached, all subsequent values should also be sentinel
values. If no sentinel value is reached, the sequence is assumed to have the
maximum length of 16.

### Holds

The holds are stored as a 2-byte integer in little-endian order. The i-th bit
indicates whether a hold was used at the i-th placement. A `0` indicates no
hold, and a `1` indicates a hold. As there is a maxium of 15 placements, the
last bit is always `0`.

### Placements

The list of placements is stored in little-endian order, and the length of this
list is always the length of the sequence minus one. Each placement is stored
as a single byte, with the following format:

| Bits | Name     | Description                                   |
| ---- | -------- | --------------------------------------------- |
| 0-1  | facing   | The direction the piece is facing when placed |
| 2-7  | position | The position of the piece. x + 10y            |

The type of piece is determined by the sequence and holds.

A value of '0' for facing indicates the piece is facing north; '1' indicates
east; '2' indicates south, and '3' indicates west.

Position is a value in the range [0, 59]. The x-coordinate is this value modulo
10, and the y-coordinate is this value divided by 10 (rounded down). The x- and
y-coordinates represent the center of the piece as defined by
[SRS true rotation](https://harddrop.com/wiki/File:SRS-true-rotations.png). The
x-axis starts at 0 at the leftmost column and increases rightwards. The y-axis
starts at 0 at the bottom row and increases upwards.

## Other interesting information

### Number of possible next sequences (with any hold)

| Length | Non-equivalent\*  |
| ------ | ----------------- |
| 0      | 7                 |
| 1      | 28                |
| 2      | 196               |
| 3      | 1,365             |
| 4      | 9,198             |
| 5      | 57,750            |
| 6      | 326,340           |
| 7      | 1,615,320         |
| 8      | 6,849,360         |
| 9      | 24,857,280        |
| 10     | 79,516,080        |
| 11     | 247,474,080       |
| 12     | 880,180,560       |
| 13     | 3,683,700,720     |
| 14     | 15,528,492,000    |
| 15     | 57,596,696,640    |
| 16     | 189,672,855,120   |
| 17     | 549,973,786,320   |
| 18     | 1,554,871,505,040 |

\*Two sequences are considered equivalent if the set of all possible structures
that can be built by each are equal.

### PC solve chances

The chances of a perfect clear being possible in certain situations with an
empty playfield are as follows:

| Type           | held piece | randomiser | rotation system | Chance  | Odds                       |
| -------------- | ---------- | ---------- | --------------- | ------- | -------------------------- |
| 2-line, opener | none       | 7-bag      | SRS             | 0%      | 0 in 5,040                 |
| 2-line         | none       | 7-bag      | SRS             | 3.3217% | 5,148 in 154,980           |
| 2-line         | any        | 7-bag      | SRS             | 3.9987% | 15,132 in 378,420          |
| 4-line, opener | none       | 7-bag      | SRS             | 100%    | 4,233,600 in 4,233,600     |
| 4-line         | none       | 7-bag      | SRS             | 100%    | 57,576,960 in 57,576,960   |
| 4-line         | any        | 7-bag      | SRS             | 99.964% | 196,791,063 in 196,862,400 |
