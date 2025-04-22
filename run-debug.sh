#!/usr/bin/env bash
#
# run-debug.sh — run GDB non‑interactively on ./build/<bin>
# Usage: ./run-debug.sh <binary> [gdb -ex args…]
#

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <binary-name> [additional -ex gdb commands...]"
  exit 1
fi

BIN="$1"; shift

# Path to executable
EXE="./build/${BIN}"

if [ ! -x "$EXE" ]; then
  echo "Error: '$EXE' not found or not executable."
  exit 2
fi

# Base GDB command:
#  - --batch: run commands and exit
#  - -q      : quiet startup
#  - -ex "run"    : start the program
#  - -ex "bt"     : backtrace
#  - -ex "info locals" : dump locals in crashing frame
GDB_ARGS=(
  --batch
  -q
  -ex "run"
  -ex "bt"
  -ex "info locals"
)

# Append any extra -ex commands passed as arguments:
# e.g. ./run-debug.sh digits -ex "print foo" -ex "print bar"
if [ "$#" -gt 0 ]; then
  GDB_ARGS+=( "$@" )
fi

# Finally, point GDB at our program:
GDB_ARGS+=( --args "$EXE" )

# Run it:
gdb "${GDB_ARGS[@]}"