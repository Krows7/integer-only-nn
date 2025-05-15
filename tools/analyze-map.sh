#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <linker-map-file>"
  exit 1
fi

MAPFILE="$1"

echo
echo "=== Section size summary ==="
awk '
  # Only lines beginning with a hex address
  /^[[:space:]]*[0-9A-Fa-f]+/ {
    # $3 is the size (hex), $5 is the output section name if it starts with "."
    size = strtonum("0x"$3)
    sec  = $5
    if (sec ~ /^\./) {
      sec_sizes[sec] += size
    }
  }
  END {
    printf "%-10s %10s\n", "Section", "Bytes"
    printf "%-10s %10s\n", "-------", "-----"
    for (s in sec_sizes) {
      printf "%-10s %10d\n", s, sec_sizes[s]
    }
  }
' "$MAPFILE" \
  | sort -nr -k2

echo
echo "=== Top 20 symbols by size ==="
awk '
  # Again, only map lines that start with an address and have at least 7 fields
  /^[[:space:]]*[0-9A-Fa-f]+/ && NF >= 7 {
    size   = strtonum("0x"$3)
    symbol = $NF
    sym_sizes[symbol] += size
  }
  END {
    for (sym in sym_sizes) {
      printf "%8d  %s\n", sym_sizes[sym], sym
    }
  }
' "$MAPFILE" \
  | sort -nr | head -20

echo
