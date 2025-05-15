#!/usr/bin/env python3
import re
import os
import sys
import argparse
from collections import defaultdict, Counter

IGNORE_SECTIONS = {'.symtab', '.strtab', '.shstrtab', '.comment'}
PRG_SECTIONS = {
    '.text', '.rodata', '.data', '.bss',
    '.noinit', '.zp', '.zp.bss', '.zp.data',
    '.dpcm', '.reset', '.vectors'
}

symbol_re = re.compile(
    r'(?P<obj>[^:]+):\(\.(?P<section>[^.]+)\.(?P<sym>[^)]+)\)'
)

def human(n: int) -> str:
    if n >= 1024*1024: return f"{n/(1024*1024):.2f} MB"
    if n >= 1024:      return f"{n/1024:.2f} KB"
    return f"{n} B"

def parse_map(filename):
    sec_totals   = {}
    sym_sizes    = Counter()
    sec_sym      = defaultdict(Counter)
    obj_sizes    = Counter()
    zp_used = 0
    wr_used = 0

    with open(filename, 'r') as f:
        for line in f:
            # look for VMA & size
            m = re.match(r'\s*([0-9A-Fa-f]+)\s+[0-9A-Fa-f]+\s+([0-9A-Fa-f]+)', line)
            if not m:
                continue
            vma  = int(m.group(1), 16)
            size = int(m.group(2), 16)

            # file-backed symbol?
            m2 = symbol_re.search(line)
            if m2:
                sec = '.' + m2.group('section')
                sym = m2.group('sym')
                obj = os.path.basename(m2.group('obj'))
                # record symbol + section
                sym_sizes[sym] += size
                sec_sym[sec][sym] += size
                # record object, but skip '<internal>'
                if obj != '<internal>':
                    obj_sizes[obj] += size
                # RAM buckets
                if vma < 0x0100:
                    zp_used += size
                elif vma < 0x0800:
                    wr_used += size
                continue

            # otherwise summary line (5 fields, valid section)
            parts = line.split()
            if len(parts) == 5 and parts[4].startswith('.') and parts[4] not in IGNORE_SECTIONS:
                sec_totals[parts[4]] = int(parts[2], 16)

    return sec_totals, sym_sizes, sec_sym, obj_sizes, zp_used, wr_used

def main():
    p = argparse.ArgumentParser(description="NES map analyzer")
    p.add_argument("mapfile", help="Linker .map file")
    p.add_argument("-n", "--top",    type=int, default=20, help="Top N entries")
    p.add_argument("-s", "--section",               help="Show top symbols in this section")
    args = p.parse_args()

    sec_totals, sym_sizes, sec_sym, obj_sizes, zp_used, wr_used = parse_map(args.mapfile)

    # PRG-ROM usage
    used  = sum(sec_totals.get(s, 0) for s in PRG_SECTIONS)
    total = 0x8000
    print(f"\nPRG-ROM used: {human(used)} / {human(total)} ({used/total*100:.1f}%)\n")

    # Section totals
    print("Section totals:")
    print(f"{'Section':12s} {'Bytes':>10s}")
    print(f"{'-'*12} {'-'*10}")
    for sec, sz in sorted(sec_totals.items(), key=lambda x: x[1], reverse=True):
        print(f"{sec:12s} {human(sz):>10s}")
    print()

    # Top symbols overall
    print(f"Top {args.top} symbols overall:")
    for sym, sz in sym_sizes.most_common(args.top):
        print(f"  {sym:40s} {human(sz):>8s}")
    print()

    # Top in specific section
    if args.section:
        sec = args.section
        if sec not in sec_sym:
            print(f"⚠️ Section '{sec}' not found.\n")
        else:
            print(f"Top {args.top} symbols in {sec}:")
            for sym, sz in sec_sym[sec].most_common(args.top):
                print(f"  {sym:40s} {human(sz):>8s}")
            print()

    # Top object-files
    print(f"Top {args.top} object-files by size:")
    for obj, sz in obj_sizes.most_common(args.top):
        print(f"  {obj:30s} {human(sz):>8s}")
    print()

    # RAM buckets
    print("RAM usage buckets:")
    print(f"  Zero-page (<0x100): {human(zp_used)} / 256 B")
    print(f"  Work-RAM  (<0x800): {human(wr_used)} / 2 048 B\n")

if __name__ == '__main__':
    main()
