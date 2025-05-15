#!/bin/bash

min=$(printf '%s\n%s\n' "$(wc -l < pc.txt)" "$(wc -l < nes.txt)" \
      | sort -n | head -1)

#colordiff \
#  <(head -n "$(printf '%s\n%s\n' "$(wc -l < pc.txt)" "$(wc -l < nes.txt)" \
#                 | sort -n | head -1)" pc.txt) \
#  <(head -n "$(printf '%s\n%s\n' "$(wc -l < pc.txt)" "$(wc -l < nes.txt)" \
#                 | sort -n | head -1)" nes.txt)

wdiff -n \
  <(head -n "$min" pc.txt) \
  <(head -n "$min" nes.txt) \
| colordiff
