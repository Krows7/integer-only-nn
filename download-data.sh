#!/usr/bin/env bash
set -e
command -v unzip >/dev/null || { echo "Please install unzip: sudo apt install -y unzip" >&2; exit 1; }
mkdir -p data/digits
wget -qO digits.zip "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
unzip -q digits.zip -d data/digits
rm digits.zip

mkdir -p data/iris
wget -qO iris.zip "https://archive.ics.uci.edu/static/public/53/iris.zip"
unzip -q iris.zip -d data/iris
rm iris.zip

python3 iris-to-nes-suitable.py
python3 iris-to-nes-header.py