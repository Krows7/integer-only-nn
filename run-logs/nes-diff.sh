#!/bin/bash

make clean all

./build/nes/iris-nes > run-logs/pc.txt

cp /home/huy/Desktop/NES/myFile.txt run-logs/nes.txt

clear

git diff --no-index --color-words run-logs/pc.txt run-logs/nes.txt