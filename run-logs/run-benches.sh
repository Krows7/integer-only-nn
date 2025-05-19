#!/bin/bash

make clean all

# To get iris-new.txt change -> BATCH_SIZE = 16 -> 4, HIDDEN_NEURONS = 64 -> 8

./build/nes/iris-nes > run-logs/iris-new-old-config.txt # Running new framework with old hyperparams
./build/nes/iris-nes-init > run-logs/iris-init-new.txt # Running old model file in new framework

./build/ext/digits > run-logs/digits-new.txt # Running new framework

cd ~/huy-project/integer-only-nn

make clean all

./build/nes/iris-nes > iris-init.txt # Running old framework
./build/ext/digits > digits-init.txt

mv iris-init.txt ~/network-project/run-logs/iris-init.txt
mv digits-init.txt ~/network-project/run-logs/digits-init.txt