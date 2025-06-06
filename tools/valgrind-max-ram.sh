#!/bin/bash

mkdir -p valgrind-logs
make clean all
valgrind --tool=massif --stacks=yes --massif-out-file=valgrind-logs/massif.out.%p ./build/nes/iris-nes & 
pid=$!
echo "Massif PID is $pid"
wait $pid
ms_print valgrind-logs/massif.out.$pid
