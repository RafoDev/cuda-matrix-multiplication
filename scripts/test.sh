#!/bin/bash

sizes=(32 64 128 256 512 1024 2048 4096 8192 16384) 

for n in "${sizes[@]}"
do
    echo "TEST $n"
    ./matrixMulKernel $n
    ./matrixMulKernelShared $n
    echo
done