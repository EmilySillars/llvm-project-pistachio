#!/bin/sh
rm out/$basename.o
clear
readonly YODEL="C to LLVM"
echo $YODEL
basename=`basename $1 | sed 's/[.][^.]*$//'`
echo "../../build-riscv/bin/clang -O1 -emit-llvm $basename.c -c -o out/$basename.bc"
../../build-riscv/bin/clang -O1 -emit-llvm $basename.c -c -o out/$basename.bc
echo "../../build-riscv/bin/llvm-dis < out/$basename.bc > out/$basename.ll"
../../build-riscv/bin/llvm-dis < out/$basename.bc > out/$basename.ll
echo "../../build-riscv/bin/clang out/$basename.ll -o out/$basename.o"
../../build-riscv/bin/clang out/$basename.ll -o out/$basename.o
echo "out/$basename.o"
echo ""
out/$basename.o