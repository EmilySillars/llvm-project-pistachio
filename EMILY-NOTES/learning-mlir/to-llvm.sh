#!/bin/sh
readonly YODEL="MLIR to LLVM"
echo $YODEL
basename=`basename $1 | sed 's/[.][^.]*$//'`
echo "mlir-opt $basename.mlir  --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm > out/$basename-in-llvm-dialect.mlir"
mlir-opt $basename.mlir  --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm > out/$basename-in-llvm-dialect.mlir
echo "mlir-translate --mlir-to-llvmir out/$basename-in-llvm-dialect.mlir > out/$basename.ll"
mlir-translate --mlir-to-llvmir out/$basename-in-llvm-dialect.mlir > out/$basename.ll
lines=`wc --lines out/$basename.ll | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/'`
minusTwo="$(($lines-2))"
head --lines 2 out/$basename.ll > out/front.ll
tail --lines $minusTwo out/$basename.ll > out/back.ll
cat out/front.ll > out/$basename-frankenstein.ll
cat middle2.ll >> out/$basename-frankenstein.ll
cat out/back.ll >> out/$basename-frankenstein.ll
echo "clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o"
clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o
echo "out/$basename-frankenstein.o"
echo ""
out/$basename-frankenstein.o