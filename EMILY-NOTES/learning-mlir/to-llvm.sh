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
head --lines 2 out/$basename.ll > out/linalg-front.ll
minusFive="$(($minusTwo-5))"
tail --lines $minusTwo out/$basename.ll > out/linalg-back.ll
head --lines $minusFive out/linalg-back.ll > out/linalg-middle.ll
cat out/linalg-front.ll > out/$basename-frankenstein.ll
cat main-func.ll >> out/$basename-frankenstein.ll
cat out/linalg-middle.ll >> out/$basename-frankenstein.ll
cat main-func-metadata.ll >> out/$basename-frankenstein.ll
echo "clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o"
clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o
echo "out/$basename-frankenstein.o"
echo ""
out/$basename-frankenstein.o