#!/bin/sh
readonly YODEL="MLIR to LLVM"
echo $YODEL
basename=`basename $1 | sed 's/[.][^.]*$//'`
mainfunc=`basename $2 | sed 's/[.][^.]*$//'`

echo "1) Converting $mainfunc.c to LLVM IR..."
echo "../../build-riscv/bin/clang -O0 -emit-llvm $mainfunc.c -c -o out/$mainfunc.bc"
../../build-riscv/bin/clang -O0 -emit-llvm $mainfunc.c -c -o out/$mainfunc.bc
echo "../../build-riscv/bin/llvm-dis < out/$mainfunc.bc > out/$mainfunc.ll"
../../build-riscv/bin/llvm-dis < out/$mainfunc.bc > out/$mainfunc.ll
# Inside the main function, replace call to modifyTensor with call to simp.
echo "awk '/modifyTensor/ && ++count==2{sub(/modifyTensor/,"simp")} 1' out/$mainfunc.ll > out/$mainfunc-calls-simp.ll"
awk '/modifyTensor/ && ++count==2{sub(/modifyTensor/,"simp")} 1' out/$mainfunc.ll > out/$mainfunc-calls-simp.ll
echo "../../build-riscv/bin/clang out/$mainfunc-calls-simp.ll -o out/$mainfunc.o"
../../build-riscv/bin/clang out/$mainfunc.ll -o out/$mainfunc.o
echo ""

echo "2) Converting $basename.mlir to LLVM IR..."
echo "mlir-opt $basename.mlir  --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm > out/$basename-in-llvm-dialect.mlir"
mlir-opt $basename.mlir  --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm > out/$basename-in-llvm-dialect.mlir
echo "mlir-translate --mlir-to-llvmir out/$basename-in-llvm-dialect.mlir > out/$basename.ll"
mlir-translate --mlir-to-llvmir out/$basename-in-llvm-dialect.mlir > out/$basename.ll
echo ""

echo "3) Splicing together an LLVM IR file with a main function..."
# trim top and bottom of compiled MLIR
lines=`wc --lines out/$basename.ll | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/'`
head --lines $(($lines-4)) out/$basename.ll > out/trimmedBottom.ll
tail --lines $(($lines-7)) out/trimmedBottom.ll > out/trimmedBottmAndTop.ll

# insert compiled MLIR into the LLVM IR file that already has a main function 
key=`grep -n "pamplemousse" out/$mainfunc-calls-simp.ll | head -n 1 | cut -d: -f1`
lines=`wc --lines out/$mainfunc-calls-simp.ll | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/'`
head --lines $(($key+1)) out/$mainfunc-calls-simp.ll > out/start.ll
tail --lines $(($lines-$key-1)) out/$mainfunc-calls-simp.ll > out/end.ll
cat out/start.ll > out/$basename-frankenstein.ll
cat out/trimmedBottmAndTop.ll >> out/$basename-frankenstein.ll
cat out/end.ll >> out/$basename-frankenstein.ll
echo ""

echo "4) Compile and run LLVM IR..."
echo "clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o"
clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o
echo "out/$basename-frankenstein.o"
echo ""
out/$basename-frankenstein.o
