#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`

mlir-opt "$basename.mlir" \
-convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm --convert-cf-to-llvm -expand-strided-metadata \
-lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts > out/$basename-llvm.mlir 

mlir-cpu-runner -e $funcname -entry-point-result=void \
-shared-libs=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so \
out/$basename-llvm.mlir 
