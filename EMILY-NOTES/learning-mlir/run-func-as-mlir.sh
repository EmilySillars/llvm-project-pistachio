#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`

mlir-opt "$basename.mlir" \
-test-linalg-transform-patterns=test-linalg-to-vector-patterns \
-empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
-bufferization-bufferize -tensor-bufferize -func-bufferize \
-finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
-convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm --convert-cf-to-llvm -expand-strided-metadata \
-lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts > hoodle.mlir 

mlir-cpu-runner -e $funcname -entry-point-result=void \
-shared-libs=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so hoodle.mlir
