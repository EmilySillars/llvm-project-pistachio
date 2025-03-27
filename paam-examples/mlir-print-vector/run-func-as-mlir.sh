#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`

mlir-opt "$basename.mlir" \
-test-linalg-transform-patterns=test-linalg-to-vector-patterns \
-empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
-bufferization-bufferize -tensor-bufferize -func-bufferize \
-finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
> out/$basename-bufferized.mlir

mlir-opt "out/$basename-bufferized.mlir" \
-convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm --convert-cf-to-llvm -expand-strided-metadata \
-lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts > out/$basename-llvm.mlir 

mlir-cpu-runner -e $funcname -entry-point-result=void \
-shared-libs=$MLIR_CPU_RUNNER_LIBS \
out/$basename-llvm.mlir 
