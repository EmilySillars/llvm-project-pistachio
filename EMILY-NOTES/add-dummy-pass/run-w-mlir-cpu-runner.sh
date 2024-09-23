#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`


mlir-opt $basename.mlir --one-shot-bufferize='bufferize-function-boundaries' > out/$basename-bufferized.mlir

mlir-opt out/$basename-bufferized.mlir \
-convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf > out/$basename-vector-scf-cf.mlir 

mlir-opt out/$basename-vector-scf-cf.mlir  \
-convert-vector-to-llvm --convert-cf-to-llvm > out/$basename-vector-scf-cf-llvm.mlir 

mlir-opt out/$basename-vector-scf-cf-llvm.mlir  \
-expand-strided-metadata -lower-affine -convert-arith-to-llvm \
--memref-expand -finalize-memref-to-llvm > out/$basename-vector-scf-cf-llvm2.mlir 

mlir-opt out/$basename-vector-scf-cf-llvm2.mlir   \
-convert-func-to-llvm > out/$basename-vector-scf-cf-llvm3.mlir 

mlir-opt out/$basename-vector-scf-cf-llvm3.mlir \
-reconcile-unrealized-casts > out/$basename-vector-scf-cf-llvm4.mlir 

mlir-cpu-runner -e $funcname -entry-point-result=void \
-shared-libs=$MLIR_CPU_RUNNER_LIBS \
out/$basename-vector-scf-cf-llvm4.mlir  > out/$basename.out