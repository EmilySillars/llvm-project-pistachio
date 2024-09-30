#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`
OUT=$3/$basename/out
CORRECT=correct/$basename/out

# make an output directory if doesn't already exist
mkdir -p $OUT

# lower mlir to llvm
mlir-opt $basename.mlir --one-shot-bufferize='bufferize-function-boundaries' > $OUT/$basename-bufferized.mlir

mlir-opt $OUT/$basename-bufferized.mlir \
--convert-vector-to-scf \
> $OUT/$basename-lowered1.mlir

mlir-opt $OUT/$basename-lowered1.mlir \
--convert-linalg-to-loops \
> $OUT/$basename-lowered2.mlir

mlir-opt $OUT/$basename-lowered2.mlir \
--lower-affine \
> $OUT/$basename-lowered3.mlir

mlir-opt $OUT/$basename-lowered3.mlir \
--convert-scf-to-cf \
> $OUT/$basename-lowered4.mlir

mlir-opt $OUT/$basename-lowered4.mlir \
--canonicalize \
> $OUT/$basename-lowered5.mlir

mlir-opt $OUT/$basename-lowered5.mlir \
--cse \
> $OUT/$basename-lowered6.mlir

mlir-opt $OUT/$basename-lowered6.mlir \
--convert-vector-to-llvm='reassociate-fp-reductions' \
> $OUT/$basename-lowered7.mlir

mlir-opt $OUT/$basename-lowered7.mlir \
--convert-math-to-llvm \
> $OUT/$basename-lowered8.mlir

mlir-opt $OUT/$basename-lowered8.mlir \
--expand-strided-metadata \
> $OUT/$basename-lowered9.mlir

mlir-opt $OUT/$basename-lowered9.mlir \
--lower-affine \
> $OUT/$basename-lowered10.mlir

mlir-opt $OUT/$basename-lowered10.mlir \
--finalize-memref-to-llvm \
> $OUT/$basename-lowered11.mlir

mlir-opt $OUT/$basename-lowered11.mlir \
--convert-func-to-llvm \
> $OUT/$basename-lowered12.mlir

mlir-opt $OUT/$basename-lowered12.mlir \
--convert-index-to-llvm \
> $OUT/$basename-lowered13.mlir

mlir-opt $OUT/$basename-lowered13.mlir \
--reconcile-unrealized-casts \
> $OUT/$basename-lowered14.mlir

cat $OUT/$basename-lowered14.mlir > $OUT/$basename-in-llvm-dialect.mlir

# run LLVM MLIR with the mlir-cpu-runner

mlir-cpu-runner -e $funcname -entry-point-result=void \
-shared-libs=$MLIR_CPU_RUNNER_LIBS \
$OUT/$basename-in-llvm-dialect.mlir  > $OUT/$basename.out

# print output of run
diff $OUT/$basename.out $CORRECT/$basename.out


