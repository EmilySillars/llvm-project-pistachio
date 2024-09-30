#!/bin/sh
# starting with a program in either linalg or affine dialect, 
# run it on the mlir-cpu-runner!
if [[ $1 == "-linalg" ]]; then
    echo "starting with linalg"
    basename=`basename $2 | sed 's/[/]*[.][^.]*$//'`
    funcname=`basename $3 | sed 's/[.][^.]*$//'`
    OUT=$4/$basename/linalg
    CORRECT=correct

    mkdir -p $OUT

    # lower mlir to llvm
    mlir-opt $2 --one-shot-bufferize='bufferize-function-boundaries' > $OUT/$basename-bufferized.mlir

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
elif [[ $1 == "-affine" ]]; then
    echo "starting with affine"
    basename=`basename $2 | sed 's/[/]*[.][^.]*$//'`
    funcname=`basename $3 | sed 's/[.][^.]*$//'`
    OUT=$4/$basename/affine
    CORRECT=correct

    mkdir -p $OUT

    mlir-opt $2 \
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
else
    echo "USAGE: sh linalg-run-w-mlir-cpu-runner.sh -<input-type> <filename.mlir> <functionName> <output-directory>"
    echo "where <input-type> can be linalg or affine"
fi



