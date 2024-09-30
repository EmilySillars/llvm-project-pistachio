#!/bin/sh
if [[ $1 == "--help" ]]; then
    echo "USAGE: sh tile-w-affine.sh <filename.mlir> <functionName> <output-directory> <tiling options>"
else
    basename=`basename $1 | sed 's/[/]*[.][^.]*$//'`
    funcname=`basename $2 | sed 's/[.][^.]*$//'`
    OUT=$3/$basename/tiling
    CORRECT=correct/
    options=$4
    mkdir -p $OUT

    # lower mlir to llvm
    mlir-opt $1 --one-shot-bufferize='bufferize-function-boundaries' > $OUT/$basename-bufferized.mlir

    mlir-opt $OUT/$basename-bufferized.mlir \
    --convert-vector-to-scf \
    > $OUT/$basename-lowered1.mlir

    mlir-opt $OUT/$basename-lowered1.mlir \
    --convert-linalg-to-affine-loops \
    > $OUT/$basename-before-tiling.mlir

    mlir-opt $OUT/$basename-before-tiling.mlir \
    --affine-loop-tile=$options \
    > $OUT/$basename-after-tiling.mlir

    # did anything happen?
    diff $OUT/$basename-before-tiling.mlir $OUT/$basename-after-tiling.mlir

    # do the tiled and untiled versions behave the same?
    sh run-w-mlir-cpu-runner.sh -affine $OUT/$basename-before-tiling.mlir main out
    sh run-w-mlir-cpu-runner.sh -affine $OUT/$basename-after-tiling.mlir main out 

    # # try tiling without intermediate files
    # mlir-opt $1 \
    # --one-shot-bufferize='bufferize-function-boundaries' \
    # --convert-vector-to-scf \
    # --convert-linalg-to-affine-loops \
    # > $OUT/$basename-before-tiling-v2.mlir

    # mlir-opt $1 \
    # --one-shot-bufferize='bufferize-function-boundaries' \
    # --convert-vector-to-scf \
    # --convert-linalg-to-affine-loops \
    # --affine-loop-tile \
    # > $OUT/$basename-after-tiling-v2.mlir

    # #  did anything happen?
    # diff $OUT/$basename-before-tiling-v2.mlir $OUT/$basename-after-tiling-v2.mlir

    # # try another tiling pipeline
    # mlir-opt \
    # --linalg-generalize-named-ops --one-shot-bufferize='bufferize-function-boundaries' --convert-vector-to-scf --convert-linalg-to-affine-loops \
    # $1 \
    # -o $OUT/matmul104x104-before-tiling-v3.mlir

    # mlir-opt \
    # --linalg-generalize-named-ops --one-shot-bufferize='bufferize-function-boundaries' --convert-vector-to-scf --convert-linalg-to-affine-loops --affine-loop-tile \
    # $1 \
    # -o $OUT/matmul104x104-after-tiling-v3.mlir

    # # did anything happen?
    # diff $OUT/matmul104x104-before-tiling-v3.mlir $OUT/matmul104x104-after-tiling-v3.mlir
    
fi



