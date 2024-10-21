#!/bin/sh
if [[ $1 == "--help" ]]; then
    echo "USAGE: sh tile-w-zigzagTile.sh <filename.mlir> <functionName> <output-directory> <tiling scheme filename>"
else
    basename=`basename $1 | sed 's/[/]*[.][^.]*$//'`
    funcname=`basename $2 | sed 's/[.][^.]*$//'`
    OUT=$3/$basename/tiling
    CORRECT=correct
    options=$4
    mkdir -p $OUT

    # starting with linalg!!
    # changed for ZigZag tiling
    optionsWrapped="builtin.module(func.func(zigzag-tile{$options}))"
    mlir-opt $basename.mlir \
    -pass-pipeline=$optionsWrapped \
    > $OUT/$basename-after-tiling.mlir

    # did anything happen?
    diff $basename.mlir $OUT/$basename-after-tiling.mlir > $OUT/$basename.log 2>&1

    # lower mlir to llvm
    mlir-opt $basename.mlir --one-shot-bufferize='bufferize-function-boundaries' > $OUT/$basename-before-tiling-bufferized.mlir
    mlir-opt $OUT/$basename-before-tiling-bufferized.mlir \
    --convert-vector-to-scf \
    > $OUT/$basename-before-tiling-lowered1.mlir
    mlir-opt $OUT/$basename-before-tiling-lowered1.mlir \
    --convert-linalg-to-affine-loops \
    > $OUT/$basename-before-tiling-lowered2.mlir
    # same for tiled version, lower it
    mlir-opt $OUT/$basename-after-tiling.mlir --one-shot-bufferize='bufferize-function-boundaries' > $OUT/$basename-after-tiling-bufferized.mlir
    mlir-opt $OUT/$basename-after-tiling-bufferized.mlir \
    --convert-vector-to-scf \
    > $OUT/$basename-after-tiling-lowered1.mlir
    mlir-opt $OUT/$basename-after-tiling-lowered1.mlir \
    --convert-linalg-to-affine-loops \
    > $OUT/$basename-after-tiling-lowered2.mlir

    echo "Do the tiled and untiled versions behave the same?" >> $OUT/$basename.log
    sh run-w-mlir-cpu-runner.sh -affine $OUT/$basename-before-tiling-lowered2.mlir main out >> $OUT/$basename.log 2>&1
    sh run-w-mlir-cpu-runner.sh -affine $OUT/$basename-after-tiling-lowered2.mlir main out >> $OUT/$basename.log 2>&1

   # cat $OUT/$basename-after-tiling.mlir
    tail $OUT/$basename.log
    
    # # run our only regression test xD
    # # which relies on us SPECIFICALLY running matmul104x104 with zigzag-tile-scheme.json
    # command
    # status=$?
    
    # ## 1. Run the diff command ##
    # cmd="diff $OUT/$basename-after-tiling.mlir $CORRECT/zigzag-loop-tile/$basename-scheme.mlir"
    # $cmd
    
    # ## 2. Get exist status  and store into '$status' var ##
    # status=$?
    
    # ## 3. Now take some decision based upon '$status' ## 
    # [ $status -eq 0 ] && echo "$cmd command was successful"
    # #DIFF=$(diff $OUT/$basename-after-tiling.mlir $CORRECT/affine-ad-hoc-loop-tile/$basename-scheme2.mlir)
    # # if [$? -eq 0]; then
    # #     echo "Regression Test Passed :)"
        
    # # else
    # #     echo "REGRESSION TEST FAILED"
    # # fi
    
fi
