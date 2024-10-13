#!/bin/sh
if [[ $1 == "--help" ]]; then
    echo "USAGE: sh tile-w-affine.sh <filename.mlir> <functionName> <output-directory> <tiling scheme filename>"
else
    basename=`basename $1 | sed 's/[/]*[.][^.]*$//'`
    funcname=`basename $2 | sed 's/[.][^.]*$//'`
    OUT=$3/$basename/tiling
    CORRECT=correct
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
    --affine-ad-hoc-loop-tile=$options \
    > $OUT/$basename-after-tiling.mlir

    # did anything happen?
    diff $OUT/$basename-before-tiling.mlir $OUT/$basename-after-tiling.mlir > $OUT/$basename.log 2>&1

    echo "Do the tiled and untiled versions behave the same?" >> $OUT/$basename.log
    sh run-w-mlir-cpu-runner.sh -affine $OUT/$basename-before-tiling.mlir main out >> $OUT/$basename.log 2>&1
    sh run-w-mlir-cpu-runner.sh -affine $OUT/$basename-after-tiling.mlir main out >> $OUT/$basename.log 2>&1

   # cat $OUT/$basename-after-tiling.mlir
    tail $OUT/$basename.log
    
    # run our only regression test xD
    # which relies on us SPECIFICALLY running matmul104x104 with zigzag-tile-scheme.json
    command
    status=$?
    
    ## 1. Run the diff command ##
    cmd="diff $OUT/$basename-after-tiling.mlir $CORRECT/affine-ad-hoc-loop-tile/$basename-scheme.mlir"
    $cmd
    
    ## 2. Get exist status  and store into '$status' var ##
    status=$?
    
    ## 3. Now take some decision based upon '$status' ## 
    [ $status -eq 0 ] && echo "$cmd command was successful"
    #DIFF=$(diff $OUT/$basename-after-tiling.mlir $CORRECT/affine-ad-hoc-loop-tile/$basename-scheme2.mlir)
    # if [$? -eq 0]; then
    #     echo "Regression Test Passed :)"
        
    # else
    #     echo "REGRESSION TEST FAILED"
    # fi
    
fi



