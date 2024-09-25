#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
#funcname=`basename $2 | sed 's/[.][^.]*$//'`

mlir-opt -pass-pipeline='any(func.func(avocado))' -mlir-pass-statistics -mlir-pass-statistics-display=list $basename.mlir 2> out/$basename-avocado.out >/dev/null

cat out/$basename-avocado.out