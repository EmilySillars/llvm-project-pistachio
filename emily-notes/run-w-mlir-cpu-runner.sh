# #!/bin/sh
# basename=`basename $1 | sed 's/[.][^.]*$//'`
# funcname=`basename $2 | sed 's/[.][^.]*$//'`


# mlir-opt $basename.mlir --one-shot-bufferize='bufferize-function-boundaries' > out/$basename-bufferized.mlir

# mlir-opt out/$basename-bufferized.mlir \
# -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf > out/$basename-vector-scf-cf.mlir 

# mlir-opt out/$basename-vector-scf-cf.mlir  \
# -convert-vector-to-llvm --convert-cf-to-llvm > out/$basename-vector-scf-cf-llvm.mlir 

# mlir-opt out/$basename-vector-scf-cf-llvm.mlir  \
# -expand-strided-metadata -lower-affine -convert-arith-to-llvm \
# --memref-expand -finalize-memref-to-llvm > out/$basename-vector-scf-cf-llvm2.mlir 

# mlir-opt out/$basename-vector-scf-cf-llvm2.mlir   \
# --finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --convert-func-to-llvm='index-bitwidth=32' > out/$basename-vector-scf-cf-llvm3.mlir 

# mlir-opt out/$basename-vector-scf-cf-llvm3.mlir \
# -reconcile-unrealized-casts > out/$basename-vector-scf-cf-llvm4.mlir 

# mlir-cpu-runner -e $funcname -entry-point-result=void \
# -shared-libs=$MLIR_CPU_RUNNER_LIBS \
# out/$basename-vector-scf-cf-llvm4.mlir  > out/$basename.out

#!/bin/sh
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`

# make an output directory if doesn't already exist
mkdir -p $basename/out

# lower mlir to llvm
mlir-opt $basename.mlir --one-shot-bufferize='bufferize-function-boundaries' > $basename/out/$basename-bufferized.mlir

mlir-opt $basename/out/$basename-bufferized.mlir \
--convert-vector-to-scf \
> $basename/out/$basename-lowered1.mlir

mlir-opt $basename/out/$basename-lowered1.mlir \
--convert-linalg-to-loops \
> $basename/out/$basename-lowered2.mlir

mlir-opt $basename/out/$basename-lowered2.mlir \
--lower-affine \
> $basename/out/$basename-lowered3.mlir

mlir-opt $basename/out/$basename-lowered3.mlir \
--convert-scf-to-cf \
> $basename/out/$basename-lowered4.mlir

mlir-opt $basename/out/$basename-lowered4.mlir \
--canonicalize \
> $basename/out/$basename-lowered5.mlir

mlir-opt $basename/out/$basename-lowered5.mlir \
--cse \
> $basename/out/$basename-lowered6.mlir

mlir-opt $basename/out/$basename-lowered6.mlir \
--convert-vector-to-llvm='reassociate-fp-reductions' \
> $basename/out/$basename-lowered7.mlir

mlir-opt $basename/out/$basename-lowered7.mlir \
--convert-math-to-llvm \
> $basename/out/$basename-lowered8.mlir

mlir-opt $basename/out/$basename-lowered8.mlir \
--expand-strided-metadata \
> $basename/out/$basename-lowered9.mlir

mlir-opt $basename/out/$basename-lowered9.mlir \
--lower-affine \
> $basename/out/$basename-lowered10.mlir

mlir-opt $basename/out/$basename-lowered10.mlir \
--finalize-memref-to-llvm \
> $basename/out/$basename-lowered11.mlir

mlir-opt $basename/out/$basename-lowered11.mlir \
--convert-func-to-llvm \
> $basename/out/$basename-lowered12.mlir

mlir-opt $basename/out/$basename-lowered12.mlir \
--convert-index-to-llvm \
> $basename/out/$basename-lowered13.mlir

mlir-opt $basename/out/$basename-lowered13.mlir \
--reconcile-unrealized-casts \
> $basename/out/$basename-lowered14.mlir

cat $basename/out/$basename-lowered14.mlir > $basename/out/$basename-in-llvm-dialect.mlir

# run LLVM MLIR with the mlir-cpu-runner

mlir-cpu-runner -e $funcname -entry-point-result=void \
-shared-libs=$MLIR_CPU_RUNNER_LIBS \
$basename/out/$basename-in-llvm-dialect.mlir  > $basename/out/$basename.out

# print output of run
cat $basename/out/$basename.out


