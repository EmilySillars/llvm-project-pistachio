# run this script from the top level directory

# normal run
# ../iree-build/tools/iree-opt iree-fork/matmul104x104.mlir

# --test-linalg-transform-patterns=test-patterns
# mlir-opt \
# --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
# ./matmul104x104.mlir \
# -o ./out/matmul104x104-before-tile.mlir

# mlir-opt \
# --linalg-generalize-named-ops --test-linalg-transform-patterns=test-patterns \
# ./matmul104x104.mlir \
# -o ./out/matmul104x104-after-tile.mlir

# diff ./out/matmul104x104-before-tile.mlir ./out/matmul104x104-after-tile.mlir

# # -test-loop-fusion -test-loop-fusion-transformation
# mlir-opt \
# --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
# ./matmul104x104.mlir \
# -o ./out/matmul104x104-before-tile2.mlir

# mlir-opt \
# --allow-unregistered-dialect --linalg-generalize-named-ops --test-loop-fusion --test-loop-fusion-transformation \
# ./matmul104x104.mlir \
# -o ./out/matmul104x104-after-tile2.mlir

# diff ./out/matmul104x104-before-tile2.mlir ./out/matmul104x104-after-tile2.mlir

# test affine dialect tiling
mlir-opt \
--linalg-generalize-named-ops --one-shot-bufferize='bufferize-function-boundaries' --convert-vector-to-scf --convert-linalg-to-affine-loops \
./matmul104x104.mlir \
-o ./out/matmul104x104-before-tile3.mlir

mlir-opt \
--linalg-generalize-named-ops --one-shot-bufferize='bufferize-function-boundaries' --convert-vector-to-scf --convert-linalg-to-affine-loops --affine-loop-tile \
./matmul104x104.mlir \
-o ./out/matmul104x104-after-tile3.mlir

diff ./out/matmul104x104-before-tile3.mlir ./out/matmul104x104-after-tile3.mlir

sh affine-run-w-mlir-cpu-runner.sh matmul104x104-before-tile3.mlir main out

sh affine-run-w-mlir-cpu-runner.sh matmul104x104-after-tile3.mlir main out




# try to test affine dialect tiling (again)
# mlir-opt \
# --convert-linalg-to-affine-loops \
# ./matmul104x104-no-main.mlir \
# -o ./out/matmul104x104-no-main-before-tile3.mlir

# mlir-opt \
# --convert-linalg-to-affine-loops --affine-loop-tile \
# ./matmul104x104-no-main.mlir \
# -o ./out/matmul104x104-no-main-after-tile3.mlir

# diff ./out/matmul104x104-no-main-before-tile3.mlir ./out/matmul104x104-no-main-after-tile3.mlir


# mlir-opt \
# --one-shot-bufferize --convert-linalg-to-affine-loops \
# ./matmul104x104-no-main.mlir \
# -o ./out/matmul104x104-no-main-before-tile4.mlir

# mlir-opt \
# --one-shot-bufferize --convert-linalg-to-affine-loops --affine-loop-tile \
# ./matmul104x104-no-main.mlir \
# -o ./out/matmul104x104-no-main-after-tile4.mlir

# diff ./out/matmul104x104-no-main-before-tile4.mlir ./out/matmul104x104-no-main-after-tile4.mlir
