# run this script from the top level directory

# normal run
# ../iree-build/tools/iree-opt iree-fork/matmul104x104.mlir


mlir-opt \
--pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
./matmul104x104.mlir \
-o ./out/matmul104x104-before-tile.mlir

mlir-opt \
--linalg-generalize-named-ops --test-linalg-transform-patterns=test-patterns \
./matmul104x104.mlir \
-o ./out/matmul104x104-after-tile.mlir

diff ./out/matmul104x104-before-tile.mlir ./out/matmul104x104-after-tile.mlir

# -test-loop-fusion -test-loop-fusion-transformation

mlir-opt \
--pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
./matmul104x104.mlir \
-o ./out/matmul104x104-before-tile2.mlir

mlir-opt \
--allow-unregistered-dialect --linalg-generalize-named-ops --test-loop-fusion --test-loop-fusion-transformation \
./matmul104x104.mlir \
-o ./out/matmul104x104-after-tile2.mlir

diff ./out/matmul104x104-before-tile2.mlir ./out/matmul104x104-after-tile2.mlir

# # ../iree-build/tools/iree-opt --iree-codegen-gpu-apply-tiling-level iree-fork/matmul104x104.mlir

# # ../iree-build/tools/iree-opt --iree-codegen-gpu-generalize-named-ops \
# # --iree-codegen-gpu-tile-reduction  iree-fork/matmul104x104.mlir \
# # -o iree-fork/out/matmul104x104-before-tile.mlir

# ../iree-build/tools/iree-opt \
# --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
# iree-fork/matmul104x104.mlir \
# -o iree-fork/out/matmul104x104-before-tile2.mlir

# ../iree-build/tools/iree-opt \
# --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops,iree-codegen-gpu-tile-reduction))" \
# iree-fork/matmul104x104.mlir \
# -o iree-fork/out/matmul104x104-after-tile2.mlir

# diff iree-fork/out/matmul104x104-before-tile2.mlir iree-fork/out/matmul104x104-after-tile2.mlir


# ../iree-build/tools/iree-opt \
# --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
# iree-fork/matmul104x104.mlir \
# -o iree-fork/out/matmul104x104-before-tile3.mlir

# ../iree-build/tools/iree-opt \
# --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops,iree-codegen-gpu-tensor-tile))" \
# iree-fork/matmul104x104.mlir \
# -o iree-fork/out/matmul104x104-after-tile3.mlir

# diff iree-fork/out/matmul104x104-before-tile3.mlir iree-fork/out/matmul104x104-after-tile3.mlir
#  --iree-codegen-gpu-generalize-named-ops                                -   Convert named Linalg ops to linalg.generic ops
#       --iree-codegen-gpu-tensor-tile                                         -   Pass to tile tensor (linalg) ops within a GPU workgroup
#       --iree-codegen-gpu-tile-reduction 
# --iree-codegen-gpu-tensor-tile