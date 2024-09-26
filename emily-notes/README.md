# What kinds of tiling does mlir-opt support?

0. make sure to set you environment variables correctly!
   ```
   export PATH=/home/hoppip/llvm-project-pistachio/build-riscv/bin:$PATH
   ```

   and for using mlir-cpu-runner, do

   ```
   export MLIR_CPU_RUNNER_LIBS=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
   ```

## Setup

1. ```
   cd llvm-project-pistachio
   ```

2. ```
   mkdir build-riscv
   ```

3. ```
   cd build-rsicv
   ```

4. ```
   cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU;RISCV" \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_USE_LINKER=lld \
   -DCMAKE_C_COMPILER=/usr/bin/clang \
   -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON
   ```

5. ```
   ninja -j 20
   ```

6. 



## Investigation

1. `mlir-opt --help | tiling tile`:

   ```
   hoodle
   ```

   

## Examples

