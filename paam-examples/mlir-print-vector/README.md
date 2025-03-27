# PAAM Resources: MLIR Print Tensor

[[Return to Homepage]](../../paam-resources/README.md)

### mlir-cpu-runner JIT

You can run MLIR source code on your native cpu using MLIR's JIT tool, `mlir-cpu-runner`. 

This is convenient for quick testing/learning how MLIR works, because `mlir-cpu-runner` includes some handy print functions. 

These functions are located in shared object files inside the `lib` directory of your mlir build folder. Therefore, when invoking the `mlir-cpu-runner`, make sure to pass paths to the following two libraries with the `-shared-libs` flag.

- `libmlir_runner_utils.so`
- `libmlir_c_runner_utils.so`

I like to save the paths two these files inside another environment variable, `MLIR_CPU_RUNNER_LIBS`.

For example,

```
export MLIR_CPU_RUNNER_LIBS=/home/emily/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/emily/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
```

### mlir-opt

We use `mlir-opt` to lower one MLIR dialect into another. This `mlir-opt` tool as well as the `mlir-cpu-runner` tool are both located in the `bin` directory of your MLIR build folder. Add this `bin` folder to your path environment variable, so it's easy to invoke the tools. For example:

```
export PATH=/home/emily/llvm-project-pistachio/build-riscv/bin/:$PATH
```

To check if you can successfully invoke `mlir-opt` / if the tool has indeed been added to your path, you can run

```
mlir-opt --help
```

## 1. Lower from tensor dialect to LLVM *dialect* (still in MLIR)

Lower to LLVM Dialect, and save output in `print-tensors-llvm.mlir`

```
mlir-opt print-tensors.mlir \
-test-linalg-transform-patterns=test-linalg-to-vector-patterns \
-empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
-bufferization-bufferize -tensor-bufferize -func-bufferize \
-finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
-convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm \
--convert-cf-to-llvm -expand-strided-metadata \
--lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
> out/print-tensors-llvm.mlir 

```

### Tips

- To see the IR after a particular pass has run, you can use the `--mlir-print-ir-after=<pass-arg> ` flag.
  For example, to lower to LLVM MLIR, but also save the IR output after the `--convert-linalg-to-loops` pass runs, do

  ```
  mlir-opt print-tensors.mlir \
  -test-linalg-transform-patterns=test-linalg-to-vector-patterns \
  -empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
  -bufferization-bufferize -tensor-bufferize -func-bufferize \
  -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
  -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm \
  --convert-cf-to-llvm -expand-strided-metadata \
  --lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
  --mlir-print-ir-after=convert-linalg-to-loops \
  > out/print-tensors-llvm.mlir 2>after-pass.mlir
  ```

- Other helpful printing options:
  ```
    --mlir-print-ir-after=<pass-arg>                           
    --mlir-print-ir-after-all                                  
    --mlir-print-ir-after-change                               
    --mlir-print-ir-after-failure                              
    --mlir-print-ir-before=<pass-arg>                         
    --mlir-print-ir-before-all                                
    --mlir-print-ir-module-scope                           
  ```

  ^^ I found these by running `mlir-opt --help | grep print`

- Specifying a sequence of mlir passes on the command line can be annoying and time consuming. It might be worth it to write a shell script to run any file through a particular sequence of mlir passes. Here is [an example of a shell script for the original lowering](run-func-as-mlir.sh) + `mlir-cpu-runner` call and how to invoke it:
  ```
  sh run-func-as-mlir.sh print-tensors.mlir main
  ```

## 2. Run with mlir-cpu-runner

Invoke `mlir-cpu-runner`

```
mlir-cpu-runner -e main -entry-point-result=void \
-shared-libs=$MLIR_CPU_RUNNER_LIBS \
out/print-tensors-llvm.mlir 
```

^^ Note that `-e` specifies the function `mlir-cpu-runner` will run ( e stands for "entry point")

Expected Output:

```
Unranked Memref base@ = 0x59ea00de0540 rank = 3 offset = 0 sizes = [1, 4, 5] strides = [20, 5, 1] data = 
[[[2.3,    2.3,    2.3,    2.3,    2.3], 
  [2.3,    2.3,    2.3,    2.3,    2.3], 
  [1,    2,    3,    2.3,    2.3], 
  [2,    3,    4,    2.3,    2.3]]]
```

### Tips

- Remember you can always use `mlir-cpu-runner --help`, and then `grep` for keywords related to what you are confused about.