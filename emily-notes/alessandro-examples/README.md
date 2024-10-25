## print out a tensor in MLIR

0. Make sure to set your environment variables correctly!
   ```
   export PATH=<path-to-llvm-repo-build-folder>/bin:$PATH
   ```

   and for using `mlir-cpu-runner`, do
   ```
   export MLIR_CPU_RUNNER_LIBS=<path-to-llvm-repo-build-folder>/lib/libmlir_c_runner_utils.so,<path-llvm-repo-build-folder>/lib/libmlir_runner_utils.so
   ```

1. Run script with desired function to run and output folder to store results.

```
sh run-w-mlir-cpu-runner.sh -linalg printVector.mlir main out
```

