# PAAM Resources: MLIR Print Memref
[[Return to Homepage]](../../paam-resources/README.md)

*Based on: https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission*

*and https://github.com/KULeuven-MICAS/snax-mlir/blob/f651860981efe0da84c0e5231bfcb03faf16890a/runtime/include/memref.h*

### The common link: LLVM IR

Since C and MLIR both compile to LLVM IR, we can link C functions into our MLIR source code, and vice versa. When memrefs get lowered to LLVM IR, their LLVM structure can be lifted to an equivalent C-struct representation. The general definition of an MLIR memref in C++ is

```
template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};
```

For our toy example, we only define a C struct that representing a 2D memref of 32 bit integers:

```
struct TwoDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  uint32_t offset;
  uint32_t shape[2];
  uint32_t stride[2];
};
typedef struct TwoDMemrefI32 TwoDMemrefI32_t;
```

In this example, we define a main function in C which drives the program. The main function launches our `matmulAndPrint` function defined in MLIR. This `matmulAndPrint` function performs a `linalg.matmul` operation defined in MLIR, followed by printing the result (using a printing helper function defined in C).

## 1. Define a print function and main function in C

- Include a header file which contains the C struct definitions of the memref types you want to print out (memref.h)
- Define a print function given the memref struct definitions in your header file, making sure to preprend your function name with "`_mlir_ciface_`" (print_memref_32_bit)
- Declare an external C function with name and types corresponding to the mlir function you wish to launch.  A memref type in MLIR corresponds to a pointer to a memref struct in C. Make sure your MLIR function is also preprended with "`_mlir_ciface_`"  (matmulAndPrint)
- Call this external C function from your main function.

## 2. Add C function prototype and emit_c attributes to MLIR source code

Given the print function with signature

```
void _mlir_ciface_print_memref_32_bit(TwoDMemrefI32_t *src)
```

the corresponding function prototype in MLIR is

```
"func.func"() <{function_type = (memref<2x2xi32, strided<[2,1], offset: ?>>) -> (), 
                sym_name = "print_memref_32_bit", 
                sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
```

**Make sure your function prototype and the MLIR code you want to call from your C main function (in this example, the `matmulAndPrint` MLIR function) contain the `llvm.emit_c_interface` attribute.**

## 3. Compile MLIR source code to an object file (.o)

The compilation consists of three steps

1. Lower to LLVM MLIR Dialect with `mlir-opt`
2. Convert LLVM MLIR Dialect to LLVM IR with `mlir-translate`
3. Compile LLVM IR to an object file with `clang`

I automated these steps in a shell script, but note that step 1 will contain different MLIR lowering passes depending on what kind of dialects your source code uses. For this example, the lowering passes in the shell script are sufficient.

```
sh lower-mlir-to-llvm.sh matmul.mlir
```

## 4. Compile C main function, linking in the MLIR

```
clang main.c -Xlinker matmul/out/matmul.o -o out/finalExecutable
```

## 5. Run Resulting Executable

```
out/finalExecutable
```

Expected Output:

```
printing memref with shape 2 x 2, offset 0: stride: [2,1]
[ 19  22 
 43  50 
]
```

