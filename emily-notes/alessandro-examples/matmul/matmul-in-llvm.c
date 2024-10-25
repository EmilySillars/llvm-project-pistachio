#include "memref.h"  // from KULeuven's snax-mlir repo: https://github.com/KULeuven-MICAS/snax-mlir/blob/4666236ffd848eff1a4634c824ad257ba68d9a64/runtime/include/memref.h
#include "data.h"
#include <stdlib.h>

/*
To compile + run with the regular clang compiler:
clang matmul-in-llvm.c; ./a.out

To compile + run with your llvm repo's clang:
<LLVM-BUILD-DIRECTORY>/bin/clang matmul-in-llvm.c; ./a.out

To compile to LLVM IR:
    1) compile to llvm bitcode
    <LLVM-BUILD-DIRECTORY>/bin/clang -O1 -emit-llvm -c matmul-in-llvm.c
    2) disassemble the llvm bitcode into human-readable LLVM IR
    <LLVM-BUILD-DIRECTORY>/bin/llvm-dis matmul-in-llvm.bc
    Notes:
    - flag -emit-llvm instructs clang to emit llvm ir bitcode
    - flag -O1 disables optimization passes
    - flag -c prevents linking (no LLVM IR generated for linked libraries)
*/

// for now, we assume a square matrix
#ifndef MAT_WIDTH
#define MAT_WIDTH 104
#endif
#define MAT_WIDTH_SQUARED (MAT_WIDTH * MAT_WIDTH)

void cCodeSquareMatmul(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y,
                       TwoDMemrefI32_t *z) {
  int z_index, x_index, y_index = 0;
  for (int d0 = 0; d0 < MAT_WIDTH; d0++) {
    for (int d1 = 0; d1 < MAT_WIDTH; d1++) {
      for (int d2 = 0; d2 < MAT_WIDTH; d2++) {
        // O[d0][d1] += I[d0][d2] * W[d2][d1]; // and this is a MAC!
        z_index = (d0 * MAT_WIDTH) + d1;
        x_index = (d0 * MAT_WIDTH) + d2;
        y_index = (d2 * MAT_WIDTH) + d1;
        z->aligned_data[z_index] +=
            x->aligned_data[x_index] * y->aligned_data[y_index];
      }
    }
  }
}

int main() {

  // Create memref objects representing each matrix
  TwoDMemrefI8_t memrefA;  // input 104x104xi8
  memrefA.data = (int8_t *)malloc(sizeof(int8_t) * MAT_WIDTH_SQUARED);
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = 104;
  memrefA.shape[1] = 104;
  memrefA.stride[0] = 104;
  memrefA.stride[1] = 1;
  TwoDMemrefI8_t memrefB;  // weight 104x104xi8
  memrefB.data = (int8_t *)malloc(sizeof(int8_t) * MAT_WIDTH_SQUARED);
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = 104;
  memrefB.shape[1] = 104;
  memrefB.stride[0] = 1;
  memrefB.stride[1] = 104;
  TwoDMemrefI32_t memrefC;  // output 104x104xi32
  memrefC.data = (int32_t *)malloc(sizeof(int32_t) * MAT_WIDTH_SQUARED);
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;
  memrefC.shape[0] = 104;
  memrefC.shape[1] = 104;
  memrefC.stride[0] = 104;
  memrefC.stride[1] = 1;

   // initialize the matrices
  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefA.aligned_data[i] = (int8_t)2;
    memrefB.aligned_data[i] = (int8_t)3;
    memrefC.aligned_data[i] = (int32_t)0;
  }
  memrefB.aligned_data[85] = 87;

  // perform C code matmul to get the ground truth
  cCodeSquareMatmul(&memrefA, &memrefB, &memrefC);

  // check for correctness
  int nerr = 0;
  for (int i = 0; i < MAT_WIDTH_SQUARED; i++) {
    int32_t error = memrefC.aligned_data[i] - golden[i];
    if (error != 0) {
      nerr += 1;
      printf(" i is %d and %d /= %d\n", i, memrefC.aligned_data[i],
             golden[i]);
      break;
    }
  }

  if (nerr != 0) {
    printf("Output does not match the golden value!\n");
    // print2DMemRefI32_t(&memrefC, 104);
    // print2DMemRefI32_t(&memrefGolden, 104);
  } else {
    printf("Output Correct\n");
  }

  // free everything before exiting!
  free(memrefA.data);
  free(memrefB.data);
  free(memrefC.data);

  return nerr;
}