#include "memref.h"
#include <stdio.h>
#include <stdlib.h>

// Define a C function with "_mlir_ciface_" prepended to its name.
// Given a C-struct representation of a 2D memref,
// this function print outs its values.
void _mlir_ciface_print_memref_32_bit(TwoDMemrefI32_t *src) {
  printf("printing memref with shape %d x %d, offset %d: stride: [%d,%d]\n[",
         src->shape[0], src->shape[1], src->offset, src->stride[0],
         src->stride[1]);
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      printf(" %d ", src->aligned_data[src->offset + (src->stride[0] * row) +
                                       (col * src->stride[1])]);
    }
    printf("\n");
  }
  printf("]\n");
}

// External function implemented in MLIR
extern void _mlir_ciface_matmulAndPrint(TwoDMemrefI32_t* a, TwoDMemrefI32_t* b, TwoDMemrefI32_t* output);

int main() {
  // let's create a 2D memref containing
  // [[1, 2],
  //  [3, 4]]
  TwoDMemrefI32_t a;
  a.data = (int32_t *)malloc(sizeof(int32_t) * 4);
  a.aligned_data = a.data;
  a.offset = 0;
  a.shape[0] = 2;
  a.shape[1] = 2;
  a.stride[0] = 2;
  a.stride[1] = 1;
  a.aligned_data[0]=1;
  a.aligned_data[1]=2;
  a.aligned_data[2]=3;
  a.aligned_data[3]=4;

  // let's create a 2D memref containing
  // [[5, 6],
  //  [7, 8]]
  TwoDMemrefI32_t b;
  b.data = (int32_t *)malloc(sizeof(int32_t) * 4);
  b.aligned_data = b.data;
  b.offset = 0;
  b.shape[0] = 2;
  b.shape[1] = 2;
  b.stride[0] = 2;
  b.stride[1] = 1;
  b.aligned_data[0]=5;
  b.aligned_data[1]=6;
  b.aligned_data[2]=7;
  b.aligned_data[3]=8;

  // let's create a 2D memref containing
  // [[0, 0],
  //  [0, 0]]
  TwoDMemrefI32_t c;
  c.data = (int32_t *)malloc(sizeof(int32_t) * 4);
  c.aligned_data = c.data;
  c.offset = 0;
  c.shape[0] = 2;
  c.shape[1] = 2;
  c.stride[0] = 2;
  c.stride[1] = 1;
  c.aligned_data[0]=0;
  c.aligned_data[1]=0;
  c.aligned_data[2]=0;
  c.aligned_data[3]=0;

  // let's call the MLIR matmul compiled to matmul.o
  _mlir_ciface_matmulAndPrint(&a,&b,&c);

  // free up space on heap
  free(a.data);
  free(b.data);
  free(c.data);
  return 0;
}