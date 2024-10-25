// copied from 
// https://github.com/KULeuven-MICAS/snax-mlir/blob/f651860981efe0da84c0e5231bfcb03faf16890a/runtime/include/memref.h
#pragma once

#include <stdint.h>
#include <stdio.h>

struct OneDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  uint32_t offset;
  uint32_t shape[1];
  uint32_t stride[1];
};

struct TwoDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  uint32_t offset;
  uint32_t shape[2];
  uint32_t stride[2];
};

struct TwoDMemrefI8 {
  int8_t *data; // allocated pointer: Pointer to data buffer as allocated,
                // only used for deallocating the memref
  int8_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                        // that memref indexes
  uint32_t offset;
  uint32_t shape[2];
  uint32_t stride[2];
};

typedef struct OneDMemrefI32 OneDMemrefI32_t;
typedef struct TwoDMemrefI8 TwoDMemrefI8_t;
typedef struct TwoDMemrefI32 TwoDMemrefI32_t;