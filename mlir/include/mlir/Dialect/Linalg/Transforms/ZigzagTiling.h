//===- Transforms.h - Linalg transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_ZIGZAGTILING_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_ZIGZAGTILING_H

#include <utility>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

namespace mlir {
namespace linalg {

namespace zigzag {
// template <typename Range>
// static LogicalResult
// // based on testTilingInterfaceTransformOps.cpp
// applyTileAndFuseToAll(RewriterBase &rewriter, Operation *transformOp,
//                       Range &&payloadOps, unsigned numLoops,
//                       ArrayRef<OpFoldResult> tileSizes,
//                       ArrayRef<int64_t> interchange, bool useForall,
//                       transform::TransformResults &transformResults);

// // based con Quidditch func
// static LogicalResult
// applyTileAndFuseToEachRoot(RewriterBase &rewriter,
//                            llvm::SmallDenseSet<TilingInterface> &payloadOps,
//                            int tilingLevel);

// SmallVector<OpFoldResult>
// ZigZagTileSizeComputation(OpBuilder &builder, Operation *operation, ArrayRef<ArrayRef<int64_t>> tileSizes);


} // namespace zigzag
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_ZIGZAGTILING_H
