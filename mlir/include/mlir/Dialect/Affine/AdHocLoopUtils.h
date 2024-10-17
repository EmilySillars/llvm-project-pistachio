//===- AdHocLoopUtils.h - Loop transformation used by the Ad-Hoc Loop Tile Pass
//--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various loop transformation utility
// methods used by the AdHocLoopTile pass. These definitions could probably be
// added to LoopUtils.h, but to help make code changes explicit, a separate
// header file will be used.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ADHOC_LOOPUTILS_H
#define MLIR_DIALECT_AFFINE_ADHOC_LOOPUTILS_H

#include "mlir/IR/Block.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include <optional>

namespace mlir {
class AffineMap;
class LoopLikeOpInterface;
class OpBuilder;
class Value;
class ValueRange;

namespace func {
class FuncOp;
} // namespace func

namespace scf {
class ForOp;
class ParallelOp;
} // namespace scf

namespace affine {
class AffineForOp;
struct MemRefRegion;

namespace AdHocLoopTile {

// This structure is used to store bound and step information about a subloop
struct LoopParams {
  AffineForOp* parent = 0; // 0 when subloop's parent is another subloop
  AffineForOp* ancestor = 0; // 0 when subloop's parent is another subloop
  size_t parentIndex = 0;  // used when subloop's parent is another subloop
  size_t parentTileSize = 0;
  size_t stepSize = 0;
  LoopParams() = default;
};

void constructDummyLoopNest(MutableArrayRef<AffineForOp> origLoops,
                                   AffineForOp rootAffineForOp, unsigned width,
                                   MutableArrayRef<AffineForOp> tiledLoops);

// LogicalResult tilePerfectlyNested(MutableArrayRef<AffineForOp> input,
//                                   ArrayRef<unsigned> tileSizes,
//                                   SmallVectorImpl<AffineForOp> *tiledNest);
template <typename t>
static LogicalResult performPreTilingChecks(MutableArrayRef<AffineForOp> input,
                                            ArrayRef<t> tileSizes);
// copied from LoopUtils.cpp and LoopTiling.cpp

void
constructTiledIndexSetHyperRect(MutableArrayRef<AffineForOp> origLoops,
                                MutableArrayRef<AffineForOp> newLoops,
                                ArrayRef<unsigned> tileSizes);
/// Move the loop body of AffineForOp 'src' from 'src' into the specified
/// location in destination's body, ignoring the terminator.
static void moveLoopBodyImpl(AffineForOp src, AffineForOp dest,
                             Block::iterator loc);

/// Move the loop body of AffineForOp 'src' from 'src' to the start of dest
/// body.
void moveLoopBody(AffineForOp src, AffineForOp dest);

} // namespace AdHocLoopTile
} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ADHOC_LOOPUTILS_H
