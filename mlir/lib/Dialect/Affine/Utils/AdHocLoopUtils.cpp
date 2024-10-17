//===- LoopUtils.cpp ---- Misc utilities for loop transformation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop transformation routines used by the AdHocLoopTiling
// pass.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/AdHocLoopUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "loop-utils"

using namespace mlir;
using namespace affine;
using namespace presburger;
using llvm::SmallMapVector;


// LogicalResult
// AdHocLoopTile::tilePerfectlyNested(MutableArrayRef<AffineForOp> input,
//                                   ArrayRef<unsigned> tileSizes,
//                                   SmallVectorImpl<AffineForOp> *tiledNest) {
//   if (input.empty())
//     return success();

//   if (failed(AdHocLoopTile::performPreTilingChecks(input, tileSizes)))
//     return failure();

// MutableArrayRef<AffineForOp> origLoops = input;
// AffineForOp rootAffineForOp = origLoops[0];

// // Note that width is at least one since the band isn't empty.
// unsigned width = input.size();
// SmallVector<AffineForOp, 6> tiledLoops(2 * width);

// // Construct a tiled loop nest without setting their bounds. Bounds are
// // set later.
// AdHocLoopTile::constructTiledLoopNest(origLoops, rootAffineForOp, width,
// tiledLoops);

// SmallVector<Value, 8> origLoopIVs;
// extractForInductionVars(input, &origLoopIVs);

// // Set loop bounds for the tiled loop nest.
// AdHocLoopTile::constructTiledIndexSetHyperRect(origLoops, tiledLoops,
// tileSizes);

// // Replace original IVs with intra-tile loop IVs.
// for (unsigned i = 0; i < width; i++)
//   origLoopIVs[i].replaceAllUsesWith(tiledLoops[i + width].getInductionVar());

// // Erase the old loop nest.
// rootAffineForOp.erase();

// if (tiledNest)
//   *tiledNest = std::move(tiledLoops);

//   return success();
// }

/// Check if the input nest is supported for tiling and whether tiling would be
/// legal or not.
template <typename t>
static LogicalResult
AdHocLoopTile::performPreTilingChecks(MutableArrayRef<AffineForOp> input,
                                      ArrayRef<t> tileSizes) {
  assert(input.size() == tileSizes.size() && "Too few/many tile sizes");

  if (llvm::any_of(input,
                   [](AffineForOp op) { return op.getNumResults() > 0; })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot tile nest where a loop has yield values\n");
    return failure();
  }

  // Check if the supplied `for` ops are all successively nested.
  if (!isPerfectlyNested(input)) {
    LLVM_DEBUG(llvm::dbgs() << "input loops not perfectly nested");
    return failure();
  }

  //  TODO: handle non hyper-rectangular spaces.
  // if (failed(checkIfHyperRectangular(input)))
  //   return failure();

  return success();
}

// create a dummy loop nest around the original nested loop's body
void AdHocLoopTile::constructDummyLoopNest(
    MutableArrayRef<AffineForOp> origLoops, AffineForOp rootAffineForOp,
    unsigned width, MutableArrayRef<AffineForOp> tiledLoops) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] inside dummy loop nest \n");
  Location loc = rootAffineForOp.getLoc();

  // The outermost among the loops as we add more..
  Operation *topLoop = rootAffineForOp.getOperation();
  AffineForOp innermostPointLoop;

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] location used is  " << loc
                          << " \n");

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] top Loop before any messing around is "
                          << *topLoop << " \n");

  // topLoop->getBlock()->getOperations()
  //  auto& hoodle = rootAffineForOp->getBody();
  //  const auto& hoodle = rootAffineForOp->getBlock()->getOperations();
  //  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] seems like the original loop
  //  body is maybe "<< hoodle[0] <<" \n");
  // rootAffineForOp.getLoc()

  // Add intra-tile (or point) loops. (LOOPS WITHIN A TILE)
  for (unsigned i = 0; i < width; i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    AffineForOp pointLoop = b.create<AffineForOp>(loc, 0, 0);
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << i << "point loop is "
                            << pointLoop << " \n");

    pointLoop.getBody()->getOperations().splice(
        pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);

    tiledLoops[2 * width - 1 - i] = pointLoop;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] tiledLoops["
                            << (2 * width - 1 - i) << "] = pointLoop; \n");

    topLoop = pointLoop.getOperation();
    if (i == 0)
      innermostPointLoop = pointLoop;
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] top Loop is " << *topLoop
                          << " \n");

  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] top Loop is possibly the inter-tile loops now... \n");
  // I think these are the between-tile loops

  // Add tile space loops;
  for (unsigned i = width; i < 2 * width;
       i++) { // some kind of assumption here that tiledLoops has twice the
              // width of origloops
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    AffineForOp tileSpaceLoop = b.create<AffineForOp>(loc, 0, 0);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] " << i << "tileSpaceLoop loop is "
               << tileSpaceLoop << " \n");

    tileSpaceLoop.getBody()->getOperations().splice(
        tileSpaceLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    tiledLoops[2 * width - i - 1] = tileSpaceLoop;

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] tiledLoops["
                            << (2 * width - 1 - i) << "] = tileSpaceLoop; \n");
    topLoop = tileSpaceLoop.getOperation();
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] now top Loop is " << *topLoop
                          << " \n");

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] innermost point loops is "
                          << innermostPointLoop << " \n");

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] origLoops.back() has size "
                          << origLoops.size() << " and is " << origLoops.back()
                          << " \n");

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] origLoops.back().body() is "
                          << origLoops.back().getBody() << " \n");

  // // Move the loop body of the original nest to the new one.
  AdHocLoopTile::moveLoopBody(origLoops.back(), innermostPointLoop);
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] about to return from dummy loop nest \n");
}

/// Constructs and sets new loop bounds after tiling for the case of
/// hyper-rectangular index sets, where the bounds of one dimension do not
/// depend on other dimensions. Bounds of each dimension can thus be treated
/// independently, and deriving the new bounds is much simpler and faster
/// than for the case of tiling arbitrary polyhedral shapes.
void AdHocLoopTile::constructTiledIndexSetHyperRect(
    MutableArrayRef<AffineForOp> origLoops,
    MutableArrayRef<AffineForOp> newLoops, ArrayRef<unsigned> tileSizes) {
  assert(!origLoops.empty());
  assert(origLoops.size() == tileSizes.size());

  OpBuilder b(origLoops[0].getOperation());
  unsigned width = origLoops.size();

  // Bounds for tile space loops.
  for (unsigned i = 0; i < width; i++) {
    OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    newLoops[i].setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
    newLoops[i].setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
    // If the step size of original loop is x and tileSize is y then after
    // tiling the tile space loops' step size becomes x*y.
    newLoops[i].setStep(tileSizes[i] * origLoops[i].getStepAsInt());
  }
  // Bounds for intra-tile loops.
  for (unsigned i = 0; i < width; i++) {
    int64_t largestDiv = getLargestDivisorOfTripCount(origLoops[i]);
    std::optional<uint64_t> mayBeConstantCount =
        getConstantTripCount(origLoops[i]);
    // The lower bound is just the tile-space loop.
    AffineMap lbMap = b.getDimIdentityMap();
    newLoops[width + i].setLowerBound(
        /*operands=*/newLoops[i].getInductionVar(), lbMap);
    // The step sizes of intra-tile loops is just the original loops' step size.
    newLoops[width + i].setStep(origLoops[i].getStepAsInt());

    // Set the upper bound.
    if (mayBeConstantCount && *mayBeConstantCount < tileSizes[i]) {
      // Trip count is less than the tile size: upper bound is lower bound +
      // trip count * stepSize.
      AffineMap ubMap = b.getSingleDimShiftAffineMap(
          *mayBeConstantCount * origLoops[i].getStepAsInt());
      newLoops[width + i].setUpperBound(
          /*operands=*/newLoops[i].getInductionVar(), ubMap);
    } else if (largestDiv % tileSizes[i] != 0) {
      // Intra-tile loop ii goes from i to min(i + tileSize * stepSize, ub_i).
      // Construct the upper bound map; the operands are the original operands
      // with 'i' (tile-space loop) appended to it. The new upper bound map is
      // the original one with an additional expression i + tileSize * stepSize
      // appended.

      // Add dim operands from original upper bound.
      SmallVector<Value, 4> ubOperands;
      AffineBound ub = origLoops[i].getUpperBound();
      ubOperands.reserve(ub.getNumOperands() + 1);
      AffineMap origUbMap = ub.getMap();
      for (unsigned j = 0, e = origUbMap.getNumDims(); j < e; ++j)
        ubOperands.push_back(ub.getOperand(j));

      // Add dim operand for new loop upper bound.
      ubOperands.push_back(newLoops[i].getInductionVar());

      // Add symbol operands from original upper bound.
      for (unsigned j = 0, e = origUbMap.getNumSymbols(); j < e; ++j)
        ubOperands.push_back(ub.getOperand(origUbMap.getNumDims() + j));

      SmallVector<AffineExpr, 4> boundExprs;
      boundExprs.reserve(1 + origUbMap.getNumResults());
      AffineExpr dim = b.getAffineDimExpr(origUbMap.getNumDims());
      // The new upper bound map is the original one with an additional
      // expression i + tileSize * stepSize (of original loop) appended.
      boundExprs.push_back(dim + tileSizes[i] * origLoops[i].getStepAsInt());
      boundExprs.append(origUbMap.getResults().begin(),
                        origUbMap.getResults().end());
      AffineMap ubMap =
          AffineMap::get(origUbMap.getNumDims() + 1, origUbMap.getNumSymbols(),
                         boundExprs, b.getContext());
      newLoops[width + i].setUpperBound(/*operands=*/ubOperands, ubMap);
    } else {
      // No need of the min expression.
      AffineExpr dim = b.getAffineDimExpr(0);
      AffineMap ubMap = AffineMap::get(
          1, 0, dim + tileSizes[i] * origLoops[i].getStepAsInt());
      newLoops[width + i].setUpperBound(newLoops[i].getInductionVar(), ubMap);
    }
  }
}

/// Move the loop body of AffineForOp 'src' from 'src' into the specified
/// location in destination's body, ignoring the terminator.
static void AdHocLoopTile::moveLoopBodyImpl(AffineForOp src, AffineForOp dest,
                                            Block::iterator loc) {
  auto &ops = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, ops, ops.begin(),
                                         std::prev(ops.end()));
}

/// Move the loop body of AffineForOp 'src' from 'src' to the start of dest
/// body.
void AdHocLoopTile::moveLoopBody(AffineForOp src, AffineForOp dest) {
  moveLoopBodyImpl(src, dest, dest.getBody()->begin());
}