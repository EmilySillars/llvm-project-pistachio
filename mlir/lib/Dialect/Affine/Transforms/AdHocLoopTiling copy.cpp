//===- AdHocLoopTiling.cpp --- Ad-Hoc Loop tiling pass
//------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile certain non-hyperrectangular loop nests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/AdHocLoopUtils.h" // specific to this pass
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h" // to parse tiling scheme
#include <fstream>             // to open tiling scheme file
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEADHOCLOOPTILING
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "affine-ad-hoc-loop-tile"

namespace {

/// A pass to perform loop tiling on all suitable loop nests of a Function.
struct AdHocLoopTiling
    : public affine::impl::AffineAdHocLoopTilingBase<AdHocLoopTiling> {
  AdHocLoopTiling() = default;
  void runOnOperation() override;
  // for exploring
  LogicalResult tilePerfectlyNested(MutableArrayRef<AffineForOp> input,
                                    SmallVectorImpl<AffineForOp> *tiledNest);
  // for transforming
  LogicalResult tilePerfectlyNested2(MutableArrayRef<AffineForOp> input,
                                     SmallVectorImpl<AffineForOp> *tiledNest);
  void constructPomegranateLoopNest(MutableArrayRef<AffineForOp> origLoops,
                                    AffineForOp rootAffineForOp, unsigned width,
                                    MutableArrayRef<AffineForOp> tiledLoops);
  // everything below relates to processing the tiling scheme as input
  LogicalResult initializeOptions(StringRef options) override;
  void parseTilingScheme(StringRef fileContent);
  void parseListOfListOfInts(llvm::json::Object *obj, std::string listName,
                             std::vector<std::vector<int>> &out);
  struct TilingScheme {
    // TODO: use SmallVector (llvm/include/llvm/ADT/SmallVector.h)
    //       instead of std::vector!
    //       Check this link about when to use Small Vector:
    //       llvm/docs/ProgrammersManual.rst#L1543-L1544
    std::vector<std::vector<int>> bounds;
    std::vector<std::vector<int>> order;
    std::vector<std::vector<int>> finalIndices;
    uint64_t totalLoopCount = 0;
    TilingScheme() = default;
    void setTotalLoopCount();
    void buildFinalIndices();

  private:
    int findSubloop(size_t i, size_t j);
  } ts;
  friend std::stringstream &
  operator<<(std::stringstream &ss, const AdHocLoopTiling::TilingScheme &ts) {
    ss << "tiling scheme: {\nbounds: [ ";
    for (const auto &sublist : ts.bounds) {
      ss << "[ ";
      for (const auto &bound : sublist) {
        ss << " " << bound << " ";
      }
      ss << "] ";
    }
    ss << "]\n";
    ss << "order: [ ";
    for (const auto &sublist : ts.order) {
      ss << "[ ";
      for (const auto &pos : sublist) {
        ss << " " << pos << " ";
      }
      ss << "] ";
    }
    ss << "]\n}";
    return ss;
  }
};

} // namespace

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAdHocLoopTilingPass() {
  return std::make_unique<AdHocLoopTiling>();
}

void AdHocLoopTiling::TilingScheme::setTotalLoopCount() {
  unsigned total = 0;
  for (const auto &bound : bounds) {
    total += (bound.size() +
              1); // for each loop getting tiled, count the extra affine loop
                  // needed to calculate the first level indexing inside a tile
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] total number of loops in tiled loop nest will be "
                   << total << " \n");
  totalLoopCount = total;
}

void AdHocLoopTiling::TilingScheme::buildFinalIndices() {
  // std::vector<std::vector<int>> bounds;
  // finalIndices
  for (size_t i = 0; i < bounds.size(); i++) {
    finalIndices.push_back(std::vector<int>());
    for (size_t j = 0; j < bounds[i].size(); j++) {
      size_t finalIndex = totalLoopCount - findSubloop(i, j) - 1;
      finalIndices[i].push_back(finalIndex);
    }
  }
}

int AdHocLoopTiling::TilingScheme::findSubloop(size_t i, size_t j) {
  for (size_t k = 0; k < order.size(); k++) {
    if (((size_t)order[k][0] == i) && ((size_t)order[k][1] == j)) {
      return k;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] Error: Could not find subloop in tiling scheme "
                             "order. Returning negative index... \n");
  return -1;
}

/* vvvvvvvvvvvvvvvvvvvvvvvvvvvv this function was completely copy-and-pasted
 * from LoopTiling.cpp vvvvvvvvvvvvvvv*/
/// Checks whether hyper-rectangular loop tiling of the nest represented by
/// `origLoops` is valid. The validity condition is from Irigoin and Triolet,
/// which states that two tiles cannot depend on each other. We simplify such
/// condition to just checking whether there is any negative dependence
/// direction, since we have the prior knowledge that the tiling results will be
/// hyper-rectangles, which are scheduled in the lexicographically increasing
/// order on the vector of loop indices. This function will return failure when
/// any dependence component is negative along any of `origLoops`.
static bool checkTilingLegality(MutableArrayRef<AffineForOp> origLoops) {
  assert(!origLoops.empty() && "no original loops provided");

  // We first find out all dependences we intend to check.
  SmallVector<Operation *, 8> loadAndStoreOps;
  origLoops[0]->walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
  });

  unsigned numOps = loadAndStoreOps.size();
  unsigned numLoops = origLoops.size();
  for (unsigned d = 1; d <= numLoops + 1; ++d) {
    for (unsigned i = 0; i < numOps; ++i) {
      Operation *srcOp = loadAndStoreOps[i];
      MemRefAccess srcAccess(srcOp);
      for (unsigned j = 0; j < numOps; ++j) {
        Operation *dstOp = loadAndStoreOps[j];
        MemRefAccess dstAccess(dstOp);

        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, /*dependenceConstraints=*/nullptr,
            &depComps);

        // Skip if there is no dependence in this case.
        if (!hasDependence(result))
          continue;

        // Check whether there is any negative direction vector in the
        // dependence components found above, which means that dependence is
        // violated by the default hyper-rect tiling method.
        LLVM_DEBUG(llvm::dbgs() << "Checking whether tiling legality violated "
                                   "for dependence at depth: "
                                << Twine(d) << " between:\n";);
        LLVM_DEBUG(srcAccess.opInst->dump(););
        LLVM_DEBUG(dstAccess.opInst->dump(););
        for (const DependenceComponent &depComp : depComps) {
          if (depComp.lb.has_value() && depComp.ub.has_value() &&
              *depComp.lb < *depComp.ub && *depComp.ub < 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Dependence component lb = " << Twine(*depComp.lb)
                       << " ub = " << Twine(*depComp.ub)
                       << " is negative  at depth: " << Twine(d)
                       << " and thus violates the legality rule.\n");
            return false;
          }
        }
      }
    }
  }

  return true;
}
/* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this function was completely copy-and-pasted
 * from LoopTiling.cpp ^^^^^^^^^^^^^^^*/

void AdHocLoopTiling::runOnOperation() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(getOperation(), &bands);
  // std::vector<SmallVector<AffineForOp, 6>> tiledBands;

  // Tile each band.
  for (auto &band : bands) {
    if (!checkTilingLegality(band)) {
      band.front().emitRemark("tiling code is illegal due to dependences");
      continue;
    }

    SmallVector<AffineForOp, 6> tiledNest;
    if (failed(AdHocLoopTiling::tilePerfectlyNested2(band, &tiledNest))) {
      // An empty band always succeeds.
      assert(!band.empty() && "guaranteed to succeed on empty bands");
      LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
      continue;
    }

    // // Separate full and partial tiles.
    // if (separate) {
    //   auto intraTileLoops =
    //       MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
    //   if (failed(separateFullTiles(intraTileLoops))) {
    //     assert(!intraTileLoops.empty() &&
    //            "guaranteed to succeed on empty bands");
    //     LLVM_DEBUG(intraTileLoops.front()->emitRemark(
    //         "separation post tiling failed!\n"));
    //   }
    // }
  }
  // Operation* op = getOperation();
  // ValueRange operands = op->getOperands();
} // end of runOnOperation

LogicalResult
AdHocLoopTiling::tilePerfectlyNested2(MutableArrayRef<AffineForOp> input,
                                      SmallVectorImpl<AffineForOp> *tiledNest) {
  if (input.empty())
    return success();

  // std::vector<std::vector<unsigned>> tileSizes;
  // tileSizes.push_back(std::vector<unsigned>());
  // for (auto &bound : ts.bounds) {
  //   unsigned tileSize = 0;
  //   // tileSizes.back().push_back(tileSize)
  // }

  MutableArrayRef<AffineForOp> origLoops = input;
  AffineForOp rootAffineForOp = origLoops[0];

  // We know exactly how many loops to create from the tiling scheme.
  unsigned width = ts.totalLoopCount;
  SmallVector<AffineForOp, 6> tiledLoops(width);

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] total loop count of transformed nest will be "
                          << width << " \n");

  // Construct a tiled loop nest without setting their bounds. Bounds are
  // set later.
  AdHocLoopTiling::constructPomegranateLoopNest(origLoops, rootAffineForOp,
                                                width, tiledLoops);

  // SmallVector<Value, 8> origLoopIVs;
  // extractForInductionVars(input, &origLoopIVs);

  // for (auto& elt : origLoopIVs){
  //   LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "<< elt<<" \n");

  // }

  // SmallVector<unsigned, 6> tileSizes;
  // tileSizes.assign(input.size(), 13);

  // // Set loop bounds for the tiled loop nest.
  // AdHocLoopTile::constructTiledIndexSetHyperRect(origLoops, tiledLoops,
  // tileSizes);

  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] after
  // constructTiledIndexSetHyperRect... \n");
  // // for(const auto& elt : tiledLoops){
  // //   LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] elt is "<< elt<<" \n");
  // // }
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] origLoops.back() has length
  // "<< origLoops.size()<<" and is "<< origLoops.back()<<" \n");

  // // Replace original IVs with intra-tile loop IVs.
  // for (unsigned i = 0; i < width; i++)
  //   origLoopIVs[i].replaceAllUsesWith(tiledLoops[i +
  //   width].getInductionVar());
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] AFTER REPLACING IV'S!!! \n");
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] tiledLoops[0] is "<<
  // tiledLoops[0]<<" \n"); LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]
  // tiledLoops[tiledLoops.size()-1] is "<< tiledLoops[tiledLoops.size()-1]<<"
  // \n"); LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] origLoops.back() has
  // length "<< origLoops.size()<<" and is "<< origLoops.back()<<" \n");
  // // Erase the old loop nest.
  // rootAffineForOp.erase();

  // if (tiledNest)
  //   *tiledNest = std::move(tiledLoops);

  return success();
}

// create a dummy loop nest around the original nested loop's body
void AdHocLoopTiling::constructPomegranateLoopNest(
    MutableArrayRef<AffineForOp> origLoops, AffineForOp rootAffineForOp,
    unsigned width, MutableArrayRef<AffineForOp> tiledLoops) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] inside pomegranate loop nest \n");
  Location loc = rootAffineForOp.getLoc();

  // // The outermost among the loops as we add more..
  Operation *topLoop = rootAffineForOp.getOperation();
  AffineForOp innermostPointLoop;

  std::vector<AffineForOp> subloops;
  std::vector<std::vector<struct AdHocLoopTile::LoopParams>> subloopsInfo;
  std::vector<struct AdHocLoopTile::LoopParams> finalSubloopsInfo;

  // CORN ON THE COB
  for (size_t i = 0; i < origLoops.size(); i++) { // for each original loop
    subloopsInfo.push_back(
        std::vector<struct AdHocLoopTile::LoopParams>()); // create a list for
                                                          // its subloops
    for (size_t j = 0; j <= ts.bounds[i].size();
         j++) { // for each of its subloops (which each have a bound)

      // declare some properties (to be initialized inside the if-else below)
      struct AdHocLoopTile::LoopParams info;
      uint64_t myGivenBound = 0;
      // TODO: remove code duplication!
      if (j == 0) { // first subloop ever made
        myGivenBound = ts.bounds[i][j];
        assert(origLoops[i].hasConstantLowerBound() &&
               "expected input loops to have constant lower bound.");
        info.parentTileSize = origLoops[i].getConstantUpperBound();
        info.parent = &origLoops[i];
        info.stepSize = info.parentTileSize / myGivenBound;
        subloopsInfo.back().push_back(info);
      } else if (j ==
                 ts.bounds[i]
                     .size()) { // this is the final subloop (no given bound)
        info.parent = 0;
        info.parentIndex = ts.finalIndices[i][j - 1];
        info.parentTileSize = subloopsInfo[i][j - 1].stepSize;
        info.stepSize = 1;
        finalSubloopsInfo.push_back(info);
      } else { // any other subloop
        myGivenBound = ts.bounds[i][j];
        info.parent = 0;
        info.parentIndex = ts.finalIndices[i][j - 1];
        info.parentTileSize = subloopsInfo[i][j - 1].stepSize;
        info.stepSize = info.parentTileSize / myGivenBound;
        subloopsInfo.back().push_back(info);
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] parentTileSize[" << info.parentTileSize
                 << "] myStepSize:[" << info.stepSize << "] \n");
    }
  }

  // create all the loops, ignoring their steps and bounds
  for (size_t i = 0; i < ts.totalLoopCount; i++) {
    OpBuilder b(topLoop);
    AffineForOp newLoop = b.create<AffineForOp>(loc, 0, i);
    newLoop.getBody()->getOperations().splice(
        newLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    topLoop = newLoop.getOperation();
    if (i == 0) {
      innermostPointLoop = newLoop;
    }
    subloops.push_back(newLoop);
  }

  for (size_t i = 0, j = 0; i < subloops.size(); i++) {
    OpBuilder b(subloops[i].getOperation());
    struct AdHocLoopTile::LoopParams info;
    AffineMap lb;
    AffineMap ub;
    // TODO: use ternary operator
    if (i > ts.order.size() - 1) { // reached the final group of subloops
      info = finalSubloopsInfo[j];
      j++;
    } else { // processing a subloop with an assigned bound
    LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] i is " << i << "] \n");
      info = subloopsInfo[ts.order[i][0]][ts.order[i][1]];
    }
    // set step size
    subloops[i].setStep(info.stepSize);
    // set bounds
    // if (!info.parent) {
    //   AffineExpr dim = b.getAffineDimExpr(0);
    //   lb = b.getDimIdentityMap();
    //   ub = AffineMap::get(1, 0, dim + info.parentTileSize);
    //   subloops[i].setLowerBound(subloops[info.parentIndex].getInductionVar(),
    //                             lb);
    //   subloops[i].setUpperBound(subloops[info.parentIndex].getInductionVar(),
    //                             ub);
    // } else {
    //   lb = info.parent->getLowerBoundMap();
    //   ub = info.parent->getUpperBoundMap();
    //   subloops[i].setLowerBound(info.parent->getLowerBoundOperands(), lb);
    //   subloops[i].setUpperBound(info.parent->getUpperBoundOperands(), ub);
    // }
  }

  for (size_t j = 0; j < subloops.size(); j++) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] hoodle " << subloops[j] << " \n");
  }

  // finally let's set the bounds,
  // 1) set bounds of the final subloops (the innermost loops)

  // let's try to manually create dependent bounds, to see if it doesn't
  // in fact throw error only threw an error because of the order of the
  // loops...
  // OpBuilder b(rootAffineForOp.getOperation());
  // AffineForOp *parent = 0;
  // uint64_t parentTileSize = 0;
  // AffineMap lbMap;
  // AffineMap ubMap;
  // parent = &subloops[0][1];
  // parentTileSize = subloops[0][1].getStepAsInt();

  // lbMap = b.getDimIdentityMap();
  // AffineExpr dim = b.getAffineDimExpr(0);
  // ubMap = AffineMap::get(1, 0, dim + parentTileSize);
  // subloops[0][0].setLowerBound(parent->getInductionVar(), lbMap);
  // subloops[0][0].setUpperBound(parent->getInductionVar(), ubMap);
  // super wacky but maybe it will work as we want! ^^^^^^

  // let's try to manually create dependent bounds, to see if it doesn't
  // in fact throw error only threw an error because of the order of the
  // loops...
  // OpBuilder b(rootAffineForOp.getOperation());
  // AffineForOp *parent = 0;
  // uint64_t parentTileSize = 0;
  // AffineMap lbMap;
  // AffineMap ubMap;
  // parent = &subloops[1][0];
  // parentTileSize = subloops[1][0].getStepAsInt();

  // lbMap = b.getDimIdentityMap();
  // AffineExpr dim = b.getAffineDimExpr(0);
  // ubMap = AffineMap::get(1, 0, dim + parentTileSize);
  // subloops[0][0].setLowerBound(parent->getInductionVar(), lbMap);
  // subloops[0][0].setUpperBound(parent->getInductionVar(), ubMap);
  // super wacky but maybe it will work as we want! ^^^^^^

  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
  //                            "] printing out my loops! subloops.size() is "
  //                         << subloops.size() << " \n");

  // for (size_t i = 0; i < subloops.size() ; i++) {
  //   for (size_t j = 0; j < subloops[i].size(); j++) {
  //     LLVM_DEBUG(llvm::dbgs()
  //                << "[" DEBUG_TYPE "] yodel " << subloops[i][j] << " \n");
  //   }
  // }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] now top Loop is " << *topLoop
                          << " \n");
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] about to return from dummy loop nest \n");
}

LogicalResult
AdHocLoopTiling::tilePerfectlyNested(MutableArrayRef<AffineForOp> input,
                                     SmallVectorImpl<AffineForOp> *tiledNest) {
  if (input.empty())
    return success();

  // std::vector<std::vector<unsigned>> tileSizes;
  // tileSizes.push_back(std::vector<unsigned>());
  // for (auto &bound : ts.bounds) {
  //   unsigned tileSize = 0;
  //   // tileSizes.back().push_back(tileSize)
  // }

  input.size();
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << "input size is  " << input.size() << " \n");

  for (size_t i = 0; i < input.size(); i++) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] i is " << i << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "input[" << i << "].getLowerBoundMap() is  "
                            << input[i].getLowerBoundMap() << " \n");
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "input[" << i << "].getUpperBoundMap() is  "
                            << input[i].getUpperBoundMap() << " \n");
  }
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] after for loop! \n");

  // if (failed(performPreTilingChecks(input, tileSizes)))
  //   return failure();

  MutableArrayRef<AffineForOp> origLoops = input;
  AffineForOp rootAffineForOp = origLoops[0];

  // // Note that width is at least one since the band isn't empty.
  unsigned width = input.size();
  SmallVector<AffineForOp, 6> tiledLoops(2 * width);

  // // Construct a tiled loop nest without setting their bounds. Bounds are
  // // set later.
  AdHocLoopTile::constructDummyLoopNest(origLoops, rootAffineForOp, width,
                                        tiledLoops);

  SmallVector<Value, 8> origLoopIVs;
  extractForInductionVars(input, &origLoopIVs);

  for (auto &elt : origLoopIVs) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << elt << " \n");
  }

  SmallVector<unsigned, 6> tileSizes;
  tileSizes.assign(input.size(), 13);

  // Set loop bounds for the tiled loop nest.
  AdHocLoopTile::constructTiledIndexSetHyperRect(origLoops, tiledLoops,
                                                 tileSizes);

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] after constructTiledIndexSetHyperRect... \n");
  // for(const auto& elt : tiledLoops){
  //   LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] elt is "<< elt<<" \n");
  // }
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] origLoops.back() has length "
                          << origLoops.size() << " and is " << origLoops.back()
                          << " \n");

  // Replace original IVs with intra-tile loop IVs.
  for (unsigned i = 0; i < width; i++)
    origLoopIVs[i].replaceAllUsesWith(tiledLoops[i + width].getInductionVar());
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] AFTER REPLACING IV'S!!! \n");
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] tiledLoops[0] is "
                          << tiledLoops[0] << " \n");
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] tiledLoops[tiledLoops.size()-1] is "
                          << tiledLoops[tiledLoops.size() - 1] << " \n");
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] origLoops.back() has length "
                          << origLoops.size() << " and is " << origLoops.back()
                          << " \n");
  // Erase the old loop nest.
  rootAffineForOp.erase();

  if (tiledNest)
    *tiledNest = std::move(tiledLoops);

  return success();
}

// static void lowerOpToLoops(Operation *op, ValueRange operands,
//                            PatternRewriter &rewriter,
//                            LoopIterationFn processIteration) {
//   auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
//   auto loc = op->getLoc();

//   // Insert an allocation and deallocation for the result of this operation.
//   auto memRefType = convertTensorToMemRef(tensorType);
//   auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

//   // Create a nest of affine loops, with one loop per dimension of the shape.
//   // The buildAffineLoopNest function takes a callback that is used to
//   construct
//   // the body of the innermost loop given a builder, a location and a range
//   of
//   // loop induction variables.
//   SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
//   SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
//   affine::buildAffineLoopNest(
//       rewriter, loc, lowerBounds, tensorType.getShape(), steps,
//       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//         // Call the processing function with the rewriter, the memref
//         operands,
//         // and the loop induction variables. This function will return the
//         value
//         // to store at the current index.
//         Value valueToStore = processIteration(nestedBuilder, operands, ivs);
//         nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
//                                                     ivs);
//       });

//   // Replace this operation with the generated alloc.
//   rewriter.replaceOp(op, alloc);
// }

// helpers for processing tiling scheme input
void AdHocLoopTiling::parseListOfListOfInts(
    llvm::json::Object *obj, std::string listName,
    std::vector<std::vector<int>> &out) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) { // getAsArray returns a (const json::Array *)
    llvm::errs() << "Error: field labeled '" << listName
                 << "' does not exist \n ";
    exit(1);
  }

  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    llvm::errs() << "Error: field labeled '" << listName
                 << "' is not a JSON array \n ";
    exit(1);
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  for (const auto &Item :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!Item.getAsArray()) {
      llvm::errs() << "Error: elt of '" << listName
                   << "' is not also a JSON array \n ";
      exit(1);
    }
    std::vector<int> sublist;
    int bound;
    for (const auto &elt :
         *(Item.getAsArray())) { // loop over a json::Array type
      if (!fromJSON(elt, bound, Root)) {
        llvm::errs() << llvm::toString(Root.getError()) << "\n";
        Root.printErrorContext(elt, llvm::errs());
        exit(1);
      }
      sublist.push_back(bound);
    }
    out.push_back(sublist);
  }
}

void AdHocLoopTiling::parseTilingScheme(StringRef fileContent) {
  llvm::Expected<llvm::json::Value> maybeParsed =
      llvm::json::parse(fileContent);
  if (!maybeParsed) {
    llvm::errs() << "Error when parsing JSON file contents: "
                 << llvm::toString(maybeParsed.takeError());
    exit(1);
  }
  // try to get the top level json object
  if (!maybeParsed->getAsObject()) {
    llvm::errs() << "Error: top-level value is not a JSON object: " << '\n';
    exit(1);
  }
  llvm::json::Object *O = maybeParsed->getAsObject();
  // try to read the two fields
  parseListOfListOfInts(O, "bounds", ts.bounds);
  parseListOfListOfInts(O, "order", ts.order);
}

LogicalResult AdHocLoopTiling::initializeOptions(StringRef options) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << "the options are  [ " << options << " ]\n");
  // try to extract file name from the options
  if (options.consume_front(StringRef("tiling-scheme="))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] "
               << "the filename is  [ " << options.data() << " ]\n");
    // try to read file
    std::ifstream ifs(options.data());
    std::stringstream ss;
    ss << ifs.rdbuf();
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "file contains... [ " << ss.str() << " ]\n");
    //  try to parse file contents
    parseTilingScheme(StringRef(ss.str()));
    std::stringstream ts_ss;
    ts_ss << ts;
    // print out what we parsed
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "the tile scheme we parsed from the json is...\n"
                            << ts_ss.str() << "\n");
    ts.setTotalLoopCount();
    ts.buildFinalIndices();
    return success();

  } else {
    return failure();
  }
}

// create loops in the correct order from the get-go
// iterate backwards (from innermost loop first) thru the tile scheme's order
// for(size_t i = ts.order.size()-1; i--;){
//   // we assume order is really a list of "tuples"

//   size_t origIndex = ts.order[i][0];
//   size_t subloopIndex = ts.order[i][1];
//   size_t tiledIndex = origIndex + tiledIndex;
//   unsigned bound = ts.bounds[origIndex][subloopIndex];

// }

// SmallVector<std::vector<AffineForOp>,6> subloops;
// subloops.assign(std::vector<AffineForOp>(), origLoops.size());