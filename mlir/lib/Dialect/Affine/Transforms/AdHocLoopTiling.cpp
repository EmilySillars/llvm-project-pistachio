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
    TilingScheme() = default;
    unsigned totalLoopCount();
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

unsigned AdHocLoopTiling::TilingScheme::totalLoopCount() {
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
  return total;
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
  unsigned width = ts.totalLoopCount();
  SmallVector<AffineForOp, 6> tiledLoops(width);

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
  std::vector<std::vector<AffineForOp>> subloops;

  // let's create the tiled loops level by level, and then at the end, stitch
  // them together!
  for (size_t i = 0; i < origLoops.size(); i++) { // for each original loop
    std::vector<AffineForOp> subs;                // create a list of subloops
    subloops.push_back(subs);
    for (size_t j = 0; j < ts.bounds[i].size();
         j++) { // for each of its subloops (which each have a bound)
      OpBuilder b(topLoop);
      // Loop bounds will be set NOW if possible!.
      AffineForOp pointLoop = b.create<AffineForOp>(loc, 0, ts.bounds[i][j]);

      topLoop = pointLoop.getOperation();
      if (i == 0)
        innermostPointLoop = pointLoop;
      subs.push_back(pointLoop);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] printing out my loops!  \n");

  for (size_t i = 0; i < subloops.size(); i++) {
    for (size_t j = 0; j < subloops[i].size(); j++) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] hoodle " << subloops[i][j] << " \n");
    }
  }

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
    return success();

  } else {
    return failure();
  }
}
