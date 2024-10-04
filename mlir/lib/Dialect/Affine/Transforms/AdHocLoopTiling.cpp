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
  void parseTilingScheme();
  LogicalResult initializeOptions(StringRef options) override;
};

} // namespace

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAdHocLoopTilingPass() {
  auto thePass = std::make_unique<AdHocLoopTiling>();
  thePass->parseTilingScheme();
  return thePass;
}

void AdHocLoopTiling::parseTilingScheme() {
  // we only want to read the requested tiling scheme once
  // std::ifstream ifs(this->tilingScheme);
  // std::stringstream ss;
  // ss << ifs.rdbuf();

  // // auto strRef = StringRef(this->tilingScheme);
  // auto strRef = this->getArgument();
  // // Json::Value people;
  // // people_file >> people;
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
  //                         << "inside the function `parseTilingScheme`\n");
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
  //                         << "file name is  [ " << strRef << " ]\n");
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
  //                         << "file contains... [ " << ss.str() << " ]\n");
}

LogicalResult AdHocLoopTiling::initializeOptions(StringRef options) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << "the options are  [ " << options << " ]\n");
  // bool 	consume_front (StringRef Prefix)
  // Returns true if this StringRef has the given prefix and removes that
  // prefix.
  if (options.consume_front(StringRef("tiling-scheme="))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] "
               << "the filename is  [ " << options.data() << " ]\n");
    std::ifstream ifs(options.data());
    std::stringstream ss;
    ss << ifs.rdbuf();
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "file contains... [ " << ss.str() << " ]\n");
    StringRef contents = StringRef(ss.str());
    // bool fromJSON(const Value &E, std::map<std::string, T> &Out, Path P) {
    // Expected<Value> E = json::parse("[1, 2, null]");
    ///   assert(E && E->kind() == Value::Array);
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << "1  ]\n");
    llvm::Expected<llvm::json::Value> E = llvm::json::parse(contents);
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << "2  ]\n");
    //EXPECT_TRUE(!!E);
    llvm::json::Object *O = E->getAsObject();
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << "3  ]\n");
    llvm::json::Value * val = O->get(StringRef("bounds"));
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << "4  ]\n");
    // assert(val && val->kind() == llvm::json::Value::Array);
    //Value *get(StringRef K)

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "contents are of type [ " << val->kind() << " ]\n");

    return success();

  } else {
    return failure();
  }
}

void AdHocLoopTiling::runOnOperation() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(getOperation(), &bands);

  //  tileSizes->assign(this->tileSizes.begin(), this->tileSizes.end());

  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "<< "Can i print out a
  // parameter passed in? It's..." << this->tilingScheme<<"
  // yodelayheeehooooo!!!\n");

  // std::ifstream tile_scheme_file(this->tilingScheme, std::ifstream::binary);
  // std::string fileContent;
  // tile_scheme_file >> fileContent;

  // std::ifstream ifs(this->tilingScheme);
  // std::stringstream ss;
  // ss << ifs.rdbuf();

  // auto strRef = StringRef(this->tilingScheme);
  // // Json::Value people;
  // // people_file >> people;
  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "<< "file name is  [ " <<
  // strRef <<" ]\n"); LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "<< "file
  // contains... [ " << ss.str() <<" ]\n");

  // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " <<
  // AdHocLoopTiling::kDefaultTileSize <<" yodelayheeehooooo\n");

  // Tile each band.
  // for (auto &band : bands) {
  //   if (!checkTilingLegality(band)) {
  //     band.front().emitRemark("tiling code is illegal due to dependences");
  //     continue;
  //   }

  //   // Set up tile sizes; fill missing tile sizes at the end with default
  //   tile
  //   // size or tileSize if one was provided.
  //   SmallVector<unsigned, 6> tileSizes;
  //   getTileSizes(band, &tileSizes);
  //   if (llvm::DebugFlag) {
  //     auto diag = band[0].emitRemark("using tile sizes [");
  //     for (unsigned tSize : tileSizes)
  //       diag << tSize << ' ';
  //     diag << "]\n";
  //   }
  //   SmallVector<AffineForOp, 6> tiledNest;
  //   if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest))) {
  //     // An empty band always succeeds.
  //     assert(!band.empty() && "guaranteed to succeed on empty bands");
  //     LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
  //     continue;
  //   }

  //   // Separate full and partial tiles.
  //   if (separate) {
  //     auto intraTileLoops =
  //         MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
  //     if (failed(separateFullTiles(intraTileLoops))) {
  //       assert(!intraTileLoops.empty() &&
  //              "guaranteed to succeed on empty bands");
  //       LLVM_DEBUG(intraTileLoops.front()->emitRemark(
  //           "separation post tiling failed!\n"));
  //     }
  //   }
  // }
}
