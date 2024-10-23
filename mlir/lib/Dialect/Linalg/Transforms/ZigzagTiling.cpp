//===- ZigzagTiling.cpp - dummy hello world pass for mlir-opt ------===//
//
// I based this pass on the file
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/JSON.h" // to parse tiling scheme
#include <fstream>             // to open tiling scheme file
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string> // for string compare

namespace mlir {
#define GEN_PASS_DEF_ZIGZAGTILING
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::transform;

#define DEBUG_TYPE "zigzag-tile"

namespace {
class ZigzagTiling : public impl::ZigzagTilingBase<ZigzagTiling> {
public:
  ZigzagTiling() = default;

private:
  SmallVector<OpFoldResult>
  ZigZagTileSizeComputation(OpBuilder &builder, Operation *operation,
                            ArrayRef<ArrayRef<int64_t>> tileSizes);

  // my own hacky exploration
  LogicalResult
  tileAndFuseEach(RewriterBase &rewriter,
                  llvm::SmallDenseSet<TilingInterface> &payloadOps,
                  int tilingLevel);

  void runOnOperation() override;
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
  friend std::stringstream &operator<<(std::stringstream &ss,
                                       const ZigzagTiling::TilingScheme &ts) {
    ss << "tiling scheme: {\nbounds: [ ";
    for (const auto &sublist : ts.bounds) {
      ss << "[ ";
      for (const auto &bound : sublist) {
        ss << " " << bound << " ";
      }
      ss << "] ";
    }
    ss << "]\n";
    ss << "finalIndices: [ ";
    for (const auto &sublist : ts.finalIndices) {
      ss << "[ ";
      for (const auto &pos : sublist) {
        ss << " " << pos << " ";
      }
      ss << "] ";
    }
    ss << "]\n}";
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

std::unique_ptr<Pass> mlir::createZigzagTilingPass() {
  return std::make_unique<ZigzagTiling>();
}

void ZigzagTiling::runOnOperation() {
  llvm::SmallDenseSet<TilingInterface> targetOps;
  // We know the operation implements a function op
  // interface because we defined this pass as an interface pass on the
  // FunctionOpInterface
  FunctionOpInterface funcOp = getOperation();
  // Pick out all the operations inside the current function
  // which implement a TilingInterface (linalg ops), and save them in a list.
  funcOp->walk([&](TilingInterface target) { targetOps.insert(target); });
  auto *context = &getContext(); // whose context? the function's context?
  // declare a pattern rewriter (Based on LinalgTransformOps tryApply function)
  struct TrivialPatternRewriter : public PatternRewriter {
  public:
    explicit TrivialPatternRewriter(MLIRContext *context)
        : PatternRewriter(context) {}
  };

  if (targetOps.size() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                               "] No Target Ops found inside this function!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Target Ops now has size "
                            << targetOps.size() << "\n");
    for (const auto &op : targetOps) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] This target Op is " << op << "\n");
    }
    // create an instance of our derived struct Pattern Rewriter.
    TrivialPatternRewriter rewriter(context);
    // Tile each Linalg Operation using a ZigZag plan
    if (failed(ZigzagTiling::tileAndFuseEach(rewriter, targetOps, 87))) {
      return signalPassFailure();
    }
  }
}

/// This collects the set of operations to tile + fuse starting from the given
/// root |op| and walking up to its producers. Stops at operations given by
/// |exclude| which are expected to receive their own independent tiling for the
/// given level.
static llvm::SmallDenseSet<Operation *>
collectTiledAndFusedOps(Operation *op,
                        const llvm::SmallDenseSet<TilingInterface> &exclude) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || producers.contains(producer) ||
          exclude.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

// my hacky tiling investigation
LogicalResult
ZigzagTiling::tileAndFuseEach(RewriterBase &rewriter,
                              llvm::SmallDenseSet<TilingInterface> &payloadOps,
                              int tilingLevel) {
  if (tilingLevel == 87) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] inside MY tiling exploration func!\n");
  }

  for (TilingInterface tilingInterfaceOp : payloadOps) {

    // TODO: what does this block do? I need to find out.
    DominanceInfo dominanceInfo(tilingInterfaceOp);
    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    // repeat tiling of each loop until we are done

    rewriter.setInsertionPoint(tilingInterfaceOp);
    scf::SCFTilingOptions tilingOptions;
    OpBuilder b(tilingInterfaceOp);
    // first level of tiling
    ArrayRef<ArrayRef<int64_t>> tileSizes = {{8}, {8}, {26}};
    const auto &ts = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // second level of tiling
    tileSizes = {{0}, {0}, {13}};
    const auto &ts2 = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // interchange vector
    ArrayRef<int64_t> interchange = {2, 0, 1};
    // ArrayRef<int64_t> interchange = {2, 1, 0};
    // ArrayRef<int64_t> interchange = {2, 3, 1, 0}; // causes stack dump
    // do something different based on the tilingLevel parameter.
    switch (tilingLevel) {
    case 87:
      // SCFTilingOptions &setTileSizes(ArrayRef<OpFoldResult> ts);
      tilingOptions.setTileSizes(ts);
      // tilingOptions.setTileSizeComputationFunction(ZigzagTiling::ZigZagTileSizeComputation);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      // tilingOptions.setInterchange(interchange); // TODO: interchange
      tilingOptions.setInterchange(interchange);
      break;
    case 88:
      tilingOptions.setTileSizes(ts2);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      tilingOptions.setInterchange(interchange);
      // tilingOptions.setTileSizeComputationFunction(
      //     [&](OpBuilder &builder, auto &&...) {
      //       SmallVector<OpFoldResult> result;

      //       SmallVector<int64_t> l1Tiles(loweringConfig.getL1Tiles());
      //       for (int64_t value : l1Tiles)
      //         result.push_back(builder.getIndexAttr(value));

      //       size_t numLoops =
      //       tilingInterfaceOp.getLoopIteratorTypes().size(); while
      //       (result.size() < numLoops)
      //         result.push_back(builder.getIndexAttr(0));

      //       return result;
      //     });
      // tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      // tilingOptions.setInterchange(interchange); // TODO: interchange
      // tilingOptions.setInterchange(loweringConfig.getL1TilesInterchange());
      break;
    default:
      tilingOptions.setTileSizes(ts2);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      tilingOptions.setInterchange(interchange);
      break;
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    // TODO: what does this block of code even do? I have to find out.
    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand) {
          Operation *owner = originalProducer.getOwner();
          bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
          bool shouldFuse = false;
          if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
            shouldFuse = !payloadOps.contains(tilingOwner);
          }
          // Do not fuse destination operands.
          shouldFuse &= !isDestinationOperand;
          return std::make_tuple(shouldFuse, yieldProducerReplacement);
        };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    // perform the tiling
    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  tileAndFuseOptions);
    if (failed(tiledResults)) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] TILING FAILED\n");
      return failure();
    } else {
      // let's print out what the heck happened
      // LLVM_DEBUG(llvm::dbgs()
      //      << "[" DEBUG_TYPE "] Hopefully the tiled function is here: "<<
      //      tilingInterfaceOp<<"\n");
      // llvm::SetVector<Operation *> tiledAndFusedOps
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE
                    "] size of tiledAndFusedOps from tiledResults is  "
                 << tiledResults->tiledAndFusedOps.size()
                 << " and loops has size " << tiledResults->loops.size()
                 << "\n");
      for (const auto &op : tiledResults->tiledAndFusedOps) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] A generated op is: " << *op << "\n");
      }

      for (const auto &loop : tiledResults->loops) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] A generated op loop: " << loop << "\n");
      }
    }

    // TODO: what does this block really do?
    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
    // when we reach here, the entire linalg op should be tiled
  }
  return success();
}

SmallVector<OpFoldResult>
ZigzagTiling::ZigZagTileSizeComputation(OpBuilder &builder,
                                        Operation *operation,
                                        ArrayRef<ArrayRef<int64_t>> tileSizes) {
  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] Inside my zigzag tile size computation function :)\n");
  SmallVector<OpFoldResult> result;
  for (auto const &tiles : tileSizes) {
    result.push_back(builder.getIndexAttr(tiles[0]));
  }
  return result;
}

void ZigzagTiling::TilingScheme::setTotalLoopCount() {
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

void ZigzagTiling::TilingScheme::buildFinalIndices() {
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

int ZigzagTiling::TilingScheme::findSubloop(size_t i, size_t j) {
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

// helpers for processing tiling scheme input
void ZigzagTiling::parseListOfListOfInts(llvm::json::Object *obj,
                                         std::string listName,
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

void ZigzagTiling::parseTilingScheme(StringRef fileContent) {
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

LogicalResult ZigzagTiling::initializeOptions(StringRef options) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << "the options are  [ " << options << " ]\n");

  // std::stringstream filename;
  // filename << options;
  //&& options.consume_back(StringRef("})"))
  // try to extract file name from the options
  // if (options.consume_front(StringRef("tiling-scheme="))) {
  if (options.consume_front(StringRef("tiling-scheme="))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] "
               << "ate the front so now filename is  [ " << options.data()
               << " ] with length [ " << strlen(options.data()) << " ]\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] couldn't eat the front :(\n");
  }

  if (options.consume_back(StringRef("}))"))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] "
               << "ate the back so now filename is  [ " << options.data()
               << " ] with length [ " << strlen(options.data()) << " ]\n");
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] "
               << "couldn't eat the back. filename is  [ " << options.data()
               << " ] with length [ " << strlen(options.data()) << " ]\n");
  }

 // find_last_of(StringRef Chars, size_t From = npos) 
 // StringRef substr(size_t Start, size_t N = npos) 
 //StringRef filename = options.slice(0,options.find_last_of(StringRef("}")));
//  LLVM_DEBUG(llvm::dbgs()
//                << "[" DEBUG_TYPE "] "
//                << "hoodle " << filename.data() << "\n");

char * aCopy = (char*) malloc(options.size()+1);
for(size_t i = 0; i < options.size()+1; i++){
  aCopy[i] = 0;

}
for(size_t i = 0; i < options.size(); i++){
  LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] ."<< (int)options.data()[i]<<". ]\n");
               aCopy[i] = options.data()[i];

}
 LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] aCopy is.["<< aCopy<<"]\n");

/*
constexpr llvm::StringRef::StringRef 	( 	const char *  	data,
		size_t  	length 
	) 	
*/

// llvm::StringRef filename(options.data(), options.size()-3);// = options.substr(0,options.size()-1);
llvm::StringRef filename= options.substr(0, 23);// = options.substr(0,options.size()-1);

 LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] new StringRef is ."<< filename.data()<<". ] with length "<< filename.size()<<"\n");

  // try to read file
  std::ifstream ifs(aCopy);
  

  if (!ifs.is_open()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] "
               << "Error opening file: " << options.data() << "\n");

    // Check for specific error conditions
    if (ifs.bad()) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                              << "Fatal error: badbit is set."
                              << "\n");
    }

    if (ifs.fail()) {
      // Print a more detailed error message using
      // strerror
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                              << "Error details: " << strerror(errno) << "\n");
    }

    // Handle the error or exit the program
    return failure();
  }

  // assert(ifs.fail() != false && "Tiling Scheme file reading error.");
  std::stringstream ss;
  ss << ifs.rdbuf();
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << "file contains... [ " << ss.str()
                          << " ] with str length [ " << ss.str().length()
                          << " ]\n");
  assert(ss.str().length() != 0 &&
         "Tiling Scheme file cannot have content length of 0");
  free(aCopy);
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
}

// // from testTilingInterfaceTransformOps.cpp
// /// Apply a tile and fuse transformation to all payload ops and store both
// the
// /// tiled operation as well as the created tile loops.
// template <typename Range>
// static LogicalResult mlir::linalg::zigzag::applyTileAndFuseToAll(
//     RewriterBase &rewriter, Operation *transformOp, Range &&payloadOps,
//     unsigned numLoops, ArrayRef<OpFoldResult> tileSizes,
//     ArrayRef<int64_t> interchange, bool useForall,
//     transform::TransformResults &transformResults) {
//   SmallVector<Operation *> tiledOps;
//   SmallVector<SmallVector<Operation *>> loopOps(numLoops);

//   // for (Operation *target : payloadOps) {
//   //   auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
//   //   if (!tilingInterfaceOp)
//   //     return transformOp->emitError("only TilingInterface ops are
//   //     supported");
//   //   DominanceInfo dominanceInfo(tilingInterfaceOp);

//   //   llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
//   //       collectTiledAndFusedOps(tilingInterfaceOp);
//   //   llvm::DenseSet<Operation *> yieldReplacementsFor;
//   //   for (auto op : tiledAndFusedOps) {
//   //     if (llvm::any_of(op->getUsers(), [&](Operation *user) {
//   //           return dominanceInfo.properlyDominates(tilingInterfaceOp,
//   user);
//   //         })) {
//   //       yieldReplacementsFor.insert(op);
//   //     }
//   //   }

//   //   scf::SCFTilingOptions tilingOptions;
//   //   tilingOptions.setTileSizes(tileSizes).setInterchange(interchange);
//   //   if (useForall) {
//   // tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
//   //   }

//   //   scf::SCFTileAndFuseOptions tileAndFuseOptions;
//   //   tileAndFuseOptions.setTilingOptions(tilingOptions);

//   //   scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
//   //       [&](tensor::ExtractSliceOp candidateSliceOp, OpResult
//   //       originalProducer,
//   //           bool isDestinationOperand) {
//   //         Operation *owner = originalProducer.getOwner();
//   //         bool yieldProducerReplacement =
//   //         yieldReplacementsFor.contains(owner); return
//   std::make_tuple(true,
//   //         yieldProducerReplacement);
//   //       };
//   //   tileAndFuseOptions.setFusionControlFn(controlFn);

//   //   rewriter.setInsertionPoint(target);
//   //   FailureOr<scf::SCFTileAndFuseResult> tiledResults =
//   //       scf::tileConsumerAndFuseProducersUsingSCF(rewriter,
//   //       tilingInterfaceOp,
//   //                                                 tileAndFuseOptions);
//   //   if (failed(tiledResults))
//   //     return failure();

//   //   // Perform the replacement of tiled and fused values.
//   //   SmallVector<Operation *> opsToReplace{target};
//   //   llvm::append_range(opsToReplace, tiledResults->fusedProducers);
//   //   for (Operation *toReplace : opsToReplace) {
//   //     for (OpResult res : toReplace->getResults())
//   //       if (auto replacement = tiledResults->replacements.lookup(res)) {
//   //         Operation *replacementOp = replacement.getDefiningOp();
//   //         rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use)
//   {
//   //           Operation *user = use.getOwner();
//   //           return dominanceInfo.properlyDominates(replacementOp, user) &&
//   //                  user->getParentOp() == replacementOp->getParentOp();
//   //         });
//   //       }

//   //     if (toReplace->use_empty()) {
//   //       rewriter.eraseOp(toReplace);
//   //     }
//   //   }

//   //   // Report back the relevant handles to the transform op.
//   //   tiledOps.push_back(tiledResults->tiledAndFusedOps.front());
//   //   assert(tiledResults->loops.size() == numLoops &&
//   //          "Mismatched number of loops, tile and fuse transform should
//   have "
//   //          "failed");
//   //   for (unsigned int i = 0; i < numLoops; ++i)
//   //     loopOps[i].push_back(tiledResults->loops[i]);
//   // }

//   // transformResults.set(transformOp->getOpResult(0), tiledOps);
//   // for (unsigned int i = 0; i < numLoops; ++i)
//   //   transformResults.set(transformOp->getOpResult(i + 1), loopOps[i]);

//   return success();
// }

// // from quidditch
// /// Apply a tile and fuse transformation to all payload ops and store both
// the
// /// tiled operation as well as the created tile loops.
// LogicalResult static mlir::linalg::zigzag::applyTileAndFuseToEachRoot(
//     RewriterBase &rewriter, llvm::SmallDenseSet<TilingInterface> &payloadOps,
//     int tilingLevel) {
//   // for (TilingInterface tilingInterfaceOp : payloadOps) {

//   //   DominanceInfo dominanceInfo(tilingInterfaceOp);

//   //   llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
//   //       collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
//   //   DenseSet<Operation *> yieldReplacementsFor;
//   //   for (auto op : tiledAndFusedOps) {
//   //     if (llvm::any_of(op->getUsers(), [&](Operation *user) {
//   //           return dominanceInfo.properlyDominates(tilingInterfaceOp,
//   user);
//   //         })) {
//   //       yieldReplacementsFor.insert(op);
//   //     }
//   //   }

//   //     rewriter.setInsertionPoint(tilingInterfaceOp);
//   // // I think lowering config only holds tile info...
//   // //
//   //
//   https://github.com/opencompl/Quidditch/blob/15935bfe2cf454a929eed37f0450ed5c4c3036cf/codegen/compiler/src/Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.cpp#L7

//   //     auto loweringConfig =
//   //         getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(
//   //             tilingInterfaceOp);
//   //     scf::SCFTilingOptions tilingOptions;
//   //     switch (tilingLevel) {
//   //     case TilingLevel::Thread:
//   // tilingOptions.setTileSizeComputationFunction(threadTileSizeComputation);
//   // tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
//   //       break;
//   //     case TilingLevel::L1:
//   //       tilingOptions.setTileSizeComputationFunction(
//   //           [&](OpBuilder &builder, auto &&...) {
//   //             SmallVector<OpFoldResult> result;

//   //             SmallVector<int64_t> l1Tiles(loweringConfig.getL1Tiles());
//   //             for (int64_t value : l1Tiles)
//   //               result.push_back(builder.getIndexAttr(value));

//   //             size_t numLoops =
//   //             tilingInterfaceOp.getLoopIteratorTypes().size(); while
//   //             (result.size() < numLoops)
//   //               result.push_back(builder.getIndexAttr(0));

//   //             return result;
//   //           });
//   //       tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
//   // tilingOptions.setInterchange(loweringConfig.getL1TilesInterchange());
//   //       break;
//   //     }

//   //     scf::SCFTileAndFuseOptions tileAndFuseOptions;
//   //     tileAndFuseOptions.setTilingOptions(tilingOptions);

//   //     scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
//   //         [&](tensor::ExtractSliceOp candidateSliceOp, OpResult
//   //         originalProducer,
//   //             bool isDestinationOperand) {
//   //           Operation *owner = originalProducer.getOwner();
//   //           bool yieldProducerReplacement =
//   //           yieldReplacementsFor.contains(owner); bool shouldFuse = false;
//   if
//   //           (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
//   //             shouldFuse = !payloadOps.contains(tilingOwner);
//   //           }
//   //           // Do not fuse destination operands.
//   //           shouldFuse &= !isDestinationOperand;
//   //           return std::make_tuple(shouldFuse, yieldProducerReplacement);
//   //         };
//   //     tileAndFuseOptions.setFusionControlFn(controlFn);

//   //     FailureOr<scf::SCFTileAndFuseResult> tiledResults =
//   //         scf::tileConsumerAndFuseProducersUsingSCF(rewriter,
//   //         tilingInterfaceOp,
//   //                                                   tileAndFuseOptions);
//   //     if (failed(tiledResults)) {
//   //       return failure();
//   //     }

//   //     // Perform the replacement of tiled and fused values.
//   //     SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
//   //     llvm::append_range(opsToReplace, tiledResults->fusedProducers);
//   //     for (Operation *toReplace : opsToReplace) {
//   //       for (OpResult res : toReplace->getResults())
//   //         if (auto replacement = tiledResults->replacements.lookup(res)) {
//   //           Operation *replacementOp = replacement.getDefiningOp();
//   //           rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand
//   &use)
//   //           {
//   //             Operation *user = use.getOwner();
//   //             return dominanceInfo.properlyDominates(replacementOp, user);
//   //           });
//   //         }

//   //       if (toReplace->use_empty()) {
//   //         rewriter.eraseOp(toReplace);
//   //       }
//   //     }
//   //   }
//   return success();
// }