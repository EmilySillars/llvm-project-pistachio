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
  // bool canScheduleOn(RegisteredOperationName opInfo) const override {
  //   return opInfo.hasInterface<FunctionOpInterface>();
  // }

  SmallVector<OpFoldResult>
  ZigZagTileSizeComputation(OpBuilder &builder, Operation *operation,
                            ArrayRef<ArrayRef<int64_t>> tileSizes);

  // my own hacky exploration
  LogicalResult
  tileAndFuseEach(RewriterBase &rewriter,
                  llvm::SmallDenseSet<TilingInterface> &payloadOps,
                  int tilingLevel);

  void runOnOperation() override;
};
} // namespace

std::unique_ptr<Pass> mlir::createZigzagTilingPass() {
  return std::make_unique<ZigzagTiling>();
}

void ZigzagTiling::runOnOperation() {
  llvm::SmallDenseSet<TilingInterface> targetOps;
  FunctionOpInterface funcOp =
      getOperation(); // I know the operation implements a function op
                      // interface
  funcOp->walk([&](TilingInterface target) { targetOps.insert(target); });
  auto *context = &getContext();
  // // ConversionTarget target(*context);
  // PatternRewriter rewriter(context);
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
    TrivialPatternRewriter rewriter(context);
    // rewriter.setInsertionPoint(funcOp); // I think I don't need this because
    // insertion point is set inside tileAndFuseEach
    if (failed(ZigzagTiling::tileAndFuseEach(rewriter, targetOps, 87))) {
      return signalPassFailure();
    }
  }
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

    // TODO: uncomment dominance info and make it work.

    // DominanceInfo dominanceInfo(tilingInterfaceOp);

    // llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
    //     collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    // DenseSet<Operation *> yieldReplacementsFor;
    // for (auto op : tiledAndFusedOps) {
    //   if (llvm::any_of(op->getUsers(), [&](Operation *user) {
    //         return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
    //       })) {
    //     yieldReplacementsFor.insert(op);
    //   }
    // }

    rewriter.setInsertionPoint(tilingInterfaceOp);
    // // I think lowering config only holds tile info...
    // //
    // https://github.com/opencompl/Quidditch/blob/15935bfe2cf454a929eed37f0450ed5c4c3036cf/codegen/compiler/src/Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.cpp#L7

    // auto loweringConfig =
    //     getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(
    //         tilingInterfaceOp);
    scf::SCFTilingOptions tilingOptions;
    OpBuilder b(tilingInterfaceOp);
    ArrayRef<ArrayRef<int64_t>> tileSizes;
    ArrayRef<int64_t> interchange;
    const auto &ts =
        ZigzagTiling::ZigZagTileSizeComputation(b, tilingInterfaceOp, tileSizes);
    switch (tilingLevel) {
    case 87:
      // SCFTilingOptions &setTileSizes(ArrayRef<OpFoldResult> ts);
      tilingOptions.setTileSizes(ts);
      // tilingOptions.setTileSizeComputationFunction(ZigzagTiling::ZigZagTileSizeComputation);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      tilingOptions.setInterchange(interchange);
      break;
    default:
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
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      tilingOptions.setInterchange(interchange);
      // tilingOptions.setInterchange(loweringConfig.getL1TilesInterchange());
      break;
    }

    //     scf::SCFTileAndFuseOptions tileAndFuseOptions;
    //     tileAndFuseOptions.setTilingOptions(tilingOptions);

    //     scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
    //         [&](tensor::ExtractSliceOp candidateSliceOp, OpResult
    //         originalProducer,
    //             bool isDestinationOperand) {
    //           Operation *owner = originalProducer.getOwner();
    //           bool yieldProducerReplacement =
    //           yieldReplacementsFor.contains(owner); bool shouldFuse = false;
    //           if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
    //             shouldFuse = !payloadOps.contains(tilingOwner);
    //           }
    //           // Do not fuse destination operands.
    //           shouldFuse &= !isDestinationOperand;
    //           return std::make_tuple(shouldFuse, yieldProducerReplacement);
    //         };
    //     tileAndFuseOptions.setFusionControlFn(controlFn);

    //     FailureOr<scf::SCFTileAndFuseResult> tiledResults =
    //         scf::tileConsumerAndFuseProducersUsingSCF(rewriter,
    //         tilingInterfaceOp,
    //                                                   tileAndFuseOptions);
    //     if (failed(tiledResults)) {
    //       return failure();
    //     }

    //     // Perform the replacement of tiled and fused values.
    //     SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    //     llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    //     for (Operation *toReplace : opsToReplace) {
    //       for (OpResult res : toReplace->getResults())
    //         if (auto replacement = tiledResults->replacements.lookup(res)) {
    //           Operation *replacementOp = replacement.getDefiningOp();
    //           rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand
    //           &use)
    //           {
    //             Operation *user = use.getOwner();
    //             return dominanceInfo.properlyDominates(replacementOp, user);
    //           });
    //         }

    //       if (toReplace->use_empty()) {
    //         rewriter.eraseOp(toReplace);
    //       }
    //     }
  }
  return success();
}

SmallVector<OpFoldResult>
ZigzagTiling::ZigZagTileSizeComputation(OpBuilder &builder,
                                        Operation *operation,
                                        ArrayRef<ArrayRef<int64_t>> tileSizes) {
  SmallVector<OpFoldResult> result;
  return result;
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