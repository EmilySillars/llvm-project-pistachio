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
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string> // for string compare

namespace mlir {
#define GEN_PASS_DEF_ZIGZAGTILING
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "zigzag-tile"

namespace {
struct ZigzagTiling : public impl::ZigzagTilingBase<ZigzagTiling> {
  ZigzagTiling() = default;

  // bool canScheduleOn(RegisteredOperationName opInfo) const override {
  //   return opInfo.hasInterface<FunctionOpInterface>();
  // }

  void runOnOperation() override {
    // auto *func = getOperation();
    // auto *context = &getContext();
    // ConversionTarget target(*context);
    // RewritePatternSet patterns(context);
    // llvm::errs() << "["<< func->getAttr("sym_name") <<"]"<<"\n";
     LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE "] ZigZag!!\n");
    
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createZigzagTilingPass() {
  return std::make_unique<ZigzagTiling>();
}

// std::unique_ptr<OperationPass<func::FuncOp>>
// mlir::createZigzagTilingPass() {
//   return std::make_unique<ZigzagTiling>();
// }

// std::unique_ptr<OperationPass<func::FuncOp>>
// mlir::affine::createAdHocLoopTilingPass() {
//   return std::make_unique<AdHocLoopTiling>();
// }

// std::unique_ptr<Pass> mlir::createLinalgElementwiseOpFusionPass() {
//   return std::make_unique<LinalgElementwiseOpFusionPass>();
// }
