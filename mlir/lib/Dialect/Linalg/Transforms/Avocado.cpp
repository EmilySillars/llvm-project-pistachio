//===- Avocado.cpp - dummy hello world pass for mlir-opt ------===//
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
#define GEN_PASS_DEF_AVOCADO
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
class AvocadoPass : public impl::AvocadoBase<AvocadoPass> {

  bool canScheduleOn(RegisteredOperationName opInfo) const override {
    return opInfo.hasInterface<FunctionOpInterface>();
  }

  void runOnOperation() final {
    auto *func = getOperation();
    // auto *context = &getContext();
    // ConversionTarget target(*context);
    // RewritePatternSet patterns(context);
    llvm::errs() << "["<< func->getAttr("sym_name") <<"]"<<"\n";
    ++funcCount;
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createAvocadoPass() {
  return std::make_unique<AvocadoPass>();
}
