//===-- Learning.h - Based on example file Hello.h ------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LEARNING_H
#define LLVM_TRANSFORMS_UTILS_LEARNING_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include <map>    // TODO: relace with an LLVM data structure
#include <set>    // TODO: relace with an LLVM data structure
#include <vector> // TODO: relace with an LLVM data structure

namespace llvm {

class LearningPass : public PassInfoMixin<LearningPass> {
  /// AliasInfo - Contains the load instructions the store instruction aliases
  /// with, as well statistics for queries involving the store.
  typedef struct AliasInfo {
    // what kind of relationship did each load have with this store?
    // NoAlias = 0 , MayAlias = 1 , PartialAlias = 2 ,  MustAlias = 3
    int kindCount[4] = {0, 0, 0, 0};
    // all the loads that do (or might) alias with this store
    std::map<Instruction *, llvm::AliasResult::Kind> loads;
    // constructor for AliasInfo
    AliasInfo() : loads(std::map<Instruction *, llvm::AliasResult::Kind>()) {}
  } AliasInfo;
  /// aliases - maps store instructions to their alias information
  std::map<Instruction *, struct AliasInfo> aliases;
  /// allLoads - all the load instructions in the current function
  std::vector<Instruction *> allLoads;
  bool checkAgainstAllLoads(AAResults &aa, Instruction &store);
  void addStore(Instruction *s);
  void recordLoad(Instruction *s, Instruction *l, llvm::AliasResult::Kind kind);
  void printAliasInfo(AliasInfo info);
  void printStoreLoadInfo();

public:
  LearningPass()
      : aliases(std::map<Instruction *, struct AliasInfo>()),
        allLoads(std::vector<Instruction *>()) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LEARNING_H