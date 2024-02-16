//===---- Learning.cpp - Based on example pass Hello.cpp -----------------===//
// This pass visits every local function in the source program,
// and then prints out the load instructions each store instruction aliases
// with.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Learning.h"

using namespace llvm;

/// run - For each function in the source program, for each store instruction,
/// find all the load instructions which alias with it, and print them out.
PreservedAnalyses LearningPass::run(Function &F, FunctionAnalysisManager &AM) {

  // // retrieve alias analysis information
  // AAResults &AA = AM.getResult<AAManager>(F);

  // // initialize alias set tracker
  // BatchAAResults BatchAA(AA);
  // AliasSetTracker AST(BatchAA);

  // // keep track of store and load instructions
  // for (auto &B : F) {
  //   for (auto &i : B) {
  //     AST.add(&i); // give every instruction to the alias set tracker
  //     if (i.getOpcode() == Instruction::Store) {
  //       addStore(&i); // add this store to aliases map
  //     }
  //     if (i.getOpcode() == Instruction::Load) {
  //       allLoads.push_back(&i); // and this load to the list of all loads
  //     }
  //   }
  // }

  // // populate a map with (instruction*, AliasInfo) key-value pairs
  // // a key is a store instruction, and a value is its aliasing info
  // for (auto &B : F) {
  //   for (auto &i : B) {
  //     if (i.getOpcode() == Instruction::Store) {
  //       checkAgainstAllLoads(AA, i);
  //     }
  //   }
  // }

  // state function name
  errs() << "Function " << F.getName() << ":\n\n";

  // // output `aliases` relationships between stores and loads
  // printStoreLoadInfo();

  // // print out alias set tracker for reality check
  // AST.print(errs());

  // errs() << "\n";

  // // free resources before returning
  // aliases.clear();
  // allLoads.clear();
  // AST.clear();

  // all analyses are still valid after this pass runs
  return PreservedAnalyses::all();
}

/// checkAgainstAllLoads - Check whether this store instruction aliases with any
/// load, then add this resulting relationship to the aliases map.
bool LearningPass::checkAgainstAllLoads(AAResults &aa, Instruction &store) {
  bool result = false;
  for (Instruction *load : allLoads) {
    // does the store alias with this load?
    AliasResult res =
        aa.alias(MemoryLocation::get(&store), MemoryLocation::get(load));
    // record result
    recordLoad(&store, load, res.operator llvm::AliasResult::Kind());
    result = res != llvm::AliasResult::Kind::NoAlias;
  }
  return result;
}

//===----------------------------------------------------------------------===//
//                        Alias Map Helper Functions
//===----------------------------------------------------------------------===//

/// addStore - add store instruction (a key) to aliases map
void LearningPass::addStore(Instruction *s) {
  std::map<Instruction *, AliasInfo>::iterator it = aliases.find(s);
  // don't add store again if already present!
  if (it == aliases.end()) {
    auto store = std::pair<Instruction *, AliasInfo>(s, AliasInfo());
    aliases.insert(store);
  }
}

/// recordLoad - add load instruction to aliases map and tally the kind of alias
/// (If load's kind is NoAlias, don't add to map, but still increment tally)
void LearningPass::recordLoad(Instruction *s, Instruction *l,
                               llvm::AliasResult::Kind kind) {
  // assume store already present in map
  std::map<Instruction *, AliasInfo>::iterator it = aliases.find(s);
  // pull out the loads this store aliases with
  std::map<Instruction *, llvm::AliasResult::Kind> &lds = it->second.loads;
  // don't add load if already present!
  if (lds.find(l) == lds.end()) {
    // record the kind of aliasing
    it->second.kindCount[kind] = it->second.kindCount[kind] + 1;
    // add the load
    if (kind != llvm::AliasResult::Kind::NoAlias) {
      auto load = std::pair<Instruction *, llvm::AliasResult::Kind>(l, kind);
      lds.insert(load);
    }
  }
}

//===----------------------------------------------------------------------===//
//                        Printing Helper Functions
//===----------------------------------------------------------------------===//

/// printStoreLoadInfo - Print out store-load alias relationships
/// for the function, including percentages of query results.
void LearningPass::printStoreLoadInfo() {
  double mustAlias = 0.0;
  double mayAlias = 0.0;
  double partialAlias = 0.0;
  double noAlias = 0.0;
  double totalQueries = 0.0;

  // for each store instruction, print its alias info
  for (auto pair : aliases) {
    errs() << *pair.first;
    if (pair.second.loads.empty()) {
      errs() << " DOES NOT ALIAS WITH ANY LOADS\n";
    } else {
      errs() << " ALIASES WITH...\n";
      printAliasInfo(pair.second);
    }
    errs() << "\n";
    // calculate statistics
    double must =
        (double)pair.second.kindCount[llvm::AliasResult::Kind::MustAlias];
    double may =
        (double)pair.second.kindCount[llvm::AliasResult::Kind::MayAlias];
    double part =
        (double)pair.second.kindCount[llvm::AliasResult::Kind::PartialAlias];
    double no = (double)pair.second.kindCount[llvm::AliasResult::Kind::NoAlias];
    double total = must + may + part + no;
    mustAlias += must;
    mayAlias += may;
    partialAlias += part;
    noAlias += no;
    totalQueries += total;
  }

  // print function statistics
  if (totalQueries > 0.0) {
    errs() << "Total Alias Queries: " << format("%0.2f", totalQueries) << "\n";
    errs() << "MustAlias: "
           << format("%0.2f", (mustAlias / totalQueries) * 100.0)
           << "% May Alias: "
           << format("%0.2f", (mayAlias / totalQueries) * 100.0)
           << "% Partial Alias: "
           << format("%0.2f", (partialAlias / totalQueries) * 100.0)
           << "% NoAlias: " << format("%0.2f", (noAlias / totalQueries) * 100.0)
           << "%\n\n";
  }
}

/// printAliasInfo - Helper for printStoreLoadInfo.
void LearningPass::printAliasInfo(AliasInfo info) {
  for (auto pair : info.loads) {
    errs() << "\t" << *pair.first << " ( ";
    switch (pair.second) {
    case llvm::AliasResult::Kind::MustAlias:
      errs() << "MustAlias";
      break;
    case llvm::AliasResult::Kind::MayAlias:
      errs() << "MayAlias";
      break;
    case llvm::AliasResult::Kind::PartialAlias:
      errs() << "PartialAlias";
      break;
    default:
      errs() << "ERROR! NO ALIAS!!!"; // should never happen
      break;
    }
    errs() << " )\n";
  }
}