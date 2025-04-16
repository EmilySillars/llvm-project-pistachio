
# PAAM Resources: Add a Custom Pass to mlir-opt
[[Return to Homepage]](../../paam-resources/README.md)

*Based on: https://mlir.llvm.org/docs/PassManagement/*

### Overview

Adding a custom compiler pass to MLIR involves two directories, `include` and `lib`. Both of these are sub-categorized by MLIR dialect.

1. `mlir/include/` <-- registration of the pass goes in here
   - modify `mlir/Dialect/<dialect-name>/Passes.h`
   - modify `mlir/Dialect/<dialect-name>/Passes.td`
2. `mlir/lib/` <-- implementation of the pass goes in here
   - modify `Dialect/<dialect-name>/Transforms/CMakeLists.txt`
   - add a `cpp` source file for your pass implementation inside `Dialect/<dialect-name>/Transforms/<custom-pass-name.cpp>`

#### Basic Steps to Add a Custom Pass

1. **Pick** the MLIR dialect level at which your pass will run.
2. **Find** an existing MLIR pass in that dialect to use as reference.
3. **Copy** changes needed to register/implement this reference pass in the corresponding `Passes.h` , `Passes.td` , and `cpp` file, replacing the name of the reference pass with your own custom pass name wherever necessary.
4. **Hack** the copied implementation `cpp` file to perform the transformation and/or analysis you desire. 

## Example Avocado Pass: count functions in MLIR source file

#### 1. Pick a dialect

We pick to run our pass at the linalg dialect level.

#### 2. Find a reference pass

We choose to use `--convert-elementwise-to-linalg` as our reference pass, which means our reference files are:

- `mlir/include/mlir/Dialect/Linalg/Passes.h`

- `mlir/include/mlir/Dialect/Linalg/Passes.td`

- `mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt`

#### 3. Copy changes, renaming with pass name as necessary

1. Inside `Passes.h`, search for instances of the string "ElementwiseToLinalg", and copy and replace with custom pass name.
   We find

   ```
   std::unique_ptr<Pass> createConvertElementwiseToLinalgPass();
   ```

   so we add
   ```
   std::unique_ptr<Pass> createAvocadoPass();
   ```

2. Inside `Passes.td` search for instances of the string "ElementwiseToLinalg"...

   We find
   ```
   def ConvertElementwiseToLinalg : Pass<"convert-elementwise-to-linalg", ""> {
     let summary = "Convert ElementwiseMappable ops to linalg";
     let description = [{
       Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.
   
       This pass only converts ops that operate on ranked tensors. It can be
       run on op which contains linalg ops (most commonly a
       FunctionOpInterface op).
     }];
     let constructor = "mlir::createConvertElementwiseToLinalgPass()";
     let dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"];
   }
   ```

   so we add a corresponding definition for our custom pass
   ```
   def Avocado : Pass<"avocado", ""> {
     let summary = "A 'hello world' pass at the linalg level";
     let description = [{
       Counts function names.
     }];
     let constructor = "mlir::createAvocadoPass()";
     let statistics = [
       Statistic<"funcCount", "func-count",
                 "how many functions are in the source file?">,
     ];
     let dependentDialects = ["linalg::LinalgDialect"];
   }
   ```
   
   ##### Q: *Wait, what are statistics? They're not present in the pass defintion for `ConvertElementwiseToLinalg`!*

   - [Statistics](https://mlir.llvm.org/docs/PassManagement/#pass-statistics) are one way to gather information as a pass runs, and then to output that information when the pass completes. 
   - Under [Declarative Pass Specification](https://mlir.llvm.org/docs/PassManagement/#declarative-pass-specification), you can see one example of how to declare statistics for your pass. 
   - **You can also looks for other MLIR passes that use statistics to figure out how to declare them**. By searching for "let statistics = [" in the mlir/include/mlir directory, I found an example pass using statistics called `OneShotBufferize` inside `mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td`

3. Inside `mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt` search for instances of the string "ElementwiseToLinalg"...

   We find inside `add_mlir_dialect_library(`
   ```
   ElementwiseToLinalg.cpp
   ```

   so we add 
   ```
   Avocado.cpp
   ```

4. We create a copy of `ElementwiseToLinalg.cpp` and rename it `Avocado.cpp`. 

#### 4. Hack Avocado.cpp

1. Keep the include statements and the pass declaration, and remove helper functions/anything else we think we don't need.
   ```
   //===- ElementwiseToLinalg.cpp - conversion of elementwise to linalg ------===//
   //
   // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   // See https://llvm.org/LICENSE.txt for license information.
   // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
   //
   //===----------------------------------------------------------------------===//
   
   #include "mlir/Dialect/Linalg/Passes.h"
   
   #include "mlir/Dialect/Arith/Utils/Utils.h"
   #include "mlir/Dialect/Linalg/IR/Linalg.h"
   #include "mlir/Dialect/Linalg/Transforms/Transforms.h"
   #include "mlir/Dialect/Linalg/Utils/Utils.h"
   #include "mlir/Transforms/DialectConversion.h"
   
   namespace mlir {
   #define GEN_PASS_DEF_CONVERTELEMENTWISETOLINALG
   #include "mlir/Dialect/Linalg/Passes.h.inc"
   } // namespace mlir
   
   using namespace mlir;
   
   // WE REMOVED A BUNCH OF HELPERS/ REWRITE PATTERNS WE DON'T NEED
   
   namespace {
   class ConvertElementwiseToLinalgPass
       : public impl::ConvertElementwiseToLinalgBase<
             ConvertElementwiseToLinalgPass> {
   
     void runOnOperation() final {
       auto *func = getOperation();
       auto *context = &getContext();
       ConversionTarget target(*context);
       RewritePatternSet patterns(context);
   
       mlir::linalg::populateElementwiseToLinalgConversionPatterns(patterns);
       target.markUnknownOpDynamicallyLegal([](Operation *op) {
         return !isElementwiseMappableOpOnRankedTensors(op);
       });
   
       if (failed(applyPartialConversion(func, target, std::move(patterns))))
         signalPassFailure();
     }
   };
   } // namespace
   
   std::unique_ptr<Pass> mlir::createConvertElementwiseToLinalgPass() {
     return std::make_unique<ConvertElementwiseToLinalgPass>();
   }
   ```

2. Replace pass name wherever necessary, update comments, and remove more of the guts of the `runOnOperation()` function to simplify more.
   ```
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
   
     void runOnOperation() final {
   	// guts removed!
     }
   };
   } // namespace
   
   std::unique_ptr<Pass> mlir::createAvocadoPass() {
     return std::make_unique<AvocadoPass>();
   }
   ```

3. Add implementation of custom pass
   ```
   namespace {
   class AvocadoPass : public impl::AvocadoBase<AvocadoPass> {
   	
     bool canScheduleOn(RegisteredOperationName opInfo) const override {
       return opInfo.hasInterface<FunctionOpInterface>();
     }
   
     void runOnOperation() final {
       auto *func = getOperation();
       llvm::errs() << "["<< func->getAttr("sym_name") <<"]"<<"\n";
       ++funcCount;
     }
   };
   } // namespace
   ```

   Note: the `canScheduleOn` function is described in the Static Schedule Filtering section [here](https://mlir.llvm.org/docs/PassManagement/#operation-pass-static-schedule-filtering).

#### 5. Re-run the CMake command and re-build

#### 6. Invoke Custom Avocado Pass

To invoke the pass on input file `matmul.mlir`, do
```
mlir-opt -pass-pipeline='any(func.func(avocado))'--mlir-disable-threading -mlir-pass-statistics -mlir-pass-statistics-display=list matmul.mlir
```





