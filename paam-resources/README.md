# PAAM Resources

Resources for PROGRAMACIÓN AVANZADA DE ARQUITECTURAS MULTINÚCLEO

## LLVM

1. clone + build LLVM
   - [official guide](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
   - [PAAM-specific setup notes](./llvm-setup.md) :frog:

2. how to invoke the LLVM compiler
   - [compile and run simple-c.c ](../paam-examples/llvm-simple-c/README.md)​ :frog:

3. how to add a custom pass to LLVM
   - [official tutorial](https://llvm.org/docs/WritingAnLLVMNewPMPass.html)
   - PAAM-specific notes (to be added ~~Monday March 31st~~; delayed, apologies!)

Useful Links:

- [LLVM Discourse Forum](https://discourse.llvm.org/)

- [LLVM Discourse Forum's Beginners Category](https://discourse.llvm.org/c/beginners/17)

- <a href="https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:21,endLineNumber:23,positionColumn:21,positionLineNumber:23,selectionStartColumn:21,selectionStartLineNumber:23,startColumn:21,startLineNumber:23),source:'%23include+%3Cstdio.h%3E%0A%0Aint+littleVector%5B5%5D+%3D+%7B0,+1,+2,+3,+4%7D%3B%0A%0A//+void+mapAdd(int+addend,+int*+vector,+int+size)%7B%0A//+++++for(size_t+i+%3D+0%3B+i+%3C+size%3B+i%2B%2B)%7B%0A//+++++++++vector%5Bi%5D%2B%3Daddend%3B%0A//+++++%7D%0A//+%7D%0A%0A%0Aint+main()%0A%7B%0A++++%0A++++printf(%22Hello+World%5Cn%22)%3B%0A++++%0A++++//+int+sizeOfLittleVector+%3D+sizeof(littleVector)/sizeof(int)%3B%0A++++%0A++++//+for(size_t+i+%3D+0%3B+i+%3C+sizeOfLittleVector%3B+i%2B%2B)%7B%0A++++//+++++printf(%22%25d%22,littleVector%5Bi%5D)%3B%0A++++//+%7D%0A++++%0A++++//+printf(%22%5Cn%22)%3B%0A++++%0A++++//+mapAdd(5,+littleVector,+sizeOfLittleVector)%3B%0A++++%0A++++//+for(size_t+i+%3D+0%3B+i+%3C+sizeOfLittleVector%3B+i%2B%2B)%7B%0A++++//+++++printf(%22%25d%22,littleVector%5Bi%5D)%3B%0A++++//+%7D%0A%0A++++return+0%3B%0A%7D%0A'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:45.31165311653116,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang_trunk,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:5,lang:c%2B%2B,libs:!(),options:'-emit-llvm',overrides:!(),selection:(endColumn:31,endLineNumber:6,positionColumn:31,positionLineNumber:6,selectionStartColumn:3,selectionStartLineNumber:5,startColumn:3,startLineNumber:5),source:1),l:'5',n:'0',o:'+x86-64+clang+(trunk)+(Editor+%231)',t:'0')),k:54.68834688346884,l:'4',m:100,n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4">Godboltorg: using clang's `--emit-llvm` flag</a>

## MLIR

1. clone + build MLIR
   - [official guide](https://urldefense.com/v3/__https://mlir.llvm.org/getting_started/__;!!D9dNQwwGXtA!SEOGnQ6kNZqQqZPcpFmf8I3xb2_IWywnYuC-onD9gFeQlo2vOJYwE8HHNPkxP2PYzkO6OKhJlpcbI77vcrKwuA$)
   - [PAAM-specific setup notes](mlir-setup.md) :frog:

2. how to invoke the MLIR compiler
   - [print out a tensor using the MLIR JIT](../paam-examples/mlir-print-vector/README.md) :frog:
   - what if I don't want to use the JIT?
     - Lower to LLVM dialect MLIR
     - `mlir-translate my-code-in-llvm-dialect.mlir > myCode.ll` 
     - follow [compilation steps for simple-c.c](../paam-examples/llvm-simple-c/README.md)
   - call MLIR from C, and C from MLIR (to be added ~~Monday March 31st~~; delayed, apologies!)
   
3. how to add a custom pass to MLIR
   - [Read this official page on pass infrastructure](https://mlir.llvm.org/docs/PassManagement/) (at least the Operation Pass and Analysis Management sections), then refer to PAAM-specific notes
   - PAAM-specific notes (to be added ~~Monday March 31st~~; delayed, apologies!)

Useful Links:

- [LLVM Discourse Forum's MLIR Category](https://discourse.llvm.org/tag/mlir)

- <a href="https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:mlir,selection:(endColumn:7,endLineNumber:7,positionColumn:7,positionLineNumber:7,selectionStartColumn:7,selectionStartLineNumber:7,startColumn:7,startLineNumber:7),source:'func.func+@simple_matmul(%0A%25arg0:+memref%3C16x16xi8%3E,+%0A%25arg1:+memref%3C16x16xi8,+strided%3C%5B1,+16%5D,+offset:0%3E%3E,+%0A%25arg2:+memref%3C16x16xi32%3E)+%7B%0A%25c0_i32+%3D+arith.constant+0+:+i32%0Alinalg.quantized_matmul+ins(%25arg0,+%25arg1,+%25c0_i32,+%25c0_i32+:+memref%3C16x16xi8%3E,+memref%3C16x16xi8,+strided%3C%5B1,+16%5D,+offset:0%3E%3E,+i32,+i32)+outs(%25arg2+:+memref%3C16x16xi32%3E)%0Areturn%0A%7D'),l:'5',n:'1',o:'MLIR+source+%231',t:'0')),k:60.176503033645886,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:output,i:(compilerName:'opt+(trunk)',editorid:1,fontScale:14,fontUsePx:'0',j:2,wrap:'1'),l:'5',n:'0',o:'Output+of+MLIR+opt+16.0.0+(Compiler+%232)',t:'0'),(h:compiler,i:(compiler:mliropt1600,filters:(b:'1',binary:'1',binaryObject:'1',commentOnly:'1',debugCalls:'1',demangle:'1',directives:'1',execute:'0',intel:'1',libraryCode:'1',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:2,lang:mlir,libs:!(),options:'--mlir-print-op-generic+--mlir-print-local-scope',overrides:!(),selection:(endColumn:14,endLineNumber:17,positionColumn:14,positionLineNumber:17,selectionStartColumn:14,selectionStartLineNumber:17,startColumn:14,startLineNumber:17),source:1),l:'5',n:'0',o:'+MLIR+opt+16.0.0+(Editor+%231)',t:'0')),k:39.823496966354114,l:'4',n:'0',o:'',s:1,t:'0')),l:'2',m:100,n:'0',o:'',t:'0')),version:4">Godboltorg: inspecting matmul in linalg dialect </a>

