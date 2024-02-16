#!/bin/sh
readonly PROLOGUE="/********************** Check Against Reference Implementations Below **********************/\\n/**                                                                                       **/\\n/**                                                                                       **/\\n/** print-alias-sets pass results:                                                        **/\\n\\n"
readonly MIDLOGUE="\\n/**                                                                                       **/\\n/**                                                                                       **/\\n/** aa-eval pass pass results:                                                            **/\\n\\n"
readonly EPILOGUE="/**                                                                                       **/\\n/**                                                                                       **/\\n/*******************************************************************************************/\\n"

echo "Building the project..."
make -j$(nproc) -C build opt

testFiles="pistachio-notes/tests/*.c"

echo "\\nTesting test folder..."
for testFile in $testFiles
do
  basename=`basename $testFile | sed 's/[.][^.]*$//'`  
  echo "$testFile.c..."
  clang -O1 -Xclang -disable-llvm-passes -emit-llvm "$testFile" -c -o "pistachio-notes/out/$basename.bc"
  llvm-dis < "pistachio-notes/out/$basename.bc" >"pistachio-notes/out/$basename-as-LLVM.ll"
  build/bin/opt -disable-output "pistachio-notes/out/$basename.bc" -passes=pistachio 2>"pistachio-notes/out/$basename-analyzed.txt"
  echo "$PROLOGUE" 1>> "pistachio-notes/out/$basename-analyzed.txt"
  build/bin/opt -disable-output -passes=print-alias-sets "pistachio-notes/out/$basename.bc" 2>> "pistachio-notes/out/$basename-analyzed.txt"
  echo "$MIDLOGUE" 1>> "pistachio-notes/out/$basename-analyzed.txt"
  build/bin/opt -disable-output -passes=aa-eval "pistachio-notes/out/$basename.bc" 2>> "pistachio-notes/out/$basename-analyzed.txt"
  echo "$EPILOGUE" 1>> "pistachio-notes/out/$basename-analyzed.txt"
done

echo "Done; results are located in pistachio-notes/out"
echo "\\nTesting cholesky.c..."
clang -O1 -Xclang -disable-llvm-passes -emit-llvm pistachio-notes/hacky-cholesky/hacky-cholesky.c -c -o pistachio-notes/hacky-cholesky/out/hacky-cholesky.bc
llvm-dis < pistachio-notes/hacky-cholesky/out/hacky-cholesky.bc >pistachio-notes/hacky-cholesky/out/cholesky-as-LLVM.ll >pistachio-notes/hacky-cholesky/out/cholesky.ll
build/bin/opt -disable-output pistachio-notes/hacky-cholesky/out/hacky-cholesky.bc -passes=pistachio 2>pistachio-notes/hacky-cholesky/out/cholesky-analyzed.txt
echo "$PROLOGUE" 1>> "pistachio-notes/hacky-cholesky/out/cholesky-analyzed.txt"
build/bin/opt -disable-output -passes=print-alias-sets "pistachio-notes/hacky-cholesky/out/hacky-cholesky.bc" 2>> "pistachio-notes/hacky-cholesky/out/cholesky-analyzed.txt"
echo "$MIDLOGUE" 1>> "pistachio-notes/hacky-cholesky/out/cholesky-analyzed.txt"
build/bin/opt -disable-output -passes=aa-eval "pistachio-notes/hacky-cholesky/out/hacky-cholesky.bc" 2>> "pistachio-notes/hacky-cholesky/out/cholesky-analyzed.txt"
echo "$EPILOGUE" 1>> "pistachio-notes/hacky-cholesky/out/cholesky-analyzed.txt"
echo "Done; results are located in pistachio-notes/hacky-cholesky/out"
