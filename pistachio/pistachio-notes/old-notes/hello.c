// testing LLVM example from https://llvm.org/docs/GettingStarted.html#simple-example

#include <stdio.h>

int main() {
  printf("hello world\n");
  return 0;
}

// Compile into native executeable with: clang hello.c -o hello
// Run executeable with: ./hello

// Compile to LLVM bitcode file with: clang -O3 -emit-llvm hello.c -c -o hello.bc
// To look at the LLVM assembly, do: llvm-dis < hello.bc | less
// Make sure you have llv installed! (sudo apt install llvm)