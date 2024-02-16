#include <stdio.h>

int foo(){
    // int a = 4;
    // int *ptrToA = &a; 
    // int b = a;
    // int c = *ptrToA;
    return 76;                 
}

/* LLVM

*/

/* Results

*/

int main() {
  return foo();
}