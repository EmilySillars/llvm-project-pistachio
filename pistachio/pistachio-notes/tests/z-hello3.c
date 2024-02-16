// let's create a more complicated load/store aliasing situation...
// An old test case.
// TODO: Document what this test case is specifically testing.

#include <stdio.h>

int foo(){
    int a = 867;
    int b = a;
    int c = b;
    int d = b + 32;
    int e = a;
    return a;
}

void bar(){}

int main() {
  printf("hello world\n");
  return foo();
}