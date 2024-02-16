// An old test case.
// TODO: Document what this test case is specifically testing.

#include <stdio.h>

int foo(){
    int arr[] = {867,777,835};
    arr[1] = 444;
    int b = arr[1];              
    int c = arr[1]; 
    return b + c;                  
}

int main() {
  return foo();
}