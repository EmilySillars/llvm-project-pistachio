// An old test case.
// TODO: Document what this test case is specifically testing.

#include <stdio.h>

int foo(){
    int arr[] = {867,777,835};
    int b = *arr;              
    int c = arr[2];            
    int e = b + c;             
    int f = arr[0] + arr[2];   
    return e;                  
}

int main() {
  return foo();
}