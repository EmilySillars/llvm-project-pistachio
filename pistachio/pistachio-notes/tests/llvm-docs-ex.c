#include <stdio.h>

/* foo
From https://llvm.org/docs/AliasAnalysis.html#representation-of-pointers
In this case, the basic-aa pass will disambiguate the stores to C[0] and C[1]
because they are accesses to two distinct locations one byte apart,
and the accesses are each one byte.
*/
void foo() {
  int i;
  char C[2];
  char A[10];
  /* ... */
  for (i = 0; i != 10; ++i) {
    C[0] = A[i];     /* One byte store */
    C[1] = A[9 - i]; /* One byte store */
  }
}

/* bar
From https://llvm.org/docs/AliasAnalysis.html#representation-of-pointers
In this case, the two stores to C do alias each other,
because the access to the &C[0] element is a two byte access.
If size information wasn’t available in the query,
even the first case would have to conservatively assume that the accesses alias.
*/
void bar() {
  int i;
  char C[2];
  char A[10];
  /* ... */
  for (i = 0; i != 10; ++i) {
    ((short *)C)[0] = A[i]; /* Two byte store! */
    C[1] = A[9 - i];        /* One byte store */
  }
}

int main() {
  foo();
  bar();
  return 5;
}
