//code/mem/matmult/bmm.c

/*
build/bin/opt pistachio-notes/out/whyNotPartialAlias.bc -passes=view-cfg-only

https://stackoverflow.com/questions/17062495/display-cfg-from-llvm-in-xvcg
https://www.graphviz.org/download/

https://askubuntu.com/questions/1181332/no-application-is-registered-as-handling-this-file-for-dot-files


*/
 void bijk(array A, array B, array C, int n, int bsize)
 {
 int i, j, k, kk, jj;
 double sum;
 int en = bsize * (n/bsize); // Amount that fits evenly into blocks

 for (i = 0; i < n; i++)
 for (j = 0; j < n; j++)
 C[i][j] = 0.0;

 for (kk = 0; kk < en; kk += bsize) {
 for (jj = 0; jj < en; jj += bsize) {
 for (i = 0; i < n; i++) {
 for (j = jj; j < jj + bsize; j++) {
 sum = C[i][j];
 for (k = kk; k < kk + bsize; k++) {
 sum += A[i][k]*B[k][j];
 }
 C[i][j] = sum;
 }
 }
 }
 }
 }