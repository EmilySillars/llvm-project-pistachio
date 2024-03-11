#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MATRIX_LEN 2048
#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_LINE_COUNT 750 // cache size is 48 KB

/*
This file contains a stripped down example of loop blocking
to make it easier to view LLVM's control flow graph

clang -O1 -Xclang -disable-llvm-passes -emit-llvm
pistachio-notes/c-matrix-mult2.c -c -o "pistachio-notes/out/c-matrix-mult2.bc"
build/bin/opt pistachio-notes/out/c-matrix-mult2.bc -passes=view-cfg-only

Tiling libraries???
https://llvm.org/doxygen/MatrixUtils_8h.html
https://discourse.llvm.org/t/the-dialect-suitable-to-describe-the-tiling-operation/2729

command to run example:
rm test.o out/regular_output out/tiled_general_output;clear; gcc c-matrix-mult.c
-o test.o;./test.o; diff out/regular_output out/tiled_general_output

reference:
http://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/ch03s02.html
https://stackoverflow.com/questions/5141960/get-the-current-time-in-c
*/

// book keeping structs/funcs
typedef struct squareMatrix {
  uint64_t **mat;
  uint64_t len;
} squareMat;
void createSquareMat(squareMat *m, uint64_t len);
void destroySquareMat(squareMat *m);
void printSquareMat(squareMat *m, FILE *out);
void fillSquareMat(squareMat *a, uint64_t num);

// interesting functions
void multmat(squareMat *a, squareMat *b, squareMat *c);
void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t block_size);

int main() {
  squareMat x, y, z, w, u;
  createSquareMat(&x, MATRIX_LEN); // dimensions are 2048 x 2048
  createSquareMat(&y, MATRIX_LEN);
  createSquareMat(&z, MATRIX_LEN);
  createSquareMat(&w, MATRIX_LEN);
  createSquareMat(&u, MATRIX_LEN);
  fillSquareMat(&x, 3);
  fillSquareMat(&y, 2);
  fillSquareMat(&z, 0);
  fillSquareMat(&w, 0);
  fillSquareMat(&u, 0);
  uint64_t block_size = CACHE_LINE_SIZE_BYTES / sizeof(uint64_t);

  // tiled execution
  multmatTiledGeneral(&x, &y, &u, block_size);

  // non-tiled execution
  multmat(&x, &y, &z);

  destroySquareMat(&x);
  destroySquareMat(&y);
  destroySquareMat(&z);
  destroySquareMat(&w);
  destroySquareMat(&u);
  return 0;
}

void multmat(squareMat *a, squareMat *b, squareMat *c) {
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      for (size_t k = 0; k < a->len;
           k++) { // sum (each elt in row * each elt in col)
        c->mat[i][j] += a->mat[i][k] * b->mat[k][j];
      }
    }
  }
}

void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t bsize) {
  int i, j, k, kk, jj;
  double sum;
  uint64_t n = c->len;
  int en = bsize * (n / bsize); // Amount that fits evenly into blocks
  for (kk = 0; kk < en; kk += bsize) {
    for (jj = 0; jj < en; jj += bsize) {
      for (i = 0; i < n; i++) {
        for (j = jj; j < jj + bsize; j++) {
          sum = c->mat[i][j];
          for (k = kk; k < kk + bsize; k++) {
            sum += a->mat[i][k] * b->mat[k][j];
          }
          c->mat[i][j] = sum;
        }
      }
    }
  }
} // end of func

// matrix helpers
void createSquareMat(squareMat *m, uint64_t len) {
  m->mat = malloc(sizeof(uint64_t *) * len);
  m->len = len;
  for (size_t i = 0; i < m->len; i++) { // for each row
    m->mat[i] = malloc(sizeof(uint64_t) * len);
  }
}

void destroySquareMat(squareMat *m) {
  for (size_t i = 0; i < m->len; i++) { // for each row
    free(m->mat[i]);
  }
  free(m->mat);
  m->len = 0;
}

void fillSquareMat(squareMat *a, uint64_t num) {
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      a->mat[i][j] = num;
    }
  }
}

/*
code/mem/matmult/bmm.c
 void bijk(array A, array B, array C, int n, int bsize)
 {
 int i, j, k, kk, jj;
 double sum;
 int en = bsize * (n/bsize);

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
*/