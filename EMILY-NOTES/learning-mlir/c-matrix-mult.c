#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MATRIX_LEN 2048
#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_LINE_COUNT 750 // cache size is 48 KB
/*
This file contains an example of loop blocking
"blocks" in this case are of size 64 Bytes, the size of a single cache line
on my laptop

command to run example:
rm test.o out/regular_output out/tiled_general_output;clear; gcc c-matrix-mult.c -o test.o;./test.o; diff out/regular_output out/tiled_general_output

reference:
http://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/ch03s02.html
https://stackoverflow.com/questions/5141960/get-the-current-time-in-c
*/

/* results from my laptop...
$ lscpu | grep "cache"
L1d cache:                          448 KiB (12 instances)
L1i cache:                          640 KiB (12 instances)
L2 cache:                           9 MiB (6 instances)
L3 cache:                           18 MiB (1 instance)

$ cd /sys/devices/system/cpu/cpu0/cache/
$ cd index0
$ cat level type coherency_line_size size
1
Data
64
48K

(48 kilobytes) / (64 bytes) = 750
*/

/*Sample Output:
A x B = C where all matrices are 2048 x 2048
Block size is 64 / 8 = 8

Comparing tiled matrix multiplication to non-tiled...

-------------------------------------------------------------------
Time to execute multmatTiledGeneral: 21.000000
Time to execute multmat: 48.000000

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
void multmatTiled(squareMat *a, squareMat *b, squareMat *c);
void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t block_size);
// time the execution of matrix operation f on parameters a, b, c
void timefunc(void (*f)(), squareMat *a, squareMat *b, squareMat *c,
              uint64_t n);

int main() {
  // create and initialize three matrices x, y, and z.
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

  // print info about run
  printf("A x B = C where all matrices are %d x %d\n",MATRIX_LEN,MATRIX_LEN);
  printf("Block size is %d / %ld = %ld\n\n", CACHE_LINE_SIZE_BYTES,sizeof(uint64_t), block_size);
  printf("Comparing tiled matrix multiplication to non-tiled...\n");
  printf("\n-------------------------------------------------------------------"
         "\n");

  // file handling
  FILE *reg_output = fopen("out/regular_output", "w");
  FILE *tiled_general_output = fopen("out/tiled_general_output", "w");

  // time general tiled execution
  printSquareMat(&u, tiled_general_output);
  timefunc(multmatTiledGeneral, &x, &y, &u, block_size);
  printSquareMat(&u, tiled_general_output);

  // time non-tiled execution
  printSquareMat(&z, reg_output);
  timefunc(multmat, &x, &y, &z, 0);
  printSquareMat(&z, reg_output);

  // close files and clean up
  fclose(reg_output);
  // fclose(tiled_output);
  fclose(tiled_general_output);
  destroySquareMat(&x);
  destroySquareMat(&y);
  destroySquareMat(&z);
  destroySquareMat(&w);
  destroySquareMat(&u);
  return 0;
}

void multmat(squareMat *a, squareMat *b, squareMat *c) {
  if (!((a->len == b->len) && (b->len == c->len))) {
    return;
  } // only square matrices allowed
  uint64_t cells = 0;
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      for (size_t k = 0; k < a->len;
           k++) { // sum (each elt in row * each elt in col)
        c->mat[i][j] += a->mat[i][k] * b->mat[k][j];
      }
      cells++;
    }
  }
}

void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t bsize) {

   int i, j, k, kk, jj;
 double sum;
 uint64_t n = c->len;
 int en = bsize * (n/bsize); // Amount that fits evenly into blocks
 //printf("en is %d \n",en);
 for (i = 0; i < n; i++)
 for (j = 0; j < n; j++)
 c->mat[i][j] = 0.0;

 for (kk = 0; kk < en; kk += bsize) {
 for (jj = 0; jj < en; jj += bsize) {
 for (i = 0; i < n; i++) {
 for (j = jj; j < jj + bsize; j++) {
 sum = c->mat[i][j];
 for (k = kk; k < kk + bsize; k++) {
 sum += a->mat[i][k]*b->mat[k][j];
 }
 c->mat[i][j] = sum;
 }
 }
 }
 }
} // end of func

void timefunc(void (*f)(), squareMat *a, squareMat *b, squareMat *c,
              uint64_t n) {
  time_t beforeTime, afterTime;
  double diff;
  char *fnm;

  if (f == multmat) {
    fnm = "multmat";
    time(&beforeTime); // save time before execution
    multmat(a, b, c);  // execute function multmat
  } 
  // else if (f == multmatTiled) {
  //   fnm = "multmatTiled";
  //   time(&beforeTime);     // save time before execution
  //   multmatTiled(a, b, c); // execute function multmatTiled
  // } 
  else if (f == multmatTiledGeneral) {
    fnm = "multmatTiledGeneral";
    time(&beforeTime);               // save time before execution
    multmatTiledGeneral(a, b, c, n); // execute function multmatTiledGeneral
  } else {
    fprintf(stderr, "ERR: function to time not recognized\n");
    return;
  }
  time(&afterTime);                       // save time after execution
  diff = difftime(afterTime, beforeTime); // compute difference
  printf("Time to execute %s: %f\n", fnm, diff);
}

// printing helper function
void printSquareMat(squareMat *m, FILE *out) {
  fprintf(out, "\n{");
  for (size_t i = 0; i < m->len; i++) { // for each row
    fprintf(out, "{ ");
    for (size_t j = 0; j < m->len; j++) { // print each elt of row
      fprintf(out, "%d ", (int)m->mat[i][j]);
    }
    fprintf(out, " }\n");
  }
  fprintf(out, "}\n");
}

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
 int en = bsize * (n/bsize); /* Amount that fits evenly into blocks

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