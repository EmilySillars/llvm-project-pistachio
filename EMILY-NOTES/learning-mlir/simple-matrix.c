#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// int64_t* maybeMat(int64_t arg0[3], int64_t arg1[3]){
//     int64_t* val = malloc(sizeof(int64_t)*3);
//     val[0] = 97;
//     val[1] = arg0[0];
//     val[2] = arg1[0];
//     return val;
// }
//define { ptr, ptr, i64, [1 x i64], [1 x i64] } 
struct fakeTensor {
    float* dim1;
    float* dim2;
    int64_t len;
    int64_t a[1];
    int64_t b[1];
};

struct fakeTensor modifyTensor(struct fakeTensor hoodle){
    // printf("Trying to print fake tensor:%ld\n", hoodle.dim1[0]);
    hoodle.dim1[0] = 98;
    return hoodle;
}

struct fakeTensor printTensor(struct fakeTensor hoodle){
    printf("Trying to print fake tensor...\n");
    printf("dim1 is %p\n", hoodle.dim1);
    printf("dim2 is %p\n", hoodle.dim1);
    printf("len is %d\n",hoodle.len);
    printf("dim1[0] is %f\n", hoodle.dim1[0]);
    printf("dim1[1] is %f\n", hoodle.dim1[1]);
    printf("dim2[0] is %f\n", hoodle.dim2[0]);
    printf("dim2[1] is %f\n", hoodle.dim2[1]);
    printf("a[0] is %d\n", hoodle.a[0]);
    printf("b[0] is %d\n", hoodle.b[0]);
   return hoodle;
}



int main(){
    printf("yodelaheehoooo~~~!\n");
    float dim1[3] = {888.0,97.0,33.0};
    float dim2[3] = {55.0,66.0,79.0};
    struct fakeTensor ft;
    ft.dim1 = dim1;
    ft.dim2 = dim2;
    struct fakeTensor retValueJill = printTensor(ft);
  //  struct fakeTensor ft2 = modifyTensor(ft);
    printTensor(retValueJill);
    return 0;
}

// old notes below
    // int64_t* val = maybeMat(a,b);
    // printf("%ld\n",val[0]);
    // free(val);
        // struct fakeTensor ft2 = eitherMat(ft);
   // printf("Trying to print fake tensor:%ld\n", ft2.dim1[0]);

/*
int yodel(int a){
    return a + a;
}
*/