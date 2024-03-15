#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// fake 1D tensor
// tensor<2xf32>
// define { ptr, ptr, i64, [1 x i64], [1 x i64] } 
struct fakeTensor1D {
    float* ptr0;
    float* ptr1;
    int64_t i64;
    int64_t arr0[1];
    int64_t arr1[1];
};

// fake 2D tensor
// tensor<3x9xf32>
// define { ptr, ptr, i64, [2 x i64], [2 x i64] }
// TODO

// fake 3D tensor
// tensor<2x3x9xf32>
// define { ptr, ptr, i64, [3 x i64], [3 x i64] }
// TODO

// dummy tensor function
struct fakeTensor1D modifyTensor(struct fakeTensor1D hoodle){
    // printf("Trying to print fake tensor:%ld\n", hoodle.dim1[0]);
    hoodle.ptr0[0] = 100000;
    return hoodle;
}

// prints out the tensor
// TODO: make more general!!
struct fakeTensor1D printTensor(struct fakeTensor1D hoodle){
    printf("|------------------------------------->\n");
    printf("ptr0 is %p\n", hoodle.ptr0);
    printf("ptr1 is %p\n", hoodle.ptr1);
    printf("i64 field is %ld\n",hoodle.i64);
    printf("ptr0[0] is %f\n", hoodle.ptr0[0]);
    printf("ptr0[1] is %f\n", hoodle.ptr0[1]);
    printf("arr0[0] is %ld\n", hoodle.arr0[0]);
    printf("arr1[0] is %ld\n", hoodle.arr1[0]);
    int64_t len = hoodle.arr0[0];
    printf("ptr1 points to [ ");
    for (size_t i = 0; i < len; i++){
        printf("%f, ",hoodle.ptr1[i]);
    }
    printf("]\n");
    printf("|_____________________________________>\n");
   return hoodle;
}

int main(){
    float ptr0[3] = {888.0,97.0,33.0};
    float ptr1[3] = {55.0,66.0,79.0};
    struct fakeTensor1D dummy;
    dummy.ptr0 = ptr0;
    dummy.ptr1 = ptr1;
    struct fakeTensor1D retValueJill = modifyTensor(dummy); // modifyTensor will be changed to simp function call
    printTensor(retValueJill);
    printf("pamplemousse!!!\n");
    return 0;
}

