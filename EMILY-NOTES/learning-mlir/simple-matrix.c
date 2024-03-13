#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

uint64_t* maybeMat(uint64_t arg0[3], uint64_t arg1[3]){
    uint64_t* val = malloc(sizeof(uint64_t)*3);
    val[0] = 97;
    val[1] = arg0[0];
    val[2] = arg1[0];
    return val;
}
int yodel(int a){
    return a + a;
}

int main(){
    uint64_t a[3] = {1,2,3};
    uint64_t b[3] = {5,6,7};
    printf("yodelaheehoooo~~~!\n");
    printf("%d\n",yodel(77));
    uint64_t* val = maybeMat(a,b);
    printf("%ld\n",val[0]);
    free(val);
    return 0;
}