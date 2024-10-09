# filling the matrices according to the values in set in the main.c 
#// initialize the matrices
#   for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
#     memrefA.aligned_data[i] = (int8_t)2;
#     memrefB.aligned_data[i] = (int8_t)3;
#     memrefC.aligned_data[i] = (int32_t)0;
#     memrefGolden.aligned_data[i] = (int32_t)0;
#   }
#   memrefB.aligned_data[85] = 87;

def printMatrixAsMLIRConstant(mat, nm, dataType):
    # assume matrix is 2D
    print(f'%const_{nm} = arith.constant dense<[')
    # print all rows except the last one
    for r in range (mat.shape[1]-1):
        print("[",end='')
        for c in range (mat.shape[0]-1):
            print(mat[r][c],end=',')
        print(mat[r][mat.shape[0]-1],end='],\n')
    # print the last row
    lastRow = mat.shape[1]-1
    print("[",end='')
    for c in range (mat.shape[0]-1):
            print(mat[lastRow][c],end=',')
    print(mat[r][mat.shape[0]-1],end=']\n')
    print(f']> : tensor<{mat.shape[0]}x{mat.shape[1]}x{dataType}>')
    return

# const int8_t A[256] = {
def printMatrixAsCArray(mat, nm, dataType):
    # assume matrix is 2D
    print(f'const {dataType} {nm}[{mat.shape[0]*mat.shape[1]}] = {"{"}')
    # print all rows except the last one
    for r in range (mat.shape[1]-1):
        
        for c in range (mat.shape[0]-1):
            print(mat[r][c],end=',')
        print(mat[r][mat.shape[0]-1],end=',\n')
    # print the last row
    lastRow = mat.shape[1]-1
    for c in range (mat.shape[0]-1):
            print(mat[lastRow][c],end=',')
    print(mat[r][mat.shape[0]-1],end='};\n')
    return

import numpy as np

MAT_WIDTH = 104
MAT_WIDTH_SQUARED = MAT_WIDTH * MAT_WIDTH

# intialize A input matrix
flatA = [2] * MAT_WIDTH_SQUARED
a = np.asarray(flatA, np.int8).reshape(MAT_WIDTH,MAT_WIDTH)

# intialize B input matrix
flatB = [3] * MAT_WIDTH_SQUARED
flatB[85] = 87
b = np.asarray(flatB, np.int8).reshape(MAT_WIDTH,MAT_WIDTH)

# intialize C output matrix
flatC = [0] * MAT_WIDTH_SQUARED
c = np.asarray(flatC, np.int32).reshape(MAT_WIDTH,MAT_WIDTH)

# initialize correct result
golden = np.matmul(a,b, dtype=np.int32)

print("VVVVV COPY AND PASTE THE FOLLOWING INTO YOUR DATA.h VVVVV")
# printMatrixAsCArray(a, "a", "int8_t")
# printMatrixAsCArray(b, "b", "int8_t")
# printMatrixAsCArray(c, "c", "int32_t")
printMatrixAsCArray(golden, "golden", "int32_t")
print("^^^^^   COPY AND PASTE THE ABOVE INTO YOUR DATA.h   ^^^^^")
