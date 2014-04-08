//#include "common.h"
extern "C" {
#include "jacobi.h"
}
#include <stdio.h>
#include <stdlib.h>
/*
* This is the core of the solver using the Jacobi Method
* http://en.wikipedia.org/wiki/Jacobi_method
* Solve Ax = b
* where b is of size 'size' and A is a square matrix of size*size
* Becuase my connectivity matrix has values on every diagonal (becauase of the fact 
* that any gates IS connected to something) jacobi is garunteed to find some solution. 
* It is an itterative method, not an exact solution. It is MUCH faster than inverting the matrix and multiplying
* Code is based on algorithm in 'NUMERICAL METHODS for Mathematics, Science and Engineering, 2nd Ed, 1992' and the
* accompanying Matlab text.
*/
__global__ void jacobi_jacobicu(float * A, float * b, float * P, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    float Bv, Av; 
    Bv = b[j]; 
    Av = A[j*size+j]; 
    if (k!=j) { 
        P[j] += (Bv - A[j*size+k]*P[k])/Av;
    }
}
/* 
* return a row/col matrix full of all zeros
*/
extern "C" float * jacobi_createMatrix(int ysize, int xsize) {
    return (float *) calloc(sizeof(float),ysize*xsize);
}
extern "C" float * jacobi_jacobi(float * A, float * b, int size, int itt) {
    float * P_d, * X_d, * A_d, * b_d; 
    cudaMalloc((void **) &A_d, size*size); 
    cudaMalloc((void **) &b_d, size); 
    cudaMalloc((void **) &P_d, size); 
    cudaMalloc((void **) &X_d, size); 
    float * X = jacobi_createMatrix(size,1);
    float * swap;

    cudaMemcpy(A_d, A, size*size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b, size*sizeof(float), cudaMemcpyHostToDevice); 
    int blockX = 16, blockY = 16; 
    dim3 blocks(blockX, blockY); 
    dim3 grid(size/blocks.x, size/blocks.y); 
    for (int i = 0; i < itt; i++) {
        jacobi_jacobicu<<<grid,blocks>>>(A_d, b_d, P_d, size); 
        swap = P_d; 
        P_d = X; 
        X_d = swap; 
    }
    cudaMemcpy(X, X_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    return X; 
}
