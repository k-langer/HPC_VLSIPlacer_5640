//#include "common.h"
extern "C" {
#include "jacobi.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>

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
/*__global__ void jacobi_jacobicu(float * A, float * b, float * P, int size,float * temp) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    float Bv, Av; 
    if ( k < size && j < size) {
        if(k == 0) {
            printf("%f\n",P[j]);
        }
        Bv = b[j]; 
        Av = A[j*size+j];
        temp[j] += A[j*size+k]*P[k]; 
    }
        __syncthreads(); 
    if ( j != k && k < size && j < size) { 
        P[j] = (Bv - temp[j])/Av;
    }
}*/


__global__ void jacobi_jacobicu(float * A, float * b, float * P, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
    printf("%d\n",j);
    float Bv, Av, Xv; 
    Bv = b[j]; 
    Av = A[j*size+j]; 
    Xv = 0.0f;
    for (int k = 0; k < size; k++) {
        Xv += A[j*size+k]*P[k];
    }
    Xv -= Av*P[j];
    P[j] = (Bv - Xv)/Av;
    } 
}

/*
__global__ void jacobi_jacobicu(float * A, float * b, float * P, int size) {
    int j = blockIdx.x; 
    int k = blockIdx.y;
    __shared__ float Xv[22];
    Xv[j] = 0;
    __syncthreads();

    float Bv, Av, Xv = 0.0f; 
    Bv = b[j]; 
    Av = A[j*size+j];
    printf("%d %d\n",j,k); 
    if (j != k)
        Xv += A[j*size+k]*P[k];
    P[j] = (Bv - Xv)/Av; 
}
*/
/* 
* return a row/col matrix full of all zeros
*/
float * jacobi_createMatrix(int ysize, int xsize) {
    return (float *) calloc(sizeof(float),ysize*xsize);
}
extern "C" float * jacobi_jacobi(float * A, float * b, int size, int itt) {
    float * P_d, * X_d, * A_d, * b_d;
    cudaMalloc((void **) &A_d, size*size*sizeof(float)); 
    cudaMalloc((void **) &b_d, size*sizeof(float)); 
    cudaMalloc((void **) &P_d, size*sizeof(float)); 
    cudaMalloc((void **) &X_d, size*sizeof(float)); 
    float * X = jacobi_createMatrix(size,1);
    float * swap;
    cudaMemcpy(A_d, A, size*size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(P_d, X, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(X_d, X, size*sizeof(float), cudaMemcpyHostToDevice); 
    for (int i = 0; i < itt; i++) {
        jacobi_jacobicu<<<size/16 + 1,16>>>(A_d, b_d, P_d, size); 
        swap = P_d; 
        P_d = X_d; 
        X_d = swap;
    }
    cudaMemcpy(X, X_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d); 
    cudaFree(b_d); 
    cudaFree(P_d); 
    cudaFree(X_d);
    return X; 
}
/*
int main() {
    int size =22; 
    float * A = new float[size*size]; 
    float * bx = new float[size];
    float * by = new float[size]; 
    std::ifstream infile;
    int count; 
    float num;
    infile.open("matrix/A.txt");
    count = 0;  
    while(infile >> num) {
        A[count] = num; 
        count++;
    } 
    infile.close();
    infile.open("matrix/bx.txt");
    count = 0;  
    while(infile >> num) {
        bx[count] = num; 
        count++;
    } 
    infile.close();
    infile.open("matrix/by.txt");
    count = 0;  
    while(infile >> num) {
        by[count] = num; 
        count++;
    }
    float * result1 = jacobi_jacobi(A, bx, size, 30);
    float * result2 = jacobi_jacobi(A, by, size, 30);
    for (int i = 0; i < size; i++) {
        printf("%f %f\n",result1[i],result2[i]);
    } 
    //infile("matrix/by.txt"); 
    //infile("matrix/bx.txt");
} 
*/
