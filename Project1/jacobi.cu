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
__global__ void jacobi_jacobicu2(float * A, float * bx, float * by, float * Px, float * Py, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
    float Bvx, Bvy, Av, Xvx, Xvy; 
    Bvx = bx[j]; 
    Bvy = by[j]; 
    Av = A[j*size+j]; 
    Xvx = 0.0f;
    Xvy = 0.0f; 
    for (int k = 0; k < size; k++) {
        Xvx += A[j*size+k]*Px[k];
        Xvy += A[j*size+k]*Py[k];
    }
    Xvx -= Av*Px[j];
    Xvy -= Av*Py[j];
    Px[j] = (Bvx - Xvx)/Av;
    Py[j] = (Bvy - Xvy)/Av;
    } 
}
__global__ void jacobi_jacobicu2s(float * A, float * bx, float * by, float * Px, float * Py, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float P_s[]; 
    float * Px_s = P_s; 
    float * Py_s = P_s + size; 
    if (j < size) {
    Px_s[j] = Px[j]; 
    Py_s[j] = Py[j]; 
    __syncthreads(); 
    float Bvx, Bvy, Av, Xvx, Xvy; 
    Bvx = bx[j]; 
    Bvy = by[j]; 
    Av = A[j*size+j]; 
    Xvx = 0.0f;
    Xvy = 0.0f; 
    for (int k = 0; k < size; k++) {
        Xvx += A[j*size+k]*Px_s[k];
        Xvy += A[j*size+k]*Py_s[k];
    }
    Xvx -= Av*Px[j];
    Xvy -= Av*Py[j];
    Px[j] = (Bvx - Xvx)/Av;
    Py[j] = (Bvy - Xvy)/Av;
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
extern "C" void jacobi_jacobi2(float * resultx, float * resulty, float * A, float * bx, float * by, int size, int itt) {
    float * Px_d, * Xx_d, * A_d, * bx_d, * Py_d, * Xy_d, *by_d ;
    cudaMalloc((void **) &A_d, size*size*sizeof(float)); 
    cudaMalloc((void **) &bx_d, size*sizeof(float)); 
    cudaMalloc((void **) &Px_d, size*sizeof(float)); 
    cudaMalloc((void **) &Xx_d, size*sizeof(float)); 
    cudaMalloc((void **) &by_d, size*sizeof(float)); 
    cudaMalloc((void **) &Py_d, size*sizeof(float)); 
    cudaMalloc((void **) &Xy_d, size*sizeof(float)); 
    //float * Xx = jacobi_createMatrix(size,1);
    //float * Xy = jacobi_createMatrix(size,1); 
    float * swap;
    cudaMemcpy(A_d, A, size*size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(bx_d, bx, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(by_d, by, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Px_d, resultx, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Xx_d, resultx, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Py_d, resulty, size*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(Xy_d, resulty, size*sizeof(float), cudaMemcpyHostToDevice); 
    for (int i = 0; i < itt; i++) {
        /*
        if (2*(sizeof(float)*size) < 48000) {
            //use shared memory
            jacobi_jacobicu2s<<<size/16 + 1,16,2*(size*sizeof(float))>>>(A_d, bx_d, by_d, Px_d, Py_d, size); 
        } else {
            jacobi_jacobicu2<<<size/16 + 1,16>>>(A_d, bx_d, by_d, Px_d, Py_d, size); 
        }
        */
        jacobi_jacobicu2<<<size/16 + 1,16>>>(A_d, bx_d, by_d, Px_d, Py_d, size); 
        swap = Px_d; 
        Px_d = Xx_d; 
        Xx_d = swap;
        
        swap = Py_d; 
        Py_d = Xy_d; 
        Xy_d = swap;
    }
    cudaMemcpy(resultx, Xx_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulty, Xy_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d); 
    cudaFree(bx_d); 
    cudaFree(Px_d); 
    cudaFree(Xx_d);
    cudaFree(by_d); 
    cudaFree(Py_d); 
    cudaFree(Xy_d);
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
