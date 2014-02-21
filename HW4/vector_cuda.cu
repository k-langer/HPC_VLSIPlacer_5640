#include <iostream>
#include <cuda.h>
#include <sys/time.h>

#include "util.h" // to check that output is correct
using namespace std;
// const int N = 16;
// const int blocksize = 16;

__global__ void add(size_t vect_len, float * vect1, float *vect2, float *result){
    size_t const tid = threadIdx.x;
    if (tid >= vect_len)
        return;
    result[tid] = vect1[tid] + vect2[tid]; 
}

int main(int argc,char ** argv){
    cudaEvent_t start=0;
    cudaEvent_t stop=0;
    float time =0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    timeval t1, t2, t3, t4;
    size_t vect_len = parse_args(argc,argv);
    if(vect_len <= 0){
        return -1; }
    printf("Elements: %d\n",vect_len);
    /*if (vect_len > 100){
        cout << "this program has very naive thread layout, "
        <<"please use vector length of less than 100" << endl;
    }*/
    float vect1[vect_len];
    float vect2[vect_len];
    float result[vect_len];
    init_vects(vect_len,vect1,vect2);
    //add vectors together
    float *vect1_d;
    float *vect2_d;
    float *result_d;    
    cudaMalloc((void**)&vect1_d,vect_len*sizeof(float));
    cudaMalloc((void**)&vect2_d,vect_len*sizeof(float));
    cudaMalloc((void**)&result_d,vect_len*sizeof(float));
    gettimeofday(&t1, NULL);
    cudaMemcpy(vect1_d,vect1,vect_len*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vect2_d,vect2,vect_len*sizeof(float),cudaMemcpyHostToDevice);
    //setup block and grid size
    dim3 dimBlock(vect_len,1);
    dim3 dimGrid(1,1);
    // call device kernel
    cudaEventRecord(start,0);
    add<<<dimGrid,dimBlock>>>(vect_len,vect1_d,vect2_d,result_d);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    // copy data from device 
    cudaMemcpy(result,result_d,vect_len*sizeof(float),cudaMemcpyDeviceToHost);
    gettimeofday(&t2, NULL);
    // free device memory    
    cudaFree(vect1_d);
    cudaFree(vect2_d);
    cudaFree(result_d);
    double  elapsedTime = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
    printf("\tGPU time (ms): %.4f\n\t\tKernel Time: %.4f\n",elapsedTime,time);
    gettimeofday(&t3, NULL);
    bool correct = verify(vect_len,vect1,vect2,result);
    gettimeofday(&t4, NULL);
    elapsedTime = (t4.tv_sec - t3.tv_sec)*1000.0 + (t4.tv_usec - t3.tv_usec)/1000.0;
    printf("\tCPU time (ms): %.4f\n",elapsedTime);
    if(correct){
    //    cout << "result is correct" << endl;
    } else{
        cout << "result is _NOT_ correct" << endl;
    }
    return 0;
}
