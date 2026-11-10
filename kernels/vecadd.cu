#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                        \
    do {                                        \
        cudaError_t err;                        \
        err = call;                             \
        if (err != cudaSuccess){                \
                printf("ERROR in file %s at line %d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
                exit(EXIT_FAILURE);             \
            }                                   \
    } while(0)



__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i;
    i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *A_d, *B_d, *C_d;
    float *A_h, *B_h, *C_h;
    int N = 1000000;
    size_t size = N * sizeof(float);

    A_h = (float *)malloc(size); 
    B_h = (float *)malloc(size);
    C_h = (float *)malloc(size);
    
    for(int i=0;i<N;i++){
        A_h[i] = float(i);
        B_h[i] = float(i)*i;

    }
    CUDA_CHECK(cudaMalloc((void**) &A_d, size));
    CUDA_CHECK(cudaMalloc((void**) &B_d, size));
    CUDA_CHECK(cudaMalloc((void**) &C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    
    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));
    
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] + B_h[i]) {
            printf("ERROR at i=%d got %f and expected %f\n", i,C_h[i],A_h[i]+B_h[i]);
            ok = 0;
            break;
        }
    }
    if (ok == 1){
        printf("OK\n");
    }    

    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
    
    return ok ? 0 : 1;
}