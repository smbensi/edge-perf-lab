
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>             // contains qsort

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



int compare_doubles(const void *a, const void *b) {
    double val_a = *(const double *)a;
    double val_b = *(const double *)b;

    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}

__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i;
    i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    float *A_d, *B_d, *C_d;
    float *A_h, *B_h, *C_h;
    int N;
    if(argc > 1){
        N = atoi(argv[1]);
    } else {
        N = 50'000'000;
    }
    int K = 1'000;
    size_t size = N * sizeof(float);
    float time, timeD2H, timeH2D;
    int blockSize = 256;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A_h = (float *)malloc(size); 
    if (A_h == NULL) { printf("ERROR allocation\n"); return 1;}
    B_h = (float *)malloc(size);
    if (B_h == NULL) { printf("ERROR allocation\n"); return 1;}
    C_h = (float *)malloc(size);
    if (C_h == NULL) { printf("ERROR allocation\n"); return 1;}
    
    for(int i=0;i<N;i++){
        A_h[i] = float(i);
        B_h[i] = float(i)*i;

    }
    CUDA_CHECK(cudaMalloc((void**) &A_d, size));
    CUDA_CHECK(cudaMalloc((void**) &B_d, size));
    CUDA_CHECK(cudaMalloc((void**) &C_d, size));

    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeH2D, start, stop));

    
    vecAddKernel<<<(N +blockSize -1 )/blockSize, blockSize>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaGetLastError());

    
    double *durations = (double *)calloc(K, sizeof(double)); 
    if (durations == NULL) { printf("ERROR allocation\n"); return 1;}
    for (int i=0; i<K; i++){
        cudaEventRecord(start, 0);
        vecAddKernel<<<(N +blockSize -1 )/blockSize, blockSize>>>(A_d, B_d, C_d, N);
        cudaEventRecord(stop, 0);
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        durations[i] = time;
    }
    qsort(durations, K, sizeof(double), compare_doubles);
    double p_50 = durations[K/2];
    double p_99 = durations[K*99/100];

    

    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeD2H, start, stop));

    double bandwidth = 3.0*N*4 / (p_50 / 1000) / 1e9;
    
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] + B_h[i]) {
            printf("ERROR at i=%d got %f and expected %f\n", i,C_h[i],A_h[i]+B_h[i]);
            ok = 0;
            break;
        }
    }
    if (ok == 1){
        printf("OK | N = %d | kernel p50 = %.4f ms | p99 = %.4f ms | %f Go/s | H2D = %f ms | D2H = %f ms\n",
               N, p_50, p_99, bandwidth, timeH2D, timeD2H);
    }    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
    free(durations);
    
    return ok ? 0 : 1;
}