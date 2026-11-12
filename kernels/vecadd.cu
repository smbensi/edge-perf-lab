#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>             // contains qsort

#define CUDA_CHECK(call)                        \
    do {                                        \
        cudaError_t err;                        \
        err = call;                             \
        if (err != cudaSuccess){                \
                fprintf(stderr, "ERROR in file %s at line %d: %s\n",            \
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

__global__ void vecAddKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
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
        if(N<=0){
            fprintf(stderr,"N must be > 0: got %d\n", N);
            return 1;
        }
    } else {
        N = 50'000'000;
    }
    int K = 1'000;
    size_t size = N * sizeof(float);
    float time, timeD2H, timeH2D;
    const int blockSize = 256;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    A_h = (float *)malloc(size); 
    if (A_h == NULL) { fprintf(stderr, "ERROR allocation\n"); return 1;}
    B_h = (float *)malloc(size);
    if (B_h == NULL) { fprintf(stderr,"ERROR allocation\n"); return 1;}
    C_h = (float *)malloc(size);
    if (C_h == NULL) { fprintf(stderr, "ERROR allocation\n"); return 1;}
    
    for(int i=0;i<N;i++){
        A_h[i] = float(i);
        B_h[i] = float(i)*i;

    }
    CUDA_CHECK(cudaMalloc((void**) &A_d, size));
    CUDA_CHECK(cudaMalloc((void**) &B_d, size));
    CUDA_CHECK(cudaMalloc((void**) &C_d, size));

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeH2D, start, stop));

    // WARMUP
    vecAddKernel<<<(N +blockSize -1 )/blockSize, blockSize>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaGetLastError());

    
    double *durations = (double *)calloc(K, sizeof(double)); 
    if (durations == NULL) { fprintf(stderr, "ERROR allocation\n"); return 1;}
    for (int i=0; i<K; i++){
        CUDA_CHECK(cudaEventRecord(start, 0));
        vecAddKernel<<<(N +blockSize -1 )/blockSize, blockSize>>>(A_d, B_d, C_d, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        durations[i] = time;
    }
    qsort(durations, K, sizeof(double), compare_doubles);
    double p_50 = durations[K/2];
    double p_99 = durations[K*99/100];

    CUDA_CHECK(cudaDeviceSynchronize());
    

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeD2H, start, stop));
    double bandwidth = 3.0*N*4 / (p_50 / 1000) / 1e9;
    
    int failed = 1;
    for (int i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] + B_h[i]) {
            fprintf(stderr, "ERROR at i=%d got %f and expected %f\n", i,C_h[i],A_h[i]+B_h[i]);
            failed = 0;
            break;
        }
    }
    if (failed == 1){
        printf("OK | N = %d | kernel p50 = %.4f ms | p99 = %.4f ms | %.1f GB/s | H2D = %f ms | D2H = %f ms\n",
               N, p_50, p_99, bandwidth, timeH2D, timeD2H);
    }    

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    free(A_h);
    free(B_h);
    free(C_h);
    free(durations);
    
    return failed ? 0 : 1;
}