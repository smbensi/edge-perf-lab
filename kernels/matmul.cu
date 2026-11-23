#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16

#define CUDA_CHECK(call)                        \
    do {                                        \
        cudaError_t err;                        \
        err = call;                             \
        if (err != cudaSuccess){                \
                fprintf(stderr,"ERROR in file %s at line %d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
                exit(EXIT_FAILURE);             \
            }                                   \
    } while(0)



__global__ void matmulKernel(float const *A, float const *B, float *C, int M, int K, int N){
    int row = ;
    int col = ;
}

void matmulRef(float const *A, float const *B, float *C, int M, int K, int N){

    float Cval = 0.0f;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; ++j){
            Cval = 0.0f;
            for(int k=0; k<K; ++k){
                Cval += A[K*i+k]*B[k*N+j];
            }
            C[i*N+j] = Cval;
        }
    }
}

int main(int argc, char *argv[]){
    float *A_d, *B_d, *C_d;
    int M, K, N;
    if (argc >= 4){
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    } else {
        M = 1000;
        K = 1000;
        N = 1000;
    }

    dim3 blockSize(16,16);
    dim3 gridSize();


    size_t size_A = (size_t)M*K*sizeof(float);
    size_t size_B = (size_t)K*N*sizeof(float);
    size_t size_C = (size_t)M*N*sizeof(float);

    float *A = (float *)malloc(size_A);
    if (A==NULL) {fprintf(stderr, "ERROR IN MALLOC ALLOCATION\n"); exit(EXIT_FAILURE);}

    float *B = (float *)malloc(size_B);
    if (B==NULL) {fprintf(stderr, "ERROR IN MALLOC ALLOCATION\n"); exit(EXIT_FAILURE);}

    float *C = (float *)malloc(size_C);
    if (C==NULL) {fprintf(stderr, "ERROR IN MALLOC ALLOCATION\n"); exit(EXIT_FAILURE);}

    float *C_ref = (float *)malloc(size_C);
    if (C_ref==NULL) {fprintf(stderr, "ERROR IN MALLOC ALLOCATION\n"); exit(EXIT_FAILURE);}

    for (int i = 0; i < M*K; i++) {
            A[i] = rand()/(float)RAND_MAX;
        }
    for (int i = 0; i < K*N; i++) {
            B[i] = rand()/(float)RAND_MAX;
        }

    matmulRef(A, B, C_ref, M, K, N);

    CUDA_CHECK(cudaMalloc((void**) &A_d, size_A));
    CUDA_CHECK(cudaMalloc((void**) &B_d, size_B));
    CUDA_CHECK(cudaMalloc((void**) &C_d, size_C));

    CUDA_CHECK(cudaMemcpy(A_d, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size_B, cudaMemcpyHostToDevice));


    matmulKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);

    CUDA_CHECK(cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost));


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);
    free(C_ref);
    return 0;
}

