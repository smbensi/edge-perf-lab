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



__global__ void matmulNaiveKernel(float const *A, float const *B, float *C, int M, int K, int N){
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;
    float Cval;
    if(row<M && col<N){
        Cval = 0;
        for(int i=0; i<K; ++i){
            Cval += A[row*K+i]*B[i*N+col];
        }
        C[row*N+col] = Cval;
    }
}

__global__ void matmulTiledKernel(float const *A, float const *B, float *C, int M, int K, int N){
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;

    float Cval;
    Cval = 0;
    for(int ph=0; ph<(K+TILE_WIDTH-1)/TILE_WIDTH; ++ph){
        if (row < M && (ph*TILE_WIDTH+tx)<K) Ads[ty][tx] = A[row*K+ph*TILE_WIDTH+tx]; else Ads[ty][tx] = 0.0f;
        if (col < N && (ph*TILE_WIDTH+ty)<K) Bds[ty][tx] = B[(ph*TILE_WIDTH+ty)*N+col]; else Bds[ty][tx] = 0.0f;
        __syncthreads();
        for (int i=0; i<TILE_WIDTH; i++){
            Cval += Ads[ty][i]*Bds[i][tx];
        } 
        __syncthreads();

    }
    if (row < M && col < N){
        C[row*N+col] = Cval;
    }
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

int compare_matrices(float *gpu_matrix, float *cpu_matrix, int M, int N){
    float diff = 0;
    int nb_errors = 0;
    for(int i=0;i<M*N;++i){
        diff = fabsf(gpu_matrix[i] - cpu_matrix[i]) > 1e-3f * fabsf(cpu_matrix[i]);
        // diff = abs(gpu_matrix[i] - cpu_matrix[i]);
        if (diff > 1e-3){
            nb_errors++;
        }
    }
    return nb_errors;
}


int main(int argc, char *argv[]){
    float *A_d, *B_d, *C_d;
    int M, K, N;
    int nb_errors;

    if (argc >= 4){
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    } else {
        M = 1000;
        K = 1000;
        N = 1000;
    }

    dim3 blockSize(TILE_WIDTH,TILE_WIDTH);
    dim3 gridSize((N+blockSize.x-1)/blockSize.x, (M+blockSize.y-1)/blockSize.y);


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
    CUDA_CHECK(cudaMemset(C_d, 0, size_C));

    printf("[TILED] M=%d, K=%d, N=%d ",M,K,N);
    matmulTiledKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost));

    nb_errors = compare_matrices(C, C_ref, M, N);
    
    if (nb_errors == 0){
            printf(" PASSED\n");
        } else {
            printf(" FAILED: %d errors\n", nb_errors);
        }

    printf("[NAIVE] M=%d, K=%d, N=%d ",M,K,N);
    CUDA_CHECK(cudaMemset(C_d, 0, size_C));
    
    matmulNaiveKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost));

    nb_errors = compare_matrices(C, C_ref, M, N);
    
    if (nb_errors == 0){
        printf(" PASSED\n");
    } else {
        printf(" FAILED: %d errors\n", nb_errors);
    }


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);
    free(C_ref);
    return (nb_errors==0) ? 0: 1;
}

