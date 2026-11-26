#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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



int compare_doubles(const void *a, const void *b) {
    double val_a = *(const double *)a;
    double val_b = *(const double *)b;

    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}


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

__global__ void matmulTiledKernel(float const *A, float const *B, float *C, int M, int K, int N, int TK, unsigned int Ads_sz, unsigned int Bds_sz){
    extern __shared__ float Ads_Bds[];

    float *Ads = Ads_Bds;
    float *Bds = Ads_Bds + Ads_sz/sizeof(float);

    //__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    //__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;

    float Cval;
    Cval = 0;
    for(int ph=0; ph<(K+TK-1)/TK; ++ph){
        if(tx<TK){
            if (row < M && (ph*TK+tx)<K) Ads[ty*TK+tx] = A[row*K+ph*TK+tx]; else Ads[ty*TK+tx] = 0.0f;
        }
        if (col < N && (ph*TK+ty)<K) Bds[ty*blockDim.x+tx] = B[(ph*TK+ty)*N+col]; else Bds[ty*blockDim.x+tx] = 0.0f;
        __syncthreads();
        for (int i=0; i<TK; i++){
            Cval += Ads[ty*TK+i]*Bds[i*blockDim.x+tx];
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

void print_results(float *p_50_99_ms,int N,const char *type){
    printf("PASSED : %s\n", type);
    printf("p_50ms = %.4f , p_99ms = %.4f\n", p_50_99_ms[0], p_50_99_ms[1]);
    double bandwidth_p50 = (2*(long)N*N*N)/(p_50_99_ms[0]/1000)/1e9;
    double bandwidth_p99 = (2*(long)N*N*N)/(p_50_99_ms[1]/1000)/1e9;
    printf(" bandwidth p_50ms = %.4f GFLOPS , bandwidth p_99ms = %.4f GFLOPS \n", bandwidth_p50, bandwidth_p99);

}

int main(int argc, char *argv[]){
    cudaEvent_t start, stop;
    float *A_d, *B_d, *C_d, time;
    int M, K, N, block_x, block_y;
    int nb_errors;
    float p_50_99_ms[2] = {0.0f};
    double p_50, p_99;
    int repetitions = 1000;

    if (argc >= 4){
        M = atoi(argv[1]);
        N = M;
        K = M;
        block_x = atoi(argv[2]);
        block_y = atoi(argv[3]);
    } else {
        M = 1000;
        N = 1000;
        K = 1000;
        block_x = 16;
        block_y = 16;
    }

    dim3 blockSize(block_x, block_y);
    dim3 gridSize((N+blockSize.x-1)/blockSize.x, (M+blockSize.y-1)/blockSize.y);

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

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

    double *latencies = (double*)malloc(repetitions * sizeof(double));
    if (latencies == NULL) {fprintf(stderr,"ERROR ALLOCATION\n"); return cudaErrorMemoryAllocation;}


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


    printf("[NAIVE] M=%d, K=%d, N=%d block_size=(%d,%d)\n",M,K,N, blockSize.x, blockSize.y);
        CUDA_CHECK(cudaMemset(C_d, 0, size_C));
        memset(latencies, 0, repetitions*sizeof(double));
        // warmup
        matmulNaiveKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);

        for (int i=0; i<repetitions; i++){
            CUDA_CHECK(cudaEventRecord(start, 0));
            matmulNaiveKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
            latencies[i] = time;
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost));
        }
        nb_errors = compare_matrices(C, C_ref, M, N);
        qsort(latencies, repetitions, sizeof(double), compare_doubles);
        p_50 = latencies[repetitions/2];
        p_99 = latencies[repetitions*99/100];
        p_50_99_ms[0] = p_50;
        p_50_99_ms[1] = p_99;
        if (nb_errors == 0){
            // printf(" PASSED\n");
            print_results(p_50_99_ms,N,"NAIVE");

        } else {
            printf(" FAILED: %d errors\n", nb_errors);
        }

    printf("[TILED] M=%d, K=%d, N=%d block_size=(%d,%d)\n",M,K,N, blockSize.x, blockSize.y);
    //warmup
    int TK = min(blockSize.x,blockSize.y);
    if (blockSize.x >= blockSize.y){
        size_t size_Ads = blockSize.y*TK*sizeof(float);
        size_t size_Bds = TK*blockSize.x*sizeof(float);
        size_t shared_size = size_Ads + size_Bds;
        matmulTiledKernel<<<gridSize, blockSize, shared_size>>>(A_d, B_d, C_d, M, K, N, TK, size_Ads, size_Bds);

        for (int i=0; i<repetitions; i++){
            CUDA_CHECK(cudaEventRecord(start, 0));
            matmulTiledKernel<<<gridSize, blockSize, shared_size>>>(A_d, B_d, C_d, M, K, N, TK, size_Ads, size_Bds);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
            latencies[i] = time;
            CUDA_CHECK(cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost));
        }
        nb_errors = compare_matrices(C, C_ref, M, N);
        qsort(latencies, repetitions, sizeof(double), compare_doubles);
        p_50 = latencies[repetitions/2];
        p_99 = latencies[repetitions*99/100];
        p_50_99_ms[0] = p_50;
        p_50_99_ms[1] = p_99;

        if (nb_errors == 0){
                // printf(" PASSED\n");
                print_results(p_50_99_ms,N,"TILED");
            } else {
                printf(" FAILED: %d errors\n", nb_errors);
            }

    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(latencies);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);
    free(C_ref);
    return (nb_errors==0) ? 0: 1;
}
