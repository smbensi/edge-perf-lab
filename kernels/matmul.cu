#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32

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

__global__ void matmulKernel(float const *A, float const *B, float *C, int M, int K, int N){

}

void matmulRef(float *A, float *B, float *c, int M, int K, int N){
    for(int i=0; i<M, i+++){
        for(int j=0; j<N; ++j){
            for(int k=0; k<K; ++k){
                
            }
        }
    }
}

int main(int argc, char argv[]){
    float *C;
    float *A_d, *B_d, *C_d;
    int M, K, N;
    if (argc > 4){
        M = argv[1];
        K = argv[2];
        N = argv[3];
    } else {
        M = 1000;
        K = 1000;
        N = 1000;
    }

    size_t size_A = M*K*sizeof(float);
    size_t size_B = N*K*sizeof(float);

    float *A = (float *)malloc(size_A);
    float *B = (float *)malloc(size_B);

}

