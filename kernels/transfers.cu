#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

cudaError_t measureCudaMemcpy(void *dest, void const *source, size_t size_bytes, cudaMemcpyKind direction, int repetitions, float* p_50_99_ms){
    if(repetitions <= 0){
        fprintf(stderr,"repetitions must be > 0\n"); return cudaErrorInvalidValue;
    }

    if(source == NULL){
        fprintf(stderr,"source is NULL\n"); return cudaErrorInvalidValue;
    }
    if(dest == NULL){
        fprintf(stderr,"dest is NULL\n"); return cudaErrorInvalidValue;
    }
    if(p_50_99_ms == NULL){
        fprintf(stderr,"p_50_99_ms is NULL\n"); return cudaErrorInvalidValue;
    }

    double *latencies = (double*)malloc(repetitions * sizeof(double));
    if (latencies == NULL) {fprintf(stderr,"ERROR ALLOCATION\n"); return cudaErrorMemoryAllocation;}

    float time;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // warmup
    CUDA_CHECK(cudaMemcpy(dest, source, size_bytes, direction));


    for(int i=0; i< repetitions; i++){
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(dest, source, size_bytes, direction));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        latencies[i] = time;
    }
    qsort(latencies, repetitions, sizeof(double), compare_doubles);
    double p_50 = latencies[repetitions/2];
    double p_99 = latencies[repetitions*99/100];
    p_50_99_ms[0] = p_50;
    p_50_99_ms[1] = p_99;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(latencies);

    return cudaSuccess;
}

void print_results(float *p_50_99_ms,size_t size,const char *type){
    printf("Transfer type : %s\n", type);
    printf("p_50ms = %.4f , p_99ms = %.4f\n", p_50_99_ms[0], p_50_99_ms[1]);
    float bandwidth_p50 = size/(p_50_99_ms[0]/1000)/1e9;
    float bandwidth_p99 = size/(p_50_99_ms[1]/1000)/1e9;
    printf(" bandwidth p_50ms = %.4f GB/s , bandwidth p_99ms = %.4f GB/s \n", bandwidth_p50, bandwidth_p99);

}

int main(int argc, char *argv[]){
    int N = 50'000'000;
    int K = 1000;
    size_t size = N*sizeof(float);
    float p_50_99_ms[2] = {0.0f};
    cudaError_t cuda_success;

    float *source = (float*)malloc(size);
    if (source==NULL) {fprintf(stderr, "ERROR IN MALLOC ALLOCATION\n"); exit(EXIT_FAILURE);}
    memset(source, 0, N*sizeof(float));



    float *source_d;
    CUDA_CHECK(cudaMalloc((void**) &source_d, size));

    cudaEvent_t start, stop;
    float *fresh = (float*)malloc(size);
    if (fresh==NULL) {fprintf(stderr, "ERROR IN MALLOC ALLOCATION\n"); exit(EXIT_FAILURE);}

    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(fresh, source_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    float cold_bw = size / (time / 1e3) / 1e9;
    printf("elapsed time for cold copy: %.4f ms and bandwitdh is %.4f GB/s\n", time, cold_bw);

    // H2D PAGEABLE
    cuda_success = measureCudaMemcpy(source_d, source, size, cudaMemcpyHostToDevice, K, p_50_99_ms);
    if (cuda_success != cudaSuccess){
        exit(EXIT_FAILURE);
    }
    print_results(p_50_99_ms, size, "H2D PAGEABLE");

    // D2H PAGEABLE
    cuda_success = measureCudaMemcpy(source, source_d, size, cudaMemcpyDeviceToHost, K, p_50_99_ms);
    if (cuda_success != cudaSuccess){
        exit(EXIT_FAILURE);
    }
    print_results(p_50_99_ms, size, "D2H PAGEABLE");



    // H2D PINNED
    float *source_pinned;
    CUDA_CHECK(cudaMallocHost((void **)&source_pinned, size));
    memset(source_pinned, 0, N*sizeof(float));

    cuda_success = measureCudaMemcpy(source_d, source_pinned, size, cudaMemcpyHostToDevice, K, p_50_99_ms);
    if (cuda_success != cudaSuccess){
        exit(EXIT_FAILURE);
    }
    print_results(p_50_99_ms, size, "H2D PINNED");


    // D2H pinned
    cuda_success = measureCudaMemcpy(source_pinned, source_d, size, cudaMemcpyDeviceToHost, K, p_50_99_ms);
    if (cuda_success != cudaSuccess){
        exit(EXIT_FAILURE);
    }
    print_results(p_50_99_ms, size, "D2H PINNED");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(source);
    cudaFreeHost(source_pinned);
    free(fresh);
    cudaFree(source_d);

    return 0 ;
}
