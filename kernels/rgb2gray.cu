#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>



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
#define CHANNELS  3


int compare_doubles(const void *a, const void *b) {
    double val_a = *(const double *)a;
    double val_b = *(const double *)b;

    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}

__global__
void rgb2gray(const unsigned char *Pin, unsigned char *Pout, 
               int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int index = row * width + col;
        unsigned char r = Pin[index * CHANNELS];
        unsigned char g = Pin[index * CHANNELS + 1];
        unsigned char b = Pin[index * CHANNELS + 2];
        Pout[index] = r * 0.21f + g * 0.72f + b * 0.07f;
    }
}

int main(int argc, char *argv[]){
    int K = 1000;
    int width, height;
    if(argc<3){
        width = 1000;
        height = 600;
    } else {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        if (width <= 0 || height <=0) {
            fprintf(stderr, "The image's dimensions must be > 0: got (%dx%d)\n",
            width, height);
            return 1;
        }
    }
    float time;
    unsigned char *Pin_d, *Pout_d;
    unsigned char *Pin_h, *Pout_h;
    
    cudaEvent_t start, stop;
    size_t size = width * height * sizeof(unsigned char);
    
    dim3 blockSize;
    if(argc<5){
        blockSize = dim3(16,16);
    } else {
        int block_x = atoi(argv[3]);
        int block_y = atoi(argv[4]);
        if (block_x <= 0 || block_y <= 0) {
            fprintf(stderr, "The block's dimensions must be > 0 (got %dx%d)\n",
                    block_x, block_y);
            return 1;
        }
        
        if (block_x * block_y > 1024){
            fprintf(stderr, "too many threads in the block: got %d > 1024\n",
                    block_x * block_y);
            return 1;
            
        }
        
            
        blockSize = dim3(block_x, block_y);
    } 
    dim3 gridSize((width + blockSize.x - 1)/ blockSize.x,
                  (height + blockSize.y - 1)/ blockSize.y);
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    Pin_h = (unsigned char*)malloc(CHANNELS * size);
    if (Pin_h == NULL) {fprintf(stderr,"ERROR ALLOCATION\n"); return 1;}
    Pout_h = (unsigned char*)malloc(size);
    if (Pout_h == NULL) {fprintf(stderr,"ERROR ALLOCATION\n"); return 1;}
    
    
    for (size_t i = 0; i < CHANNELS * size; i++) {
        Pin_h[i] = rand() % 256;
    }
    
    
    CUDA_CHECK(cudaMalloc((void**) &Pin_d, CHANNELS*size));
    CUDA_CHECK(cudaMalloc((void**) &Pout_d, size));
    
    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, CHANNELS * size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start,0));
    rgb2gray<<<gridSize, blockSize>>>(Pin_d, Pout_d, width, height);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaEventRecord(stop,0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time,start,stop));
    
    double *durations = (double*)calloc(K, sizeof(double));
    if (durations == NULL) {fprintf(stderr,"ERROR ALLOCATION\n"); return 1;}
    
    for(int i=0;i<K;i++){
        CUDA_CHECK(cudaEventRecord(start,0));
        rgb2gray<<<gridSize, blockSize>>>(Pin_d, Pout_d, width, height);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaEventRecord(stop,0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time,start,stop));
        durations[i] = time;
    }
    qsort(durations, K, sizeof(double), compare_doubles);
    double p_50 = durations[K/2];
    double p_99 = durations[K*99/100];
    
    
    CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));
    
    
    int failed = 0;
    int errors = 0;
    float gray, dist;
    float max_dist = 0;
    for(size_t i=0; i < size; i++){
        gray = 0.21f * Pin_h[i*CHANNELS] + 0.72f * Pin_h[i*CHANNELS + 1] + 0.07f * Pin_h[i*CHANNELS + 2];
        dist = fabsf(Pout_h[i] - gray); 
        if (dist > max_dist){
            max_dist = dist;
        }
        if(dist > 1 ) {
            failed = 1;
            errors++;
        }
    }
    printf("Number of errors: %d and max error is %.5f\n", errors, max_dist);
    if (failed == 0){
        double bandwidth = 4.0 * width * height / (p_50/1000) / 1e9;
        printf("OK | bandwidth : %.2f GB/s | p50_ms = %.4f | p99_ms = %.4f "
               "| blockSize(%d,%d) | image_size = %dx%d\n",
               bandwidth, p_50, p_99, blockSize.x, blockSize.y, width, height);
        printf("%% of peak BW 320 GB/s: %.1f%% \n", bandwidth/320*100);
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));
    free(Pin_h);
    free(Pout_h);
    free(durations);
    return failed ? 1 : 0;
}