The goal of this repo is to compare performance of different models for ASR running on the device.
We mean by performance: accuracy, speed, memory used, time to load, time to process 
The main language studied will be : English, French, Hebrew and Hindi


# vecadd on T4 (Colab)



### Method
- Hardware: T4 (colab) date: 2026-11-11
- Kernel: C[i] = A[i] + B[i], float32, block size 256, grid = ceil(N / block size)
- Timing: cudaEvent around each launch, 1 warmup launch, K = 1000 timed runs
- Percentiles: p50 = sorted[K/2], p99 = sorted[K*99/100] (no interpolation,  np.percentile)
- Bandwidth: bytes moved = 3 × N × 4 (2 reads + 1 write) / p50
- Peak reference: 320 GB/s (T4 datasheet)
- H2D / D2H: single cudaMemcpy, pageable host memory (malloc)
- Run: nvcc -O3 -o vecadd kernels/vecadd.cu && ./vecadd <N>

### Results

| N   | Kernel p50 | Kernel p99 | GB/s  | % of max 320 GB/s | H2D       | D2H       |
|----:|-----------:|-----------:|------:|------------------:|----------:|----------:|
| 1M  | 0.0512 ms  | 0.0553 ms  | 234.4 | 73.2 %            | 1.97 ms   | 2.79 ms   |
| 16M | 0.7498 ms  | 0.7721 ms  | 256.1 | 80.0 %            | 32.50 ms  | 50.42 ms  |
| 50M | 2.3266 ms  | 2.3400 ms  | 257.9 | 80.6 %            | 85.83 ms  | 138.74 ms |
| 64M | 2.9757 ms  | 2.9921 ms  | 258.1 | 80.7 %            | 117.12 ms | 175.36 ms |


### Analysis
From this table we can see that even though the max bandwidth of the T4 is given as 320 GB/s we manage to get only ~80% of it
Also we can see the huge price of transferring data between host and device it's a factor of ~95x compared to the kernel time 

The interesting point is when N=1M we get only 73% and we can conclude that there are some fixed costs that are responsible for it
At 257 GB/s
for N = 1M , 12MB should take 3*N*4/257e9~46 us but we measure ~50us so ~4 µs is a fixed cost (GPU-side kernel start-up and tail effect). 


# rgb2gray on T4 (Colab)

Converts an RGB image (3 bytes/pixel) to grayscale (1 byte/pixel)
with one thread per pixel, using a 2D grid

### Method

- GPU: T4 model colab, peak memory bandwidth 320 GB/s
- Random input image, result checked against CPU reference (tolerance 1)
- 1 warmup launch, then K=1000 times launches with cudaEvent
- effective bandwidth = 4 (3 read + 1 write) * width * height / p50



### Results

| Image size | Block size | p50       | p99       | GB/s  | % of max 320 GB/s | 
|-----------:|-----------:|----------:|----------:|------:|------------------:|
| 1000x600   | (16,16)    | 0.0195 ms | 0.0250 ms | 123.4 | 38.5 %            |
| 1000x600   | (32,8)     | 0.0164 ms | 0.0189 ms | 146.8 | 45.9 %            |
| 4000x3000  | (16,16)    | 0.3287 ms | 0.3724 ms | 146.0 | 45.6 %            |
| 4000x3000  | (32,8)     | 0.2705 ms | 0.2780 ms | 177.5 | 55.5 %            |


### Takeaways

1. (32,8) blocks are ~20 % faster than (16,16) (+19 % on 1000×600, +22 % on 4000×3000), with the same number of threads per block. Hypothesis: 32-wide blocks give more contiguous memory accesses (memory coalescing). To be confirmed after studying warps (PMPP ch. 4) and with ncu in the next PR.

2. the small image got only 46% . In this image we deplace 2.4MB at the speed of the big image 177.5GB/s we would have get 13.5us but now we have 18.5 so it must be the fixed costs of launching the kernel and dispatch the work to SMs and the closing that takes around 5us

3. The large image plateaus at 55.5 % of peak, vs 80 % for vecAdd, so the kernel itself is less efficient than vecAdd, not just the fixed cost. Open question, to be checked with ncu in the next PR.

### How to run
nvcc -O3 -o rgb2gray rgb2gray.cu
./rgb2gray <width> <height> [block_x block_y]



# PINNED TRANSFERS (T4 on Colab)


### prediction 
we have seen in pageable memory that the performance for transfer is around 30% of the maximal theory number with pinned memory we can achieve 80% of the max

| Transfer | Pageable (measured) | Pinned (predicted) | Predicted gain |
|:---------|---------------------------:|-------------------:|---------------:|
| H2D      | 4.66 GB/s                  | 12.6 GB/s          | ≈ 2.7×         |
| D2H      | 1.44 GB/s                  | 12.6 GB/s          | ≈ 8.7×         |

PCIe Gen3 x16 ceiling: 15.75 GB/s
the 1.44 GB/s D2H was a cold first copy from PR#4, not steady-state pageable; warm pageable D2H = 4.64 GB/s

### content

we measured the data bandwidth between host and device in both directions using a T4 gpu running in Colab

### method

K = 1000
memory copy = 200 MB 
one warmup
measure p50 and p99 using cuda events


### Results (measured)

Transfer size: 200 MB (bandwidth × p50).

| Transfer (200 MB, T4, K=1000) | p50 ms | p99 ms | GB/s @p50 | GB/s @p99 latency | % of PCIe ceiling (p50) |
|---|---|---|---|---|---|
| H2D pageable | 42.96 | 53.04 | 4.66 | 3.77 | 29.6 % |
| D2H pageable | 43.08 | 51.50 | 4.64 | 3.88 | 29.5 % |
| H2D pinned   | 16.19 | 16.23 | 12.35 | 12.32 | 78.4 % |
| D2H pinned   | 15.25 | 15.30 | 13.12 | 13.07 | 83.3 % |
| Cold first copy (single run) | 131.97 | — | 1.52 | — | 9.7 % |


The T4's PCIe Gen3 x16 theoretical peak is 15.75 GB/s per direction
In our tests we measured
For pageable memory , p50=4.66 GB/s that is around 30% of peak
For pinned memory p50=12.35 GB/s  , around 80% of peak
For cold copy, single run = 1.52 GB/s it's very slow because of all the work to allocate physical pages on first touch (page faults)


### conclusion

1. We can observe the huge difference between pageable and pinned memory (2.7x faster 12.35GB/s vs 4.66GB/s) due to the need to create a staging copy in pageable memory to let the DMA access the CPU memory to copy to the GPU. This copy is not needed in pinned memory but it's important to avoid overusing it because pinned pages cannot be swapped out so too much pinned memory takes RAM away from the OS.
2. the pageable memory is irregular we have big difference between p50 and p99 (for H2D pageable 4.66GB/s at p50 latency and 3.77GB/s at p99 latency) in pinned we don't see this irregularity (H2D pinned 12.35GB/s at p50 latency and 12.32GB/s at p99 latency)
3. the cold copy doesn't reflect the reality so it explains the low bandwidth observed in PR#4 which was measured without warmup

### Instruction

`nvcc -arch=sm_75 transfers.cu -o transfers && ./transfers`

# Matrix Multiplication

We will compare a naive and a tiled matrix multiplication implementations using CUDA

### predictions

For a NxNxN matrix multiplications we compute 2*N^3 operations.
For the naive implementation, we load for each thread one line from matrix A and one column from matrix B so we will load 2*4(float)*N Bytes for each thread and we will compute 2*N FLOPS so our compute intensity is 0.25 FLOP/B.
Nevertheless , we know that for each warp if we have a block size of 16x16, in the same warp we will load the 2 rows of A (broadcast) and 16  columns of B shared by both half-warps so the number of loads in the warp will be (2+16)*4*N bytes and the operations number will be 32*2*N so we have a compute intensity of 0.88 FLOP/B.
In the tiled implementation we load each block in the shared memory so we divide the number of global memory access by the width of the tiled matrix so if TILED_WIDTH=16 we'll get a compute intensity of 4 FLOP/B 


### Prediction table (T4: 320 GB/s, 8.1 TFLOPS FP32, ridge point ≈ 25 FLOP/B)

Hypothesis: each kernel reaches ~80 % of its roofline ceiling, as vecAdd did (257 / 320 GB/s).
The model does not depend on N, so the same prediction holds for N = 1024 and N = 2048.

| Kernel | Block | Model | Intensity (FLOP/B) | Roofline ceiling (GFLOPS) | Predicted (GFLOPS) |
|:-------|:-----:|:------|-------------------:|--------------------------:|-------------------:|
| naive  | 16×16 | per thread, no reuse (1st guess)          | 0.25 |   80 |   64 |
| naive  | 32×8  | per warp, A broadcast, B coalesced        | 0.48 |  155 |  124 |
| naive  | 16×16 | per warp, 2 rows of A, B shared by halves | 0.89 |  285 |  228 |
| tiled  | 16×16 | global loads ÷ TILE_WIDTH (16)            | 4    | 1280 | 1024 |

All kernels are memory-bound (intensity < 25 FLOP/B).

Without the 2nd `__syncthreads()` (5 runs): prediction = 5 wrong runs beause of the write-after-read


### Results

| Kernel | Block | p50 (ms) | p99 (ms) | GFLOPS @p50 | GFLOPS @p99 | Predicted (GFLOPS) | Measured / predicted | Roofline ceiling (GFLOPS) | % of ceiling | % of FP32 peak (8.1 TFLOPS) |
|:-------|:-----:|---------:|---------:|------------:|------------:|-------------------:|---------------------:|--------------------------:|-------------:|----------------------------:|
| naive  | 16×16            | 4.1096 | 9.1853 | 522.6 | 233.8 | 228   | 2.29× | 285   | 183 % | 6.5 % |
| tiled  | 16×16            | 3.4916 | 3.5712 | 615.0 | 601.3 | 1024  | 0.60× | 1280  | 48 %  | 7.6 % |
| naive  | 32×8             | 3.3845 | 7.1032 | 634.5 | 302.3 | 124   | 5.12× | 155   | 409 % | 7.8 % |
| tiled  | 32×8 (TK=8)      | 4.3085 | 4.3831 | 498.4 | 489.9 | 819\* | 0.61× | 1024\* | 49 %  | 6.2 % |

\* Not in the prediction table: TK = 8 gives an intensity of 2·8·256 / ((8·8 + 8·32)·4) = 3.2 FLOP/B, so a ceiling of 3.2 × 320 = 1024 GFLOPS and 80 % of it = 819 GFLOPS.

### instruction

!nvcc -arch=sm_75 matmul.cu -o matmul && ./matmul 1024 32 8
