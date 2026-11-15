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


