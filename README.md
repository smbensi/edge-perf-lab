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
for N = 1M , 12MB should take 3*N*4/257e9~46 us but we measure ~50us so ~4 µs is a fixed cost (GPU-side kernel start-up and tail effect). »
