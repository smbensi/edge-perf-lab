The goal of this repo is to compare performance of different models for ASR running on the device.
We mean by performance: accuracy, speed, memory used, time to load, time to process 
The main language studied will be : English, French, Hebrew and Hindi


# computation of vecadd on T4 colab

| N   | Kernel p50 | Kernel p99 | GB/s  | % of max 320 GB/s | H2D       | D2H       |
|----:|-----------:|-----------:|------:|------------------:|----------:|----------:|
| 1M  | 0.0512 ms  | 0.0553 ms  | 234.4 | 73.2 %            | 1.97 ms   | 2.79 ms   |
| 16M | 0.7498 ms  | 0.7721 ms  | 256.1 | 80.0 %            | 32.50 ms  | 50.42 ms  |
| 50M | 2.3266 ms  | 2.3400 ms  | 257.9 | 80.6 %            | 85.83 ms  | 138.74 ms |
| 64M | 2.9757 ms  | 2.9921 ms  | 258.1 | 80.7 %            | 117.12 ms | 175.36 ms |
