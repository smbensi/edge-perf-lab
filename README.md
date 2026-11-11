The goal of this repo is to compare performance of different models for ASR running on the device.
We mean by performance: accuracy, speed, memory used, time to load, time to process 
The main language studied will be : English, French, Hebrew and Hindi


# computation of vecadd on T4 

| N   | Kernel p50 | Go/s | % of max 320 Go/s |
|----:|-----------:|-----:|---------------:|
| 1M  | 0,05 ms    | 235  | 73 %           |
| 16M | 0,75 ms    | 257  | 80 %           |
| 50M | 2,33 ms    | 257  | 80 %           |
| 64M | 2,98 ms    | 258  | 81 %           |
