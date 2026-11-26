# H2D and D2H (T4 colab)
| Transfer (200 MB, T4, K=1000) | p50 ms | p99 ms | GB/s @p50 | GB/s @p99 latency | % of PCIe ceiling (p50) |
|---|---|---|---|---|---|
| H2D pageable | 42.96 | 53.04 | 4.66 | 3.77 | 29.6 % |
| D2H pageable | 43.08 | 51.50 | 4.64 | 3.88 | 29.5 % |
| H2D pinned   | 16.19 | 16.23 | 12.35 | 12.32 | 78.4 % |
| D2H pinned   | 15.25 | 15.30 | 13.12 | 13.07 | 83.3 % |
| Cold first copy (single run) | 131.97 | — | 1.52 | — | 9.7 % |

PCIe ceiling = 15.75 GB/s (PCIe 3.0 x16 theoretical, T4).

# MatMul Naive and Tiled (T4 Colab)

| Kernel | Block | p50 (ms) | p99 (ms) | GFLOPS @p50 | GFLOPS @p99 | Predicted (GFLOPS) | Measured / predicted | Roofline ceiling (GFLOPS) | % of ceiling | % of FP32 peak (8.1 TFLOPS) |
|:-------|:-----:|---------:|---------:|------------:|------------:|-------------------:|---------------------:|--------------------------:|-------------:|----------------------------:|
| naive  | 16×16            | 4.1096 | 9.1853 | 522.6 | 233.8 | 228   | 2.29× | 285   | 183 % | 6.5 % |
| tiled  | 16×16            | 3.4916 | 3.5712 | 615.0 | 601.3 | 1024  | 0.60× | 1280  | 48 %  | 7.6 % |
| naive  | 32×8             | 3.3845 | 7.1032 | 634.5 | 302.3 | 124   | 5.12× | 155   | 409 % | 7.8 % |
| tiled  | 32×8 (TK=8)      | 4.3085 | 4.3831 | 498.4 | 489.9 | 819\* | 0.61× | 1024\* | 49 %  | 6.2 % |

\* Not in the prediction table: TK = 8 gives an intensity of 2·8·256 / ((8·8 + 8·32)·4) = 3.2 FLOP/B, so a ceiling of 3.2 × 320 = 1024 GFLOPS and 80 % of it = 819 GFLOPS.
