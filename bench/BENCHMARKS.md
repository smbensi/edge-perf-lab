# H2D and D2H (T4 colab)
| Transfer (200 MB, T4, K=1000) | p50 ms | p99 ms | GB/s @p50 | GB/s @p99 latency | % of PCIe ceiling (p50) |
|---|---|---|---|---|---|
| H2D pageable | 42.96 | 53.04 | 4.66 | 3.77 | 29.6 % |
| D2H pageable | 43.08 | 51.50 | 4.64 | 3.88 | 29.5 % |
| H2D pinned   | 16.19 | 16.23 | 12.35 | 12.32 | 78.4 % |
| D2H pinned   | 15.25 | 15.30 | 13.12 | 13.07 | 83.3 % |
| Cold first copy (single run) | 131.97 | — | 1.52 | — | 9.7 % |

PCIe ceiling = 15.75 GB/s (PCIe 3.0 x16 theoretical, T4).
