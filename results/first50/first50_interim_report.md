# First-50 Interim Report

Selection: first 50 benchmark samples for each split-task, keeping failed generations in the denominator for success rate.

## Real

| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 50/50 (100.0%) | 18062.9 | 59.16 | 640.08 | 0.2739 | 3.40 | 4.24 | - | - | - |
| Text-Image-to-Lottie | 50/50 (100.0%) | 20197.9 | 66.11 | 690.47 | 0.2745 | 4.20 | 4.76 | - | - | - |
| Video-to-Lottie | 50/50 (100.0%) | 22086.4 | 62.53 | 635.48 | - | - | - | 0.81 | 0.034 | 0.627 |

## Synthetic

| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 50/50 (100.0%) | 12716.0 | 37.69 | 608.53 | 0.2726 | 3.92 | 2.68 | - | - | - |
| Text-Image-to-Lottie | 50/50 (100.0%) | 16670.0 | 54.28 | 693.41 | 0.2827 | 5.36 | 3.16 | - | - | - |
| Video-to-Lottie | 49/50 (98.0%) | 33211.6 | 108.95 | 719.88 | - | - | - | 1.16 | 0.038 | 0.482 |

## Table 1 Reference

### Real

| Task | Time | Tokens | Success | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 65.76 | 13.4 | 88.3% | 202.14 | 0.3029 | 4.62 | 5.89 | - | - | - |
| Text-Image-to-Lottie | 65.55 | 16.5 | 93.3% | 180.27 | 0.2997 | 4.96 | 4.24 | - | - | - |
| Video-to-Lottie | 111.9 | 40.2 | 88.1% | 227.11 | - | - | - | 16.08 | 0.82 | 0.92 |

### Synthetic

| Task | Time | Tokens | Success | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 37.93 | 13.4 | 82.1% | 206.35 | 0.2748 | 4.31 | 5.63 | - | - | - |
| Text-Image-to-Lottie | 84.8 | 16.3 | 92.9% | 225.45 | 0.2666 | 4.44 | 3.98 | - | - | - |
| Video-to-Lottie | 109.53 | 41.4 | 80.7% | 342.65 | - | - | - | 15.76 | 0.79 | 0.88 |

