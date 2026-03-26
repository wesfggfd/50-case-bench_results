# Final 150 Metrics

Time/Tokens/Success/FVD/CLIP/Obj/Motion come from the original pipeline. PSNR/SSIM/DINO are refreshed from the corrected video-pair metric implementation.

## Real

| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 150/150 (100.0%) | 20460.9 | 59.36 | 539.20 | 0.2654 | 3.36 | 3.17 | - | - | - |
| Text-Image-to-Lottie | 148/150 (98.7%) | 22112.3 | 68.87 | 532.76 | 0.2703 | 4.03 | 3.93 | - | - | - |
| Video-to-Lottie | 150/150 (100.0%) | 20763.8 | 61.16 | 539.98 | - | - | - | 12.70 | 0.788 | 0.771 |

## Synthetic

| Task | Success | Tokens | Time/sample | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 150/150 (100.0%) | 12882.0 | 38.60 | 517.47 | 0.2742 | 4.46 | 3.21 | - | - | - |
| Text-Image-to-Lottie | 150/150 (100.0%) | 17571.4 | 55.99 | 564.77 | 0.2775 | 6.44 | 3.77 | - | - | - |
| Video-to-Lottie | 149/150 (99.3%) | 38741.5 | 118.90 | 567.36 | - | - | - | 12.26 | 0.834 | 0.563 |

