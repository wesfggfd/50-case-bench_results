# OmniLottie Table 1 Reproduction Report

## Final 150 Summary
- Recomputed with the corrected PSNR/SSIM/DINO implementation and the user-specified FVD sampling protocol.
- Obj/Motion use the user-specified `https://ai.hhhl.cc` + `claude-sonnet-3-5-latest` local judge.
- `judge_count` can be lower than `success_count` when a task has fewer successful text-task generations in the benchmark results.

### Real

| Task | Time | Tokens | Success | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 59.3581 | 20460.8867 | 100.0000 | 539.1959 | 0.2654 | 3.3600 | 3.1667 | null | null | null |
| Text-Image-to-Lottie | 68.8690 | 22112.2838 | 98.6667 | 532.7569 | 0.2703 | 4.0272 | 3.9320 | null | null | null |
| Video-to-Lottie | 61.1554 | 20763.8467 | 100.0000 | 539.9755 | null | null | null | 12.6965 | 0.7881 | 0.7709 |

### Synthetic

| Task | Time | Tokens | Success | FVD | CLIP | Obj | Motion | PSNR | SSIM | DINO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-Lottie | 38.6040 | 12882.0467 | 100.0000 | 517.4699 | 0.2742 | 4.4600 | 3.2067 | null | null | null |
| Text-Image-to-Lottie | 55.9932 | 17571.4200 | 100.0000 | 564.7740 | 0.2775 | 6.4400 | 3.7733 | null | null | null |
| Video-to-Lottie | 118.8951 | 38741.5235 | 99.3333 | 567.3586 | null | null | null | 12.2612 | 0.8339 | 0.5635 |

## Comparison Against PDF Table 1
| Subset | Task | Metric | Paper Table 1 | Reproduction | Note |
| --- | --- | --- | ---: | ---: | --- |
| real | Text-to-Lottie | time | 65.7600 | 59.3581 | numeric comparison |
| real | Text-to-Lottie | tokens | 13.4000 | 20460.8867 | numeric comparison |
| real | Text-to-Lottie | success | 88.3000 | 100.0000 | numeric comparison |
| real | Text-to-Lottie | fvd | 202.1400 | 539.1959 | numeric comparison |
| real | Text-to-Lottie | clip | 0.3029 | 0.2654 | numeric comparison |
| real | Text-to-Lottie | obj | 4.6200 | 3.3600 | numeric comparison |
| real | Text-to-Lottie | motion | 5.8900 | 3.1667 | numeric comparison |
| real | Text-Image-to-Lottie | time | 65.5500 | 68.8690 | numeric comparison |
| real | Text-Image-to-Lottie | tokens | 16.5000 | 22112.2838 | numeric comparison |
| real | Text-Image-to-Lottie | success | 93.3000 | 98.6667 | numeric comparison |
| real | Text-Image-to-Lottie | fvd | 180.2700 | 532.7569 | numeric comparison |
| real | Text-Image-to-Lottie | clip | 0.2997 | 0.2703 | numeric comparison |
| real | Text-Image-to-Lottie | obj | 4.9600 | 4.0272 | numeric comparison |
| real | Text-Image-to-Lottie | motion | 4.2400 | 3.9320 | numeric comparison |
| real | Video-to-Lottie | time | 111.9000 | 61.1554 | numeric comparison |
| real | Video-to-Lottie | tokens | 40.2000 | 20763.8467 | numeric comparison |
| real | Video-to-Lottie | success | 88.1000 | 100.0000 | numeric comparison |
| real | Video-to-Lottie | fvd | 227.1100 | 539.9755 | numeric comparison |
| real | Video-to-Lottie | psnr | 16.0800 | 12.6965 | numeric comparison |
| real | Video-to-Lottie | ssim | 0.8200 | 0.7881 | numeric comparison |
| real | Video-to-Lottie | dino | 0.9200 | 0.7709 | numeric comparison |
| synthetic | Text-to-Lottie | time | 37.9300 | 38.6040 | numeric comparison |
| synthetic | Text-to-Lottie | tokens | 13.4000 | 12882.0467 | numeric comparison |
| synthetic | Text-to-Lottie | success | 82.1000 | 100.0000 | numeric comparison |
| synthetic | Text-to-Lottie | fvd | 206.3500 | 517.4699 | numeric comparison |
| synthetic | Text-to-Lottie | clip | 0.2748 | 0.2742 | numeric comparison |
| synthetic | Text-to-Lottie | obj | 4.3100 | 4.4600 | numeric comparison |
| synthetic | Text-to-Lottie | motion | 5.6300 | 3.2067 | numeric comparison |
| synthetic | Text-Image-to-Lottie | time | 84.8000 | 55.9932 | numeric comparison |
| synthetic | Text-Image-to-Lottie | tokens | 16.3000 | 17571.4200 | numeric comparison |
| synthetic | Text-Image-to-Lottie | success | 92.9000 | 100.0000 | numeric comparison |
| synthetic | Text-Image-to-Lottie | fvd | 225.4500 | 564.7740 | numeric comparison |
| synthetic | Text-Image-to-Lottie | clip | 0.2666 | 0.2775 | numeric comparison |
| synthetic | Text-Image-to-Lottie | obj | 4.4400 | 6.4400 | numeric comparison |
| synthetic | Text-Image-to-Lottie | motion | 3.9800 | 3.7733 | numeric comparison |
| synthetic | Video-to-Lottie | time | 109.5300 | 118.8951 | numeric comparison |
| synthetic | Video-to-Lottie | tokens | 41.4000 | 38741.5235 | numeric comparison |
| synthetic | Video-to-Lottie | success | 80.7000 | 99.3333 | numeric comparison |
| synthetic | Video-to-Lottie | fvd | 342.6500 | 567.3586 | numeric comparison |
| synthetic | Video-to-Lottie | psnr | 15.7600 | 12.2612 | numeric comparison |
| synthetic | Video-to-Lottie | ssim | 0.7900 | 0.8339 | numeric comparison |
| synthetic | Video-to-Lottie | dino | 0.8800 | 0.5635 | numeric comparison |

## Notes
- These tables summarize the selected 150 benchmark results for each subset-task, so success-related metrics still reflect generation failures inside that selection.
- `real / Text-Image-to-Lottie` shows `judge_count=147` because only 147 successful generated outputs were available for text-task judging in the selected benchmark results.
- `pair_count` and generated feature counts can be below 150 for video metrics when individual generated outputs fail decoding or rendering during metric computation.
