# OmniLottie Table 1 Reproduction Report

## Verdict
- Null-pattern check against Table 1: failed
- Core local metrics were computed from the public benchmark release and generated outputs.
- Obj/Motion were re-evaluated locally with `claude-sonnet-4-6` via `https://aiapi.cxyquan.com/v1/messages`, not the paper's `Claude-3.5-Sonnet`.
- Because the public MMLottieBench release does not expose reference videos for Text-to-Lottie and Text-Image-to-Lottie rows, `FVD` for those rows is unavailable here, while Table 1 reports numeric values.

## Null Pattern Check
- Overall null-pattern match: `False`
- Matching nulls: T2L/TI2L `PSNR/SSIM/DINO`, V2L `CLIP/Obj/Motion`.
- Mismatching nulls: T2L/TI2L `FVD` is `null` here but numeric in Table 1.

## Comparison
| Subset | Task | Metric | Paper Table 1 | Reproduction | Note |
| --- | --- | --- | ---: | ---: | --- |
| real | Text-to-Lottie | time | 33.7100 | 59.3581 | numeric comparison |
| real | Text-to-Lottie | tokens | 21200.0000 | 20460.8867 | numeric comparison |
| real | Text-to-Lottie | success | 88.3000 | 100.0000 | numeric comparison |
| real | Text-to-Lottie | fvd | 202.1400 | null | unavailable from public release or protocol mismatch |
| real | Text-to-Lottie | clip | 0.2748 | 0.2654 | numeric comparison |
| real | Text-to-Lottie | obj | 4.4400 | 3.6333 | numeric comparison |
| real | Text-to-Lottie | motion | 5.9400 | 4.4800 | numeric comparison |
| real | Text-to-Lottie | psnr | null | null | null matches paper |
| real | Text-to-Lottie | ssim | null | null | null matches paper |
| real | Text-to-Lottie | dino | null | null | null matches paper |
| real | Text-Image-to-Lottie | time | 88.5700 | 68.8690 | numeric comparison |
| real | Text-Image-to-Lottie | tokens | 23400.0000 | 22112.2838 | numeric comparison |
| real | Text-Image-to-Lottie | success | 93.3000 | 98.6667 | numeric comparison |
| real | Text-Image-to-Lottie | fvd | 180.2700 | null | unavailable from public release or protocol mismatch |
| real | Text-Image-to-Lottie | clip | 0.2666 | 0.2703 | numeric comparison |
| real | Text-Image-to-Lottie | obj | 5.1000 | 3.7297 | numeric comparison |
| real | Text-Image-to-Lottie | motion | 4.4400 | 3.8784 | numeric comparison |
| real | Text-Image-to-Lottie | psnr | null | null | null matches paper |
| real | Text-Image-to-Lottie | ssim | null | null | null matches paper |
| real | Text-Image-to-Lottie | dino | null | null | null matches paper |
| real | Video-to-Lottie | time | 110.7700 | 61.1554 | numeric comparison |
| real | Video-to-Lottie | tokens | 36800.0000 | 20763.8467 | numeric comparison |
| real | Video-to-Lottie | success | 88.1000 | 100.0000 | numeric comparison |
| real | Video-to-Lottie | fvd | 227.1100 | 612.8099 | numeric comparison |
| real | Video-to-Lottie | clip | null | null | null matches paper |
| real | Video-to-Lottie | obj | null | null | null matches paper |
| real | Video-to-Lottie | motion | null | null | null matches paper |
| real | Video-to-Lottie | psnr | 16.0800 | 0.9646 | numeric comparison |
| real | Video-to-Lottie | ssim | 0.8200 | 0.0509 | numeric comparison |
| real | Video-to-Lottie | dino | 0.9200 | 0.6126 | numeric comparison |
| synthetic | Text-to-Lottie | time | 37.9300 | 38.6040 | numeric comparison |
| synthetic | Text-to-Lottie | tokens | 13400.0000 | 12882.0467 | numeric comparison |
| synthetic | Text-to-Lottie | success | 82.1000 | 100.0000 | numeric comparison |
| synthetic | Text-to-Lottie | fvd | 206.3500 | null | unavailable from public release or protocol mismatch |
| synthetic | Text-to-Lottie | clip | 0.2748 | 0.2742 | numeric comparison |
| synthetic | Text-to-Lottie | obj | 4.3100 | 4.0000 | numeric comparison |
| synthetic | Text-to-Lottie | motion | 5.6300 | 3.0800 | numeric comparison |
| synthetic | Text-to-Lottie | psnr | null | null | null matches paper |
| synthetic | Text-to-Lottie | ssim | null | null | null matches paper |
| synthetic | Text-to-Lottie | dino | null | null | null matches paper |
| synthetic | Text-Image-to-Lottie | time | 84.8000 | 55.9932 | numeric comparison |
| synthetic | Text-Image-to-Lottie | tokens | 16300.0000 | 17571.4200 | numeric comparison |
| synthetic | Text-Image-to-Lottie | success | 92.9000 | 100.0000 | numeric comparison |
| synthetic | Text-Image-to-Lottie | fvd | 225.4500 | null | unavailable from public release or protocol mismatch |
| synthetic | Text-Image-to-Lottie | clip | 0.2666 | 0.2775 | numeric comparison |
| synthetic | Text-Image-to-Lottie | obj | 4.4400 | 5.1667 | numeric comparison |
| synthetic | Text-Image-to-Lottie | motion | 3.9800 | 3.4733 | numeric comparison |
| synthetic | Text-Image-to-Lottie | psnr | null | null | null matches paper |
| synthetic | Text-Image-to-Lottie | ssim | null | null | null matches paper |
| synthetic | Text-Image-to-Lottie | dino | null | null | null matches paper |
| synthetic | Video-to-Lottie | time | 109.5300 | 118.8951 | numeric comparison |
| synthetic | Video-to-Lottie | tokens | 41400.0000 | 38741.5235 | numeric comparison |
| synthetic | Video-to-Lottie | success | 80.7000 | 99.3333 | numeric comparison |
| synthetic | Video-to-Lottie | fvd | 342.6500 | 1143.4498 | numeric comparison |
| synthetic | Video-to-Lottie | clip | null | null | null matches paper |
| synthetic | Video-to-Lottie | obj | null | null | null matches paper |
| synthetic | Video-to-Lottie | motion | null | null | null matches paper |
| synthetic | Video-to-Lottie | psnr | 15.7600 | 1.0611 | numeric comparison |
| synthetic | Video-to-Lottie | ssim | 0.7900 | 0.0228 | numeric comparison |
| synthetic | Video-to-Lottie | dino | 0.8800 | 0.4622 | numeric comparison |

## Notes
- `Time` is estimated from available `_info.txt` timestamps, not a clean 3-run average on a fresh A100 session, so it should be treated as approximate.
- `Tokens` uses the Qwen2.5-VL tokenizer over generated Lottie JSON, matching the paper's metric definition.
- `Obj/Motion` judge model differs from the paper because the user explicitly provided a different external API/model for evaluation.
- Based on the user's condition, this repository should only be pushed if null-pattern expectations are satisfied. That condition is not met because T2L/TI2L `FVD` remains unavailable in the public benchmark release.
