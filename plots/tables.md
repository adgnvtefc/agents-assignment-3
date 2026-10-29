# Summary tables

## Per-run summary

| run | final eval/success_rate | final eval/pass@4 | step-0 eval/success_rate | mean step_time_s | median step_time_s | total train time (min) | mean frac_degenerate_groups | groups_consumed | groups_dropped_stale |
|---|---|---|---|---|---|---|---|---|---|
| p2a_number_search_reinforce | 0.455 (step 50) | 0.880 | 0.325 | 26.4 | 18.7 | 22.0 | N/A | - | - |
| p2b_number_search_grpo | 0.985 (step 50) | 1.000 | 0.345 | 44.9 | 33.7 | 37.4 | 0.575 | - | - |
| p41_number_search_grpo_g2 | 0.490 (step 50) | 0.840 | 0.305 | 29.2 | 20.0 | 24.3 | 0.670 | - | - |
| p3_number_search_async | 0.995 (step 50) | 1.000 | 0.315 | 30.4 | 24.6 | 25.3 | 0.438 | 400 | 38 |
| p43_number_search_async_stale4_iw | 0.980 (step 50) | 1.000 | 0.310 | 29.2 | 24.3 | 24.3 | 0.395 | 400 | 0 |
| p43_number_search_async_stale4_noiw | 0.975 (step 50) | 1.000 | 0.280 | 28.0 | 22.3 | 23.3 | 0.345 | 400 | 0 |
| p5_textcraft_grpo | 0.590 (step 50) | 0.740 | 0.115 | 856.4 | 888.2 | 713.6 | 0.451 | 800 | 0 |

## Importance-weight statistics over training

| run | mean of importance_weight_mean | min of importance_weight_min | max of importance_weight_max | mean of importance_weight_std | max of importance_weight_std |
|---|---|---|---|---|---|
| p2a_number_search_reinforce | 0.9997 | 0.1601 | 1.6524 | 0.0226 | 0.0358 |
| p2b_number_search_grpo | 1.0000 | 0.5914 | 1.8671 | 0.0177 | 0.0279 |
| p41_number_search_grpo_g2 | 0.9998 | 0.4623 | 1.4959 | 0.0211 | 0.0347 |
| p3_number_search_async | 0.9999 | 0.3103 | 2.1540 | 0.0236 | 0.0376 |
| p43_number_search_async_stale4_iw | 0.9998 | 0.1069 | 2.2401 | 0.0261 | 0.0444 |
| p43_number_search_async_stale4_noiw | 1.0000 | 0.0192 | 3.6414 | 0.0296 | 0.0818 |
| p5_textcraft_grpo | 1.0000 | 0.0393 | 4.4806 | 0.0264 | 0.0286 |

## Part 5 final evaluation (k = 4)

| model | split | success_rate | pass@4 | avg_steps | abandoned_episodes |
|---|---|---|---|---|---|
| fine-tuned (v50 adapter) | medium (213 tasks) | 0.634 | 0.756 | 48.9 | 3 |
| fine-tuned (v50 adapter) | hard (136 tasks) | 0.000 | 0.000 | 72.4 | 1 |
| base Qwen3.5-4B | medium (213 tasks) | 0.130 | 0.338 | 62.3 | 2 |
| base Qwen3.5-4B | hard (136 tasks) | 0.000 | 0.000 | 69.9 | 3 |
