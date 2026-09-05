# 20260905-none2qw08b-B3-s17-2f0450fb

**Depth CE LoRA fine-tune, 1 epoch (B3)**

| Field | Value |
|---|---|
| Recipe | `B3` |
| Status | **PILOT** |
| Split | `val` |
| Student model | `Qwen/Qwen3.5-0.8B` |
| Teacher model | `—` |
| Inference modality | `depth` |
| Distillation mode | `none` |
| Prompt style | `terse` |
| Seed | 17 |
| Recorded (UTC) | 2026-09-05T19:15:22+00:00 |

**Macro accuracy: 40.8%**  ·  invalid outputs: 0.0%

| Type | n | accuracy | invalid |
|---|---:|---:|---:|
| existence | 352 | 56.0% | 0.0% |
| identify_superlative | 341 | 16.4% | 0.0% |
| left_right | 340 | 58.8% | 0.0% |
| nearest_object | 343 | 12.5% | 0.0% |
| relative_depth | 344 | 60.2% | 0.0% |

Reference baselines on this split (macro): chance 30.3%, random 29.9%, majority 33.5%, question_only 34.0%

> **PILOT.** Produced on a 16 GB RTX 4080 SUPER under Option A of `docs/New_Submission/experiment_protocol.md` §9.5. Not a confirmatory result and must not appear in a main or ablation table.

## Notes

One-seed PILOT on 16GB. LoRA r16 on q/k/v/o_proj, lr 1e-5, effective batch 16, 954 optimizer steps, 101.9 min. The strong supervised comparator: per protocol section 4, a KD gain over a weak CE student proves nothing, so this must be tuned fairly before any KD claim rests on it.

## Reproduce

```bash
python evaluate.py --predictions runs/pilot/20260905-none2qw08b-B3-s17-2f0450fb/predictions.csv \
    --split val --model-name "Depth CE LoRA fine-tune, 1 epoch (B3)"
```
