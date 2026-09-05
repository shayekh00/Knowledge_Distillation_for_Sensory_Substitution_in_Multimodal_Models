# VQA-SUNRGBD-v2 — single-reviewer gold verification report

Reviewer: R1
Protocol: one reviewer inspects the RGB image, evidence overlay, question, and gold answer, then records correct, incorrect, or ambiguous.
This verifies sampled gold labels; it does not measure inter-rater reliability.
Acceptance rule (§8.3): gold accuracy ≥ 95% and ambiguous share ≤ 3%.

| Type | Sampled | Reviewed | Gold accuracy | Corrections matching gold | Ambiguous | Meets acceptance |
|---|---|---|---|---|---|---|
| existence | 150 | 150 | 82.0% | 0.0% | 0.0% | no |
| identify_superlative | 150 | 150 | 97.3% | — | 2.7% | yes |
| left_right | 150 | 150 | 95.3% | 0.0% | 2.0% | yes |
| nearest_object | 150 | 150 | 100.0% | — | 0.0% | yes |
| relative_depth | 150 | 150 | 100.0% | — | 0.0% | yes |
