# Round-4 Distillation Report (validated, 20261006_1244)

## e2b-distill vs e2b-cal2

| metric | cal2 | distill |
| :--- | :--- | :--- |
| gate floor | 0.8717 | 0.8675 |
| gate hard | 0.5602 | 0.5695 |
| gate medium | 0.8371 | 0.8302 |
| ANLI acc | 0.5519 | 0.5509 |
| RAGTruth served F1@0.5 | 0.6972 | 0.7150 |
| MMLU-Pro (engine) | - | 0.2800 |
| McNemar hard | - | 0.93pp (p=3e-05) |
| McNemar ALL | - | 0.56pp (p=0.00189) |

## e4b-distill vs e4b-cal2

| metric | cal2 | distill |
| :--- | :--- | :--- |
| gate floor | 0.8605 | 0.8661 |
| gate hard | 0.6511 | 0.6507 |
| gate medium | 0.8153 | 0.8203 |
| ANLI acc | 0.6297 | 0.6266 |
| RAGTruth served F1@0.5 | 0.6411 | 0.6809 |
| MMLU-Pro (engine) | - | 0.3500 |
| McNemar hard | - | -0.04pp (p=0.92195) |
| McNemar ALL | - | 0.1pp (p=0.60574) |


## Verdict rule (pre-registered)
Keep distill lineage only if: RAGTruth F1 not worse by >0.02, AND no McNemar-significant
gate regression (p<0.01), AND ANLI neutral predictions stable. Otherwise cal2 stands.
