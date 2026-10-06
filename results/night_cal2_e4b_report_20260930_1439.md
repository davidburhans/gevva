# e4b Cal-Round-2 Report (validated, 20260930_1439)

| metric | value |
| :--- | :--- |
| gate floor | 0.8605 (n=717) |
| gate hard | 0.6511 (n=5291) |
| gate medium | 0.8153 (n=1007) |
| ANLI acc / neutral preds | 0.6297 / 1172 |
| RAGTruth served F1 @0.5 | 0.6411 (prec 0.6979, rec 0.5928, yes 0.2967) |
| MMLU-Pro (engine framing) | 0.3440 (n=500) |
| McNemar hard vs r3 | delta 0.08pp, p=0.775 |
| McNemar ALL vs r3 | delta 0.07pp, p=0.73079 |

## Comparison (RAGTruth served F1@0.5)
| model | F1 | yes_rate |
| :--- | :--- | :--- |
| e4b v1 | 0.3668 | - |
| e4b r3 (cal-1, rejected) | 0.2048 | 0.664 |
| e4b cal-2 | 0.6411 | 0.2967 |
| e2b cal-2 | 0.6972 | 0.2819 |
