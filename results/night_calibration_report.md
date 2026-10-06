# Night Calibration Report (validated)

## e2b 1.1 (baseline)

| metric | value |
| :--- | :--- |
| gate floor | 0.8703 (n=717) |
| gate hard | 0.5721 (n=5291) |
| gate medium | 0.8352 (n=1007) |
| ANLI acc / neutral preds | 0.5541 / 1171 |

## e2b 1.1+xopt

| metric | value |
| :--- | :--- |
| gate floor | 0.8661 (n=717) |
| gate hard | 0.5628 (n=5291) |
| gate medium | 0.8371 (n=1007) |
| ANLI acc / neutral preds | 0.5472 / 1240 |
| MMLU-Pro sample acc | 0.1320 (n=500) |

## e2b cal

| metric | value |
| :--- | :--- |
| gate floor | 0.8759 (n=717) |
| gate hard | 0.5540 (n=5291) |
| gate medium | 0.8361 (n=1007) |
| ANLI acc / neutral preds | 0.5463 / 1204 |
| RAGTruth served F1 @0.5 | 0.1637 (prec 0.1359, rec 0.2057) |
| MMLU-Pro sample acc | 0.1320 (n=500) |

## e4b r3 (baseline)

| metric | value |
| :--- | :--- |
| gate floor | 0.8619 (n=717) |
| gate hard | 0.6503 (n=5291) |
| gate medium | 0.8133 (n=1007) |
| ANLI acc / neutral preds | 0.6275 / 1181 |

## e4b cal

| metric | value |
| :--- | :--- |
| gate floor | 0.8675 (n=717) |
| gate hard | 0.6564 (n=5291) |
| gate medium | 0.8153 (n=1007) |
| ANLI acc / neutral preds | 0.6291 / 1242 |
| RAGTruth served F1 @0.5 | 0.2048 (prec 0.1562, rec 0.2969) |
| MMLU-Pro sample acc | 0.1320 (n=500) |

## MMLU-Pro listwise A/B (phase5 vs xopt)
- phase5 (cross-option 0.0): 0.1320
- xopt    (cross-option 0.5): 0.1320
- e2b-cal (xopt + cal recipe): see table above
