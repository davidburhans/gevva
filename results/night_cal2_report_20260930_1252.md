# Cal-Round-2 Report (validated, 20260930_1252)

## e2b 1.1 (phase5 base)

| metric | value |
| :--- | :--- |
| gate floor | 0.8703 (n=717) |
| gate hard | 0.5721 (n=5291) |
| gate medium | 0.8352 (n=1007) |
| ANLI acc / neutral preds | 0.5541 / 1171 |
| MMLU-Pro (engine framing) | 0.2560 (n=500) |

## e2b 1.1+xopt

| metric | value |
| :--- | :--- |
| gate floor | 0.8661 (n=717) |
| gate hard | 0.5628 (n=5291) |
| gate medium | 0.8371 (n=1007) |
| ANLI acc / neutral preds | 0.5472 / 1240 |
| MMLU-Pro (engine framing) | 0.2620 (n=500) |

## e2b cal-round-1 (rejected)

| metric | value |
| :--- | :--- |
| gate floor | 0.8759 (n=717) |
| gate hard | 0.5540 (n=5291) |
| gate medium | 0.8361 (n=1007) |
| ANLI acc / neutral preds | 0.5463 / 1204 |
| RAGTruth served F1 @0.5 | 0.1637 (prec 0.1359, rec 0.2057, yes 0.5285) |
| MMLU-Pro (engine framing) | 0.1320 (n=500) |

## e2b cal-round-2

| metric | value |
| :--- | :--- |
| gate floor | 0.8717 (n=717) |
| gate hard | 0.5602 (n=5291) |
| gate medium | 0.8371 (n=1007) |
| ANLI acc / neutral preds | 0.5519 / 1193 |
| RAGTruth served F1 @0.5 | 0.6972 (prec 0.7806, rec 0.6299, yes 0.2819) |
| MMLU-Pro (engine framing) | 0.2680 (n=500) |

## TR-21 listwise A/B (first valid measurement; engine framing)
- phase5 (cross-option 0.0): 0.2560
- xopt    (cross-option 0.5): 0.2620

## Round-1 rejection criteria replay
- cal2 F1 0.6972 vs cal1 0.1637 | yes_rate 0.2819 -> PASS
