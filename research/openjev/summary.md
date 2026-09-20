### NLI sanity (accuracy)

| model | MNLI-m | MNLI-mm |
|---|---|---|
| Qwen3.5-0.8B full FT | 0.869 | 0.874 |
| Qwen3.5-2B full FT | 0.886 | 0.889 |
| Qwen3.5-4B full FT | 0.904 | 0.907 |
| Qwen3.5-2B head-only | 0.829 | 0.841 |
| ModernCE-large-nli (ref) | 0.909 | 0.921 |

### Multiple choice, rerank without reference (blog #3): premise = question, pick argmax P(entailment)

| model | gpqa | mmlu | arc_easy | arc_challenge | winogrande | chess |
|---|---|---|---|---|---|---|
| random | 0.250 | 0.250 | 0.250 | 0.250 | 0.500 | 0.250 |
| Qwen3.5-0.8B full FT | 0.207 | 0.351 | 0.555 | 0.375 | 0.504 | 0.280 |
| Qwen3.5-2B full FT | 0.237 | 0.394 | 0.629 | 0.491 | 0.534 | 0.324 |
| Qwen3.5-4B full FT | 0.273 | 0.472 | 0.769 | 0.592 | 0.586 | 0.240 |
| Qwen3.5-2B head-only | 0.253 | 0.378 | 0.596 | 0.497 | 0.544 | 0.334 |
| ModernCE-large-nli (ref) | 0.242 | 0.354 | 0.607 | 0.416 | 0.569 | 0.260 |

### Multiple choice, grading with reference (blog #6): premise = question + gold, entailment <=> option is gold (acc / F1)

| model | gpqa | mmlu | arc_easy | arc_challenge | winogrande | chess |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B full FT | 0.939 / 0.891 | 0.951 / 0.909 | 0.974 / 0.951 | 0.958 / 0.922 | 0.866 / 0.877 | 0.996 / 0.992 |
| Qwen3.5-2B full FT | 0.956 / 0.918 | 0.968 / 0.940 | 0.984 / 0.970 | 0.972 / 0.947 | 0.828 / 0.852 | 0.997 / 0.994 |
| Qwen3.5-4B full FT | 0.968 / 0.940 | 0.974 / 0.949 | 0.993 / 0.986 | 0.987 / 0.975 | 0.843 / 0.863 | 0.997 / 0.993 |
| Qwen3.5-2B head-only | 0.932 / 0.865 | 0.897 / 0.807 | 0.938 / 0.887 | 0.932 / 0.869 | 0.716 / 0.761 | 0.929 / 0.876 |
| ModernCE-large-nli (ref) | 0.912 / 0.846 | 0.953 / 0.912 | 0.969 / 0.941 | 0.964 / 0.931 | 0.916 / 0.916 | 0.975 / 0.948 |

### GSM8K (200 test questions, candidates from Qwen3.5-4B: greedy + 4 samples)

| model | greedy | pass@1 | maj@4 | NLI rerank@4 | oracle@4 | grade acc / F1 |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B full FT | 0.880 | 0.859 | 0.915 | 0.865 | 0.940 | 0.986 / 0.992 |
| Qwen3.5-2B full FT | 0.880 | 0.859 | 0.915 | 0.855 | 0.940 | 0.996 / 0.998 |
| Qwen3.5-4B full FT | 0.880 | 0.859 | 0.915 | 0.845 | 0.940 | 0.996 / 0.998 |
| Qwen3.5-2B head-only | 0.880 | 0.859 | 0.915 | 0.885 | 0.940 | 0.582 / 0.681 |
| ModernCE-large-nli (ref) | 0.880 | 0.859 | 0.915 | 0.850 | 0.940 | 0.984 / 0.991 |
