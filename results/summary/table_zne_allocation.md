# Variance-optimal ZNE shot allocation vs. uniform

Same total budget (3072 shots) split uniformly (1024 each) or proportional to |γ_i| (the variance-minimizing allocation). For each circuit instance the empirical std is taken over 50 repetitions; each row is the mean ± SD of that std over the 5 circuit instances (seeds) of a fixed (n, L, noise). The optimal split has the lower mean std in 12 of 12 conditions. Theoretical std assumes σ_i ≈ 1 at every scale.

| n | L | noise | scheme | shots (λ1,λ3,λ5) | mean bias | empirical std (mean ± SD) | mean rmse | theoretical std |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | low | optimal | 1646,1097,329 | -0.0041 | 0.0473 ± 0.0167 | 0.0583 | 0.0631 |
| 2 | 2 | low | uniform | 1024,1024,1024 | -0.0105 | 0.0502 ± 0.0091 | 0.0602 | 0.0714 |
| 2 | 2 | moderate | optimal | 1646,1097,329 | -0.0110 | 0.0514 ± 0.0063 | 0.0986 | 0.0631 |
| 2 | 2 | moderate | uniform | 1024,1024,1024 | -0.0114 | 0.0567 ± 0.0085 | 0.0987 | 0.0714 |
| 2 | 4 | low | optimal | 1646,1097,329 | +0.0090 | 0.0533 ± 0.0078 | 0.0576 | 0.0631 |
| 2 | 4 | low | uniform | 1024,1024,1024 | +0.0146 | 0.0632 ± 0.0091 | 0.0681 | 0.0714 |
| 2 | 4 | moderate | optimal | 1646,1097,329 | +0.0381 | 0.0542 ± 0.0042 | 0.0856 | 0.0631 |
| 2 | 4 | moderate | uniform | 1024,1024,1024 | +0.0396 | 0.0709 ± 0.0033 | 0.0988 | 0.0714 |
| 4 | 2 | low | optimal | 1646,1097,329 | +0.0009 | 0.0531 ± 0.0026 | 0.0665 | 0.0631 |
| 4 | 2 | low | uniform | 1024,1024,1024 | -0.0029 | 0.0653 ± 0.0066 | 0.0793 | 0.0714 |
| 4 | 2 | moderate | optimal | 1646,1097,329 | -0.0003 | 0.0616 ± 0.0048 | 0.1396 | 0.0631 |
| 4 | 2 | moderate | uniform | 1024,1024,1024 | +0.0012 | 0.0647 ± 0.0073 | 0.1411 | 0.0714 |
| 4 | 4 | low | optimal | 1646,1097,329 | +0.0034 | 0.0601 ± 0.0049 | 0.0689 | 0.0631 |
| 4 | 4 | low | uniform | 1024,1024,1024 | +0.0045 | 0.0631 ± 0.0087 | 0.0763 | 0.0714 |
| 4 | 4 | moderate | optimal | 1646,1097,329 | +0.0119 | 0.0616 ± 0.0096 | 0.1322 | 0.0631 |
| 4 | 4 | moderate | uniform | 1024,1024,1024 | +0.0098 | 0.0673 ± 0.0038 | 0.1352 | 0.0714 |
| 6 | 2 | low | optimal | 1646,1097,329 | -0.0193 | 0.0559 ± 0.0023 | 0.0705 | 0.0631 |
| 6 | 2 | low | uniform | 1024,1024,1024 | -0.0260 | 0.0669 ± 0.0052 | 0.0811 | 0.0714 |
| 6 | 2 | moderate | optimal | 1646,1097,329 | -0.0693 | 0.0611 ± 0.0051 | 0.1476 | 0.0631 |
| 6 | 2 | moderate | uniform | 1024,1024,1024 | -0.0713 | 0.0723 ± 0.0042 | 0.1456 | 0.0714 |
| 6 | 4 | low | optimal | 1646,1097,329 | -0.0300 | 0.0611 ± 0.0039 | 0.0760 | 0.0631 |
| 6 | 4 | low | uniform | 1024,1024,1024 | -0.0329 | 0.0698 ± 0.0058 | 0.0852 | 0.0714 |
| 6 | 4 | moderate | optimal | 1646,1097,329 | -0.1035 | 0.0615 ± 0.0055 | 0.1670 | 0.0631 |
| 6 | 4 | moderate | uniform | 1024,1024,1024 | -0.1054 | 0.0702 ± 0.0055 | 0.1735 | 0.0714 |

## Pooled over every condition

| scheme | mean empirical std | mean rmse | theoretical std |
| --- | ---: | ---: | ---: |
| uniform | 0.0651 | 0.1036 | 0.0714 |
| optimal | 0.0569 | 0.0974 | 0.0631 |

Empirical std ratio (optimal / uniform): **0.874**. Theoretical ratio: **0.885**.
