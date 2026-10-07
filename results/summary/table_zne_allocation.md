# Variance-optimal ZNE shot allocation vs. uniform

Same total budget (3072 shots) split uniformly (1024 each) or proportional to |γ_i| (the variance-minimizing allocation). Each row aggregates 50 repetitions at a fixed (n, L, noise). Theoretical std assumes σ_i ≈ 1 at every scale.

| n | L | noise | scheme | shots (λ1,λ3,λ5) | bias | empirical std | rmse | theoretical std |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | low | optimal | 1646,1097,329 | -0.0308 | 0.0446 | 0.0538 | 0.0631 |
| 2 | 2 | low | uniform | 1024,1024,1024 | -0.0300 | 0.0524 | 0.0599 | 0.0714 |
| 2 | 2 | moderate | optimal | 1646,1097,329 | -0.0812 | 0.0486 | 0.0944 | 0.0631 |
| 2 | 2 | moderate | uniform | 1024,1024,1024 | -0.0882 | 0.0616 | 0.1072 | 0.0714 |
| 2 | 4 | low | optimal | 1646,1097,329 | +0.0202 | 0.0529 | 0.0562 | 0.0631 |
| 2 | 4 | low | uniform | 1024,1024,1024 | +0.0201 | 0.0610 | 0.0636 | 0.0714 |
| 2 | 4 | moderate | optimal | 1646,1097,329 | +0.0733 | 0.0511 | 0.0891 | 0.0631 |
| 2 | 4 | moderate | uniform | 1024,1024,1024 | +0.0711 | 0.0695 | 0.0989 | 0.0714 |
| 4 | 2 | low | optimal | 1646,1097,329 | -0.0304 | 0.0544 | 0.0618 | 0.0631 |
| 4 | 2 | low | uniform | 1024,1024,1024 | -0.0488 | 0.0648 | 0.0806 | 0.0714 |
| 4 | 2 | moderate | optimal | 1646,1097,329 | -0.0891 | 0.0698 | 0.1127 | 0.0631 |
| 4 | 2 | moderate | uniform | 1024,1024,1024 | -0.0810 | 0.0712 | 0.1074 | 0.0714 |
| 4 | 4 | low | optimal | 1646,1097,329 | -0.0229 | 0.0666 | 0.0698 | 0.0631 |
| 4 | 4 | low | uniform | 1024,1024,1024 | -0.0401 | 0.0514 | 0.0648 | 0.0714 |
| 4 | 4 | moderate | optimal | 1646,1097,329 | -0.1093 | 0.0596 | 0.1242 | 0.0631 |
| 4 | 4 | moderate | uniform | 1024,1024,1024 | -0.1264 | 0.0706 | 0.1445 | 0.0714 |
| 6 | 2 | low | optimal | 1646,1097,329 | -0.0466 | 0.0586 | 0.0744 | 0.0631 |
| 6 | 2 | low | uniform | 1024,1024,1024 | -0.0508 | 0.0737 | 0.0889 | 0.0714 |
| 6 | 2 | moderate | optimal | 1646,1097,329 | -0.1436 | 0.0561 | 0.1540 | 0.0631 |
| 6 | 2 | moderate | uniform | 1024,1024,1024 | -0.1375 | 0.0781 | 0.1577 | 0.0714 |
| 6 | 4 | low | optimal | 1646,1097,329 | +0.0401 | 0.0568 | 0.0691 | 0.0631 |
| 6 | 4 | low | uniform | 1024,1024,1024 | +0.0385 | 0.0741 | 0.0828 | 0.0714 |
| 6 | 4 | moderate | optimal | 1646,1097,329 | +0.1293 | 0.0614 | 0.1428 | 0.0631 |
| 6 | 4 | moderate | uniform | 1024,1024,1024 | +0.1332 | 0.0631 | 0.1471 | 0.0714 |

## Pooled over every condition

| scheme | mean empirical std | mean rmse | theoretical std |
| --- | ---: | ---: | ---: |
| uniform | 0.0651 | 0.1036 | 0.0714 |
| optimal | 0.0569 | 0.0974 | 0.0631 |

Empirical std ratio (optimal / uniform): **0.874**. Theoretical ratio: **0.885**.
