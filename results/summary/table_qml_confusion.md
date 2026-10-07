# Confusion matrices, summed over 5 seeds (n = 4, L = 2)

Rows: TP/FN/FP/TN with versicolor (+1) as the positive class, pooled over the 5 seeds' 20-sample test sets (100 predictions per condition).

| condition | TP | FN | FP | TN | accuracy (pooled) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 47 | 3 | 4 | 46 | 0.930 |
| SVM | 46 | 4 | 5 | 45 | 0.910 |
| VQC, exact | 48 | 2 | 4 | 46 | 0.940 |
| VQC, moderate / none | 48 | 2 | 4 | 46 | 0.940 |
| VQC, moderate / zne_rem | 47 | 3 | 4 | 46 | 0.930 |
| VQC, high / none | 47 | 3 | 4 | 46 | 0.930 |
| VQC, high / zne_rem | 46 | 4 | 4 | 46 | 0.920 |
