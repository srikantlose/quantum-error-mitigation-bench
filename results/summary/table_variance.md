# Variance amplification and residual bias of ZNE

Residuals of each estimate against its own exact infinite-shot value (density-matrix simulation of the base and folded circuits), so they contain shot noise only. The predicted ratio is the mean of zne_std / est_std(none); the theoretical value at E = 0 is √5.22 ≈ 2.28. The last two columns are the error ZNE would leave with unlimited shots: as run, and with the exact readout attenuation (1 − 2p_ro)ⁿ divided out, which isolates the extrapolation error on gate noise alone.

| noise | runs | RMS shot residual none | RMS shot residual zne | empirical std ratio | predicted std ratio | zne \|err\|, infinite shots | zne \|err\|, infinite shots, readout removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal | 30 | 0.0270 | 0.0641 | 2.38 | 2.29 | 0.0000 | 0.0000 |
| low | 30 | 0.0273 | 0.0661 | 2.42 | 2.30 | 0.0349 | 0.0003 |
| moderate | 30 | 0.0205 | 0.0554 | 2.70 | 2.32 | 0.1127 | 0.0214 |
| high | 30 | 0.0321 | 0.0664 | 2.07 | 2.30 | 0.2238 | 0.1184 |
