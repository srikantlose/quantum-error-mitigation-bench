# Extrapolator comparison

Mean absolute error of each extrapolator at the noisy levels, pooled over L and seeds (`all` pools n too). The exp error averages only the fits that succeeded.

| noise | method | n | runs | Richardson \|err\| | linear \|err\| | exp \|err\| (valid fits) | Richardson out of [−1, 1] | exp fit failed (NaN) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | zne | 2 | 10 | 0.058 | 0.034 | 0.033 | 0 | 0 |
| low | zne | 4 | 10 | 0.072 | 0.062 | 0.061 | 0 | 0 |
| low | zne | 6 | 10 | 0.061 | 0.064 | 0.045 | 0 | 0 |
| low | zne | all | 30 | 0.064 | 0.053 | 0.047 | 0 | 0 |
| low | zne_rem | 2 | 10 | 0.044 | 0.025 | 0.028 | 0 | 0 |
| low | zne_rem | 4 | 10 | 0.060 | 0.035 | 0.042 | 0 | 0 |
| low | zne_rem | 6 | 10 | 0.059 | 0.028 | 0.043 | 0 | 0 |
| low | zne_rem | all | 30 | 0.054 | 0.029 | 0.038 | 0 | 0 |
| moderate | zne | 2 | 10 | 0.044 | 0.088 | 0.070 | 0 | 0 |
| moderate | zne | 4 | 10 | 0.121 | 0.170 | 0.135 | 0 | 1 |
| moderate | zne | 6 | 10 | 0.126 | 0.198 | 0.156 | 0 | 1 |
| moderate | zne | all | 30 | 0.097 | 0.152 | 0.119 | 0 | 2 |
| moderate | zne_rem | 2 | 10 | 0.056 | 0.038 | 0.042 | 1 | 0 |
| moderate | zne_rem | 4 | 10 | 0.054 | 0.084 | 0.083 | 0 | 1 |
| moderate | zne_rem | 6 | 10 | 0.088 | 0.116 | 0.079 | 0 | 1 |
| moderate | zne_rem | all | 30 | 0.066 | 0.079 | 0.067 | 1 | 2 |
