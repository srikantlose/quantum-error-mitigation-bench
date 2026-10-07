# Extrapolator comparison

Mean absolute error of each extrapolator, pooled over L and seeds (`all` pools n too). The exp error averages only the fits that succeeded. `high` is the optional course stress level, where the folded signal is expected to decay fastest.

| noise | method | n | runs | Richardson \|err\| | linear \|err\| | exp \|err\| (valid fits) | Richardson out of [−1, 1] | exp fit failed (NaN) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | zne | 2 | 10 | 0.058 | 0.035 | 0.036 | 0 | 0 |
| low | zne | 4 | 10 | 0.087 | 0.063 | 0.059 | 0 | 0 |
| low | zne | 6 | 10 | 0.049 | 0.051 | 0.046 | 0 | 0 |
| low | zne | all | 30 | 0.065 | 0.050 | 0.047 | 0 | 0 |
| low | zne_rem | 2 | 10 | 0.044 | 0.029 | 0.031 | 0 | 0 |
| low | zne_rem | 4 | 10 | 0.075 | 0.029 | 0.028 | 0 | 0 |
| low | zne_rem | 6 | 10 | 0.045 | 0.024 | 0.027 | 0 | 0 |
| low | zne_rem | all | 30 | 0.055 | 0.027 | 0.029 | 0 | 0 |
| moderate | zne | 2 | 10 | 0.044 | 0.090 | 0.073 | 0 | 0 |
| moderate | zne | 4 | 10 | 0.128 | 0.167 | 0.125 | 0 | 1 |
| moderate | zne | 6 | 10 | 0.127 | 0.196 | 0.167 | 0 | 1 |
| moderate | zne | all | 30 | 0.100 | 0.151 | 0.120 | 0 | 2 |
| moderate | zne_rem | 2 | 10 | 0.056 | 0.037 | 0.041 | 1 | 0 |
| moderate | zne_rem | 4 | 10 | 0.059 | 0.077 | 0.066 | 0 | 1 |
| moderate | zne_rem | 6 | 10 | 0.079 | 0.109 | 0.097 | 0 | 1 |
| moderate | zne_rem | all | 30 | 0.064 | 0.074 | 0.067 | 1 | 2 |
| high | zne | 2 | 10 | 0.170 | 0.227 | 0.146 | 0 | 1 |
| high | zne | 4 | 10 | 0.232 | 0.317 | 0.280 | 0 | 4 |
| high | zne | 6 | 10 | 0.311 | 0.315 | 0.295 | 0 | 3 |
| high | zne | all | 30 | 0.238 | 0.286 | 0.230 | 0 | 8 |
| high | zne_rem | 2 | 10 | 0.086 | 0.134 | 0.126 | 0 | 1 |
| high | zne_rem | 4 | 10 | 0.128 | 0.219 | 0.198 | 0 | 4 |
| high | zne_rem | 6 | 10 | 0.243 | 0.250 | 0.154 | 0 | 6 |
| high | zne_rem | all | 30 | 0.152 | 0.201 | 0.155 | 0 | 11 |
