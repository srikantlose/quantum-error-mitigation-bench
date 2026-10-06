# Overhead per estimate

Executions and shots needed for one mitigated estimate, the largest executed depth relative to the base circuit (range over L), and total CNOTs executed relative to the base circuit. Calibration circuits contain no CNOTs.

| method | n=2 circuits | n=2 shots | n=2 max/base depth | n=2 total/base CX | n=4 circuits | n=4 shots | n=4 max/base depth | n=4 total/base CX | n=6 circuits | n=6 shots | n=6 max/base depth | n=6 total/base CX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 1 | 1024 | 1.00 | 1 | 1 | 1024 | 1.00 | 1 | 1 | 1024 | 1.00 | 1 |
| rem | 5 | 5120 | 1.00 | 1 | 17 | 17408 | 1.00 | 1 | 65 | 66560 | 1.00 | 1 |
| zne | 3 | 3072 | 5.00 | 9 | 3 | 3072 | 5.00 | 9 | 3 | 3072 | 5.00 | 9 |
| zne_rem | 7 | 7168 | 5.00 | 9 | 19 | 19456 | 5.00 | 9 | 67 | 68608 | 5.00 | 9 |

## Wall time per estimate

Simulator time of the circuits a method needs and classical post-processing time, averaged over L, noise levels and seeds. Simulator time is not hardware time.

| method | n=2 quantum (s) | n=4 quantum (s) | n=6 quantum (s) | n=2 classical (ms) | n=4 classical (ms) | n=6 classical (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.0026 | 0.0037 | 0.0058 | 0.035 | 0.047 | 0.081 |
| rem | 0.0094 | 0.0369 | 0.1703 | 0.157 | 0.268 | 0.819 |
| zne | 0.0095 | 0.0148 | 0.0312 | 0.232 | 0.280 | 0.375 |
| zne_rem | 0.0162 | 0.0480 | 0.1957 | 0.340 | 0.517 | 1.419 |
