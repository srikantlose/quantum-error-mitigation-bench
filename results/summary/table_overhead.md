# Overhead per estimate

Executions and shots needed for one mitigated estimate, the largest executed depth relative to the base circuit (range over L), and total CNOTs executed relative to the base circuit. Calibration circuits contain no CNOTs.

| method | n=2 circuits | n=2 shots | n=2 max/base depth | n=2 total/base CX | n=4 circuits | n=4 shots | n=4 max/base depth | n=4 total/base CX | n=6 circuits | n=6 shots | n=6 max/base depth | n=6 total/base CX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 1 | 1024 | 1.00 | 1 | 1 | 1024 | 1.00 | 1 | 1 | 1024 | 1.00 | 1 |
| rem | 5 | 5120 | 1.00 | 1 | 17 | 17408 | 1.00 | 1 | 65 | 66560 | 1.00 | 1 |
| rem_tensored | 3 | 3072 | 1.00 | 1 | 3 | 3072 | 1.00 | 1 | 3 | 3072 | 1.00 | 1 |
| zne | 3 | 3072 | 5.00 | 9 | 3 | 3072 | 5.00 | 9 | 3 | 3072 | 5.00 | 9 |
| zne_rem | 7 | 7168 | 5.00 | 9 | 19 | 19456 | 5.00 | 9 | 67 | 68608 | 5.00 | 9 |

## Wall time per estimate

Simulator time of the circuits a method needs and classical post-processing time, averaged over L, noise levels and seeds. Simulator time is not hardware time.

| method | n=2 quantum (s) | n=4 quantum (s) | n=6 quantum (s) | n=2 classical (ms) | n=4 classical (ms) | n=6 classical (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.0026 | 0.0037 | 0.0061 | 0.164 | 0.186 | 0.389 |
| rem | 0.0100 | 0.0385 | 0.1838 | 0.164 | 0.186 | 0.389 |
| rem_tensored | 0.0065 | 0.0086 | 0.0125 | 0.164 | 0.186 | 0.389 |
| zne | 0.0096 | 0.0150 | 0.0324 | 0.164 | 0.186 | 0.389 |
| zne_rem | 0.0169 | 0.0499 | 0.2101 | 0.164 | 0.186 | 0.389 |
