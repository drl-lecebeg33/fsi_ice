# FSI2 accepted-history analysis

Status: **INCOMPLETE** — Periodic response has not stabilized

Accepted time: 12 / 12 s; steps: 2400 / 2400.

Accepted physics gates: True; periodic status: INCOMPLETE.

Estimates marked provisional are not a stable, completed benchmark verdict.

| Quantity | Value | Reference | Relative error | Gate | Provisional |
| --- | ---: | ---: | ---: | ---: | --- |
| tip_y_amplitude | 0.074457556 | 0.0816 | 8.753% | 10.0% | True |
| frequency | 2.199896 | 1.93 | 13.984% | 5.0% | True |
| drag_mean | 194.9372 | 215.06 | 9.357% | 10.0% | True |

| Signal | Window (s) | Physical cycle rate (Hz) | Amplitude | Mean | Period / amplitude / mean / waveform variation |
| --- | --- | ---: | ---: | ---: | --- |
| tip_y | 10.50936–11.87306 | 2.199896 | 0.07445756 | 0.001012698 | 0.055% / 0.081% / 0.053% / 0.481% |
| drag | 10.50936–11.87306 | 2.199896 | 89.53872 | 194.9372 | 0.055% / 0.008% / 0.010% / 2.046% |
| lift | 10.50936–11.87306 | 2.199896 | 300.4451 | 2.885655 | 0.055% / 1.070% / 0.043% / 3.066% |

Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak. All signals use tip-y physical cycles; force cycle rates are not dominant spectral frequencies.

Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.

```json
{
  "fluid_solves": 36027,
  "solid_solves": 36027,
  "newton_iterations": 113493,
  "linear_solves": 126086,
  "coupling_iterations": 28827,
  "iqn_updates": 24027,
  "iqn_restarts": 0,
  "accepted_solve_seconds": 10974.097648009716,
  "fair_timing_evidence": false,
  "scope": "Accepted step solves only; excludes setup, output and failed attempts. Concurrent activity and repeatability are not certified."
}
```
