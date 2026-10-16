# FSI2 accepted-history analysis

Status: **INCOMPLETE** — Periodic response has not stabilized

Accepted time: 12 / 12 s; steps: 1200 / 1200.

Accepted physics gates: True; periodic status: INCOMPLETE.

Estimates marked provisional are not a stable, completed benchmark verdict.

| Quantity | Value | Reference | Relative error | Gate | Provisional |
| --- | ---: | ---: | ---: | ---: | --- |
| tip_y_amplitude | 0.07348816 | 0.0816 | 9.941% | 10.0% | True |
| frequency | 2.1846496 | 1.93 | 13.194% | 5.0% | True |
| drag_mean | 195.12908 | 215.06 | 9.268% | 10.0% | True |

| Signal | Window (s) | Physical cycle rate (Hz) | Amplitude | Mean | Period / amplitude / mean / waveform variation |
| --- | --- | ---: | ---: | ---: | --- |
| tip_y | 10.55690–11.93012 | 2.18465 | 0.07348816 | 0.001014472 | 0.038% / 0.083% / 0.045% / 0.406% |
| drag | 10.55690–11.93012 | 2.18465 | 86.45197 | 195.1291 | 0.038% / 0.565% / 0.048% / 2.405% |
| lift | 10.55690–11.93012 | 2.18465 | 263.6927 | 2.906993 | 0.038% / 0.617% / 0.051% / 1.103% |

Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak. All signals use tip-y physical cycles; force cycle rates are not dominant spectral frequencies.

Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.

```json
{
  "fluid_solves": 16957,
  "solid_solves": 16957,
  "newton_iterations": 54167,
  "linear_solves": 60465,
  "coupling_iterations": 13357,
  "iqn_updates": 10957,
  "iqn_restarts": 0,
  "accepted_solve_seconds": 5726.445682897873,
  "fair_timing_evidence": false,
  "scope": "Accepted step solves only; excludes setup, output and failed attempts. Concurrent activity and repeatability are not certified."
}
```
