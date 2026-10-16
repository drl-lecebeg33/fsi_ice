# FSI2 accepted-history analysis

Status: **INCOMPLETE** — Periodic response has not stabilized

Accepted time: 12 / 12 s; steps: 2400 / 2400.

Accepted physics gates: True; periodic status: INCOMPLETE.

Estimates marked provisional are not a stable, completed benchmark verdict.

| Quantity | Value | Reference | Relative error | Gate | Provisional |
| --- | ---: | ---: | ---: | ---: | --- |
| tip_y_amplitude | 0.074483717 | 0.0816 | 8.721% | 10.0% | True |
| frequency | 2.1992125 | 1.93 | 13.949% | 5.0% | True |
| drag_mean | 194.9682 | 215.06 | 9.342% | 10.0% | True |

| Signal | Window (s) | Physical cycle rate (Hz) | Amplitude | Mean | Period / amplitude / mean / waveform variation |
| --- | --- | ---: | ---: | ---: | --- |
| tip_y | 10.51009–11.87422 | 2.199212 | 0.07448372 | 0.001050136 | 0.054% / 0.083% / 0.051% / 0.501% |
| drag | 10.51009–11.87422 | 2.199212 | 89.62268 | 194.9682 | 0.054% / 0.080% / 0.012% / 2.003% |
| lift | 10.51009–11.87422 | 2.199212 | 300.6886 | 2.946613 | 0.054% / 1.250% / 0.045% / 2.955% |

Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak. All signals use tip-y physical cycles; force cycle rates are not dominant spectral frequencies.

Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.

```json
{
  "fluid_solves": 36018,
  "solid_solves": 36018,
  "newton_iterations": 113428,
  "linear_solves": 126016,
  "coupling_iterations": 28818,
  "iqn_updates": 24018,
  "iqn_restarts": 0,
  "accepted_solve_seconds": 10451.462820600835,
  "fair_timing_evidence": false,
  "scope": "Accepted step solves only; excludes setup, output and failed attempts. Concurrent activity and repeatability are not certified."
}
```
