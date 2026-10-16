# FSI2 accepted-history analysis

Status: **PASS** — Stable-cycle reference comparison

Accepted time: 12 / 12 s; steps: 1200 / 1200.

Accepted physics gates: True; periodic status: PASS.

Estimates marked provisional are not a stable, completed benchmark verdict.

| Quantity | Value | Reference | Relative error | Gate | Provisional |
| --- | ---: | ---: | ---: | ---: | --- |
| tip_y_amplitude | 0.080718411 | 0.0816 | 1.080% | 10.0% | False |
| frequency | 1.9593566 | 1.93 | 1.521% | 5.0% | False |
| drag_mean | 201.38344 | 215.06 | 6.359% | 10.0% | False |

| Signal | Window (s) | Physical cycle rate (Hz) | Amplitude | Mean | Period / amplitude / mean / waveform variation |
| --- | --- | ---: | ---: | ---: | --- |
| tip_y | 10.02939–11.56051 | 1.959357 | 0.08071841 | 0.001098795 | 0.017% / 0.169% / 0.036% / 0.304% |
| drag | 10.02939–11.56051 | 1.959357 | 77.53772 | 201.3834 | 0.017% / 0.354% / 0.124% / 1.274% |
| lift | 10.02939–11.56051 | 1.959357 | 243.359 | 2.882058 | 0.017% / 0.284% / 0.018% / 0.453% |

Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak. All signals use tip-y physical cycles; force cycle rates are not dominant spectral frequencies.

Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.

```json
{
  "fluid_solves": 17157,
  "solid_solves": 17157,
  "newton_iterations": 55661,
  "linear_solves": 61961,
  "coupling_iterations": 13557,
  "iqn_updates": 11157,
  "iqn_restarts": 0,
  "accepted_solve_seconds": 19781.13597507612,
  "fair_timing_evidence": false,
  "scope": "Accepted step solves only; excludes setup, output and failed attempts. Concurrent activity and repeatability are not certified."
}
```
