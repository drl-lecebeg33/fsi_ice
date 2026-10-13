# FSI2 accepted-history analysis

Status: **INCOMPLETE** — Periodic response has not stabilized

Accepted time: 10 / 10 s; steps: 1000 / 1000.

Accepted physics gates: True; periodic status: INCOMPLETE.

Estimates marked provisional are not a stable, completed benchmark verdict.

| Quantity | Value | Reference | Relative error | Gate | Provisional |
| --- | ---: | ---: | ---: | ---: | --- |
| tip_y_amplitude | 0.062869252 | 0.0816 | 22.954% | 10.0% | True |
| frequency | 2.219358 | 1.93 | 14.993% | 5.0% | True |
| drag_mean | 177.36126 | 215.06 | 17.529% | 10.0% | True |

| Signal | Window (s) | Frequency (Hz) | Amplitude | Mean | Period / amplitude / mean variation |
| --- | --- | ---: | ---: | ---: | --- |
| tip_y | 8.29094–9.64268 | 2.219358 | 0.06286925 | 0.000281635 | 1.113% / 30.863% / 1.257% |
| drag | 9.17069–9.85479 | 4.385328 | 81.6448 | 191.5589 | 0.491% / 9.633% / 2.105% |
| lift | 8.49544–9.85121 | 2.212765 | 238.2895 | 3.658319 | 1.845% / 26.410% / 3.881% |

Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak.

Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.

```json
{
  "fluid_solves": 17950,
  "solid_solves": 17950,
  "newton_iterations": 45356,
  "linear_solves": 45356,
  "coupling_iterations": 17950,
  "iqn_updates": 15520,
  "iqn_restarts": 430,
  "accepted_solve_seconds": 4481.798812860332,
  "fair_timing_evidence": false,
  "scope": "Accepted step solves only; excludes setup, output and failed attempts. Concurrent activity and repeatability are not certified."
}
```
