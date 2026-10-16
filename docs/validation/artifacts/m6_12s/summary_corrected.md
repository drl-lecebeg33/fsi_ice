# FSI2 accepted-history analysis

Status: **FAIL** — Stable-cycle reference comparison

Accepted time: 12 / 12 s; steps: 1200 / 1200.

Accepted physics gates: True; periodic status: FAIL.

Estimates marked provisional are not a stable, completed benchmark verdict.

| Quantity | Value | Reference | Relative error | Gate | Provisional |
| --- | ---: | ---: | ---: | ---: | --- |
| tip_y_amplitude | 0.073530716 | 0.0816 | 9.889% | 10.0% | False |
| frequency | 2.1841818 | 1.93 | 13.170% | 5.0% | False |
| drag_mean | 195.18632 | 215.06 | 9.241% | 10.0% | False |

| Signal | Window (s) | Frequency (Hz) | Amplitude | Mean | Period / amplitude / mean variation |
| --- | --- | ---: | ---: | ---: | --- |
| tip_y | 10.55761–11.93112 | 2.184182 | 0.07353072 | 0.001043735 | 0.038% / 0.194% / 0.046% |
| drag | 11.22802–11.91451 | 4.370042 | 86.19776 | 195.1303 | 0.127% / 0.965% / 0.019% |
| lift | 10.30859–11.68212 | 2.184153 | 264.0976 | 2.933403 | 0.096% / 0.657% / 0.078% |

Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak.

Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.

```json
{
  "fluid_solves": 20733,
  "solid_solves": 20733,
  "newton_iterations": 57346,
  "linear_solves": 57346,
  "coupling_iterations": 20733,
  "iqn_updates": 17897,
  "iqn_restarts": 436,
  "accepted_solve_seconds": 5361.058244220301,
  "fair_timing_evidence": false,
  "scope": "Accepted step solves only; excludes setup, output and failed attempts. Concurrent activity and repeatability are not certified."
}
```
