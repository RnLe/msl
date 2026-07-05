# TE X-Point Comparison Report

Figure: [ea_x_te_1ret_3rem_lowest_vs_fdfd_res8_sig002.png](/home/renlephy/msl/research/moire_envelope/thesis_results/ea_x_te_1ret_3rem_lowest_vs_fdfd_res8_sig002.png)

## Setup

- Crystal: square lattice moire supercell
- Polarization: TE
- k-point: X, using `q = (π, 0)` in the direct solver
- Direct solver: FDFD, `8 px/cell`, `sigma_omega = 0.02`
- EA datasets included:
  - `ea_x_te_1ret_3rem_lowest_{angle}.npz`
  - `ea_x_te_4ret_0rem_lowest_{angle}.npz`
- Common EA metadata:
  - `lambda_ref = 5.4492357154741295`
  - `sigma_omega = 0.02`
  - `sigma_delta = -5.433444348432387`

## Notes

- `valid modes` counts frequencies after omitting `NaN` entries.
- `bandwidth` means `max(valid) - min(valid)`.
- Comparison metrics use only modes where the EA frequency is valid, and compare against the FDFD mode with the same index.
- Relative errors are reported against the FDFD frequency magnitude.

## 8.17°

| Dataset | Total modes | Valid modes | NaN modes | Min | Max | Bandwidth | Mean | Median | Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FDFD | 30 | 30 | 0 | 0.026955 | 0.164510 | 0.137555 | 0.110951 | 0.111706 | 0.038988 |
| EA 1 ret, 3 rem, lowest | 30 | 14 | 16 | 0.038469 | 0.260068 | 0.221598 | 0.177241 | 0.191135 | 0.071077 |
| EA 4 ret, 0 rem, lowest | 30 | 30 | 0 | 0.204041 | 0.298743 | 0.094703 | 0.265031 | 0.269467 | 0.027608 |

| Comparison vs FDFD | Comparable modes | Mean abs diff | RMSE | Mean signed diff | Mean abs rel % | Max abs rel % |
|---|---:|---:|---:|---:|---:|---:|
| EA 1 ret, 3 rem, lowest | 14 | 0.064288 | 0.069062 | 0.031692 | 43.600 | 71.174 |
| EA 4 ret, 0 rem, lowest | 30 | 0.154080 | 0.154596 | 0.154080 | 181.699 | 675.678 |

## 3.01°

| Dataset | Total modes | Valid modes | NaN modes | Min | Max | Bandwidth | Mean | Median | Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FDFD | 50 | 50 | 0 | 0.009957 | 0.079608 | 0.069652 | 0.053176 | 0.054444 | 0.018873 |
| EA 1 ret, 3 rem, lowest | 50 | 24 | 26 | 0.019734 | 0.162542 | 0.142808 | 0.108068 | 0.112139 | 0.040533 |
| EA 4 ret, 0 rem, lowest | 50 | 50 | 0 | 0.241952 | 0.269615 | 0.027663 | 0.260190 | 0.261946 | 0.007327 |

| Comparison vs FDFD | Comparable modes | Mean abs diff | RMSE | Mean signed diff | Mean abs rel % | Max abs rel % |
|---|---:|---:|---:|---:|---:|---:|
| EA 1 ret, 3 rem, lowest | 24 | 0.044693 | 0.051670 | 0.038626 | 61.696 | 104.178 |
| EA 4 ret, 0 rem, lowest | 50 | 0.207014 | 0.207342 | 0.207014 | 512.889 | 2345.080 |

## 1.005°

| Dataset | Total modes | Valid modes | NaN modes | Min | Max | Bandwidth | Mean | Median | Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FDFD | 50 | 50 | 0 | 0.009984 | 0.027103 | 0.017119 | 0.020248 | 0.021104 | 0.005168 |
| EA 1 ret, 3 rem, lowest | 50 | 34 | 16 | 0.062912 | 0.116630 | 0.053718 | 0.095020 | 0.098491 | 0.015820 |
| EA 4 ret, 0 rem, lowest | 50 | 50 | 0 | 0.275689 | 0.317271 | 0.041583 | 0.308569 | 0.310579 | 0.009074 |

| Comparison vs FDFD | Comparable modes | Mean abs diff | RMSE | Mean signed diff | Mean abs rel % | Max abs rel % |
|---|---:|---:|---:|---:|---:|---:|
| EA 1 ret, 3 rem, lowest | 34 | 0.071799 | 0.072942 | 0.071799 | 307.177 | 331.790 |
| EA 4 ret, 0 rem, lowest | 50 | 0.288321 | 0.288369 | 0.288321 | 1536.694 | 2680.434 |

## Short Takeaways

- The `EA 1 ret, 3 rem, lowest` family is consistently closer to FDFD than `EA 4 ret, 0 rem, lowest` at all three angles.
- Even so, the `EA 1 ret, 3 rem, lowest` spectra remain shifted upward relative to FDFD across all three angles.
- The mismatch becomes more severe as the angle decreases:
  - mean absolute relative error rises from `43.6%` at `8deg` to `307.2%` at `1deg` for `EA 1 ret, 3 rem, lowest`
  - `EA 4 ret, 0 rem, lowest` is much farther away throughout
- The EA runs contain many omitted modes only in the `1 ret, 3 rem` family:
  - `16` NaNs at `8deg`
  - `26` NaNs at `3deg`
  - `16` NaNs at `1deg`
- The `4 ret, 0 rem, lowest` family has no NaNs in this dataset, but its full spectrum is centered much higher than the FDFD target window.