# Final Sanity Check Report for the Multiband Envelope-Approximation Pipeline

## Purpose

This document summarizes the final validation pass for the multiband moire envelope-approximation pipeline used in the thesis. The goal is to determine whether the Berry-connection and Born-Huang outputs, which can look noisy in raw plots, are physically consistent and whether the assembled effective Hamiltonian is trustworthy for research-grade results.

The diagnostics were run with:

- Script: `research/moire_envelope/thesis_results/T_sanity_check/compute.py`
- Output JSON: `research/moire_envelope/thesis_results/T_sanity_check/sanity_check_results.json`
- Generated figures:
  - `gauge_inv_norms_square_M_b3.{png,pdf}`
  - `gauge_inv_norms_hex_M_b1.{png,pdf}`
  - `gauge_inv_norms_honeycomb_K_b1.{png,pdf}`
  - `berry_curvature_square_M_b3.{png,pdf}`
  - `berry_curvature_hex_M_b1.{png,pdf}`
  - `berry_curvature_honeycomb_K_b1.{png,pdf}`

## Executive Summary

### Bottom-line assessment

For the square and hexagonal thesis candidates, the framework is physically consistent and sufficiently validated to be treated as trustworthy for research-grade use.

For the honeycomb Dirac candidate, the pipeline is very likely still physically correct, but it does not yet meet the same validation standard as the other two cases. The decisive issue is that the pre-enforcement Hamiltonian anti-Hermitian residual is larger by about 3 to 4 orders of magnitude than in the square and hex cases. This does not indicate a clear failure, but it does justify one more focused investigation before presenting the honeycomb result as equally settled.

### Practical thesis statement

The final sanity checks support the conclusion that the visually noisy Berry-connection plots are not, by themselves, evidence of a gauge or symmetry failure. After converting to gauge-invariant diagnostics and testing the assembled Hamiltonian directly, the square and hexagonal cases pass the relevant physical consistency checks. The honeycomb Dirac case shows the same qualitative pattern but retains a larger pre-symmetrization anti-Hermitian residual, so it should be presented as validated with a remaining caution note rather than as fully closed.

## What Was Tested

The final diagnostic script evaluated six classes of checks:

1. Hermiticity of the Berry connection `A_berry`.
2. Hermiticity and positivity of the Born-Huang potential `Phi_BH`.
3. Gauge-invariant scalar norms derived from `A_berry`.
4. Crystal-symmetry consistency of the gauge-invariant quantities.
5. Gauge smoothness via the non-Abelian Berry curvature and nearest-neighbor variation.
6. Anti-Hermitian residual of the full effective Hamiltonian before and after the explicit `(H + H^\dagger)/2` enforcement.

## Important Interpretation Principle

The raw Berry connection should not be judged directly from visual appearance alone.

Because the Bloch functions are normalized using an `epsilon`-weighted electric-field inner product, and because `epsilon(r; R)` depends on the moire coordinate `R`, the Berry connection generically obeys

$$
A + A^\dagger = -i\langle u_m | \partial_R \varepsilon | u_n \rangle,
$$

which means that `A_berry` is not expected to be Hermitian pointwise. Therefore, a raw lack of visual symmetry in `A_berry` is not automatically unphysical. The decisive physical test is whether the full Hamiltonian assembled from all terms is Hermitian to high accuracy before the explicit numerical enforcement step.

## Candidate Set

The checks were run on the three thesis candidates:

| Candidate | Symmetry | Polarization | Bands | Run directory |
| --- | --- | --- | ---: | --- |
| `square_M_b3` | `C4` | TM | 5 | `runsV3/thesis_square_M_b3_20260209_173724/candidate_0000` |
| `hex_M_b1` | `C2` | TE | 4 | `runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000` |
| `honeycomb_K_b1` | `C6` | TM | 2 | `runsV3/thesis_honeycomb_K_b1_20260307_171424/candidate_0000` |

## Final Cross-Candidate Verdict Table

| Check | `square_M_b3` | `hex_M_b1` | `honeycomb_K_b1` |
| --- | --- | --- | --- |
| `A_hermiticity` | `INFO_METRIC` | `INFO_METRIC` | `INFO_METRIC` |
| `BH_hermiticity` | `SKIP_ZERO` | `SKIP_ZERO` | `SKIP_ZERO` |
| `BH_positivity` | `SKIP_ZERO` | `SKIP_ZERO` | `SKIP_ZERO` |
| `gauge_inv_symmetry` | `PASS` | `PASS` | `PASS` |
| `gauge_smoothness` | `MARGINAL` | `MARGINAL` | `MARGINAL` |
| `H_antihermitian` | `MARGINAL` | `MARGINAL` | `MARGINAL_HIGH` |
| Overall | `PASS` | `PASS` | `REVIEW` |

## Quantitative Results

### 1. Berry-Connection Hermiticity

The raw Hermiticity test fails in all three cases, but this is interpreted as an expected metric effect rather than a physical inconsistency.

| Candidate | Global relative residual | Hermitian-part norm | Anti-Hermitian-part norm | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `square_M_b3` | 1.4207433117 | 235.1489 | 237.3355 | Expected metric correction |
| `hex_M_b1` | 1.7521069421 | 279.4453 | 507.6771 | Expected metric correction |
| `honeycomb_K_b1` | 1.4768820344 | 94.5073 | 103.4943 | Expected metric correction |

### 2. Born-Huang Potential

The Born-Huang term is identically zero in the saved phase-2 outputs for all three candidates.

| Candidate | `max_abs(Phi_BH)` | Status |
| --- | ---: | --- |
| `square_M_b3` | 0.0 | Placeholder zero, skipped |
| `hex_M_b1` | 0.0 | Placeholder zero, skipped |
| `honeycomb_K_b1` | 0.0 | Placeholder zero, skipped |

This is consistent with the current pipeline configuration and means the Born-Huang term was not an active contributor in these validation runs.

### 3. Symmetry of Gauge-Invariant Scalars

The symmetry check was performed on scalar gauge invariants such as the Frobenius norm of the Berry connection and the magnitude of a representative off-diagonal entry. Vector-like componentwise quantities were not used in the final pass/fail criterion because they rotate into each other under `C4` and `C6` operations.

| Candidate | `A_frob` symmetry error | `A01_mag` symmetry error | Raw `A_berry` vector symmetry error | Verdict |
| --- | ---: | ---: | ---: | --- |
| `square_M_b3` | 6.8811e-17 | 9.8972e-17 | 1.0123e-16 | PASS |
| `hex_M_b1` | 0.0 | 0.0 | 0.0 | PASS |
| `honeycomb_K_b1` | 1.4808e-16 | 2.4147e-16 | 4.0184e-16 | PASS |

These are essentially machine-precision symmetry errors and are strong evidence that the gauge fixing and symmetrization are behaving correctly at the physically meaningful level.

### 4. Gauge Smoothness

The Berry-curvature fields are smooth enough to avoid indicating an obvious gauge pathology, but the variation is not uniformly small. The outcome is therefore marked `MARGINAL` rather than `PASS`.

| Candidate | Curvature max | Curvature mean | Spike ratio | Robust NN jump x | Robust NN jump y | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `square_M_b3` | 4.4029 | 2.7880 | 1.5792 | 1.8722 | 1.8722 | MARGINAL |
| `hex_M_b1` | 18.0628 | 3.8935 | 4.6392 | 2.9999 | 3.0595 | MARGINAL |
| `honeycomb_K_b1` | 2.0412 | 0.8966 | 2.2767 | 2.5165 | 2.5165 | MARGINAL |

This should be interpreted as moderate spatial variation rather than evidence of discontinuous gauge jumps. The earlier enormous nearest-neighbor ratios were traced to divisions by nearly zero norms at nodal points and were therefore not physically meaningful.

### 5. Full Hamiltonian Hermiticity

This is the decisive physical validation table.

| Candidate | `||H - H^\dagger|| / ||H||` before enforcement | Verdict before enforcement | After `(H+H^\dagger)/2` | Assessment |
| --- | ---: | --- | --- | --- |
| `square_M_b3` | 1.4216e-08 | MARGINAL | exactly Hermitian | Excellent |
| `hex_M_b1` | 8.2450e-09 | MARGINAL | exactly Hermitian | Excellent |
| `honeycomb_K_b1` | 8.5664e-05 | MARGINAL_HIGH | exactly Hermitian | Needs targeted follow-up |

The square and hexagonal cases are already in a regime where the anti-Hermitian contamination is negligible for practical purposes. The honeycomb case remains small in absolute relative terms, but it is clearly not as tight as the other two and should not be dismissed without explanation.

## Interpretation by Candidate

### Square candidate

The square case is validated to a high standard. Gauge-invariant symmetry is exact up to machine precision, and the full Hamiltonian is Hermitian to about `1e-8` before explicit enforcement. This is strong evidence that the pipeline is physically consistent in this regime.

### Hexagonal candidate

The hexagonal case is also validated to a high standard. It behaves similarly to the square case, with excellent Hamiltonian Hermiticity and exact gauge-invariant symmetry restoration. This result is suitable for confident use in the thesis.

### Honeycomb Dirac candidate

The honeycomb case is qualitatively consistent with the validated pattern: gauge-invariant symmetry passes at machine precision, the Berry-connection anti-Hermiticity is explained by the metric term, and the post-enforcement Hamiltonian is exactly Hermitian. However, the pre-enforcement full-Hamiltonian residual is `8.6e-5`, which is substantially larger than for the other two candidates. That makes this case credible but not yet as fully validated as the square and hexagonal benchmarks.

## Research-Grade Judgment

### What is already justified

The framework and pipeline are physically valid in the sense required for the thesis, with one important qualification.

The square and hexagonal cases can be treated as validated and trustworthy. They show:

- physically correct gauge interpretation of `A_berry`,
- exact symmetry restoration at the gauge-invariant level,
- absence of obvious gauge discontinuities,
- and an almost perfectly Hermitian effective Hamiltonian before explicit numerical cleanup.

### What should still be presented with caution

The honeycomb Dirac result should be described as strongly suggestive and likely correct, but still carrying one unresolved validation item. The current evidence does not suggest that the pipeline is wrong there. It does suggest that this case is numerically or structurally more delicate, and that the thesis should state this clearly.

## Recommended Additional Investigations Before Final Thesis Submission

The following follow-up items would materially strengthen the work from good research quality to a more defensible research-grade standard.

### Priority 1: Explain the honeycomb Hamiltonian residual

Measure how the honeycomb pre-enforcement Hermiticity error changes with:

- moire grid refinement,
- finite-difference step size in the Berry-connection calculation,
- phase-2 interpolation settings,
- and band-count enlargement around the Dirac point.

If the `8.6e-5` residual decreases systematically under refinement, the remaining concern becomes a numerical-convergence issue rather than a conceptual one.

### Priority 2: Quantify spectral sensitivity to Hermitian enforcement

Compare a few low-energy eigenvalues and eigenvectors before and after the final `(H+H^\dagger)/2` projection, especially for the honeycomb case.

The key question is not only whether `H` is slightly non-Hermitian, but whether that non-Hermiticity changes the physics of interest. If the low-energy minibands and mode profiles are stable under the projection, the practical risk is much lower.

### Priority 3: Validate the Dirac case against symmetry-expected physics

For the honeycomb case, directly verify that the expected symmetry signatures near `K` survive:

- degeneracy structure,
- rotational character of the low-energy states,
- and stability of the effective velocity or cone structure under modest numerical refinement.

This is especially important because the honeycomb candidate is the one most likely to be discussed as a conceptually interesting result rather than just a benchmark.

### Priority 4: If feasible, activate one nonzero Born-Huang test run

The present validation cannot say much about `Phi_BH` because it is zeroed out in all saved runs. If the thesis intends to discuss the Born-Huang term beyond saying it was negligible and omitted, then at least one dedicated nonzero validation run should be included.

## Recommended Thesis Wording

### Short version

Final gauge and symmetry sanity checks showed that raw Berry-connection plots can appear irregular without indicating a physical inconsistency. After transforming to gauge-invariant diagnostics, all thesis candidates exhibited the expected crystal symmetry to machine precision. The assembled effective Hamiltonian was nearly Hermitian before explicit numerical symmetrization in the square and hexagonal cases, and remained acceptably close in the honeycomb Dirac case, where the residual was larger but still small. These checks support the validity of the pipeline while identifying the honeycomb case as numerically more delicate.

### More cautious version

The final validation campaign indicates that the envelope-approximation framework is reliable for the square and hexagonal benchmark systems and likely reliable for the honeycomb Dirac case as well. The latter, however, exhibits a noticeably larger pre-symmetrization anti-Hermitian residual in the effective Hamiltonian and should therefore be interpreted with an explicit convergence and robustness caveat.

## Recommended Figures and Tables for the Thesis

Include the following materials in the thesis or appendix:

1. A three-column table containing the full-Hamiltonian Hermiticity residual before and after explicit symmetrization.
2. A three-column table with gauge-invariant symmetry errors for `A_frob` and `A01_mag`.
3. One figure per candidate showing the gauge-invariant norm maps.
4. One figure per candidate showing the Berry-curvature magnitude map.
5. A short explanatory paragraph stating why raw `A_berry` visual noise is not itself a failure criterion.

## Final Conclusion

The strongest defensible conclusion is the following:

The pipeline is physically trustworthy and validated for the square and hexagonal thesis systems. For the honeycomb Dirac system, the present evidence is favorable and probably sufficient for inclusion in the thesis, but a focused refinement study on the pre-enforcement Hermiticity residual would materially strengthen the result and remove the main remaining research-grade caveat.

If no further investigation is performed, the honeycomb result should still be usable, but it should be framed as numerically more delicate rather than as equally settled as the other two benchmark cases.