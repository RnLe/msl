# 10° Full-Theory Validation Findings

1. The old direct-validation EA path was not the full theory: it explicitly zeroed Berry and Born-Huang terms and disabled off-diagonal Berry coupling.
2. Phase 3 had a real implementation bug: it could use the wrong moire length, and the mass-trace regularization setting was not actually threaded through the solve.
3. The first full-theory square run still used the generic small-angle registry map instead of the exact commensurate supercell geometry; for 10° this was a factor-of-2 length-scale error.
4. Registry band identities were not tracked across stacking space, so raw MPB band indices could scramble near crossings; overlap-based reordering was required.
5. Phase 3 was initially targeting the wrong spectral region: it found the lowest bound states instead of the folded band-3 comparison window near $\omega_0$.

With those five issues fixed, the 10° full-theory EA vs FDFD comparison improved from a catastrophic mismatch to about $1.10\times10^{-2}$ RMS on the current high-fidelity run.