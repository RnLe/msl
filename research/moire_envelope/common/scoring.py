"""
Scoring utilities for candidate evaluation
"""
import math
import numpy as np


def score_candidate(row, config):
    """
    Compute scores for a candidate based on various metrics
    
    Args:
        row: Dictionary with candidate metrics
        config: Configuration dictionary with weights and reference values
        
    Returns:
        dict: Score components and total score
    """
    # Helper utilities
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _safe_positive(value, default=0.0):
        return max(0.0, _safe_float(value, default))

    # Extract configuration parameters
    kappa_0 = config.get('kappa_0', 1.0)  # Reference curvature
    Delta_0 = config.get('Delta_0', 0.1)  # Reference gap
    parab_span_ref = float(config.get('parab_span_ref', 0.35))
    vg_ref = max(1e-6, float(config.get('vg_ref', 0.08)))
    symmetry_ref = max(1e-6, float(config.get('symmetry_ref', 1.0)))
    
    # Extract weights
    # Note: increased w_flat (curvature) and w_vg to emphasize band extrema
    w_flat = config.get('w_flat', 0.60)
    w_gap = config.get('w_gap', 0.35)
    w_parab = config.get('w_parab', 0.25)
    w_vg = config.get('w_vg', 0.15)
    w_linear = config.get('w_linear', 0.20)
    w_sym = config.get('w_sym', 0.15)
    
    # Safety parameters
    alpha_parab = config.get('alpha_parab', 0.3)
    
    # 1. Band curvature score (reward LARGE curvature for strong extrema)
    # Higher curvature = stronger band extremum = better for EA
    kappa = abs(_safe_float(row.get('curvature_trace', 0.01), 0.01))
    S_flat = min(1.0, kappa / kappa_0)  # Reward high curvature, cap at 1.0
    
    # 2. Spectral isolation score
    gap_above = _safe_float(row.get('gap_above', 0.0), 0.0)
    gap_below = _safe_float(row.get('gap_below', 0.0), 0.0)
    Delta = max(0.0, min(gap_above, gap_below))
    S_gap = Delta / (Delta + Delta_0)
    
    # 3. Parabolic validity score (reward wider quadratic region)
    k_parab = _safe_positive(row.get('k_parab'), 0.0)
    k_parab_far = _safe_positive(row.get('k_parab_far'), k_parab)
    kxx = _safe_float(row.get('curvature_xx', 0.0), 0.0)
    kyy = _safe_float(row.get('curvature_yy', 0.0), 0.0)
    kxy = _safe_float(row.get('curvature_xy', 0.0), 0.0)
    trace = kxx + kyy
    det = kxx * kyy - kxy ** 2
    discriminant = max(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = math.sqrt(discriminant)
    k1 = 0.5 * (trace + sqrt_disc)
    k2 = 0.5 * (trace - sqrt_disc)
    abs_sum = abs(k1) + abs(k2) + 1e-9
    anisotropy_ratio = min(1.0, abs(k1 - k2) / abs_sum)
    saddle_penalty = 1.0 if det <= 0 else 0.0
    symmetry_penalty = 0.6 * anisotropy_ratio + 0.4 * saddle_penalty
    symmetry_residual = row.get('parab_error_far')
    symmetry_near = row.get('parab_error_near')
    residual_candidates = [
        abs(_safe_float(val))
        for val in (symmetry_near, symmetry_residual)
        if val is not None and np.isfinite(val)
    ]
    parab_residual = max(residual_candidates) if residual_candidates else None
    if parab_residual is not None:
        symmetry_penalty += 0.2 * min(1.0, parab_residual / (symmetry_ref * 2.0))
    S_parab = max(0.0, 1.0 - symmetry_penalty)

    # 4. Parabola span score (penalize extrema that diverge quickly away from k0)
    parab_span = max(k_parab, k_parab_far)
    S_vg = parab_span / (parab_span + parab_span_ref)

    # 5. Group velocity / linearity penalty (Dirac-like extrema have high |vg|)
    vg_norm = _safe_positive(row.get('vg_norm', 0.0), 0.0)
    S_linear = 1.0 / (1.0 + (vg_norm / vg_ref)**2)

    # 6. Symmetry consistency from near/far samples
    if parab_residual is None:
        symmetry_error = symmetry_ref
    else:
        symmetry_error = parab_residual
    S_sym = 1.0 / (1.0 + (symmetry_error / symmetry_ref)**2)

    # Total weighted score (no normalization of weights)
    S_total = (w_flat * S_flat +
               w_gap * S_gap +
               w_parab * S_parab +
               w_vg * S_vg +
               w_linear * S_linear +
               w_sym * S_sym)
    
    # Validity flag: require sufficiently wide parabolic region
    valid_ea = parab_span >= alpha_parab * max(kappa_0, 1e-6)
    
    return {
        'S_flat': S_flat,
        'S_gap': S_gap,
        'S_parab': S_parab,
        'S_vg': S_vg,
        'S_linear': S_linear,
        'S_sym': S_sym,
        'S_total': S_total,
        'valid_ea_flag': valid_ea,
    }


def participation_ratio(field):
    """
    Compute participation ratio for a field distribution
    
    PR = (∫ |ψ|² dV)² / ∫ |ψ|⁴ dV
    
    Higher PR means more delocalized, lower PR means more localized
    
    Args:
        field: 2D array of field values
        
    Returns:
        float: Participation ratio
    """
    field_abs2 = np.abs(field)**2
    field_abs4 = field_abs2**2
    
    numerator = np.sum(field_abs2)**2
    denominator = np.sum(field_abs4)
    
    if denominator > 0:
        pr = numerator / denominator
    else:
        pr = 0.0
    
    return pr


def field_entropy(field):
    """
    Compute entropy of a normalized field distribution
    
    S = -∫ |ψ|² log(|ψ|²) dV
    
    Args:
        field: 2D array of field values
        
    Returns:
        float: Entropy
    """
    field_abs2 = np.abs(field)**2
    prob = field_abs2 / np.sum(field_abs2)
    
    # Avoid log(0)
    prob = prob[prob > 1e-15]
    
    entropy = -np.sum(prob * np.log(prob))
    
    return entropy


def localization_length(field, R_grid):
    """
    Compute characteristic localization length
    
    Args:
        field: 2D array of field values [Nx, Ny]
        R_grid: Spatial grid [Nx, Ny, 2]
        
    Returns:
        float: RMS radius
    """
    field_abs2 = np.abs(field)**2
    total_weight = np.sum(field_abs2)
    
    if total_weight == 0:
        return 0.0
    
    # Compute center of mass
    r_cm = np.zeros(2)
    for i in range(2):
        r_cm[i] = np.sum(field_abs2 * R_grid[..., i]) / total_weight
    
    # Compute RMS radius
    r_sq = 0.0
    for i in range(2):
        r_sq += np.sum(field_abs2 * (R_grid[..., i] - r_cm[i])**2) / total_weight
    
    return np.sqrt(r_sq)
