"""
Scoring utilities for candidate evaluation
"""
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
    # Extract configuration parameters
    kappa_0 = config.get('kappa_0', 1.0)  # Reference curvature
    Delta_0 = config.get('Delta_0', 0.1)  # Reference gap
    v_0 = config.get('v_0', 1.0)  # Reference group velocity
    eps_bg_max = config.get('eps_bg_max', 12.0)
    
    # Extract weights
    # Note: increased w_flat (curvature) and w_vg to emphasize band extrema
    w_flat = config.get('w_flat', 0.40)
    w_gap = config.get('w_gap', 0.20)
    w_parab = config.get('w_parab', 0.15)
    w_vg = config.get('w_vg', 0.20)
    w_contrast = config.get('w_contrast', 0.05)
    
    # Safety parameters
    alpha_parab = config.get('alpha_parab', 0.3)
    beta_parab = config.get('beta_parab', 1.5)
    
    # 1. Band curvature score (reward LARGE curvature for strong extrema)
    # Higher curvature = stronger band extremum = better for EA
    kappa = abs(row.get('curvature_trace', 0.01))
    S_flat = min(1.0, kappa / kappa_0)  # Reward high curvature, cap at 1.0
    
    # 2. Spectral isolation score
    gap_above = row.get('gap_above', 0.0)
    gap_below = row.get('gap_below', 0.0)
    Delta = min(gap_above, gap_below)
    S_gap = Delta / (Delta + Delta_0)
    
    # 3. Parabolic validity score (reward wider quadratic region)
    k_parab = row.get('k_parab', 0.0)
    S_parab = min(1.0, k_parab / (beta_parab * max(kappa_0, 1e-6)))
    
    # 4. Group velocity score (strongly penalize non-zero vg at band extrema)
    # Use quadratic penalty to emphasize zero group velocity
    vg_norm = row.get('vg_norm', 0.0)
    S_vg = 1.0 / (1.0 + (vg_norm / v_0)**2)
    
    # 5. Dielectric contrast score
    eps_bg = row.get('eps_bg', 1.0)
    S_contrast = (eps_bg - 1.0) / (eps_bg_max - 1.0)
    
    # Total weighted score
    S_total = (w_flat * S_flat + 
               w_gap * S_gap + 
               w_parab * S_parab + 
               w_vg * S_vg + 
               w_contrast * S_contrast)
    
    # Validity flag: require sufficiently wide parabolic region
    valid_ea = k_parab >= alpha_parab * max(kappa_0, 1e-6)
    
    return {
        'S_flat': S_flat,
        'S_gap': S_gap,
        'S_parab': S_parab,
        'S_vg': S_vg,
        'S_contrast': S_contrast,
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
