//! Moiré stacking registries (monatomic case) and utilities
//!
//! This module implements robust detection of monatomic stacking registries,
//! i.e. the surface-science positions "top / bridge / hollow". AB/BA are not
//! defined for a single-site basis and are intentionally *not* used here.
//!
//! Storage & conventions:
//! - All inputs are taken in Cartesian coordinates.
//! - For registry *types*, labeled shifts τ are stored in the *reference layer-1*
//!   primitive cell as 2D vectors (z = 0 when embedded in 3D).
//! - Centers are computed as r_τ = M^{-1}(τ - d0) and wrapped into a moiré cell.
//!
//! Math (see stacking_registries.md for derivation):
//! - F := R_θ T,   T := A2 A1^{-1}  (mismatch/strain followed by twist)
//! - M := I - F
//! - L := M^{-1} A1  are moiré primitives (2×2)
//!
//! Public API returns Vecs of plain types so it is easy to expose to Python.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use std::collections::HashMap;

use crate::lattice::lattice_types::Bravais2D;
use crate::lattice::lattice2d::Lattice2D;
// NOTE: adjust this path to your crate layout if needed.
use crate::moire_lattice::moire2d::Moire2D;

/// Labeled registry center (monatomic case).
#[derive(Debug, Clone)]
pub struct RegistryCenter {
    /// Human-readable label for the registry type (e.g., "top", "bridge_a1", "hollow_2").
    pub label: String,
    /// Registry shift τ in Cartesian (z = 0).
    pub tau: Vector3<f64>,
    /// Wrapped center position in Cartesian (z = 0) within one moiré cell.
    pub position: Vector3<f64>,
}

// ========================= Small helpers (2D <-> 3D) =========================

#[inline]
fn mat3_to_mat2_xy(m: &Matrix3<f64>) -> Matrix2<f64> {
    Matrix2::new(m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)])
}

#[inline]
fn vec3_to_vec2_xy(v: &Vector3<f64>) -> Vector2<f64> {
    Vector2::new(v[0], v[1])
}

#[inline]
fn vec2_to_vec3_xy(v: &Vector2<f64>) -> Vector3<f64> {
    Vector3::new(v[0], v[1], 0.0)
}

#[inline]
fn rot2(theta: f64) -> Matrix2<f64> {
    let c = theta.cos();
    let s = theta.sin();
    Matrix2::new(c, -s, s, c)
}

#[inline]
fn solve_2x2(m: &Matrix2<f64>, rhs: Vector2<f64>) -> Result<Vector2<f64>, String> {
    m.try_inverse()
        .ok_or_else(|| "Singular 2x2 matrix; cannot invert".to_string())
        .map(|inv| inv * rhs)
}

#[inline]
fn inv2(m: &Matrix2<f64>) -> Result<Matrix2<f64>, String> {
    m.try_inverse()
        .ok_or_else(|| "Singular 2x2 matrix; cannot invert".to_string())
}

/// Reduce x into the fundamental parallelogram spanned by columns of A (2×2).
/// Wraps to [0, 1) in fractional coordinates.
fn wrap_in_cell(x: Vector2<f64>, a: &Matrix2<f64>) -> Result<Vector2<f64>, String> {
    let c = solve_2x2(a, x)?; // fractional coords in basis A
    let c_wrapped = Vector2::new(c[0] - c[0].floor(), c[1] - c[1].floor());
    Ok(a * c_wrapped)
}

// ========================= Moiré geometry =========================

/// M = I - R_θ * (A2 A1^{-1}) for explicit-twist use.
pub fn moire_matrix(
    a1: &Matrix2<f64>,
    a2: &Matrix2<f64>,
    theta: f64,
) -> Result<Matrix2<f64>, String> {
    let a1_inv = a1
        .try_inverse()
        .ok_or_else(|| "A1 is singular; cannot invert".to_string())?;
    let t = a2 * a1_inv; // mismatch / strain mapping
    let f = rot2(theta) * t; // include twist explicitly
    Ok(Matrix2::identity() - f)
}

/// Use when layer-2 lattice already includes the full transform vs. layer-1:
/// M = I - (A2 A1^{-1}) (i.e., do NOT multiply R_θ again).
pub fn moire_matrix_from_layers(l1: &Lattice2D, l2: &Lattice2D) -> Result<Matrix2<f64>, String> {
    let a1 = mat3_to_mat2_xy(l1.direct_basis());
    let a2 = mat3_to_mat2_xy(l2.direct_basis());
    let a1_inv = a1
        .try_inverse()
        .ok_or_else(|| "Layer-1 basis is singular; cannot invert".to_string())?;
    let t = a2 * a1_inv;
    Ok(Matrix2::identity() - t)
}

/// L = M^{-1} A1
pub fn moire_primitives_from_m(
    a1: &Matrix2<f64>,
    m: &Matrix2<f64>,
) -> Result<Matrix2<f64>, String> {
    Ok(inv2(m)? * a1)
}

pub fn moire_primitives(
    a1: &Matrix2<f64>,
    a2: &Matrix2<f64>,
    theta: f64,
) -> Result<Matrix2<f64>, String> {
    let m = moire_matrix(a1, a2, theta)?;
    moire_primitives_from_m(a1, &m)
}

pub fn moire_primitives_from_layers(
    l1: &Lattice2D,
    l2: &Lattice2D,
) -> Result<Matrix2<f64>, String> {
    let a1 = mat3_to_mat2_xy(l1.direct_basis());
    let m = moire_matrix_from_layers(l1, l2)?;
    moire_primitives_from_m(&a1, &m)
}

// ========================= Stable inverse for twist-only case =========================

/// Analytic inverse of (I - R_θ), numerically stable for small θ.
/// Correct closed form:
///   (I - R_θ)^{-1} = 1/2 * [ I + cot(θ/2) J ],  with J = [[0,-1],[1,0]]
fn inv_i_minus_rot(theta: f64) -> Result<Matrix2<f64>, String> {
    let half = 0.5 * theta;
    let s = half.sin();
    if s.abs() < 1e-16 {
        return Err("Angle too close to 0 mod 2π; moiré period diverges".to_string());
    }
    let cot = half.cos() / s;
    let j = Matrix2::new(0.0, -1.0, 1.0, 0.0);
    let result = 0.5 * (Matrix2::identity() + cot * j);

    // DEBUG: Print the computed matrix and its norm
    eprintln!(
        "  inv_i_minus_rot(θ = {:.6} rad = {:.2}°):",
        theta,
        theta.to_degrees()
    );
    eprintln!("    half = {:.6}, sin(half) = {:.6}", half, s);
    eprintln!("    cot(half) = {:.6}", cot);
    eprintln!(
        "    M^-1 = [{:.6}, {:.6}; {:.6}, {:.6}]",
        result[(0, 0)],
        result[(0, 1)],
        result[(1, 0)],
        result[(1, 1)]
    );
    eprintln!("    ||M^-1|| = {:.6}", result.norm());

    Ok(result)
}

/// Check if a 2x2 matrix is a rotation matrix (orthogonal with determinant +1).
fn is_rotation_matrix(m: &Matrix2<f64>, eps: f64) -> bool {
    // Check determinant ≈ 1
    let det = m.determinant();
    if (det - 1.0).abs() > eps {
        return false;
    }

    // Check orthogonality: M * M^T ≈ I
    let mt = m.transpose();
    let should_be_identity = m * mt;
    let identity = Matrix2::identity();

    // Check each element
    let diff_norm = (should_be_identity - identity).norm();
    diff_norm <= eps
}

/// Extract rotation angle from a 2x2 rotation matrix.
/// Returns angle in radians, in the range (-π, π].
fn extract_rotation_angle(r: &Matrix2<f64>) -> Result<f64, String> {
    if !is_rotation_matrix(r, 1e-10) {
        return Err("Matrix is not a valid rotation matrix".to_string());
    }

    // For rotation matrix R = [[cos θ, -sin θ], [sin θ, cos θ]]
    // We can extract θ = atan2(R[1,0], R[0,0])
    let theta = r[(1, 0)].atan2(r[(0, 0)]);
    Ok(theta)
}

// ========================= Monatomic registry τ-sets =========================

/// Return a labeled set of registry shifts τ for a monatomic Bravais lattice.
/// Labels are chosen to be stable across Bravais types.
/// All τ are given in the *reference* layer-1 primitive cell (Cartesian).
pub fn monatomic_tau_set_for_bravais(
    a1: Vector2<f64>,
    a2: Vector2<f64>,
    bravais: &Bravais2D,
) -> Vec<(String, Vector2<f64>)> {
    match bravais {
        // Hexagonal ↔ triangular Bravais in 2D
        Bravais2D::Hexagonal => {
            let mut out = Vec::new();
            out.push(("top".to_string(), Vector2::new(0.0, 0.0)));
            out.push(("bridge_a1".to_string(), 0.5 * a1));
            out.push(("bridge_a2".to_string(), 0.5 * a2));
            out.push(("bridge_a1_plus_a2".to_string(), 0.5 * (a1 + a2)));
            // Three distinct triangle barycenters per moiré cell
            out.push(("hollow_1".to_string(), (a1 + a2) / 3.0));
            out.push(("hollow_2".to_string(), (a1 + 2.0 * a2) / 3.0));
            out.push(("hollow_3".to_string(), (2.0 * a1 + a2) / 3.0));
            out
        }
        Bravais2D::Square => {
            let mut out = Vec::new();
            out.push(("top".to_string(), Vector2::new(0.0, 0.0)));
            out.push(("bridge_a1".to_string(), 0.5 * a1));
            out.push(("bridge_a2".to_string(), 0.5 * a2));
            out.push(("hollow".to_string(), 0.5 * (a1 + a2))); // parallelogram center
            out
        }
        // Rectangular, Centered-Rectangular, Oblique:
        // use two edge midpoints and the cell center as “hollow”.
        Bravais2D::Rectangular | Bravais2D::CenteredRectangular | Bravais2D::Oblique => {
            let mut out = Vec::new();
            out.push(("top".to_string(), Vector2::new(0.0, 0.0)));
            out.push(("bridge_a1".to_string(), 0.5 * a1));
            out.push(("bridge_a2".to_string(), 0.5 * a2));
            out.push(("hollow".to_string(), 0.5 * (a1 + a2)));
            out
        }
    }
}

/// Build the monatomic τ-set for a given layer (uses its Bravais type).
pub fn monatomic_tau_set_for_layer(layer: &Lattice2D) -> Vec<(String, Vector2<f64>)> {
    let (a1_3, a2_3) = layer.direct_base_vectors();
    let a1 = vec3_to_vec2_xy(&a1_3);
    let a2 = vec3_to_vec2_xy(&a2_3);
    monatomic_tau_set_for_bravais(a1, a2, &layer.bravais_type())
}

// ========================= Registry centers (monatomic) =========================

/// Compute registry centers for a supplied τ-list, using *explicit θ*.
/// This must only be used if `a2` is the *unrotated* lattice and a twist is
/// being applied by this function. For near twist-only (T ≈ I), a stable
/// analytic inverse of (I - R_θ) is used.
pub fn registry_centers(
    a1: &Matrix2<f64>,
    a2: &Matrix2<f64>,
    theta: f64,
    d0: Vector2<f64>,
    tau_list: &[(String, Vector2<f64>)],
) -> Result<(Matrix2<f64>, Vec<RegistryCenter>), String> {
    // Build T and test if near-identity (twist-only)
    let a1_inv = a1
        .try_inverse()
        .ok_or_else(|| "A1 is singular; cannot invert".to_string())?;
    let t = a2 * a1_inv;
    let near_identity = (t - Matrix2::identity()).norm() < 1e-12;

    // M and its inverse
    let m = Matrix2::identity() - rot2(theta) * t;
    let m_inv = if near_identity {
        // Use negative angle to match moiré lattice orientation convention
        inv_i_minus_rot(-theta)? // stable
    } else {
        inv2(&m)? // general case
    };

    // L = M^{-1} A1 (used as moiré primitives for wrapping, unless a known moiré basis exists)
    let l = m_inv * a1;

    let mut centers = Vec::with_capacity(tau_list.len());
    for (label, tau) in tau_list.iter() {
        let r = m_inv * (*tau - d0);
        let r_wrapped = wrap_in_cell(r, &l)?;
        centers.push(RegistryCenter {
            label: label.clone(),
            tau: vec2_to_vec3_xy(tau),
            position: vec2_to_vec3_xy(&r_wrapped),
        });
    }
    Ok((l, centers))
}

/// Compute registry centers when layer-2 already includes the full transform
/// relative to layer-1. This does *not* apply the twist again.
/// Centers are wrapped using the *actual* moiré basis if available.
pub fn registry_centers_from_layers(
    l1: &Lattice2D,
    l2: &Lattice2D,
    d0_cart: Vector2<f64>,
    tau_list: &[(String, Vector2<f64>)],
) -> Result<(Matrix2<f64>, Vec<RegistryCenter>), String> {
    let a1 = mat3_to_mat2_xy(l1.direct_basis());
    let m = moire_matrix_from_layers(l1, l2)?;
    let l = moire_primitives_from_m(&a1, &m)?;
    let m_inv = inv2(&m)?;

    let mut centers = Vec::with_capacity(tau_list.len());
    for (label, tau) in tau_list.iter() {
        let r = m_inv * (*tau - d0_cart);
        let r_wrapped = wrap_in_cell(r, &l)?;
        centers.push(RegistryCenter {
            label: label.clone(),
            tau: vec2_to_vec3_xy(tau),
            position: vec2_to_vec3_xy(&r_wrapped),
        });
    }
    Ok((l, centers))
}

// ========================= Moire2D convenience methods =========================

impl Moire2D {
    /// Monatomic τ-set for this moiré system, labeled and returned as 3D vectors (z = 0).
    /// Uses the Bravais type and primitive vectors of `lattice_1` as the reference cell.
    pub fn monatomic_tau_set(&self) -> Vec<(String, Vector3<f64>)> {
        let tau2 = monatomic_tau_set_for_layer(&self.lattice_1);
        tau2.into_iter()
            .map(|(lbl, v2)| (lbl, vec2_to_vec3_xy(&v2)))
            .collect()
    }

    /// M = I - (A2 A1^{-1}) computed from the two layer lattices.
    pub fn moire_matrix_from_layers_2d(&self) -> Result<Matrix2<f64>, String> {
        moire_matrix_from_layers(&self.lattice_1, &self.lattice_2)
    }

    /// Moiré primitives computed directly from the two layer lattices.
    pub fn moire_primitives_from_layers_2d(&self) -> Result<Matrix2<f64>, String> {
        moire_primitives_from_layers(&self.lattice_1, &self.lattice_2)
    }

    /// Registry centers (monatomic) using the existing transformed layers.
    /// `d0` is the global in-plane shift (z ignored). Centers are wrapped using
    /// the *actual* moiré basis (`self.direct`) to ensure consistency.
    ///
    /// **DEPRECATED**: Use `registry_centers_monatomic()` instead for better numerical stability.
    #[deprecated(
        since = "0.1.2",
        note = "Use registry_centers_monatomic() for automatic numerical optimization"
    )]
    pub fn registry_centers_monatomic_from_layers(
        &self,
        d0: Vector3<f64>,
    ) -> Result<(Matrix2<f64>, Vec<RegistryCenter>), String> {
        let d0_xy = vec3_to_vec2_xy(&d0);
        let tau_list2d = monatomic_tau_set_for_layer(&self.lattice_1);

        let a1 = mat3_to_mat2_xy(self.lattice_1.direct_basis());
        let m = moire_matrix_from_layers(&self.lattice_1, &self.lattice_2)?;
        let l = moire_primitives_from_m(&a1, &m)?;
        let m_inv = inv2(&m)?;
        let moire_basis_2d = mat3_to_mat2_xy(&self.direct);

        let mut centers = Vec::with_capacity(tau_list2d.len());
        for (label, tau) in tau_list2d.iter() {
            let r = m_inv * (*tau - d0_xy);
            // Wrap using actual moiré lattice basis, not the derived L, for safety.
            let r_wrapped = wrap_in_cell(r, &moire_basis_2d)?;
            centers.push(RegistryCenter {
                label: label.clone(),
                tau: vec2_to_vec3_xy(tau),
                position: vec2_to_vec3_xy(&r_wrapped),
            });
        }
        Ok((l, centers))
    }

    /// Registry centers (monatomic) using explicit θ. Use only if `lattice_2`
    /// is the *unrotated* primitive and the twist is applied here. Otherwise
    /// the rotation would be applied twice.
    ///
    /// **DEPRECATED**: Use `registry_centers_monatomic()` instead for automatic detection and optimization.
    #[deprecated(
        since = "0.1.2",
        note = "Use registry_centers_monatomic() for automatic numerical optimization"
    )]
    pub fn registry_centers_monatomic_with_theta(
        &self,
        d0: Vector3<f64>,
    ) -> Result<(Matrix2<f64>, Vec<RegistryCenter>), String> {
        let a1 = mat3_to_mat2_xy(self.lattice_1.direct_basis());
        let a2 = mat3_to_mat2_xy(self.lattice_2.direct_basis());
        let d0_xy = vec3_to_vec2_xy(&d0);
        let tau_list2d = monatomic_tau_set_for_layer(&self.lattice_1);
        registry_centers(&a1, &a2, self.twist_angle, d0_xy, &tau_list2d)
    }

    /// Registry centers (monatomic) using the most appropriate numerical method.
    ///
    /// This is the **recommended API** for computing registry centers. It automatically:
    /// 1. Detects if the transformation is pure twist vs. general (twist + strain/shear)

    /// Build preliminary local Bravais lattices + minimal bases at each stacking registry.
    ///
    /// This helper returns, for every identified monatomic stacking registry ("top", "bridge_*", "hollow_*"),
    /// a tuple consisting of:
    ///   * A Bravais lattice (`Lattice2D`) cloned from layer 1 primitives (no moiré supercell) – suffixed "_preliminary"
    ///   * A minimal basis (Vec<Vector3<f64>>) giving atomic positions inside that local cell in Cartesian coordinates.
    ///     For the current monoatomic implementation, the basis contains two atoms:
    ///       - layer-1 atom at (0,0,0)
    ///       - layer-2 atom at the registry shift τ (z = 0)
    ///
    /// Return format: HashMap<label, (Lattice2D, Vec<Vector3<f64>>)>.
    ///
    /// Arguments:
    /// - `d0`: Global in-plane shift applied before determining registry center positions.
    ///
    /// Notes / Limitations:
    /// - The returned lattices are monoatomic Bravais cells; the multi-atomic (Bravais + basis) abstraction
    ///   is not yet integrated into `Lattice2D`. Hence the suffix `_preliminary` and separate basis vector list.
    /// - τ is taken directly from the internal registry data (stored in layer-1 primitive cell coordinates).
    /// - Positions are not wrapped further; τ already lies in the reference primitive cell by construction.
    /// - For hexagonal systems a stray hollow site may have been removed upstream following existing logic.
    ///
    /// TODO: Once multi-atomic lattices are supported, replace this with a proper constructor returning
    ///       a lattice object that embeds the basis internally.
    pub fn local_bravais_and_basis_from_registries_preliminary(
        &self,
        d0: Vector3<f64>,
    ) -> Result<HashMap<String, (Lattice2D, Vec<Vector3<f64>>)>, String> {
        // Use the numerically stable registry computation with L (but we only need τ here).
        let (_l_wrap, centers) = self.registry_centers_monatomic_with_l(d0.clone())?;

        // Clone layer-1 lattice to serve as the local Bravais lattice (no moiré enlargement).
        let base_lat = self.lattice_1.clone();

        let mut map: HashMap<String, (Lattice2D, Vec<Vector3<f64>>)> = HashMap::new();
        for c in centers.into_iter() {
            // Minimal two-atom basis: layer1 at origin; layer2 at τ
            let mut basis = Vec::with_capacity(2);
            basis.push(Vector3::new(0.0, 0.0, 0.0));
            basis.push(c.tau); // τ already z=0
            map.insert(c.label.clone(), (base_lat.clone(), basis));
        }
        Ok(map)
    }

    /// Build a preliminary local Bravais lattice + minimal basis at an arbitrary in-plane point.
    ///
    /// This selects the nearest stacking registry center (in-plane Euclidean distance) to `point`
    /// using the UNWRAPPED registry centers for continuity, then returns the same data format as
    /// `local_bravais_and_basis_from_registries_preliminary` for that single site.
    ///
    /// Return: (registry_label, Lattice2D, basis_vectors)
    ///
    /// Arguments:
    /// - `point`: Cartesian 3D vector (z ignored) inside the moiré plane where a local approximation is desired.
    /// - `d0`: Global in-plane shift (z ignored).
    ///
    /// Strategy:
    /// 1. Compute unwrapped registry centers r_τ = M^{-1}(τ - d0).
    /// 2. Find the label with minimal |r_τ - point| in the XY plane.
    /// 3. Return layer-1 Bravais lattice clone + two-site basis [ (0,0,0), τ ].
    ///
    /// Limitations:
    /// - Nearest-center approximation; for points far from any registry midpoint, a future interpolation scheme
    ///   (e.g. barycentric across triangle of centers) could yield smoother local disregistry.
    /// - Monoatomic only; see TODO regarding multi-atomic support.
    ///
    /// TODO: Generalize to interpolate τ for arbitrary points (not just snapping to nearest registry).
    pub fn local_bravais_and_basis_at_point_preliminary(
        &self,
        point: Vector3<f64>,
        d0: Vector3<f64>,
    ) -> Result<(String, Lattice2D, Vec<Vector3<f64>>), String> {
        // Unwrapped centers for continuous positions
        let centers = self.registry_centers_monatomic_unwrapped(d0.clone())?;
        if centers.is_empty() {
            return Err("No registry centers available".to_string());
        }
        // Find nearest in-plane
        let mut best = None;
        for c in centers.iter() {
            let dx = c.position.x - point.x;
            let dy = c.position.y - point.y;
            let dist2 = dx * dx + dy * dy;
            if best.as_ref().map(|(_, d)| dist2 < *d).unwrap_or(true) {
                best = Some((c, dist2));
            }
        }
        let (nearest, _d2) = best.unwrap();
        let base_lat = self.lattice_1.clone();
        let basis = vec![Vector3::new(0.0, 0.0, 0.0), nearest.tau];
        Ok((nearest.label.clone(), base_lat, basis))
    }
    /// 2. Uses the numerically stable method for pure twists (small angle expansion)
    /// 3. Uses general matrix inversion for mixed transformations
    /// 4. Always wraps using the actual moiré basis from `self.direct` for consistency
    ///
    /// Arguments:
    /// - `d0`: Global in-plane shift vector (z component ignored)
    ///
    /// Returns: Registry centers wrapped in the moiré unit cell
    pub fn registry_centers_monatomic(
        &self,
        d0: Vector3<f64>,
    ) -> Result<Vec<RegistryCenter>, String> {
        let d0_xy = vec3_to_vec2_xy(&d0);
        let tau_list2d = monatomic_tau_set_for_layer(&self.lattice_1);

        // PRIORITY: Use stored twist_angle for stable computation
        // For moiré lattices created with explicit twist, always use the stable formula
        eprintln!(
            "Using stored twist_angle = {:.6} rad for stable M^-1 = (I - R_θ)^-1",
            self.twist_angle
        );
        // Use negative angle so that the resulting L matches the moiré lattice orientation
        let m_inv = inv_i_minus_rot(-self.twist_angle)?;

        // Compute the moiré primitive basis L = M^{-1} A1 and use it for wrapping
        let a1 = mat3_to_mat2_xy(self.lattice_1.direct_basis());
        let l = m_inv * a1;
        eprintln!(
            "Using wrapping basis L = M^-1 A1: [ [{:.6}, {:.6}], [{:.6}, {:.6}] ]",
            l[(0, 0)],
            l[(0, 1)],
            l[(1, 0)],
            l[(1, 1)]
        );

        let mut centers = Vec::with_capacity(tau_list2d.len());
        for (label, tau) in tau_list2d.iter() {
            let r = m_inv * (*tau - d0_xy);
            let r_wrapped = wrap_in_cell(r, &l)?;
            centers.push(RegistryCenter {
                label: label.clone(),
                tau: vec2_to_vec3_xy(tau),
                position: vec2_to_vec3_xy(&r_wrapped),
            });
        }
        // Remove the stray hollow for hexagonal Bravais
        if matches!(self.lattice_1.bravais_type(), Bravais2D::Hexagonal) {
            centers.retain(|c| c.label != "hollow_1");
        }

        Ok(centers)
    }

    /// Registry centers (monatomic), UNWRAPPED in Cartesian coordinates.
    /// Returns r_τ = M^{-1}(τ - d0) without wrapping into the moiré cell.
    /// Use this for continuous tracking across twist angle changes; wrap in the UI if needed.
    pub fn registry_centers_monatomic_unwrapped(
        &self,
        d0: Vector3<f64>,
    ) -> Result<Vec<RegistryCenter>, String> {
        let d0_xy = vec3_to_vec2_xy(&d0);
        let tau_list2d = monatomic_tau_set_for_layer(&self.lattice_1);

        // PRIORITY: Use stored twist_angle for moiré lattices created with explicit twist
        // For twisted bilayers created via create_twisted_bilayer(), always use the stable formula
        eprintln!(
            "  Using stored twist_angle = {:.6} rad for stable M^-1 = (I - R_θ)^-1",
            self.twist_angle
        );
        eprintln!("  d0 = ({:.6}, {:.6})", d0_xy.x, d0_xy.y);
        // Use negative angle so positions follow the moiré lattice orientation
        let m_inv = inv_i_minus_rot(-self.twist_angle)?;

        let mut centers = Vec::with_capacity(tau_list2d.len());
        for (label, tau) in tau_list2d.iter() {
            let r = m_inv * (*tau - d0_xy);
            eprintln!(
                "  {} tau=({:.6}, {:.6}) -> r=({:.6}, {:.6})",
                label, tau.x, tau.y, r.x, r.y
            );
            centers.push(RegistryCenter {
                label: label.clone(),
                tau: vec2_to_vec3_xy(tau),
                position: vec2_to_vec3_xy(&r), // UNWRAPPED
            });
        }
        // Remove the stray hollow for hexagonal Bravais
        if matches!(self.lattice_1.bravais_type(), Bravais2D::Hexagonal) {
            centers.retain(|c| c.label != "hollow_1");
        }
        Ok(centers)
    }

    /// Registry centers (monatomic) plus wrapping basis L = M^{-1} A1.
    /// Use this to compute fractional coordinates in the correct moiré basis.
    pub fn registry_centers_monatomic_with_l(
        &self,
        d0: Vector3<f64>,
    ) -> Result<(Matrix2<f64>, Vec<RegistryCenter>), String> {
        let d0_xy = vec3_to_vec2_xy(&d0);
        let tau_list2d = monatomic_tau_set_for_layer(&self.lattice_1);

        // Stable inverse for twist-only systems (use negative angle to match moiré orientation)
        let m_inv = inv_i_minus_rot(-self.twist_angle)?;

        // L = M^{-1} A1 is the correct wrapping basis
        let a1 = mat3_to_mat2_xy(self.lattice_1.direct_basis());
        let l = m_inv * a1;

        let mut centers = Vec::with_capacity(tau_list2d.len());
        for (label, tau) in tau_list2d.iter() {
            let r = m_inv * (*tau - d0_xy);
            let r_wrapped = wrap_in_cell(r, &l)?;
            centers.push(RegistryCenter {
                label: label.clone(),
                tau: vec2_to_vec3_xy(tau),
                position: vec2_to_vec3_xy(&r_wrapped),
            });
        }
        // Remove the stray hollow for hexagonal Bravais
        if matches!(self.lattice_1.bravais_type(), Bravais2D::Hexagonal) {
            centers.retain(|c| c.label != "hollow_1");
        }
        Ok((l, centers))
    }
}

// ========================= Future work (poly-basis) =========================
// - Extend to multi-site bases: generate τ_{αβ} = δ^{(2)}_β − δ^{(1)}_α,
//   and reuse the same center computation. Monatomic reduces to the current set.
// - Provide a generic Delaunay-based τ generator to compute bridges/hollows
//   on arbitrary oblique cells (optional; current formulas cover common Bravais).
