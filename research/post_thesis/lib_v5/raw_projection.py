"""Raw projected envelope operator (model raw_direct_projection_v1), built on the
product space (slow torus grid) x (monolayer plane waves) by direct operator
composition — no hand-expanded coefficient algebra anywhere.

The hermitized TM lifted operator is exactly quadratic in (D_r + eta D_R):

    L = -rho (D_r + eta D_R)^2 rho,   rho(r; delta(R)) = eps_loc^{-1/2},

so the eta-expansion L = L0 + eta L1 + eta^2 L2 is EXACT (three orders, no truncation).
On the product space, D_r is diagonal in the plane-wave factor (carrier included),
D_R = spectral derivative on the slow grid (exact on grid functions), and rho is a
block-diagonal multiplication operator (its registry dependence delta(R) makes the two
factors couple; the operator composition executes every product rule mechanically).

The raw projection restricts the fast factor to a frozen retained frame U(delta_bar):
H_P = (I x U)^dag L (I x U). The only approximations relative to the exact commensurate
supercell problem are then the two-scale lift itself (diagonal restriction R = eta r)
and the frame truncation — exactly what the raw-vs-direct comparison measures.
"""
import numpy as np
import scipy.linalg as sla

from . import lattice as lat
from . import materials as mat
from . import micro_pwe as mp


def _mult_matrix(field_fft, ns):
    """Plane-wave multiplication matrix from a fine-grid FFT coefficient array
    (indexed by wrapped integers), restricted to the PW index set ns."""
    Nf = field_fft.shape[0]
    h = np.asarray(ns)
    d1 = (h[:, 0][:, None] - h[:, 0][None, :]) % Nf
    d2 = (h[:, 1][:, None] - h[:, 1][None, :]) % Nf
    return field_fft[d1, d2]


def rho_tables(coeffs_fn, deltas, gmax, fine=64, orders=(0,)):
    """Exact-to-machine multiplication matrices of rho = eps^{-1/2} (and registry
    derivatives if requested) at each registry point. coeffs_fn(delta) -> Fourier dict.
    rho is smooth and positive, so its Fourier series decays exponentially; a fine
    sampling grid of 64^2 reaches ~1e-14 truncation."""
    ns = mp.pw_set(gmax)
    out = []
    for d in deltas:
        e = mat.sample(coeffs_fn(d), fine, fine)
        assert e.min() > 0
        rho = e ** -0.5
        out.append(_mult_matrix(np.fft.fft2(rho) / fine ** 2, ns))
    return out


def _spectral_d(N):
    """Exact spectral derivative matrix d/ds on the periodic unit grid (N points)."""
    F = np.fft.fft(np.eye(N), axis=0)
    ik = 2j * np.pi * np.fft.fftfreq(N) * N
    return np.real_if_close(np.fft.ifft(ik[:, None] * F, axis=0), tol=1e6)


def product_space_orders(lattice, coeffs_fn, kappa0, gmax, Ns, registry_of_R,
                        fine=64):
    """The three exact orders (L0, L1, L2) of the lifted hermitized operator on the
    product space (slow Ns x Ns grid, C-order) x (monolayer PWs).

    registry_of_R(s1, s2) -> delta: the registry map evaluated on slow fractional
    coordinates (the caller supplies the exact commensurate map, e.g. delta = A2-based
    winding for the moire cell; the map must wind integer times around the torus).
    Slow derivatives are spectral (exact for grid-resolved fields); D_R here is the
    derivative with respect to slow FRACTIONAL coordinates contracted with the moire
    reciprocal basis is the caller's responsibility via the returned raw operators:
    L1/L2 use d/ds_a paired with the metric g^{ab} of the slow cell passed as `gslow`.
    """
    B0 = lat.monolayer_basis(lattice)
    ns = mp.pw_set(gmax)
    npw = len(ns)
    Brec = 2 * np.pi * np.linalg.inv(B0).T
    k0 = Brec @ np.array([float(kappa0[0]), float(kappa0[1])])
    kG = np.array([k0 + Brec @ np.array(h, float) for h in ns])
    Dr = [np.diag(1j * kG[:, i]) for i in (0, 1)]

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [registry_of_R(S1.reshape(-1)[j], S2.reshape(-1)[j])
              for j in range(Ns * Ns)]
    rho = rho_tables(coeffs_fn, deltas, gmax, fine=fine)

    nslow = Ns * Ns
    dim = nslow * npw

    def blockdiag(mats):
        M = np.zeros((dim, dim), complex)
        for j in range(nslow):
            M[j * npw:(j + 1) * npw, j * npw:(j + 1) * npw] = mats[j]
        return M

    R = blockdiag(rho)
    D1s = _spectral_d(Ns)
    # slow derivatives on the C-order (s1, s2) grid, acting on the slow factor
    P1 = np.kron(np.kron(D1s, np.eye(Ns)), np.eye(npw))
    P2 = np.kron(np.kron(np.eye(Ns), D1s), np.eye(npw))
    Dfast = [np.kron(np.eye(nslow), d) for d in Dr]

    def dot(Aops, Bops):
        return sum(a @ b for a, b in zip(Aops, Bops))

    L0 = -(R @ dot(Dfast, Dfast) @ R)
    return {"L0": L0, "R": R, "Dfast": Dfast, "Pslow": (P1, P2),
            "npw": npw, "nslow": nslow, "ns": ns, "kG": kG, "deltas": deltas}


def assemble(orders, eta, slow_to_cart):
    """Full lifted operator L0 + eta L1 + eta^2 L2 with the slow gradient expressed in
    Cartesian: D_R_i = sum_a slow_to_cart[i, a] * d/ds_a  (slow_to_cart = the inverse
    transpose of the slow cell basis, so that D_r and D_R live in the same Cartesian
    frame)."""
    R = orders["R"]
    P1, P2 = orders["Pslow"]
    DR = [slow_to_cart[i, 0] * P1 + slow_to_cart[i, 1] * P2 for i in (0, 1)]
    Df = orders["Dfast"]
    L1 = -(R @ (sum(Df[i] @ DR[i] + DR[i] @ Df[i] for i in (0, 1))) @ R)
    L2 = -(R @ (sum(DR[i] @ DR[i] for i in (0, 1))) @ R)
    return orders["L0"] + eta * L1 + eta ** 2 * L2, L1, L2


def mono_hermitized(coeffs, k_cart, B0, gmax, fine=64):
    """Truncated hermitized-collocation monolayer operator h = M[rho] |k+G|^2 M[rho]
    on the PW window — the SAME discretization family as the product-space operator
    (note (P eps P)^{-1/2} != P rho P: the generalized pencil truncated at the same
    gmax is a DIFFERENT operator at O(truncation); one family must be used
    throughout)."""
    ns = mp.pw_set(gmax)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(B0, float)).T
    kG = np.array([np.asarray(k_cart, float) + Brec @ np.array(h, float)
                   for h in ns])
    e = mat.sample(coeffs, fine, fine)
    R = _mult_matrix(np.fft.fft2(e ** -0.5) / fine ** 2, ns)
    h = R @ np.diag((kG ** 2).sum(1)) @ R
    return 0.5 * (h + h.conj().T), R, ns, kG


def frozen_frame(lattice, coeffs_fn, delta_bar, kappa0, gmax, band_ids, fine=64):
    """Frozen-frame columns: eigenvectors of the truncated hermitized monolayer
    operator at the carrier (orthonormal exactly; same family as the product space)."""
    B0 = lat.monolayer_basis(lattice)
    Brec = 2 * np.pi * np.linalg.inv(B0).T
    k0 = Brec @ np.array([float(kappa0[0]), float(kappa0[1])])
    h, R, ns, kG = mono_hermitized(coeffs_fn(delta_bar), k0, B0, gmax, fine)
    w, V = sla.eigh(h)
    return V[:, list(band_ids)], w[list(band_ids)]


def project(L, U, nslow):
    """H_P = (I x U)^dag L (I x U) on the slow-grid x retained-band space."""
    npw = U.shape[0]
    nb = U.shape[1]
    T = np.kron(np.eye(nslow), U)
    H = T.conj().T @ L @ T
    herm = np.linalg.norm(H - H.conj().T) / max(np.linalg.norm(H), 1e-300)
    return H, herm
