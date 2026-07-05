Here is the clean version.

**What MPB actually solves**

MPB solves the **Maxwell eigenproblem for the magnetic field** (H) in a periodic dielectric. In the docs/manual this is the eigenproblem
[
\hat\Theta H = \left(\frac{\omega}{c}\right)^2 H,
\qquad
\hat\Theta = \nabla \times \varepsilon^{-1} \nabla \times,
]
with the transverse constraint (\nabla!\cdot H = 0). So the underlying eigenvalue is **((\omega/c)^2)**, not (\lambda). ([mpb.readthedocs.io][1])

**What MPB returns to you**

The user-facing output variable `freqs` is a list of **frequencies** for the bands at the current (k)-point. MPB’s tutorial states that these frequency eigenvalues are returned in units of **(c/a)**, where (a) is the chosen unit of distance in the model. So MPB does **not** directly return wavelength (\lambda). It returns a **normalized frequency**. ([mpb.readthedocs.io][2])

A very useful way to write the returned number is
[
\tilde f ;=; \frac{f}{c/a}
;=; \frac{\omega a}{2\pi c}
;=; \frac{a}{\lambda_0},
]
because the docs also say the corresponding vacuum wavelength is
[
\lambda_0 = \frac{a}{\tilde f}.
]
So if you were asking “is it (\lambda) or (\omega)?”, the precise answer is:

* **internally:** the eigenproblem is in **(\omega^2/c^2)**,
* **returned/output:** MPB gives **normalized frequency** (\tilde f = \omega a/(2\pi c)=a/\lambda_0),
* **not** wavelength directly. ([mpb.readthedocs.io][1])

**Where the lattice constant (a) comes in**

In MPB, (a) is fundamentally the **distance unit / lattice unit** used to nondimensionalize the problem. Geometry is specified in lattice coordinates, the lattice vectors are set by `geometry-lattice`, and the actual lattice-vector lengths are given by `size × basis-size`. Resolution is specified in **pixels per lattice unit**. So all lengths are dimensionless until you decide what one lattice unit means physically. ([mpb.readthedocs.io][2])

That means:

* if you model a crystal with physical lattice constant (a_{\rm phys}), then you typically set **1 MPB length unit = (a_{\rm phys})**;
* all band frequencies are then reported in units of **(c/a_{\rm phys})**;
* conversion back to SI is
  [
  f_{\rm phys} = \tilde f,\frac{c}{a_{\rm phys}},\qquad
  \omega_{\rm phys} = 2\pi \tilde f,\frac{c}{a_{\rm phys}},\qquad
  \lambda_0 = \frac{a_{\rm phys}}{\tilde f}.
  ]
  This is exactly why changing only the absolute scale (a) rescales the frequencies inversely, while leaving the dimensionless band diagram unchanged. ([mpb.readthedocs.io][3])

**How (k) is normalized**

`k_points` are specified in the basis of the **reciprocal lattice vectors** (G_j), defined by
[
R_i\cdot G_j = 2\pi \delta_{ij}.
]
So (a) also enters through reciprocal space via (G\sim 2\pi/a). A dimensionless entry like `0.5` in a `k_point` means “half of the relevant reciprocal-lattice basis vector,” not “0.5 m(^{-1})”. ([mpb.readthedocs.io][2])

**What the eigenvectors are**

MPB stores/works with eigenvectors in a **transverse plane-wave basis**. The low-level `get_eigenvectors` / `save_eigenvectors` routines expose these as the raw planewave amplitudes. The physical fields (E,H,\dots) are Bloch fields; MPB stores the periodic Bloch envelope and multiplies by (e^{ik\cdot x}) when requested/output unless you disable the Bloch phase. ([mpb.readthedocs.io][1])

**One important edge case**

If you enable the experimental negative-(\varepsilon) mode, MPB says it outputs **real frequency squared** instead of possibly imaginary frequencies. So the normal “`freqs` = normalized frequency” interpretation has that exception. ([mpb.readthedocs.io][4])

## Bottom line

For normal MPB runs:

* **MPB solves:** (\hat\Theta H = (\omega/c)^2 H)
* **MPB returns:** `freqs` = **dimensionless normalized frequencies**
* **Those are:** (\tilde f = \omega a/(2\pi c) = a/\lambda_0)
* **So not (\lambda) directly**
* **(a)** is the length scale used to nondimensionalize the entire problem, entering both real-space geometry and reciprocal-space (k)-normalization. ([mpb.readthedocs.io][1])

[1]: https://mpb.readthedocs.io/en/latest/Developer_Information/ "Developer Information - MPB Documentation"
[2]: https://mpb.readthedocs.io/en/latest/Scheme_User_Interface/ "User Interface - MPB Documentation"
[3]: https://mpb.readthedocs.io/en/latest/Python_Tutorial/?utm_source=chatgpt.com "Tutorial - MPB Documentation - Read the Docs"
[4]: https://mpb.readthedocs.io/en/stable/Python_User_Interface/ "User Interface - MPB Documentation"

# Geometry Conventions

The official MPB convention is this: `geometry_lattice` defines a cell **centered on the origin**; `basis1/2/3` specify only the **directions** of the lattice basis vectors; their lengths are set separately by `basis_size`; and the actual periodic lattice vectors are then determined by `size × basis_size`. Resolution is in pixels per lattice unit, and `k_points` are given in the basis of the reciprocal lattice vectors. ([mpb.readthedocs.io][2])

For geometric objects, MPB takes a list of objects and a default material. If objects overlap, **later objects in the list win**. If `ensure_periodicity=True` (the default), MPB treats each object as repeated by all lattice translations. All object vectors — including `center`, axes, etc. — are specified in the **lattice basis**, not raw Cartesian coordinates. In 2D, only the intersection with the **xy-plane** matters. ([mpb.readthedocs.io][3])

For materials, isotropic media use scalar `epsilon`, while anisotropic media use `epsilon_diag` and `epsilon_offdiag`, and those tensor components are defined with respect to the **Cartesian x/y/z axes**, not the lattice basis. When MPB outputs epsilon, it can output the Cartesian tensor components and inverse-tensor components. ([mpb.readthedocs.io][1])

For boundaries, MPB does not just do a crude pointwise mask. Internally it averages (\varepsilon) and (\varepsilon^{-1}) over a local mesh around each grid point, and at dielectric interfaces it constructs an **effective dielectric tensor** at boundary points. So the geometry you describe by objects becomes a smoothed/effective grid representation before the eigensolver sees it. ([mpb.readthedocs.io][4])

The one thing I would **not** promote to an official MPB law from your note is the exact statement that the extracted `get_epsilon()` array is sampled at ((i+0.5)/N) and therefore always needs an `np.roll(..., N//2, ...)`. Your empirical rule may be perfectly valid for your pipeline alignment, but I did **not** find that exact indexing convention documented explicitly in the MPB manual. I would keep that as an **empirically validated interoperability rule**, not as the canonical MPB convention.  ([mpb.readthedocs.io][5])

## How MPB assembles (\varepsilon) for a 2-atomic basis

If your primitive cell has two basis objects, say A and B, with centers (\tau_A) and (\tau_B) in **lattice coordinates**, then MPB conceptually builds a periodic dielectric by repeating both objects over all lattice translations (T=n_1R_1+n_2R_2). At a point (r), the material is:

[
\varepsilon(r)=
\begin{cases}
\varepsilon_A & r \in \bigcup_T (O_A+\tau_A+T),[2mm]
\varepsilon_B & r \in \bigcup_T (O_B+\tau_B+T),[2mm]
\varepsilon_{\text{default}} & \text{otherwise,}
\end{cases}
]

with the caveat that if two objects overlap, the **later one in the geometry list overrides the earlier one**. If A or B is anisotropic, then inside that object MPB assigns the corresponding **Cartesian dielectric tensor** instead of a scalar epsilon. After that, MPB performs its mesh averaging / boundary tensor construction on the grid. ([mpb.readthedocs.io][3])

## Small MPB-only geometric convention note

You can use this as the clean reference:

### MPB geometric conventions

**1. Computational cell**
`geometry_lattice` defines the periodic cell, and the cell is centered on the origin. `basis1/2/3` specify lattice **directions** only; `basis_size` sets their lengths; `size` gives the cell size in units of those basis vectors. The actual lattice vectors are therefore determined by `size × basis_size`, not by the raw magnitudes of `basis1/2/3`. ([mpb.readthedocs.io][2])

**2. Coordinate convention**
Object centers, axes, and other 3-vectors are specified in the **lattice basis**. In other words, a vector (u=(u_1,u_2,u_3)) means the Cartesian position obtained by expanding in the lattice basis, not a raw Cartesian xyz triple unless the lattice basis happens to be Cartesian. In 2D, only the xy cross-section is used. ([mpb.readthedocs.io][1])

**3. Objects and periodicity**
The dielectric structure is defined by `geometry` plus `default_material`. With `ensure_periodicity=True`, objects are periodically repeated by lattice translations. If objects overlap, the one appearing later in `geometry` takes precedence. MPB has no special notion of “atom A” and “atom B”; they are simply separate objects in the geometry list. ([mpb.readthedocs.io][3])

**4. Materials**
For isotropic media, use scalar `epsilon`. For anisotropic media, use `epsilon_diag` and `epsilon_offdiag`. These tensor components are defined in the **Cartesian** xyz frame, not the lattice frame. ([mpb.readthedocs.io][1])

**5. Distances**
Scalar distances like `radius` are ordinary scalar lengths in your chosen length unit. They are **not** defined by the norm of `basis1`. If you want a radius (r/a=0.2), you should express that scalar consistently in the same length unit used for the lattice. ([mpb.readthedocs.io][6])

**6. Grid / epsilon representation**
MPB evaluates the geometry as a periodic material function, then averages (\varepsilon) and (\varepsilon^{-1}) on a local mesh and constructs an effective boundary tensor where needed. `get_epsilon()` returns this grid dielectric representation. The exact array-index origin convention is not clearly documented, so any extra rolling/alignment step should be treated as a pipeline-specific postprocessing rule unless re-verified empirically. ([mpb.readthedocs.io][4])

The uploaded note I checked is here: [GEOMETRIC_CONVENTIONS.md](sandbox:/mnt/data/GEOMETRIC_CONVENTIONS.md). 

The biggest correction to lock in is this: **do not think “raw `basis1/basis2` = actual lattice vectors.”** In official MPB, that is false. That one misunderstanding alone can absolutely create the kind of half-off, scale-off, and coordinate-off mismatches you are trying to eliminate.

[1]: https://mpb.readthedocs.io/en/latest/Scheme_User_Interface/ "User Interface - MPB Documentation"
[2]: https://mpb.readthedocs.io/en/stable/Scheme_User_Interface/ "User Interface - MPB Documentation"
[3]: https://mpb.readthedocs.io/en/stable/Python_User_Interface/?utm_source=chatgpt.com "Python User Interface - MPB Documentation - Read the Docs"
[4]: https://mpb.readthedocs.io/en/latest/Developer_Information/?utm_source=chatgpt.com "Developer Information - MPB Documentation"
[5]: https://mpb.readthedocs.io/en/latest/Python_User_Interface/ "User Interface - MPB Documentation"
[6]: https://mpb.readthedocs.io/en/latest/Python_Tutorial/?utm_source=chatgpt.com "Tutorial - MPB Documentation - Read the Docs"
