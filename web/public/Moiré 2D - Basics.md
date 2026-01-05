# 2D Moiré Lattice

## Primitive vectors and transformation

- Let $\{\vec{a}_1,\vec{a}_2\}$ be the primitive vectors of the first lattice.  
- Define the second lattice by
  $$
  \vec{a}_i' \;=\; s\,R(\vartheta)\,\vec{a}_i,
  \quad
  s\in\mathbb{Q},\;\vartheta\in\mathbb{R}.
  $$

---

## General affine transformations (to be explored)

The transformation above uses uniform scaling and rotation. More general transformations in GL(2,ℝ) are possible:

| Transformation | Matrix Form | Parameters | Physical Meaning |
|----------------|-------------|------------|------------------|
| Uniform scaling | $\begin{pmatrix} s & 0 \\ 0 & s \end{pmatrix}$ | $s > 0$ | Isotropic expansion/contraction |
| Anisotropic scaling | $\begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$ | $s_x, s_y > 0$ | Different scaling along axes |
| Rotation | $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ | $\theta \in \mathbb{R}$ | Rigid rotation |
| Shear (x-direction) | $\begin{pmatrix} 1 & h \\ 0 & 1 \end{pmatrix}$ | $h \in \mathbb{R}$ | Skewing along x |
| Shear (y-direction) | $\begin{pmatrix} 1 & 0 \\ h & 1 \end{pmatrix}$ | $h \in \mathbb{R}$ | Skewing along y |
| Reflection | Any matrix with $\det(M) = -1$ | - | Mirror transformation |
| General GL(2,ℝ) | $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ | $ad - bc \neq 0$ | Any invertible linear map |

**Note:** Translation doesn't affect the moiré pattern due to translational invariance.

---

## Coincidence (commensurability) condition

A common lattice vector $\vec{L}$ must satisfy
$$
\vec{L}
= m_1\,\vec{a}_1 + m_2\,\vec{a}_2
= n_1\,\vec{a}_1' + n_2\,\vec{a}_2',
\quad
m_i,n_i\in\mathbb{Z}\setminus\{0\}.
$$

Inserting $\vec{a}_i' = s\,R(\vartheta)\,\vec{a}_i$ yields
$$
m_1\,\vec{a}_1 + m_2\,\vec{a}_2
= s\,R(\vartheta)\bigl(n_1\,\vec{a}_1 + n_2\,\vec{a}_2\bigr).
$$

---

## Complex‐number representation

1.  **Basis choice**  
    Map the real basis to the complex plane:
    $$
    \vec{a}_1 \mapsto 1,
    \quad
    \vec{a}_2 \mapsto \tau = \tau_x + i\,\tau_y \in \mathbb{C}.
    $$
2.  **Integer combinations**  
    $$
    u \;=\; m_1 + m_2\,\tau,
    \quad
    z \;=\; n_1 + n_2\,\tau.
    $$
3.  **Transformed combination**  
    $$
    n_1\,\vec{a}_1' + n_2\,\vec{a}_2'
    \;\longmapsto\;
    s\,e^{i\vartheta}\,z.
    $$
4.  **Coincidence condition**  
    $$
    u = s\,e^{i\vartheta}\,z.
    $$

---

## Scale (magnitude) condition

Taking absolute values,
$$
|u| = s\,|z|
\;\Longrightarrow\;
s^2
= \frac{|u|^2}{|z|^2}
= \frac{(m_1 + m_2\,\tau_x)^2 + (m_2\,\tau_y)^2}
       {(n_1 + n_2\,\tau_x)^2 + (n_2\,\tau_y)^2}.
$$

---

## Rotation (angular) condition

Taking arguments,
$$
\vartheta = \arg(u) - \arg(z),
$$
where
$$
\arg(u)
= \atan2\!\bigl(\Im\,u,\;\Re\,u\bigr)
= \atan2\!\bigl(m_2\,\tau_y,\;m_1 + m_2\,\tau_x\bigr),
$$
$$
\arg(z)
= \atan2\!\bigl(\Im\,z,\;\Re\,z\bigr)
= \atan2\!\bigl(n_2\,\tau_y,\;n_1 + n_2\,\tau_x\bigr).
$$

---

## Closed‐form for the twist angle

A concise expression for the moiré rotation is
$$
\boxed{
\vartheta(m_1,m_2,n_1,n_2;\tau_x,\tau_y)
= \atan2\!\bigl(m_2\,\tau_y,\;m_1 + m_2\,\tau_x\bigr)
- \atan2\!\bigl(n_2\,\tau_y,\;n_1 + n_2\,\tau_x\bigr).
}
$$

---

## Number‐theoretic remark

- Although each integer tuple $(m_1,m_2,n_1,n_2)$ yields finite $s$ and $\vartheta$,  
  elementary number theory provides **no closed‐form enumeration** of all commensurate solutions.
