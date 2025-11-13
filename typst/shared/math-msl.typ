// shared/math-msl.typ
// Common math helpers for MSL / moiré lattice theory.

// Shortcuts for common sets.
#let RR = $ bold(R) $
#let CC = $ bold(C) $
#let ZZ = $ bold(Z) $

// Vector / ket / bra helpers.
#let v(vec) = $ bold(#vec) $
#let ket(x) = $ lr(| #x angle.r) $
#let bra(x) = $ lr(angle.l #x |) $
#let braket(a, b) = $ lr(angle.l #a | #b angle.r) $

// Simple operator shorthands.
#let op(name) = $ hat(#name) $
#let dby(sym, var) = $ (diff #sym) / (diff #var) $

// Placeholder macro for moiré lattice parameter sets.
#let moire-params(theta, a, t) = [
  Moiré parameters: $ theta = #theta, space a = #a, space t = #t $.
]
