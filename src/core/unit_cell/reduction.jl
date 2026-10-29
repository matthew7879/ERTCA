# Lattice-basis reduction: successive minima, the Minkowski (Semaev greedy) reduction, the
# Lagrange/Gauss pair reduction it recurses on, and the 2D canonical reduction the Bravais
# classifier maps tabulated k-points through.
#
# This is the FLOAT counterpart of `utils/intlinalg.jl`. The split is by charter, not by
# subject: exact integer arithmetic there, metric geometry carrying a tolerance here. Both are
# written once at `{D}` and never per dimension.
#
# These routines lived in `spectra/bz_mesh.jl` and `crystallography/bravais_types.jl`, which is
# why `identify_bravais` and `lattice_point_group` both refused in 3D with the message "3D
# reduction is not implemented here yet" while a 3D-capable reduction sat in the tree, four
# hundred lines away in a file about Brillouin-zone meshes. A reduction is not a property of
# the thing that first needed it.
#
# Included EARLY -- before `geometry/` -- because the moire coincidence search needs the same
# reduction the classifier does.

"""
    _reduced_norms(A) -> SVector{D,Float64}

The successive minima of the row lattice of `A`, ascending: the lengths of a Minkowski-reduced
basis, which for `D ≤ 3` -- everything `_hnf_candidates` enumerates -- are exactly the
lengths of the `D` shortest independent lattice vectors. Floats by nature (it scores real
cluster shapes), so it lives here and not in `utils/intlinalg.jl`, whose charter is exactness.

The reduction is Semaev's greedy algorithm (`_greedy_reduce!`). The pairwise (Lagrange)
reduction alone, which this used to be, is optimal only in `D = 2`: in `D = 3` it stops at
bases no PAIR can shorten while a combination of all three rows is far shorter. On
`[8 0 0; 4 2 0; 7 0 2]` it reported `λ₁ = 4`, though `-(8,0,0) + (7,0,2) = (-1,0,2)` has
length `√5`. That overestimate is not benign noise here, because [`optimal_bz_mesh`](@ref)
MAXIMIZES `λ₁` over candidate clusters -- an overestimate is precisely what wins the argmax,
so the search was steered onto the skewed clusters the metric was wrong on.
"""
function _reduced_norms(A::SMatrix{D,D,Float64})::SVector{D,Float64} where {D}
    rows = MVector{D,SVector{D,Float64}}(ntuple(i -> A[i, :], Val(D)))
    _greedy_reduce!(rows)
    return sort(SVector{D,Float64}(ntuple(i -> norm(rows[i]), Val(D))))
end

"""
    _greedy_reduce!(rows) -> rows

Semaev's greedy lattice-basis reduction, in place. One pass sorts the rows by length,
Lagrange-reduces the leading `D-1` of them among themselves, and then replaces the longest row
by the shortest representative of its coset modulo the sublattice those span
(`_coset_shortest`). Passes repeat while the longest row strictly shortens -- a positive
quantity strictly decreasing, so it terminates. The output is Minkowski-reduced, hence realizes
the successive minima, for the `D ≤ 3` used here (Semaev proves it through `D = 4`); the
pairwise step alone would do so only for `D ≤ 2`.
"""
function _greedy_reduce!(rows::MVector{D,SVector{D,Float64}}) where {D}
    D == 1 && return rows
    while true
        _sort_by_length!(rows)
        ### the recursion of the greedy algorithm: the leading rows must themselves be reduced
        ### before the coset step below, which is what makes its bounded scan exhaustive
        for i in 1:(D - 2), j in (i + 1):(D - 1)
            rows[i], rows[j] = _lagrange_pair(rows[i], rows[j])
        end
        cand = _coset_shortest(rows)
        norm(cand) < norm(rows[D]) * (1 - 1e-12) || return rows
        rows[D] = cand
    end
end

@inline function _sort_by_length!(rows::MVector{D,SVector{D,Float64}}) where {D}
    for i in 1:(D - 1), j in (i + 1):D
        norm(rows[j]) < norm(rows[i]) && ((rows[i], rows[j]) = (rows[j], rows[i]))
    end
    return rows
end

### Lagrange (Gauss) reduction of one pair, returned shortest-first: the D = 2 case in full,
### and the base of the greedy recursion above.
function _lagrange_pair(u::SVector{D,Float64}, v::SVector{D,Float64}) where {D}
    while true
        norm(v) < norm(u) && ((u, v) = (v, u))
        n2 = dot(u, u)
        n2 > 0 || return (u, v)                     # degenerate input; nothing to reduce with
        w = v - round(dot(u, v) / n2) * u           # the shortest v + ℤu, exactly
        ### strict-decrease guard: at a fixed point `round` may oscillate between two
        ### equal-length representatives; accepting only real improvements terminates
        norm(w) < norm(v) * (1 - 1e-12) || return (u, v)
        v = w
    end
end

### The closest-vector step: the shortest representative of the longest row modulo the
### sublattice the leading `D-1` rows span. Babai size reduction (subtract the rounded
### projection on each leading row, to a fixed point) lands within one coefficient step of the
### closest vector once those rows are reduced -- which the pass above guarantees -- so the
### surrounding ±2 scan is exhaustive for the D ≤ 3 this is used at.
function _coset_shortest(rows::MVector{D,SVector{D,Float64}}) where {D}
    v = rows[D]
    while true
        shortened = false
        for i in 1:(D - 1)
            n2 = dot(rows[i], rows[i])
            n2 > 0 || continue
            w = v - round(dot(v, rows[i]) / n2) * rows[i]
            norm(w) < norm(v) * (1 - 1e-12) || continue
            v = w
            shortened = true
        end
        shortened || break
    end
    best = v
    for dc in Iterators.product(ntuple(_ -> -2:2, Val(D - 1))...)
        cand = v
        for i in 1:(D - 1)
            cand -= dc[i] * rows[i]
        end
        norm(cand) < norm(best) && (best = cand)
    end
    return best
end


"""
    _greedy_reduce(A::SMatrix{D,D,Float64}) -> (A_red, U)

The Minkowski-reduced basis of the ROW lattice of `A`, together with the exact unimodular `U`
that produces it: `A_red = U * A`, with `det U = +1`.

[`_greedy_reduce!`](@ref) discards `U`, because its original consumer (`_reduced_norms`) reads
only lengths. Everything that carries a TABULATED quantity through a reduction needs it --
`identify_bravais` maps high-symmetry points through, `lattice_point_group` conjugates its
search back -- and that missing return, not the reduction, is what both 3D methods were
actually waiting on.

`U` is PROPER, not merely unimodular. The reduction is defined up to the sign of a basis vector,
so `round.(A_red * A⁻¹)` lands on `det = -1` about half the time (measured: 187-217 of every 400
scrambled bases). An improper `U` is not harmless here: `identify_bravais` maps tabulated
high-symmetry points through it, and an improper map silently exchanges the chirality-sensitive
pair -- K and K' -- which is the same failure `_unimodular_matches` avoids by returning proper
matches first. The last row of `A_red` and of `U` is negated together when needed, which flips
the determinant and leaves `A_red = U * A` and the reduction both intact.

`U` is recovered as `round.(A_red * A⁻¹)` rather than accumulated through the reduction, which is
the idiom `identify_bravais` already uses to rebuild its own map. That recovery goes through a
float inverse, so it DEGRADES on an ill-conditioned basis -- and the failure is in the recovery,
not in the reduction: measured on rhombohedral and triclinic cells, the reduced basis is
correctly sorted and spans the right lattice in 400/400 cases at every conditioning tried, while
the rounding residual `max|A_red A⁻¹ - round(A_red A⁻¹)|` exceeds 1e-6 in 5/400 at cond ~4e3 and
382/400 at cond ~4e10.

That residual is therefore the certificate, and it is checked BEFORE the rounding is trusted.
Measured over 4000 cells across five conditioning decades and three Bravais types, it admits
zero wrong results (0 missed in every bin), at the cost of refusing some recoverable ones on very
skew input (5/400 at cond 4e3, 140/400 at cond 3e6). Refusing is the right side to err on: the
alternative that shipped before this check was a plausible integer matrix that is not the
transformation, which nothing downstream can detect. A caller that hits the refusal should reduce
its basis in stages, or supply a less skew one. The exact fix -- accumulating `U` through the
reduction steps instead of recovering it -- is recorded in the design ledger.
"""
function _greedy_reduce(A::SMatrix{D,D,Float64}) where {D}
    rows = MVector{D,SVector{D,Float64}}(ntuple(i -> A[i, :], Val(D)))
    _greedy_reduce!(rows)
    Ared = _rows_to_matrix(Tuple(rows))
    Uf = Ared * inv(A)
    resid = maximum(abs, Uf .- round.(Uf))
    resid <= GREEDY_U_ATOL || throw(ErrorException(
        "`_greedy_reduce`: the basis change is not recoverable from this basis -- " *
        "`A_red * A⁻¹` misses the integers by $resid (tolerance $GREEDY_U_ATOL). The reduction " *
        "itself succeeded; it is the float inverse that has lost the transformation, which " *
        "happens on a very skew basis (measured: from about cond(A) = 4e3 upward). Reduce the " *
        "basis in stages, or supply a less skew one. Returning the rounded matrix anyway would " *
        "give a plausible integer matrix that is NOT the transformation, which nothing " *
        "downstream can detect."))
    U = SMatrix{D,D,Int}(round.(Int, Uf))
    _int_det_abs(U) == 1 || throw(ErrorException(
        "`_greedy_reduce`: the recovered basis change has |det| = $(_int_det_abs(U)) ≠ 1 on\n" *
        "$A.\nThe reduction is unimodular by construction, so this is an internal " *
        "inconsistency, not a usage error."))
    ### proper, for the reason in the docstring: negate the last row of BOTH, which flips det
    ### and preserves `A_red = U * A` exactly
    if _int_det(U) < 0
        U = _rows_to_matrix_int(ntuple(i -> i == D ? -U[i, :] : U[i, :], Val(D)))
        Ared = _rows_to_matrix(ntuple(i -> i == D ? -Ared[i, :] : Ared[i, :], Val(D)))
    end
    isapprox(SMatrix{D,D,Float64}(U) * A, Ared; atol=1e-8 * maximum(abs, Ared)) || throw(ErrorException(
        "`_greedy_reduce`: the recovered basis change does not reproduce the reduced basis " *
        "on\n$A.\nThis is an internal inconsistency, not a usage error."))
    return Ared, U
end

### the integer sibling of `_rows_to_matrix`, for the sign fix above
@inline _rows_to_matrix_int(rows::NTuple{D,SVector{D,Int}}) where {D} =
    SMatrix{D,D,Int,D * D}(ntuple(k -> rows[1 + (k - 1) % D][1 + (k - 1) ÷ D], D * D))


#############################################################
###### The 2D canonical reduction                       #####
#############################################################

### `_lagrange_reduce` and `_lagrange_pair` above are the same textbook algorithm serving two
### different contracts, and they are deliberately NOT one function -- recorded here, beside
### both, so the merge is not re-proposed a third time.
###
### `_lagrange_reduce` is 2D-only and returns the unimodular `U`. Its contract is a canonical
### representative that is STABLE ACROSS SCALE, because `identify_bravais` maps the tabulated
### high-symmetry points through that `U`: a representative that flips when the lattice is
### expressed in nm rather than Angstrom changes the reported K point. That is what the
### `round(Int, r - 1e-9)` bias buys.
###
### `_lagrange_pair` is D-generic, returns no transform, and is reached only from
### `_greedy_reduce!`. Its consumers read LENGTHS, and two tied representatives have equal
### length by construction, so the bias would buy nothing there; its strict-decrease guard
### exists for a separate reason (oscillation between equal-length representatives at a fixed
### point). Forcing one contract onto both would change `optimal_bz_mesh`'s cluster selection
### for no gain.

"""
    _lagrange_reduce(lat::BravaisLattice{2}) -> (a1, a2, U)

The Lagrange (2D Gauss) reduced basis of a lattice: the two successive minima, ordered
`|a₁| ≤ |a₂|` and sign-normalized so the angle between them lies in `[60°, 90°]`, together with
the unimodular integer matrix `U` such that `[a₁; a₂] = U * lat.a`.

Notes:
------
The reduced cell is what makes classification possible at all: the raw primitive vectors of a
lattice are an arbitrary unimodular relabeling, so their lengths and angle mean nothing until
reduced to the canonical domain. Reduction is unique up to the point symmetries of the lattice
(a square lattice has eight reduced bases), which is why `identify_bravais` matches METRICS and
then searches the small symmetry ambiguity explicitly rather than assuming this function's
tie-breaking.
"""
function _lagrange_reduce(lat::BravaisLattice{2})
    a1 = primitive_vector(lat, 1)
    a2 = primitive_vector(lat, 2)
    U = SMatrix{2,2,Int}(1, 0, 0, 1)
    ### the classic two-step loop terminates because |a2| strictly decreases at every reduction
    for _ in 1:64
        if norm(a1) > norm(a2)
            a1, a2 = a2, a1
            U = SMatrix{2,2,Int}(0, 1, 1, 0) * U
        end
        ### the reduction ratio is dimensionless, so the ONLY scale dependence can come from
        ### rounding ties: at exactly γ = 60° the ratio is 1/2 ± float noise, and a plain
        ### `round` would flip the reduced representative with the lattice constant (K coming
        ### back as (1/3,1/3) at one `a` and (-2/3,1/3) at another). A small downward bias
        ### makes every near-half-integer tie resolve the SAME way at every scale.
        r = dot(a1, a2) / dot(a1, a1)
        m = round(Int, r - 1e-9)
        m == 0 && break
        a2 = a2 - m * a1
        ### row₂ -> row₂ - m·row₁ is [1 0; -m 1]; SMatrix fills column-major, so the shear entry
        ### is the SECOND argument -- the transposed fill was a real (and once-shipped) bug
        U = SMatrix{2,2,Int}(1, -m, 0, 1) * U
    end
    if norm(a1) > norm(a2)                        # one final swap can be pending
        a1, a2 = a2, a1
        U = SMatrix{2,2,Int}(0, 1, 1, 0) * U
    end
    ### the sign flip needs the same noise immunity: an exactly-right angle (dot = 0 ± noise)
    ### must not flip a₂ scale-dependently
    if dot(a1, a2) < -1e-12 * norm(a1) * norm(a2)  # angle into [60°, 90°]
        a2 = -a2
        U = SMatrix{2,2,Int}(1, 0, 0, -1) * U
    end
    return a1, a2, U
end




#############################################################
###### The Niggli (Krivy-Gruber) canonical reduction    #####
#############################################################

"""
    _metric_scale(G::SMatrix{D,D,Float64}) -> Float64

The scale every metric comparison in this file is measured against: `det(G)^(1/D)`.

It is exactly unimodular-invariant -- `det(U G Uᵀ) = det(G)` for `|det U| = 1` -- and it carries
the units of the metric's own entries (length²), so one comparison answers consistently across
the many bases a single reduction visits. `maximum(abs, G)` does neither: it is a property of
the basis you were handed, so the same lattice compares differently before and after a step. A
floor like `max(..., 1.0)` is worse still, because it makes a 3 Å cell sit above the floor and
the same cell in nanometres below it -- the tolerance would then depend on the unit.
"""
@inline _metric_scale(G::SMatrix{D,D,Float64}) where {D} = abs(det(G))^(1 / D)

### the eight Krivy-Gruber transformations, ROW convention: each LEFT-multiplies the basis, and
### each has det +1 so the reduction never changes handedness (see `_greedy_reduce`).
const _KG_M1 = SMatrix{3,3,Int}(0, -1, 0, -1, 0, 0, 0, 0, -1)
const _KG_M2 = SMatrix{3,3,Int}(-1, 0, 0, 0, 0, -1, 0, -1, 0)

"""
    _kg_shear(i, j, v) -> SMatrix{3,3,Int}

The identity with entry `(i, j)` set to `v`: left-multiplying by it sends row `i` to
`row i + v * row j`, and nothing else changes. Determinant 1 for `i != j`.

It exists because the four shear steps of Krivy-Gruber were written as column-major `SMatrix`
literals and ALL FOUR came out transposed -- `a → a - s*c` where the algorithm wants
`c → c - s*a`. The transposed operations are still unimodular and still reduce something, so the
reduction still terminated and still looked canonical on most inputs; it failed by refusing to
converge on a handful of face-centred cells, which is the only reason it was caught. A1 and A2
are symmetric matrices and so were immune, which is exactly the asymmetry that once made
`is_lattice_symmetry` reject 8 of a triangular lattice's 12 operations.
"""
@inline function _kg_shear(i::Int, j::Int, v::Int)
    return SMatrix{3,3,Int}(ntuple(k -> begin
        r, c = (k - 1) % 3 + 1, (k - 1) ÷ 3 + 1
        (r == c ? 1 : 0) + (r == i && c == j ? v : 0)
    end, Val(9)))
end

"""
    _niggli_reduce(A::SMatrix{3,3,Float64}; rtol=1e-8, maxiter=100) -> (A_red, U)

The NIGGLI-reduced basis of the row lattice of `A`, with the exact unimodular `U` giving
`A_red = U * A` and `det U = +1`. Krivy & Gruber, Acta Cryst A32 (1976) 297, steps A1-A8.

This is the CANONICAL reduction, and that is the whole reason it exists beside
[`_greedy_reduce`](@ref), which is not. Minkowski reduction fixes the successive minima but not
the cell: measured on this package's own templates, greedy leaves **65 distinct reduced
signatures across the 14 Bravais types** where a canonical reduction must leave 14. fcc alone
admits **five** all-shortest (Buerger) angle triples -- (60,60,60), (60,60,90), (60,90,120),
(60,120,120), (90,120,120) -- and greedy lands on (60,90,120) more often than on the package's
own 60/60/60 template, so a table keyed on the greedy cell would see the "wrong" cell most of
the time. Niggli's special conditions are exactly what resolve that.

`_greedy_reduce` runs FIRST, as a conditioner. It is not redundant: it hands Krivy-Gruber a
basis of successive minima, after which every STRICT branch of A5-A8 is already satisfied and
only the tie branches and the sign normalisations ever fire. That makes the tolerance below a
TIE-DETECTION tolerance rather than a convergence knob, and it is why one tolerance suffices.

Traps, both of which produce a silently non-canonical cell rather than an error:

- **A7's tie-break is `2ξ < η`, not `2ξ < ζ`.** A6's is `2ξ < ζ`, and copying A6's predicate into
  A7 is the natural slip because the two lines are otherwise identical. Measured on 3847 random
  integer bases with four re-spellings each: correct gives 0 non-convergent runs and 0
  non-canonical lattices; A6's predicate in A7 gives 1056 non-convergent runs and 264
  non-canonical lattices; omitting the tie-break entirely gives 0 non-convergences and **72
  silently non-canonical lattices** -- the worst of the three, because nothing complains.
- **Every comparison is against `eps = rtol * _metric_scale(G)`**, recomputed from the exact
  accumulated `U` each pass rather than updated in place, so float drift cannot accumulate
  across iterations.

The iteration cap refuses by name rather than returning whatever it has reached; a Krivy-Gruber
loop that does not terminate has hit a tolerance problem, and continuing would return a cell that
is not reduced.
"""
function _niggli_reduce(A::SMatrix{3,3,Float64}; rtol::Float64=1e-8, maxiter::Int=100)
    Ag, Ug = _greedy_reduce(A)
    U = one(SMatrix{3,3,Int})
    it = 0
    while true
        it += 1
        it <= maxiter || throw(ErrorException(
            "`_niggli_reduce`: did not converge in $maxiter iterations on\n$A\nA Krivy-Gruber " *
            "loop that does not terminate has hit a tolerance problem, not a hard case; " *
            "returning the current cell would return one that is not reduced."))
        B = SMatrix{3,3,Float64}(U) * Ag
        G = B * transpose(B)
        eps = rtol * _metric_scale(G)
        a, b, c = G[1, 1], G[2, 2], G[3, 3]
        xi, eta, ze = 2G[2, 3], 2G[1, 3], 2G[1, 2]

        ### A1 -- and it FALLS THROUGH to A2 rather than restarting
        if a > b + eps || (abs(a - b) <= eps && abs(xi) > abs(eta) + eps)
            U = _KG_M1 * U
            B = SMatrix{3,3,Float64}(U) * Ag; G = B * transpose(B)
            a, b, c = G[1, 1], G[2, 2], G[3, 3]
            xi, eta, ze = 2G[2, 3], 2G[1, 3], 2G[1, 2]
        end
        ### A2
        if b > c + eps || (abs(b - c) <= eps && abs(eta) > abs(ze) + eps)
            U = _KG_M2 * U
            continue
        end

        ### A3 / A4 -- the sign normalisation, and the step no Minkowski reduction has. It is
        ### what makes "the three angles share a sign" true BY CONSTRUCTION, which is in turn
        ### what makes the reciprocal route to the Setyawan-Curtarolo triclinic setting work.
        ###
        ### The branch is on the PRODUCT `sgn(ξ)sgn(η)sgn(ζ)`: strictly positive selects type I
        ### (A3, make all three positive), anything else selects type II (A4, make all three
        ### non-positive). Testing the counts of positive and negative entries instead is wrong
        ### whenever an entry is exactly zero -- the product is then 0, so A4 must fire, while a
        ### count-based test sends it to A3. Measured: that error left tI, oS, hP and mP each
        ### with TWO reduced cells differing only in the signs of (ξ, η, ζ), i.e. exactly the
        ### type I/II split, with (A, B, C) identical.
        ###
        ### `diag(i, j, k)` acting on the ROWS sends ξ → jkξ, η → ikη, ζ → ijζ, so the choice is
        ### a constraint satisfaction over eight sign patterns rather than a formula. It is
        ### searched rather than derived: the closed form is easy to transcribe with the wrong
        ### index association, and the search is eight iterations of integer arithmetic.
        sgn(x) = x > eps ? 1 : (x < -eps ? -1 : 0)
        sx, sy, sz = sgn(xi), sgn(eta), sgn(ze)
        want = sx * sy * sz > 0 ? 1 : -1
        found = (1, 1, 1)
        for i in (1, -1), j in (1, -1), k in (1, -1)
            ### a zero entry is unconstrained: it stays zero under any sign choice
            (sx == 0 || j * k * sx == want) || continue
            (sy == 0 || i * k * sy == want) || continue
            (sz == 0 || i * j * sz == want) || continue
            found = (i, j, k)
            break
        end
        if found != (1, 1, 1)
            U = SMatrix{3,3,Int}(found[1], 0, 0, 0, found[2], 0, 0, 0, found[3]) * U
            continue
        end

        ### A5-A8. After the greedy conditioner the STRICT branches never fire and these are
        ### pure tie-breaking; the tie predicates are where the canonicality lives.
        if abs(xi) > b + eps || (abs(xi - b) <= eps && 2eta < ze - eps) ||
           (abs(xi + b) <= eps && ze < -eps)
            s = xi > 0 ? 1 : -1
            U = _kg_shear(3, 2, -s) * U          # c -> c - sign(xi) * b
            continue
        end
        if abs(eta) > a + eps || (abs(eta - a) <= eps && 2xi < ze - eps) ||
           (abs(eta + a) <= eps && ze < -eps)
            s = eta > 0 ? 1 : -1
            U = _kg_shear(3, 1, -s) * U          # c -> c - sign(eta) * a
            continue
        end
        ### `2ξ < η` here, NOT `2ξ < ζ` -- see the docstring; the wrong one is silent
        if abs(ze) > a + eps || (abs(ze - a) <= eps && 2xi < eta - eps) ||
           (abs(ze + a) <= eps && eta < -eps)
            s = ze > 0 ? 1 : -1
            U = _kg_shear(2, 1, -s) * U          # b -> b - sign(zeta) * a
            continue
        end
        if xi + eta + ze + a + b < -eps ||
           (abs(xi + eta + ze + a + b) <= eps && 2(a + eta) + ze > eps)
            U = (_kg_shear(3, 1, 1) * _kg_shear(3, 2, 1)) * U   # c -> c + a + b
            continue
        end
        break
    end
    Utot = U * Ug
    ### The A3/A4 diagonal steps may be improper. In an ODD dimension `-I` has determinant -1
    ### and preserves every dot product, so handedness is repaired at the end for free, without
    ### touching the reduced metric. An improper map would otherwise exchange K and K' silently
    ### when `identify_bravais` carries tabulated points through it.
    _int_det(Utot) < 0 && (Utot = -Utot)
    return SMatrix{3,3,Float64}(Utot) * A, Utot
end
