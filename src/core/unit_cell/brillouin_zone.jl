##################################################################
###### export only the following functions from this module ######
##################################################################

export bz_corners, fold_to_bz


### The FIRST Brillouin zone as a shape, rather than as a sampling. Everything else in this
### package works in the fractional box `[0,1)^D`, which is the right fundamental domain for
### integration and indexing -- it tiles, it has no seam, and its index arithmetic is exact.
### It is the wrong domain for a PICTURE: drawn in fractional coordinates every lattice looks
### like a unit square, so a hexagonal zone does not look hexagonal and a k-point's distance
### from the zone boundary is not the distance on the page. These two functions are the
### translation, and they live here -- with `reduction.jl` and `gram.jl` -- because they are
### metric geometry on a `BravaisLattice` and nothing more. Filing them in the plotting layer
### would have made the shape of the Brillouin zone a property of having Plots loaded.

"""
    bz_corners(lattice::BravaisLattice{2}; shells=3) -> Vector{SVector{2,Float64}}

The corners of the first Brillouin zone -- the Wigner-Seitz cell of the RECIPROCAL lattice --
in cartesian reciprocal coordinates, counter-clockwise from the positive-x axis and **not**
repeating the first corner at the end.

The zone is the set of momenta closer to the origin than to any other reciprocal lattice point,
`{x : x·ĝ ≤ |g|/2 for every reciprocal vector g}`, so it is the intersection of the half-planes
bounded by the perpendicular bisectors. This returns that polygon: four corners on a rectangular
lattice, six on a hexagonal or a generic oblique one.

Args:
-----
- `lattice`: a fully periodic 2D lattice. A partially periodic one is refused -- an aperiodic
  direction has no reciprocal vector, so there is no bisector and no bounded cell.
- `shells`: how many reciprocal shells to include as candidate bisectors. Three is ample for any
  2D lattice (the zone is bounded by the first or second shell even for a very oblique cell);
  the knob exists so a pathological basis can be given more room rather than silently losing a
  facet.

Notes:
------
The candidates are generated from the lattice's own basis, NOT from a reduced one, and every
`shells`-range vector is tested against every other -- an `O(n²)` intersection over a few dozen
half-planes, microseconds, and correct for a skewed basis where a reduction-first approach would
have to justify that it kept the right vectors. Corners are deduplicated on a tolerance scaled
to the reciprocal basis, so a lattice whose zone has a degenerate (four-bisector) corner returns
it once rather than four times.

Example:
--------
    length(bz_corners(bravais_lattice(:square; a=1.0)))       # 4
    length(bz_corners(bravais_lattice(:triangular; a=1.0)))   # 6
"""
function bz_corners(lattice::BravaisLattice{2}; shells::Int=3)
    nperiodic(lattice) == 2 || throw(ArgumentError(
        "`bz_corners`: the first Brillouin zone is bounded by the perpendicular bisectors of " *
        "the reciprocal lattice vectors, and this lattice is periodic along " *
        "$(nperiodic(lattice)) of its 2 directions -- an aperiodic direction contributes no " *
        "reciprocal vector and the cell is unbounded. There is no polygon to return."))
    shells >= 1 || throw(ArgumentError(
        "`bz_corners`: `shells` must be at least 1; got $shells."))
    B = recip_matrix(lattice)
    scale = max(norm(B * SVector(1.0, 0.0)), norm(B * SVector(0.0, 1.0)))
    atol = 1e-9 * scale

    gs = SVector{2,Float64}[]
    for n1 in (-shells):shells, n2 in (-shells):shells
        (n1 == 0 && n2 == 0) && continue
        push!(gs, B * SVector{2,Float64}(n1, n2))
    end

    corners = SVector{2,Float64}[]
    for i in eachindex(gs), j in (i + 1):length(gs)
        g1, g2 = gs[i], gs[j]
        d = g1[1] * g2[2] - g1[2] * g2[1]
        abs(d) > atol * scale || continue                  # parallel bisectors never meet
        c1, c2 = dot(g1, g1) / 2, dot(g2, g2) / 2
        x = SVector{2,Float64}((c1 * g2[2] - c2 * g1[2]) / d,
                               (g1[1] * c2 - g2[1] * c1) / d)
        ### a bisector intersection is a zone CORNER only if it is inside every other half-plane
        inside = true
        for g in gs
            if dot(x, g) > dot(g, g) / 2 + atol * scale
                inside = false
                break
            end
        end
        inside || continue
        any(c -> norm(c - x) <= 1e-7 * scale, corners) || push!(corners, x)
    end
    isempty(corners) && throw(ErrorException(
        "`bz_corners`: no zone corner survived the half-plane test on a lattice with " *
        "reciprocal basis $(Tuple.(eachrow(lattice.b))). This is an internal inconsistency; " *
        "raising `shells` is the only knob."))
    sort!(corners; by=c -> atan(c[2], c[1]))
    return corners
end

"""
    fold_to_bz(lattice::BravaisLattice{2}, k; shells=2) -> SVector{2,Float64}

The cartesian reciprocal point equivalent to fractional momentum `k`, translated by whichever
reciprocal lattice vector brings it into the first Brillouin zone (the one returned by
[`bz_corners`](@ref)).

This is what turns the fractional `[0,1)²` box every mesh samples into a picture of the actual
zone: the box is a parallelogram of the right AREA but the wrong shape, and folding maps it onto
the Wigner-Seitz cell one point at a time.

Args:
-----
- `k`: fractional momentum, any real values -- it need not already be in `[0,1)²`.
- `shells`: search radius in reciprocal lattice vectors. Two is enough for any `k` in the
  fractional box (the furthest corner is `b₁ + b₂`).

Notes:
------
Exactly on a zone boundary two images are equidistant and the choice between them is arbitrary;
the search runs in a fixed order and keeps the first, so the answer is deterministic but the
partner is equally correct. A caller tiling the zone with per-point patches will see such a
point's patch straddle the boundary, which is the honest picture of a sample that sits on it.
"""
function fold_to_bz(lattice::BravaisLattice{2}, k::AbstractVector; shells::Int=2)
    nperiodic(lattice) == 2 || throw(ArgumentError(
        "`fold_to_bz`: this lattice is periodic along $(nperiodic(lattice)) of its 2 " *
        "directions, so there is no bounded Brillouin zone to fold into."))
    length(k) == 2 || throw(DimensionMismatch(
        "`fold_to_bz`: `k` must have 2 components; got $(length(k))."))
    B = recip_matrix(lattice)
    x0 = B * SVector{2,Float64}(k)
    best = x0
    bestn = dot(x0, x0)
    for n1 in (-shells):shells, n2 in (-shells):shells
        (n1 == 0 && n2 == 0) && continue
        x = x0 - B * SVector{2,Float64}(n1, n2)
        n = dot(x, x)
        ### strict, with a tolerance: an exact tie is a boundary point and keeps the incumbent
        if n < bestn - 1e-12 * max(1.0, bestn)
            bestn = n
            best = x
        end
    end
    return best
end
