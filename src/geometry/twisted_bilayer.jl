##################################################################
###### export only the following functions from this module ######
##################################################################

export TwistedBilayer, twisted_bilayer, commensurate_twist_angles
export twist_angle, n_monolayer_cells, monolayer_bz_gvectors
export moire_lattice, layer_lattice, monolayer_lattice, supercell_matrix




#############################################################
###### Rotations and coincidence (moiré) lattices       #####
#############################################################

"""
    _rotation2(theta) -> SMatrix{2,2,Float64}

The counter-clockwise rotation matrix `R(θ)` acting on cartesian column vectors.

This is the ONLY 2x2 rotation in the package, and it is deliberately shared rather than
re-derived: `interfaces/moiredb/moiredb.jl` (the Cartesian-frame reconstruction) and
`hamiltonians/ertca/` (the disregistry map and the mean-orientation rotation) both call it.
They each used to carry a private `_rot2` with an identical body, and because `src/` is a
single module those were not two functions but two METHODS of one generic -- one typed
`::Float64`, one untyped -- so a `Float64` call from the moire database dispatched into
ERTCA's copy. The bodies agreed, so nothing ever failed; editing either would silently have
changed the other's callers. Add no local rotation helper: call this one.
"""
@inline _rotation2(theta::Real) =
    SMatrix{2,2,Float64}(cos(theta), sin(theta), -sin(theta), cos(theta))

"""
    _rotated_lattice(lat, theta) -> BravaisLattice{2}

The lattice rigidly rotated by `θ` about the origin. With **rows** of `a` as the primitive
vectors, each row transforms as `v -> R v`, i.e. `a_new = a * Rᵀ`.
"""
_rotated_lattice(lat::BravaisLattice{2}, theta::Real)::BravaisLattice{2} =
    BravaisLattice(lat.a * transpose(_rotation2(theta)))



#############################################################
###### The twisted-bilayer geometry                     #####
#############################################################

"""
    TwistedBilayer

The complete geometry of a commensurate twisted bilayer: the monolayer, the two rotated layers,
the moiré (coincidence) superlattice, and the integer matrices tying them together.

Fields:
-------
- `monolayer::BravaisLattice{2}`      the reference (untwisted) layer lattice
- `layers::NTuple{2,BravaisLattice{2}}`  layer 1 rotated by `-θ/2`, layer 2 by `+θ/2`
- `moire::BravaisLattice{2}`          the primitive coincidence lattice of the two layers
- `M::NTuple{2,SMatrix{2,2,Int,4}}`   `a_moire = M[l] * a_layer[l]`; `|det M[1]| = |det M[2]| = N`
- `theta::Float64`                    the twist angle, in radians
- `indices::Union{Nothing,Tuple{Int,Int}}`  the `(m, n)` the twist was built from, if any

Conventions:
------------
Both layers are rotated about the **origin** (a common Bravais point), symmetrically by `∓θ/2`.
The stacking registry (AA vs AB, layer separation) lives with the basis states, not here: this is
Bravais-level geometry, shared by every stacking.

The Brillouin zones:
--------------------
Everything reciprocal follows from the stored lattices -- `reciprocal(moire_lattice(tb))` is the
moiré BZ's lattice, `reciprocal(layer_lattice(tb, l))` the monolayer BZ of layer `l`. Because
`a_moire = M a_layer` forces `b_layer = Mᵀ b_moire`, each layer's reciprocal lattice is the
integer sublattice `M[l] ℤ²` of the moiré reciprocal lattice, so the monolayer BZ is tiled by
exactly `N` copies of the moiré BZ. The `N` moiré-G labels doing the tiling are what
[`monolayer_bz_gvectors`](@ref) returns -- the natural plane-wave set for a continuum model that
must cover the whole monolayer BZ rather than a shell-truncated neighbourhood of its center.
"""
struct TwistedBilayer
    monolayer::BravaisLattice{2}
    layers::NTuple{2,BravaisLattice{2}}
    moire::BravaisLattice{2}
    M::NTuple{2,SMatrix{2,2,Int,4}}
    theta::Float64
    indices::Union{Nothing,Tuple{Int,Int}}
end

"""
    _twisted_bilayer(lat, theta, indices, rcut0; max_grow, maxpoints, atol) -> TwistedBilayer

Shared core of the two public constructors: rotate, intersect, and validate.
"""
function _twisted_bilayer(lat::BravaisLattice{2}, theta::Float64,
                          indices::Union{Nothing,Tuple{Int,Int}}, rcut0::Float64;
                          max_grow::Int, maxpoints::Int, atol::Float64)::TwistedBilayer
    l1 = _rotated_lattice(lat, -theta / 2)
    l2 = _rotated_lattice(lat, +theta / 2)
    moire = _coincidence_lattice(l1, l2, rcut0; max_grow=max_grow, maxpoints=maxpoints,
                                 atol=atol, known_commensurate=!isnothing(indices))
    M1 = _integer_expansion(moire.a, l1.a; atol=atol)
    M2 = _integer_expansion(moire.a, l2.a; atol=atol)
    ### This is the "same number of cells in both layers" guarantee: the layers share the moiré
    ### cell and have equal primitive volumes, so the counts must agree. Checked exactly on the
    ### integer matrices -- a mismatch means the search returned a cell common to neither.
    abs(det(M1)) == abs(det(M2)) || throw(ErrorException(
        "The two layers decompose the moiré cell into different numbers of unit cells " *
        "($(abs(det(M1))) vs $(abs(det(M2)))); this is an internal inconsistency, not a usage error."
    ))
    return TwistedBilayer(lat, (l1, l2), moire, (M1, M2), theta, indices)
end

"""
    twisted_bilayer(lat, m, n; rcut=nothing, max_grow=4, maxpoints=4_000_000,
                    atol=INT_ROUND_ATOL) -> TwistedBilayer

The twisted bilayer indexed by the coprime integers `(m, n)`: the twist angle is the angle
between the lattice vectors `v₁ = m a₁ + n a₂` and `v₂ = n a₁ + m a₂`. Rotating the layers by
`∓θ/2` lands `v₁` of one layer exactly on `v₂` of the other, so ONE coincidence vector always
exists -- and the twist is genuinely commensurate (a full 2D coincidence lattice exists)
whenever `a₁·a₂/|a₁|²` is RATIONAL: always for the triangular (±1/2) and square (0) lattices,
and for rhombic lattices only at rational values. A rhombic lattice with irrational
`a₁·a₂/|a₁|²` shares nothing beyond the `v₁` row at any radius, and the search raises.

For the triangular lattice this is the standard bilayer-graphene / TMD indexing,

    cos θ = (m² + 4mn + n²) / (2 (m² + mn + n²))

so the `(m, m+1)` family gives the familiar sequence θ = 21.79°, 13.17°, 9.43°, 7.34°, 6.01°,
5.09°, 4.41°, 3.89°, ... with `N = m² + mn + n²` monolayer cells per moiré cell -- except when
`3 | m - n`, where the primitive coincidence cell is three times smaller; the search returns the
primitive cell either way, so `n_monolayer_cells` is always the true count.

The indices are convention-independent: they always refer to the **acute** equal-length basis of
the lattice. A triangular lattice is written with γ = 60° in some conventions and γ = 120° in
others (`bravais_lattice(:triangular)` uses 120°), and taking the given basis literally would
make `(2, 1)` mean the literature 21.79° twist in one convention and the trivial 60° symmetry
rotation in the other. When the given basis is obtuse it is therefore re-indexed as
`(-a₂, a₁)` -- same lattice, same lengths, same orientation, acute angle -- before `v₁`, `v₂`
are formed.

Args:
-----
- `lat`: the monolayer Bravais lattice; the construction requires `|a₁| = |a₂|` (triangular,
  square, rhombic). For unequal-length primitive vectors give the twist angle directly.
- `m`, `n`: positive coprime integers with `m ≠ n`
- `rcut`: search radius for the coincidence lattice (default `1.2 |v₁|`, grown automatically)

Returns:
--------
- `TwistedBilayer`

Notes:
------
The moiré cell is found by an exhaustive coincidence search rather than by closed-form supercell
vectors, so it is guaranteed primitive and the "same `N` in both layers" invariant is verified on
the integer matrices rather than assumed.
"""
function twisted_bilayer(lat::BravaisLattice{2}, m::Integer, n::Integer;
                         rcut::Union{Nothing,Real}=nothing, max_grow::Int=8,
                         maxpoints::Int=4_000_000, atol::Float64=INT_ROUND_ATOL)::TwistedBilayer
    (m > 0 && n > 0) || throw(ArgumentError(
        "Twist indices must be positive; got (m, n) = ($m, $n)."))
    m != n || throw(ArgumentError(
        "(m, n) = ($m, $n) gives a zero twist angle; the two layers coincide."))
    g = gcd(m, n)
    g == 1 || throw(ArgumentError(
        "Twist indices must be coprime: ($m, $n) = $g × ($(m ÷ g), $(n ÷ g)) describes the same " *
        "twist as the reduced pair."))

    a1 = primitive_vector(lat, 1)
    a2 = primitive_vector(lat, 2)
    ### `v₂` is `v₁` with the indices swapped, so `|v₁| = |v₂|` -- the premise of the whole
    ### construction -- holds iff `|a₁| = |a₂|`. Physical near-equality is what matters here
    ### (the same reasoning as SHELL_RTOL), not the float-noise tolerances.
    isapprox(norm(a1), norm(a2); rtol=SHELL_RTOL) || throw(ArgumentError(
        "The (m, n) construction requires primitive vectors of equal length; this lattice has " *
        "|a₁| = $(norm(a1)) and |a₂| = $(norm(a2)). Give the twist angle directly: " *
        "`twisted_bilayer(lat, theta)`."))

    ### Index in the ACUTE basis (see the docstring): `(-a₂, a₁)` is the acute, right-handed,
    ### length-preserving re-indexing of any obtuse equal-length basis, so the literature
    ### `(m, n)` angles come out whether the lattice was written with γ = 60° or γ = 120°.
    if dot(a1, a2) < 0
        a1, a2 = -a2, a1
    end

    v1 = m * a1 + n * a2
    v2 = n * a1 + m * a2
    theta = acos(clamp(dot(v1, v2) / (norm(v1) * norm(v2)), -1.0, 1.0))
    ### `rcut` default: `v₁` bisected is itself a coincidence vector, so the shortest moiré basis
    ### vector is never longer than |v₁|; the margin covers ties, and the growth loop covers
    ### low-symmetry (rhombic) lattices whose second basis vector can be longer.
    r0 = isnothing(rcut) ? 1.2 * norm(v1) : Float64(rcut)
    return _twisted_bilayer(lat, theta, (Int(m), Int(n)), r0;
                            max_grow=max_grow, maxpoints=maxpoints, atol=atol)
end

twisted_bilayer(lat::BravaisLattice{2}, mn::Tuple{Integer,Integer}; kwargs...) =
    twisted_bilayer(lat, mn[1], mn[2]; kwargs...)

"""
    twisted_bilayer(lat, theta; degrees=true, rcut=nothing, max_grow=4,
                    maxpoints=4_000_000, atol=INT_ROUND_ATOL) -> TwistedBilayer

The twisted bilayer at an explicit twist angle. The angle must be commensurate: the constructor
searches for the coincidence lattice of the two rotated layers and raises if none exists within
its (growing) search radius.

Args:
-----
- `theta`: the twist angle, in degrees by default (`degrees=false` for radians), strictly
  between 0 and 180°
- `rcut`: initial search radius (default `3 ℓ / (2 sin(θ/2))`, three times the naive moiré
  length scale, grown automatically up to `max_grow` doublings)

Notes:
------
Prefer the integer-index constructor when the indices are known -- it derives the angle exactly
and seeds the search with the right radius. This method exists for angles taken from elsewhere
(a paper's θ = 21.787°, a `commensurate_twist_angles` row) and verifies commensurability rather
than assuming it. An angle rounded to a few digits is NOT the commensurate angle it abbreviates:
pass the exact value (e.g. from the indices) or expect the incommensurability error.
"""
function twisted_bilayer(lat::BravaisLattice{2}, theta::Real;
                         degrees::Bool=true, rcut::Union{Nothing,Real}=nothing,
                         max_grow::Int=4, maxpoints::Int=4_000_000,
                         atol::Float64=INT_ROUND_ATOL)::TwistedBilayer
    th = degrees ? deg2rad(float(theta)) : float(theta)
    ### compared against Float64(π), not the Irrational: `deg2rad(180.0) < π` is TRUE because
    ### Float64(π) rounds below the true π, so the Irrational comparison lets 180° through
    0 < th < Float64(π) || throw(DomainError(theta,
        "The twist angle must lie strictly between 0 and 180 degrees."))
    lmax = maximum(norm(primitive_vector(lat, i)) for i in 1:2)
    r0 = isnothing(rcut) ? max(3 * lmax / (2 * sin(th / 2)), 2 * lmax) : Float64(rcut)
    return _twisted_bilayer(lat, th, nothing, r0;
                            max_grow=max_grow, maxpoints=maxpoints, atol=atol)
end




#############################################################
###### Accessors                                        #####
#############################################################

"""
    twist_angle(tb; degrees=true) -> Float64

The twist angle between the two layers, in degrees by default.
"""
twist_angle(tb::TwistedBilayer; degrees::Bool=true)::Float64 =
    degrees ? rad2deg(tb.theta) : tb.theta

"""
    n_monolayer_cells(tb) -> Int

The number `N` of monolayer unit cells inside the moiré cell, per layer -- equal for the two
layers by construction. This is also the number of moiré Brillouin zones tiling the monolayer
BZ, i.e. the length of [`monolayer_bz_gvectors`](@ref).
"""
n_monolayer_cells(tb::TwistedBilayer)::Int = abs(det(tb.M[1]))

"""
    monolayer_lattice(tb) -> BravaisLattice{2}

The reference (untwisted) monolayer lattice.
"""
monolayer_lattice(tb::TwistedBilayer)::BravaisLattice{2} = tb.monolayer

"""
    moire_lattice(tb) -> BravaisLattice{2}

The primitive moiré (coincidence) lattice. Its reciprocal, `reciprocal(moire_lattice(tb))`,
is the moiré Brillouin zone's lattice.
"""
moire_lattice(tb::TwistedBilayer)::BravaisLattice{2} = tb.moire

"""
    layer_lattice(tb, l) -> BravaisLattice{2}

The lattice of layer `l`: layer 1 is the monolayer rotated by `-θ/2`, layer 2 by `+θ/2`.
"""
function layer_lattice(tb::TwistedBilayer, l::Integer)::BravaisLattice{2}
    l in (1, 2) || throw(ArgumentError("Layer index must be 1 or 2; got $l."))
    return tb.layers[l]
end

"""
    supercell_matrix(tb, l) -> SMatrix{2,2,Int}

The integer matrix `M` with `a_moire = M * a_layer(l)`. Feeds directly into `supercell`: in
`:unit_expand` mode it maps a tight-binding model of layer `l` onto the moiré cell, and in
passed to `fold` it folds that layer's Brillouin zone onto the moiré one.
"""
function supercell_matrix(tb::TwistedBilayer, l::Integer)::SMatrix{2,2,Int}
    l in (1, 2) || throw(ArgumentError("Layer index must be 1 or 2; got $l."))
    return tb.M[l]
end




#############################################################
###### Padding the monolayer BZ with moiré G vectors    #####
#############################################################

"""
    monolayer_bz_gvectors(tb, layer) -> Vector{SVector{2,Int}}
    monolayer_bz_gvectors(tb)        -> Vector{SVector{2,Int}}

The moiré reciprocal vectors (as integer coordinates on the moiré `b` grid) that tile layer
`layer`'s monolayer Brillouin zone -- exactly `N = n_monolayer_cells(tb)` of them, one per moiré
BZ copy, each the minimal-length representative of its class (i.e. folded into the monolayer BZ).
Without a layer argument, the union of the two layers' sets.

Purpose:
--------
A continuum model that must be valid over the WHOLE monolayer BZ -- not just near its center --
needs its plane-wave basis to cover that zone, and shell truncation (`plane_wave_cell(...;
shells=n)`) is the wrong selector for that: it grows isotropically around Γ. The right set is
one moiré G per coset of the layer's reciprocal lattice `M ℤ²` inside the moiré reciprocal
lattice `ℤ²` (`b_layer = Mᵀ b_moire`), which is what this returns. Feed it straight into
`plane_wave_cell(moire_lattice(tb); gvectors=monolayer_bz_gvectors(tb), schema=...)`.

Notes:
------
The two layers' sets are representative choices of DIFFERENT coset structures (each layer's
reciprocal is its own integer sublattice `M[l] ℤ²` of the moiré reciprocal) -- yet in practice
they typically coincide element-for-element: the minimal-length representative of each coset
tends to lie inside both layers' reciprocal Voronoi cells at once. Either way, the load-bearing
property is one-representative-per-coset: a set that represents some coset of a layer twice
(any shell or radius truncation with margin) duplicates that layer's monolayer states -- two
basis states one layer reciprocal apart are the SAME Wannier comb up to a phase. On
BZ-boundary ties (several equivalent G of equal length) the lexicographically
smallest integer label is kept, so the set is deterministic. The result contains `G = 0` and is
sorted by cartesian length, then lexicographically.
"""
function monolayer_bz_gvectors(tb::TwistedBilayer, layer::Integer)::Vector{SVector{2,Int}}
    M = supercell_matrix(tb, layer)
    N = n_monolayer_cells(tb)
    B = recip_matrix(tb.moire)
    bmin = minimum(norm(primitive_recip_vector(tb.moire, m)) for m in 1:2)
    datol = LATTICE_ATOL * bmin
    ### EXACT integer coset label: adj(M)·g = det(M)·M⁻¹g, so `adj(M)·g mod |det M|` names the
    ### coset of `g` in ℤ²/Mℤ² with no tolerance at all. The float label wrap(M⁻¹g) rounded to a
    ### digit grid split cosets whose coordinates (multiples of 1/N) straddled a rounding
    ### boundary -- first seen at N = 61, invisible at the small N of the early tests.
    adjM = _int_adjugate(M)

    ### Search ball: the minimal representative of every coset lies inside the monolayer BZ
    ### (the Wigner-Seitz cell of the layer's reciprocal lattice), whose covering radius is at
    ### most half the diagonal of its primitive parallelogram, ≤ (|b₁ˡ| + |b₂ˡ|)/2.
    lay = layer_lattice(tb, layer)
    Rmax = sum(norm(primitive_recip_vector(lay, m)) for m in 1:2) / 2 + datol
    ### the g_shells box bound: the m-th moiré coordinate of a cartesian G is aₘ⋅G / 2π
    box = ntuple(m -> ceil(Int, Rmax * norm(primitive_vector(tb.moire, m)) / (2π)) + 1, 2)

    ### One pass finds the cosets AND their minimal representatives, keyed on the exact integer
    ### label above -- every member of a coset produces the identical key, bit for bit.
    best = Dict{SVector{2,Int},Tuple{Float64,SVector{2,Int}}}()
    for n1 in -box[1]:box[1], n2 in -box[2]:box[2]
        g = SVector{2,Int}(n1, n2)
        d = norm(B * g)
        d <= Rmax || continue
        key = _coset_label(adjM, N, g)
        cur = get(best, key, nothing)
        if isnothing(cur) || d < cur[1] - datol ||
           (abs(d - cur[1]) <= datol && Tuple(g) < Tuple(cur[2]))
            best[key] = (d, g)
        end
    end
    length(best) == N || throw(ErrorException(
        "Found $(length(best)) reciprocal cosets instead of N = $N inside a provably-sufficient " *
        "ball; this is an internal inconsistency, not a usage error."
    ))

    out = [v[2] for v in values(best)]
    sort!(out; by = g -> (norm(B * g), Tuple(g)))
    return out
end

function monolayer_bz_gvectors(tb::TwistedBilayer)::Vector{SVector{2,Int}}
    B = recip_matrix(tb.moire)
    gs = collect(Set(vcat(monolayer_bz_gvectors(tb, 1), monolayer_bz_gvectors(tb, 2))))
    sort!(gs; by = g -> (norm(B * g), Tuple(g)))
    return gs
end




#############################################################
###### Browsing the commensurate angles                 #####
#############################################################

"""
    commensurate_twist_angles(lat; max_index=5, degrees=true, kwargs...) -> Vector{NamedTuple}

A table of the commensurate twist angles of `lat` reachable with indices up to `max_index`:
one row `(m, n, theta, N, moire_a)` per coprime pair `m > n ≥ 1`, sorted by angle. `N` is the
number of monolayer cells per moiré cell (per layer) and `moire_a` the moiré lattice parameter.

Notes:
------
The angles are reported exactly as the `(m, n)` construction produces them; pairs related by the
lattice's point-group rotations describe the same physical moiré and appear as separate rows with
equal `N` (on the triangular lattice, θ and 60° - θ). Each row is built by the full coincidence
search, so `N` is the primitive count, formula subtleties included.
"""
function commensurate_twist_angles(lat::BravaisLattice{2}; max_index::Int=5,
                                   degrees::Bool=true, kwargs...)
    max_index >= 2 || throw(DomainError(max_index, "`max_index` must be at least 2."))
    rows = [begin
                tb = twisted_bilayer(lat, m, n; kwargs...)
                (m=m, n=n, theta=twist_angle(tb; degrees=degrees),
                 N=n_monolayer_cells(tb), moire_a=lattice_parameters(tb.moire).a)
            end
            for m in 2:max_index for n in 1:m-1 if gcd(m, n) == 1]
    sort!(rows; by = r -> r.theta)
    return rows
end




#############################################################
###### Display                                          #####
#############################################################

function Base.show(io::IO, ::MIME"text/plain", tb::TwistedBilayer)
    N = n_monolayer_cells(tb)
    pm = lattice_parameters(tb.monolayer)
    po = lattice_parameters(tb.moire)
    println(io, "TwistedBilayer")
    print(io, "  twist angle: θ = ", _fmt(rad2deg(tb.theta)), "°")
    isnothing(tb.indices) ? println(io) : println(io, "   from indices (m, n) = ", tb.indices)
    println(io, "  layers: monolayer rotated by -θ/2 (layer 1) and +θ/2 (layer 2)")
    println(io, "  commensuration: N = ", N, " monolayer cells per moiré cell, per layer")
    println(io, "  moiré cell: a = ", _fmt(po.a), " = ", _fmt(po.a / pm.a), " × monolayer a,",
                "  γ = ", _fmt(po.gamma), "°")
    print(io, "  monolayer BZ = ", N, " moiré BZs (see `monolayer_bz_gvectors`)")
end

Base.show(io::IO, tb::TwistedBilayer) =
    print(io, "TwistedBilayer(θ = ", _fmt(rad2deg(tb.theta)), "°, N = ",
          n_monolayer_cells(tb), ")")
