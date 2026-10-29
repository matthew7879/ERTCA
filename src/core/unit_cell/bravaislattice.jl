##################################################################
###### export only the following functions from this module ######
##################################################################

export BravaisLattice, unimodular_transformation
export ndim, nperiodic, primitive_vectors, primitive_recip_vectors
export primitive_vector, primitive_recip_vector
export direct_matrix, recip_matrix, metric, recip_metric
export cell_volume, bz_volume, length_scale, distance_atol
export lattice_parameters, lattice_from_parameters, reciprocal, same_basis
export fractional_to_real, real_to_fractional
export fractional_to_reciprocal, reciprocal_to_fractional
export unimodular_transformation





#####################################################################
###### Define the Bravais lattice structure and minimal checks ######
#####################################################################

"""
    BravaisLattice{D}

A structure representing a Bravais lattice in D dimensions.
- `a`: D×D matrix of real-space lattice vectors (**rows** are the primitive vectors: `a[i,:]` is aᵢ)
- `b`: D×D matrix of reciprocal lattice vectors (**rows** are the primitive reciprocal vectors)

Notes:
------
The row convention is what `primitive_vectors` returns and what every derived formula assumes.
It gives `aᵀb = 2πI` and `abᵀ = 2πI`, hence `b = 2π*inv(aᵀ)` and `(aᵀ)⁻¹ = b/2π`.

Two consequences worth knowing, since the whole Hamiltonian layer relies on them:
- `fractional_to_real(v, lat) = transpose(a) * v`, so `k_cart ⋅ R_cart = 2π (k_frac ⋅ R_frac)`
  exactly. Bloch phases are therefore computed entirely in fractional coordinates, free of the
  lattice metric.
- The m-th fractional component of a cartesian displacement `d` is `(1/2π) * b[m,:] ⋅ d`, which
  bounds integer translation searches by `|n_m| ≤ ‖b[m,:]‖ * R_cut / 2π` (see `neighbors.jl`).

How the vectors are stored:
---------------------------
`a` and `b` are **properties**, not fields: internally the primitive vectors are held as
`NTuple{D,SVector{D,Float64}}` (one `SVector` per row) and the matrices are rebuilt on access.

Partially periodic lattices (`nperiodic < D`):
----------------------------------------------
`nperiodic` (default `D`) splits the rows into `P = nperiodic` genuine primitive vectors and
`D − P` orthonormal PADDING rows: the object stays a square, invertible coordinate frame, but
only the first `P` directions are translation symmetries. This is how a ribbon (1-periodic in
2D space), a slab (2-periodic in 3D) or a layered continuum cell with a real `z` coordinate is
represented. The padding rows must be unit-norm and orthogonal to every other row -- that keeps
`cell_volume = |det a|` equal to the `P`-volume of the true cell and keeps fractional transverse
coordinates equal to cartesian ones. Everything periodic (wrapping, `ΔR`, `G`, `k`) lives in the
first `P` components only; the aperiodic components of positions never wrap, and `k`, `G`, `ΔR`
must vanish there (enforced by `UnitCell` and the assembly funnels). `nperiodic == 0` is the
degenerate end of the same family -- a Γ-only finite cluster (an orthonormal coordinate frame
with no translation symmetry at all), produced by `open_boundary` opening the last periodic
direction; the only legal momentum there is `k = 0`.
"""
struct BravaisLattice{D}
    arows::NTuple{D,SVector{D,Float64}}   # rows of `a`: the primitive vectors
    brows::NTuple{D,SVector{D,Float64}}   # rows of `b`: the primitive reciprocal vectors
    nperiodic::Int                        # rows 1:nperiodic are periodic; the rest are padding

    function BravaisLattice{D}(a, b; force::Bool=false, atol::Float64=INT_ROUND_ATOL,
                               nperiodic::Int=D) where {D}
        if !(D in (1,2,3)) && !force
            throw(ArgumentError(
                "Dimension D=$D is outside the standard range (1,2,3).\n" *
                "Set `force=true` if you really want to construct a `BravaisLattice` with D=$D."
            ))
        end
        ### `nperiodic == 0` is legal: a Γ-only finite cluster (every row a padding axis), the
        ### frame `open_boundary` produces when it opens the LAST periodic direction. All rows
        ### then go through the orthonormal-padding check below.
        (0 <= nperiodic <= D) || throw(ArgumentError(
            "`nperiodic` must lie in 0:$D; got $nperiodic."))

        M = transpose(a) * b / (2π)
        I_D = one(SMatrix{D,D,Float64})
        if !isapprox(M, I_D; atol=atol)
            err = maximum(abs.(M .- I_D))
            @warn "Reciprocal lattice vectors are not consistent with `a`. " *
                    "Overriding `b` with `2π * inv(a')`." *
                    " max(abs(diff)) = $err > $atol."
            b = 2π * inv(transpose(a))
        end

        ### padding rows must be an orthonormal complement of the periodic rows: that is what
        ### keeps |det a| the P-volume and transverse fractional = cartesian. `force=true` skips
        ### the check (the `reciprocal` involution's padding rows carry norm 2π).
        if nperiodic < D && !force
            for m in (nperiodic + 1):D
                am = a[m, :]
                isapprox(norm(am), 1.0; atol=LATTICE_ATOL) || throw(ArgumentError(
                    "Padding row a$m has norm $(norm(am)); the aperiodic rows of a partially " *
                    "periodic lattice must be unit vectors (they are coordinate axes, not " *
                    "translations)."))
                for l in 1:D
                    l == m && continue
                    abs(dot(am, a[l, :])) <= LATTICE_ATOL * max(1.0, norm(a[l, :])) ||
                        throw(ArgumentError(
                        "Padding row a$m is not orthogonal to row a$l " *
                        "(dot = $(dot(am, a[l, :]))). The aperiodic directions must be an " *
                        "orthonormal complement of the periodic rows."))
                end
            end
        end

        new{D}(_matrix_to_rows(a, Val(D)), _matrix_to_rows(b, Val(D)), nperiodic)
    end
end

"""
    _matrix_to_rows(m, ::Val{D}) -> NTuple{D,SVector{D,Float64}}
    _rows_to_matrix(rows)        -> SMatrix{D,D,Float64,D*D}

Convert between the stored row tuple and the matrix view of it.

`_rows_to_matrix` fills column-major, so linear index `k` maps to row `1+(k-1)%D` and column
`1+(k-1)÷D`. Both are compile-time unrolled for the small `D` this package supports, so the
round trip is free -- the matrix never reaches the heap.
"""
@inline _matrix_to_rows(m::AbstractMatrix, ::Val{D}) where {D} =
    ntuple(i -> SVector{D,Float64}(ntuple(j -> Float64(m[i, j]), D)), D)

@inline _rows_to_matrix(rows::NTuple{D,SVector{D,Float64}}) where {D} =
    SMatrix{D,D,Float64,D*D}(ntuple(k -> rows[1 + (k - 1) % D][1 + (k - 1) ÷ D], D * D))

"""
    getproperty(lattice::BravaisLattice, name)

Expose the stored rows as the matrices `a` and `b`.

`lat.a` and `lat.b` behave exactly as if they were fields -- same values, same `SMatrix` type,
inferred precisely, and no allocation. The underlying fields are `arows` and `brows`.
"""
@inline function Base.getproperty(lattice::BravaisLattice{D}, name::Symbol) where {D}
    name === :a && return _rows_to_matrix(getfield(lattice, :arows))
    name === :b && return _rows_to_matrix(getfield(lattice, :brows))
    return getfield(lattice, name)
end

Base.propertynames(::BravaisLattice) = (:a, :b, :arows, :brows, :nperiodic)





##################################
###### Flexible constructor ######
##################################

# using abtsract matrices for the constructor
"""
    BravaisLattice(a, b; force=false)

Construct a Bravais lattice from abstract matrices `a` and `b`. The matrices are converted to static matrices internally.
"""
function BravaisLattice(a::AbstractMatrix{<:Real},
                        b::AbstractMatrix{<:Real};
                        kwargs...)::BravaisLattice
    size(a,1) == size(a,2) || throw(DimensionMismatch("Bravais lattice matrix `a` must be square"))
    D = size(a,1)
    size(b) == (D,D) || throw(DimensionMismatch("Reciprocal bravais lattice matrix `b` must be $D×$D"))

    aS = SMatrix{D,D,Float64}(a)
    bS = SMatrix{D,D,Float64}(b)

    return BravaisLattice{D}(aS, bS; kwargs...)
end

# BravaisLattice constructor from a single lattice matrix
"""
    BravaisLattice(A; reciprocal=false, kwargs...)

Construct a Bravais lattice from a single matrix `A`.
- If `reciprocal=false` (default), `A` is treated as the real-space lattice `a` and
  the reciprocal lattice `b` is computed.
- If `reciprocal=true`, `A` is treated as the reciprocal lattice `b` and the real-
  space lattice `a` is computed.
"""
function BravaisLattice(A::AbstractMatrix{<:Real};
                        reciprocal::Bool=false, rtol::Float64=LINDEP_RTOL,
                        kwargs...)::BravaisLattice
    # Checks
    D = size(A,1)
    if size(A,2) != D 
        throw(DimensionMismatch("Bravais lattice matrix `A` must be square"))
    end
    if abs(det(A)) / prod([norm(A[i,:]) for i in 1:D]) < rtol
        throw(DomainError(
            "The real-space lattice vectors are linearly dependent (within the tolerance $rtol). \n" *
            "Check the lattice vectors, or lower `tolerance` if you are sure the lattice vectors are correct."
        ))
    end

    # Invert and compute
    invA = 2π * inv(transpose(A))
    if reciprocal
        return BravaisLattice(invA, A; kwargs...)
    else
        return BravaisLattice(A, invA; kwargs...)
    end
end

# BravaisLattice constructor from a list of lattice vectors
"""
    BravaisLattice(lattice_vectors; reciprocal=false, kwargs...)
Construct a Bravais lattice from a list of lattice vectors.
- `lattice_vectors`: A vector of D lattice vectors, each of length D.
"""
function BravaisLattice(lattice_vectors::AbstractVector{<:AbstractVector{<:Real}};
                        kwargs...)::BravaisLattice
    A = transpose(hcat(lattice_vectors...))
    return BravaisLattice(A; kwargs...)
end





####################################################
###### Accessors for the BravaisLattice class ######
####################################################

"""
    ndim(lattice::BravaisLattice)

Return the embedding dimension of the Bravais lattice (the length of positions and rows).
"""
function ndim(lattice::BravaisLattice{D})::Int where {D}
    return D
end

"""
    nperiodic(lattice::BravaisLattice) -> Int

The number of PERIODIC directions `P ≤ D`: rows `1:P` of `a` are genuine primitive vectors,
rows `P+1:D` are orthonormal padding (aperiodic coordinate axes). Equal to `ndim` for every
fully periodic lattice.
"""
@inline nperiodic(lattice::BravaisLattice)::Int = getfield(lattice, :nperiodic)

"""
    _require_periodic_shift(lattice, G, caller) -> Nothing

Refuse an integer reciprocal shift `G` that is non-zero along an aperiodic direction, naming
`caller`.

Notes:
------
The ONE owner of this invariant, in the spirit of [`_endpoint_q`](@ref): every site that turns
an integer `G` into a reciprocal vector -- `bloch_periodicity`, the form-factor relabelling --
goes through here, so the check cannot be present in one and missing in the other.

This matters more than it looks. A partially periodic lattice pads rows `P+1:D` with UNIT
vectors to stay invertible, so `recip_matrix` reports a perfectly finite `b_m = 2π` per length
unit along a direction that has no reciprocal lattice at all. Nothing downstream throws: the
shifted quantity is simply, silently, the unshifted one — the exact shape of a wrong answer
this package ranks highest.
"""
@inline function _require_periodic_shift(lattice::BravaisLattice{D},
                                         G::AbstractVector{<:Integer},
                                         caller::AbstractString) where {D}
    P = nperiodic(lattice)
    P == D && return nothing
    all(m -> G[m] == 0, (P + 1):D) || throw(ArgumentError(
        "`$caller`: G = $(Tuple(G)) is non-zero along an aperiodic direction; there is no " *
        "reciprocal lattice there to shift by. This lattice is periodic along its first " *
        "$P direction$(P == 1 ? "" : "s") only -- the padding rows of `recip_matrix` are " *
        "coordinate axes, not reciprocal vectors, so a shift along them is not an umklapp."))
    return nothing
end

"""
    primitive_vectors(lattice) -> Vector{SVector{D,Float64}}
    primitive_recip_vectors(lattice) -> Vector{SVector{D,Float64}}

All `D` primitive (or primitive reciprocal) vectors, as a freshly allocated `Vector`. Inside a
loop, prefer the non-allocating scalar accessors [`primitive_vector`](@ref) /
[`primitive_recip_vector`](@ref).
"""
function primitive_vectors(lattice::BravaisLattice{D})::Vector{SVector{D,Float64}} where {D}
    return [lattice.a[i,:] for i in 1:D]
end
function primitive_recip_vectors(lattice::BravaisLattice{D})::Vector{SVector{D,Float64}} where {D}
    return [lattice.b[i,:] for i in 1:D]
end
@doc (@doc primitive_vectors) primitive_recip_vectors

"""
    primitive_vector(lattice, i) -> SVector{D,Float64}
    primitive_recip_vector(lattice, i) -> SVector{D,Float64}

Return the `i`-th primitive (or primitive reciprocal) vector, without allocating.

Notes:
------
`primitive_vectors` builds a heap-allocated `Vector` on every call. That is fine at the top of a
function and wasteful inside a loop over candidate translations, which is why the scalar accessors
exist. Rows of a static matrix are themselves `SVector`s, so these return stack-allocated values.
"""
@inline function primitive_vector(lattice::BravaisLattice{D}, i::Integer)::SVector{D,Float64} where {D}
    1 <= i <= D || throw(ArgumentError("Primitive-vector index $i is out of range 1:$D."))
    return getfield(lattice, :arows)[i]
end

@inline function primitive_recip_vector(lattice::BravaisLattice{D}, i::Integer)::SVector{D,Float64} where {D}
    1 <= i <= D || throw(ArgumentError("Primitive-vector index $i is out of range 1:$D."))
    return getfield(lattice, :brows)[i]
end
@doc (@doc primitive_vector) primitive_recip_vector

"""
    direct_matrix(lattice) -> SMatrix{D,D,Float64}
    recip_matrix(lattice)  -> SMatrix{D,D,Float64}

The matrices that send fractional coordinates to cartesian ones:

    r_cart = direct_matrix(lat) * r_frac        # real space
    k_cart = recip_matrix(lat)  * k_frac        # reciprocal space

These are `transpose(a)` and `transpose(b)`. They are given names because the transpose is a direct
consequence of the **row** convention for `a` and `b`, and repeating `transpose(lat.a)` at every
call site duplicates that piece of knowledge -- a convention change would then have to be chased
through the whole package rather than through these two definitions.

`direct_matrix` is also the object to hoist out of an inner loop: `fractional_to_real` allocates
nothing but does re-derive the transpose on each call.
"""
@inline direct_matrix(lattice::BravaisLattice{D}) where {D} =
    SMatrix{D,D,Float64}(transpose(lattice.a))

@inline recip_matrix(lattice::BravaisLattice{D}) where {D} =
    SMatrix{D,D,Float64}(transpose(lattice.b))
@doc (@doc direct_matrix) recip_matrix

"""
    metric(lattice) -> SMatrix{D,D,Float64}

The real-space metric tensor `gᵢⱼ = aᵢ ⋅ aⱼ`, i.e. `a * aᵀ`.

Purpose:
--------
Lengths and angles of fractional vectors without converting to cartesian coordinates:

    |v|² = vᵀ g v      for `v` in fractional coordinates

Useful wherever only distances matter and the cartesian embedding is irrelevant -- neighbour
distances, and the crystallographic lattice parameters returned by `lattice_parameters`.
"""
@inline metric(lattice::BravaisLattice{D}) where {D} =
    SMatrix{D,D,Float64}(lattice.a * transpose(lattice.a))

"""
    recip_metric(lattice) -> SMatrix{D,D,Float64}

The reciprocal-space metric tensor `bᵢ ⋅ bⱼ`, i.e. `b * bᵀ`.

Gives `|k|²` directly from fractional momentum coordinates, which is what a continuum kinetic
energy needs when it is expressed without leaving the fractional basis.
"""
@inline recip_metric(lattice::BravaisLattice{D}) where {D} =
    SMatrix{D,D,Float64}(lattice.b * transpose(lattice.b))

"""
    cell_volume(lattice) -> Float64

The `D`-dimensional measure of the primitive unit cell, `|det a|`.

This is a length in 1D, an area in 2D and a volume in 3D; the name is used for all of them.
Needed by anything that normalizes over the cell -- densities of states, Brillouin-zone integrals,
form factors.
"""
@inline function cell_volume(lattice::BravaisLattice{D})::Float64 where {D}
    return abs(det(lattice.a))
end

"""
    bz_volume(lattice) -> Float64

The measure of the Brillouin zone: `(2π)^P / |det a|` with `P = nperiodic`, which is `|det b|`
on a fully periodic lattice.

Notes:
------
The Brillouin zone is a `P`-dimensional object, and `P` is what this counts. On a partially
periodic lattice `|det b|` is NOT it: the padding rows are unit coordinate axes, so each
contributes a spurious factor of `2π` to `|det b|` while contributing 1 to `|det a|` -- a 2D
lattice embedded in 3D came out `2π` times too large. That is not a cosmetic error, because this
is the plaquette normalizer of every Berry-curvature density in `analysis/topology/`; it was
invisible only because those entry points are `HamiltonianModel{2}`-only and so never saw a
padded frame. [`length_scale`](@ref) already took the `P`-th root for the same reason, and
`cell_volume` is safe only because the padding rows are unit-norm by contract.
"""
@inline function bz_volume(lattice::BravaisLattice{D})::Float64 where {D}
    return (2π)^nperiodic(lattice) / cell_volume(lattice)
end

"""
    length_scale(lattice) -> Float64

The characteristic length of the cell, `|det a|^(1/D)`.

Purpose:
--------
The scale against which a *cartesian* tolerance must be measured. `LATTICE_ATOL` bounds errors in
**fractional** coordinates, which live in `[0,1)` and are therefore dimensionless; a distance is in
whatever units the lattice vectors carry, so comparing one against the other is a category error
that only goes unnoticed when `|a| ≈ 1`.

See [`distance_atol`](@ref).
"""
@inline function length_scale(lattice::BravaisLattice{D})::Float64 where {D}
    ### unit-norm padding rows contribute a factor 1 to |det a|, so the P-th root is the
    ### characteristic length of the TRUE cell on partially periodic lattices. A 0-periodic
    ### (finite-cluster) frame is all unit rows -- fractional IS cartesian -- so its scale is 1
    ### exactly; the P-th-root formula would raise float noise around |det a| = 1 to the power
    ### Inf and return 0 or Inf.
    P = nperiodic(lattice)
    P == 0 && return 1.0
    return cell_volume(lattice)^(1 / P)
end

"""
    distance_atol(lattice; atol=LATTICE_ATOL) -> Float64

The absolute tolerance for comparing cartesian distances in this lattice: `atol * length_scale`.

Purpose:
--------
Makes every distance comparison **unit-system independent**. Expressing the same model in Ångström,
bohr or metres rescales every distance, and a fixed absolute floor would then mean something
different in each -- at a lattice constant near the floor it silently drops the nearest-neighbour
shell, so `shell=1` starts selecting second neighbours with no diagnostic.

Notes:
------
Only the floor is affected. `SHELL_RTOL` dominates for any distance well above it, which is why the
defect is invisible at the `|a| ≈ 1` used by every model in the test suite.
"""
@inline function distance_atol(lattice::BravaisLattice{D};
                               atol::Float64=LATTICE_ATOL)::Float64 where {D}
    return atol * length_scale(lattice)
end

"""
    reciprocal(lattice) -> BravaisLattice{D}

The reciprocal lattice, as a `BravaisLattice` in its own right: its primitive vectors are the
`bᵢ` of the original, and its reciprocal vectors are the original `aᵢ`.

Notes:
------
The pair `(b, a)` satisfies the same invariant as `(a, b)`: transposing `aᵀb = 2πI` gives
`bᵀa = 2πI`. So this is an involution, `reciprocal(reciprocal(lat))` recovers the original.

Useful for Brillouin-zone work -- a k-space mesh, a Wigner-Seitz construction, or a folding
transform is the same geometry applied to this lattice.
"""
reciprocal(lattice::BravaisLattice{D}) where {D} =
    BravaisLattice{D}(lattice.b, lattice.a; force=true, nperiodic=nperiodic(lattice))

"""
    same_basis(l1, l2; atol=LATTICE_ATOL) -> Bool

Whether two lattices have the **same primitive vectors**, not merely describe the same lattice.

Why this is separate from `==`:
-------------------------------
`==` on `BravaisLattice` is *geometric equivalence*: it returns `true` whenever the two are related
by a unimodular change of basis, which is the right meaning of "the same Bravais lattice". But
almost everything built on top of a lattice is expressed in a **specific** basis -- fractional
positions, `k_frac`, the integer translations `ΔR` of a bond list. Two lattices that compare `==`
can have entirely different primitive vectors, so `==` does not license comparing or merging their
basis-dependent data.

    l1 = [1 0; 0 1],  l2 = [1 1; 0 1]   ->   l1 == l2         (same lattice)
                                             same_basis false  (different description)

Use `==` to ask "same lattice?", and this to ask "same coordinates?".
"""
function same_basis(l1::BravaisLattice{D1}, l2::BravaisLattice{D2};
                    atol::Float64=LATTICE_ATOL)::Bool where {D1,D2}
    D1 == D2 || return false
    ### `atol` is a fractional (dimensionless) tolerance, but the entries of `a` are cartesian
    ### lengths -- the category error `length_scale` documents. Unscaled, the compare is
    ### unit-system dependent: metre-scale lattices differing by percents pass, and large-unit
    ### lattices rebuilt through a product can fail on float noise.
    return isapprox(l1.a, l2.a; atol=atol * length_scale(l1))
end




#############################################################
###### Crystallographic lattice parameters              #####
#############################################################

"""
    lattice_parameters(lattice; degrees=true) -> NamedTuple

The crystallographic cell parameters of a lattice.

Returns:
--------
- `D = 1`: `(a = |a₁|,)`
- `D = 2`: `(a, b, gamma)` with `gamma` the angle between `a₁` and `a₂`
- `D = 3`: `(a, b, c, alpha, beta, gamma)` with the standard assignment
  `alpha = ∠(a₂,a₃)`, `beta = ∠(a₁,a₃)`, `gamma = ∠(a₁,a₂)`

Args:
-----
- `degrees`: return angles in degrees (default) or radians

Notes:
------
This is the language every external format speaks -- CIF, Quantum ESPRESSO, ASE, Crystalline -- so
it is the natural bridge for import and export. Together with `lattice_from_parameters` it forms a
round trip *up to orientation*: the parameters fix the cell's shape, not its embedding in space, so
recovering a lattice from its parameters yields a rigidly rotated copy with the same `metric`.

Computed from the metric tensor, so no cartesian intermediates are involved.
"""
function lattice_parameters(lattice::BravaisLattice{D}; degrees::Bool=true) where {D}
    ### Rows `P+1:D` are orthonormal coordinate axes, not translations: their lengths are all 1
    ### and their angles all right BY CONSTRUCTION, so reporting them as `c`, `alpha`, `beta`
    ### describes the frame and not the lattice. Refused rather than answered, for the reason
    ### `_assert_fully_periodic` gives in `crystallography/bravais_types.jl` -- where the same
    ### mistake once shipped, `identify_bravais` returning `:lp` with `a = 1.0`, the length of a
    ### padding axis. Returning the `P` real parameters instead is not an option here: `P` is a
    ### runtime value, so the NamedTuple type would vary and this is an allocation-free accessor.
    nperiodic(lattice) == D || throw(ArgumentError(
        "`lattice_parameters`: this lattice is periodic along only $(nperiodic(lattice)) of " *
        "its $D directions, and rows $(nperiodic(lattice) + 1):$D are orthonormal padding " *
        "axes. Their lengths (all 1) and angles (all right) are properties of the coordinate " *
        "frame, not of the lattice, so there is no honest `$(D)`-parameter answer. Ask the " *
        "periodic block instead: `lattice_parameters(periodic_block(lattice))`."))
    g = metric(lattice)
    len(i) = sqrt(g[i, i])
    ang(i, j) = begin
        c = clamp(g[i, j] / (len(i) * len(j)), -1.0, 1.0)
        degrees ? rad2deg(acos(c)) : acos(c)
    end

    if D == 1
        return (a = len(1),)
    elseif D == 2
        return (a = len(1), b = len(2), gamma = ang(1, 2))
    elseif D == 3
        return (a = len(1), b = len(2), c = len(3),
                alpha = ang(2, 3), beta = ang(1, 3), gamma = ang(1, 2))
    else
        throw(ArgumentError(
            "`lattice_parameters` is defined for D = 1, 2, 3; got D = $D. " *
            "Use `metric(lattice)` directly for higher dimensions."
        ))
    end
end

"""
    lattice_from_parameters(; a, b, c, alpha, beta, gamma, D, degrees=true) -> BravaisLattice{D}

Build a lattice from crystallographic cell parameters.

Args:
-----
- `D = 1`: `a`
- `D = 2`: `a`, `b`, `gamma`
- `D = 3`: `a`, `b`, `c`, `alpha`, `beta`, `gamma`
- `degrees`: interpret the angles as degrees (default) or radians

Returns:
--------
- `BravaisLattice{D}` in the standard orientation: `a₁` along x, `a₂` in the xy-plane, `a₃`
  completing a right-handed set.

Notes:
------
The construction is the conventional crystallographic-to-cartesian one,

    a₁ = (a, 0, 0)
    a₂ = (b cos γ, b sin γ, 0)
    a₃ = (c cos β, c (cos α − cos β cos γ)/sin γ, c ν/sin γ)

with `ν = √(1 − cos²α − cos²β − cos²γ + 2 cos α cos β cos γ)`, giving cell volume `abc ν`.

Not every combination of angles describes a real cell: `ν²` must be positive, which is exactly the
condition that the three vectors are linearly independent. Parameters violating it are rejected
rather than producing a degenerate or complex cell.
"""
function lattice_from_parameters(; a::Real,
                                 b::Union{Nothing,Real}=nothing,
                                 c::Union{Nothing,Real}=nothing,
                                 alpha::Union{Nothing,Real}=nothing,
                                 beta::Union{Nothing,Real}=nothing,
                                 gamma::Union{Nothing,Real}=nothing,
                                 D::Union{Nothing,Int}=nothing,
                                 degrees::Bool=true)
    ### infer D from which parameters were supplied, unless it was given
    Dim = if !isnothing(D)
        D
    elseif !isnothing(c) || !isnothing(alpha) || !isnothing(beta)
        3
    elseif !isnothing(b) || !isnothing(gamma)
        2
    else
        1
    end

    a > 0 || throw(DomainError(a, "Lattice parameter `a` must be positive."))
    torad(x) = degrees ? deg2rad(x) : float(x)

    if Dim == 1
        return BravaisLattice(SMatrix{1,1,Float64}(Float64(a)))

    elseif Dim == 2
        isnothing(b) && throw(ArgumentError("A 2D lattice needs `b`."))
        isnothing(gamma) && throw(ArgumentError("A 2D lattice needs `gamma`."))
        b > 0 || throw(DomainError(b, "Lattice parameter `b` must be positive."))
        g = torad(gamma)
        (0 < g < π) || throw(DomainError(gamma, "`gamma` must lie strictly between 0 and 180 degrees."))
        ### `SMatrix` fills column-major, so this lays down rows (a, 0) and (b cos γ, b sin γ)
        return BravaisLattice(SMatrix{2,2,Float64}(Float64(a), b * cos(g), 0.0, b * sin(g)))

    elseif Dim == 3
        for (nm, v) in ((:b, b), (:c, c), (:alpha, alpha), (:beta, beta), (:gamma, gamma))
            isnothing(v) && throw(ArgumentError("A 3D lattice needs `$nm`."))
        end
        (b > 0 && c > 0) || throw(DomainError((b, c), "Lattice parameters `b` and `c` must be positive."))
        al, be, ga = torad(alpha), torad(beta), torad(gamma)
        for (nm, v, raw) in ((:alpha, al, alpha), (:beta, be, beta), (:gamma, ga, gamma))
            (0 < v < π) || throw(DomainError(raw, "`$nm` must lie strictly between 0 and 180 degrees."))
        end

        ca, cb, cg, sg = cos(al), cos(be), cos(ga), sin(ga)
        nu2 = 1 - ca^2 - cb^2 - cg^2 + 2 * ca * cb * cg
        nu2 > 0 || throw(DomainError((alpha, beta, gamma),
            "The angles (α=$alpha, β=$beta, γ=$gamma) do not describe a valid cell: " *
            "1 − cos²α − cos²β − cos²γ + 2cosα cosβ cosγ = $nu2 ≤ 0, so the three lattice " *
            "vectors would be linearly dependent."
        ))

        a1 = SVector(Float64(a), 0.0, 0.0)
        a2 = SVector(b * cg, b * sg, 0.0)
        a3 = SVector(c * cb, c * (ca - cb * cg) / sg, c * sqrt(nu2) / sg)
        return BravaisLattice([a1, a2, a3])
    else
        throw(ArgumentError("`lattice_from_parameters` supports D = 1, 2, 3; got D = $Dim."))
    end
end




#############################################################
###### Display                                          #####
#############################################################

function Base.show(io::IO, ::MIME"text/plain", lattice::BravaisLattice{D}) where {D}
    p = lattice_parameters(lattice)
    P = nperiodic(lattice)
    println(io, P == D ? "BravaisLattice{$D}" :
                "BravaisLattice{$D} ($P periodic direction$(P == 1 ? "" : "s") + $(D-P) aperiodic)")
    println(io, "  primitive vectors (rows of `a`):")
    for i in 1:D
        println(io, "    a$i = ", _fmt_vec(primitive_vector(lattice, i)),
                i > P ? "   (aperiodic padding)" : "")
    end
    println(io, "  reciprocal vectors (rows of `b`):")
    for i in 1:D
        println(io, "    b$i = ", _fmt_vec(primitive_recip_vector(lattice, i)))
    end
    if D == 1
        println(io, "  parameters: a = ", _fmt(p.a))
    elseif D == 2
        println(io, "  parameters: a = ", _fmt(p.a), ", b = ", _fmt(p.b),
                    ", γ = ", _fmt(p.gamma), "°")
    elseif D == 3
        println(io, "  parameters: a = ", _fmt(p.a), ", b = ", _fmt(p.b), ", c = ", _fmt(p.c))
        println(io, "              α = ", _fmt(p.alpha), "°, β = ", _fmt(p.beta),
                    "°, γ = ", _fmt(p.gamma), "°")
    end
    print(io, "  cell volume: ", _fmt(cell_volume(lattice)),
              "   BZ volume: ", _fmt(bz_volume(lattice)))
end

Base.show(io::IO, lattice::BravaisLattice{D}) where {D} =
    print(io, "BravaisLattice{$D}(", join((_fmt_vec(primitive_vector(lattice, i)) for i in 1:D), ", "), ")")

_fmt(x::Real) = string(round(x; digits=6))
_fmt_vec(v::AbstractVector{<:Real}) = "[" * join((_fmt(x) for x in v), ", ") * "]"




############################################################
###### Geometric equivalence between Bravais lattices ######
############################################################

"""
    ==(bl1::BravaisLattice, bl2::BravaisLattice)

Compare two Bravais lattices for equality. Two lattices represent the same Bravais lattice
if the transformation between their primitive vectors is unimodular and integer-valued.

Specifically: if A_1 = A_2 · M where M is an integer matrix with det(M) = ±1 (unimodular),
then both lattices describe the same Bravais lattice.
"""
function Base.:(==)(bl1::BravaisLattice{D1}, bl2::BravaisLattice{D2})::Bool where {D1, D2}
    if D1 != D2 || nperiodic(bl1) != nperiodic(bl2)
        return false
    end
    # Compute transformation matrix: M = inv(a₁) · a₂
    M = bl1.a * transpose(bl2.b) / (2π) # M = a₁ · inv(a₂) = a₁ · b₂.T / (2π)
    ### Integrality is tested RELATIVE to the size of M: the float error of the matrix product
    ### grows with |M| and its conditioning, so a fixed atol rejects legitimate large unimodular
    ### relations (entries ~150 already exceed INT_ROUND_ATOL by a factor of a few).
    tol = INT_ROUND_ATOL * max(1.0, maximum(abs, M))
    if !isapprox(M, round.(M); atol=tol)
        return false
    end
    ### The determinant is then checked EXACTLY on the integer matrix: the float determinant of a
    ### large M is far less accurate than its entries, and `unimodular_transformation` already
    ### promises an exact check for the same quantity.
    return abs(det(round.(Int, M))) == 1
end

function Base.:(!=)(bl1::BravaisLattice, bl2::BravaisLattice)::Bool
    return !(bl1 == bl2)
end

"""
    hash(lattice::BravaisLattice, h::UInt)

Hash consistent with `==`.

Notes:
------
`==` is equivalence under a unimodular (integer, `|det| = 1`) change of primitive vectors, and no
continuous quantity survives that equivalence *exactly*: the cell volume is invariant only up to
float noise, so folding a rounded volume into the hash would let two `==` lattices land on opposite
sides of a rounding boundary. Only the dimension is hashed -- coarse but exact, the same trade the
`UnitCell` hash makes.
"""
Base.hash(l::BravaisLattice{D}, h::UInt) where {D} =
    hash(:BravaisLattice, hash(D, hash(nperiodic(l), h)))





############################################
###### transformations of the lattice ######
############################################

"""
    unimodular_transformation(lattice, M) -> BravaisLattice{D}

Re-express a Bravais lattice in a different set of primitive vectors describing the *same* lattice.

Args:
-----
- `lattice`: the lattice to re-express
- `M`: an integer matrix with `det M = ±1`; the new primitive vectors are `M * a`

Returns:
--------
- `BravaisLattice{D}` satisfying `lattice == result` (geometric equivalence) but generally
  `same_basis(lattice, result) == false`.

Notes:
------
The determinant of an integer matrix is an integer, so unimodularity is checked **exactly** rather
than within a tolerance -- a near-miss here means the caller passed something that is not a change
of basis at all, not that a rounding error crept in.
"""
function unimodular_transformation(lattice::BravaisLattice{D},
                                   M::AbstractMatrix{<:Integer})::BravaisLattice{D} where {D}
    size(M) == (D, D) || throw(DimensionMismatch(
        "Transformation matrix must be $D×$D; got $(size(M))."
    ))
    d = det(SMatrix{D,D,Int}(M))
    abs(d) == 1 || throw(ArgumentError(
        "Transformation matrix M must be unimodular (determinant ±1); got det M = $d. " *
        "A determinant of magnitude $(abs(d)) would describe a supercell, not a change of basis -- " *
        "use `supercell` for that."
    ))
    P = nperiodic(lattice)
    ### on a partially periodic lattice only the periodic block may be re-expressed: mixing a
    ### padding row into a primitive vector (or vice versa) is not a change of basis of the
    ### physical lattice
    if P < D
        Mi = SMatrix{D,D,Int}(M)
        (all(iszero, Mi[1:P, (P+1):D]) && all(iszero, Mi[(P+1):D, 1:P]) &&
         Mi[(P+1):D, (P+1):D] == one(SMatrix{D-P,D-P,Int})) || throw(ArgumentError(
            "This lattice is periodic only along its first $P direction(s); a change of basis " *
            "must act on that block alone (M = blockdiag(M_P, I)). Got M = $M."))
    end
    return BravaisLattice(SMatrix{D,D,Int}(M) * lattice.a; force=true, nperiodic=P)
end

### `embed_into_higher_dimension` and its inverse `periodic_block` live in
### `core/unit_cell/embedding.jl`, which owns the D <-> nperiodic boundary.





############################################################
###### Transformation of attributes to physical units ######
############################################################

"""
    fractional_to_real(r_frac, lattice::BravaisLattice)

Transform fractional coordinates to real-space coordinates.

Args:
- `r_frac`: Fractional coordinates (length D)

Returns:
- `r_real`: Real-space coordinates as an `SVector{D,Float64}` (length D)

Notes:
------
The return type is statically sized so that call sites stay type-stable. Annotating it as
`AbstractVector` erases the size and makes every caller box its result, which matters because
this is called once per candidate translation in a neighbour search.
"""
function fractional_to_real(r_frac::AbstractVector{<:Real},
                            lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(r_frac) == D || throw(DimensionMismatch(
        "Input dimension $(length(r_frac)) does not match lattice dimension $D"
    ))
    return transpose(lattice.a) * SVector{D,Float64}(r_frac)
end

"""
    fractional_to_reciprocal(k_frac, lattice::BravaisLattice) -> SVector{D,Float64}

Transform fractional coordinates of the RECIPROCAL lattice to Cartesian momentum: `bᵀ · k`.

The reciprocal-lattice sibling of [`fractional_to_real`](@ref). Until 2026-09-16 these were one
function with a `mode = :real | :momentum` keyword, which meant the call that returns a MOMENTUM
was spelled `fractional_to_real(..., mode = :momentum)` -- a name and a flag saying opposite
things, with an `if` in the body picking `a` or `b`. Two functions, no flag, no `if`.
"""
function fractional_to_reciprocal(k_frac::AbstractVector{<:Real},
                                  lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(k_frac) == D || throw(DimensionMismatch(
        "Input dimension $(length(k_frac)) does not match lattice dimension $D"
    ))
    return transpose(lattice.b) * SVector{D,Float64}(k_frac)
end

"""
    real_to_fractional(r_real, lattice::BravaisLattice)

Transform real-space coordinates to fractional coordinates.

Args:
- `r_real`: Real-space coordinates (length D)

Returns:
- `r_frac`: Fractional coordinates (length D)

Notes:
------
The DIRECT lattice only. For a Cartesian momentum back to reciprocal-fractional coordinates,
use [`reciprocal_to_fractional`](@ref).
"""
function real_to_fractional(r_real::AbstractVector{<:Real},
                            lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(r_real) == D || throw(DimensionMismatch(
        "Input dimension $(length(r_real)) does not match lattice dimension $D"
    ))
    return (lattice.b * SVector{D,Float64}(r_real)) ./ (2π)
end

"""
    reciprocal_to_fractional(k_cart, lattice::BravaisLattice) -> SVector{D,Float64}

Transform a Cartesian momentum to fractional coordinates of the RECIPROCAL lattice:
`(a · k) / 2π`.

The reciprocal-lattice sibling of [`real_to_fractional`](@ref); see
[`fractional_to_reciprocal`](@ref) for why these are two functions and not one with a mode.
"""
function reciprocal_to_fractional(k_cart::AbstractVector{<:Real},
                                  lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(k_cart) == D || throw(DimensionMismatch(
        "Input dimension $(length(k_cart)) does not match lattice dimension $D"
    ))
    return (lattice.a * SVector{D,Float64}(k_cart)) ./ (2π)
end
