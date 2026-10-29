##################################################################
###### export only the following functions from this module ######
##################################################################

export g_shells, g_star, plane_wave_cell, free_electron_model




#############################################################
###### Reciprocal-lattice shells                        #####
#############################################################

"""
    _recip_scale(lattice) -> (bmin, datol)

The shortest primitive reciprocal length and the absolute distance floor derived from it -- the
unit-system-independent scale every reciprocal-space comparison in this file anchors to
(`LATTICE_ATOL` is dimensionless; multiplying by `bmin` ties it to the lattice's own scale).
"""
function _recip_scale(lattice::BravaisLattice{D}) where {D}
    ### periodic rows only: the padding rows of a partially periodic lattice carry |b| = 2π
    ### by construction and are not reciprocal vectors
    nperiodic(lattice) >= 1 || throw(ArgumentError(
        "This lattice has no periodic direction (a Γ-only finite cluster): it has no " *
        "reciprocal vectors, so reciprocal-space shells and plane-wave bases are not defined " *
        "on it."))
    bmin = minimum(norm(primitive_recip_vector(lattice, m)) for m in 1:nperiodic(lattice))
    return bmin, LATTICE_ATOL * bmin
end

"""
    g_shells(lattice, nshells) -> (radii, shells)
    g_shells(lattice; shells) -> (radii, shells)

The first `nshells` shells of **non-zero** reciprocal lattice vectors, as integer coordinate
vectors, grouped by cartesian length `|Σ nₘ bₘ|`. The keyword form spells the count the same
way `plane_wave_cell(lat; shells=n)` does.

Returns:
--------
- `radii::Vector{Float64}`: the shell radii, ascending
- `shells::Vector{Vector{SVector{D,Int}}}`: shell `s` holds the integer `G`s at `radii[s]`,
  sorted lexicographically for a deterministic order

Notes:
------
This is the reciprocal-space counterpart of the direct-space neighbour search, with the same shell
policy: distances are clustered with `SHELL_RTOL`, and the absolute floor scales with the
lattice's own reciprocal length, so the grouping is unit-system independent.

The enumeration box is exact, not heuristic: `nₘ = aₘ⋅G / 2π`, so every `G` with `|G| ≤ r` has
`|nₘ| ≤ r·|aₘ|/2π`, and the `s`-th distinct radius is at most `s` times the shortest non-zero
reciprocal vector. Together these bound the search box that provably contains `nshells` complete
shells.
"""
g_shells(lattice::BravaisLattice; shells::Int) = g_shells(lattice, shells)

"""
    _g_shells_within(lattice, r) -> (radii, shells)

Every non-zero reciprocal vector with `|G| <= r`, clustered into shells by the shell policy of
`g_shells`. Within a shell the order is the enumeration's, not lexicographic: sorting is
`_sort_shells!`, applied by the caller to the shells it keeps.

Notes:
------
The enumeration box is exact: `n_m = a_m ⋅ G / 2π`, so `|G| <= r` forces
`|n_m| <= r * |a_m| / 2π`. The cost is therefore `O((r/b_min)^P)` -- the number of vectors
actually returned -- with no search over shell COUNTS. Both callers want a radius: `g_shells`
converts its shell count to the radius that provably contains it, and the `gcut=` selector of
`plane_wave_cell` already has one.

Clustering is greedy from the smallest distance upward and each cluster is measured from its own
representative, so enlarging `r` never changes a cluster whose members all lie below the old `r`.
That is what lets the two callers share this function and still agree shell for shell.
"""
function _g_shells_within(lattice::BravaisLattice{D}, r::Float64) where {D}
    bmin, datol = _recip_scale(lattice)
    P = nperiodic(lattice)
    amax = maximum(norm(primitive_vector(lattice, m)) for m in 1:P)
    N = ceil(Int, r * amax / (2π)) + 1

    entries = Tuple{Float64,SVector{D,Int}}[]
    ### aperiodic directions carry no reciprocal lattice: their index is pinned to 0
    for n in Iterators.product(ntuple(m -> m <= P ? (-N:N) : (0:0), D)...)
        nv = SVector{D,Int}(n)
        iszero(nv) && continue
        g = sum(nv[m] * primitive_recip_vector(lattice, m) for m in 1:D)
        d = norm(g)
        d <= r + datol && push!(entries, (d, nv))
    end
    sort!(entries; by = first)

    radii = Float64[]
    shells = Vector{Vector{SVector{D,Int}}}()
    for (d, nv) in entries
        if isempty(radii) || d - radii[end] > max(datol, SHELL_RTOL * d)
            push!(radii, d)
            push!(shells, SVector{D,Int}[])
        end
        push!(shells[end], nv)
    end
    ### The lexicographic sort WITHIN each shell is left to the caller, applied to the shells it
    ### actually keeps. Doing it here would sort every shell the enumeration box happens to
    ### resolve -- and `g_shells(n)`'s box provably contains `n` shells by resolving O(n^2) of
    ### them, so sorting them all costs more than the search.
    return radii, shells
end

"""
    _sort_shells!(shells) -> shells

Order each shell lexicographically, in place: the deterministic within-shell order both callers
promise.
"""
function _sort_shells!(shells::Vector{Vector{SVector{D,Int}}}) where {D}
    for s in shells
        sort!(s; by = Tuple)
    end
    return shells
end

function g_shells(lattice::BravaisLattice{D}, nshells::Int) where {D}
    nshells >= 1 || throw(DomainError(nshells, "Number of reciprocal shells must be at least 1."))

    bmin, datol = _recip_scale(lattice)

    # the nshells-th distinct radius is <= nshells * (shortest reciprocal vector) <= nshells * bmin
    r = nshells * bmin * (1 + SHELL_RTOL) + datol
    radii, shells = _g_shells_within(lattice, r)

    length(shells) >= nshells || throw(ErrorException(
        "g_shells resolved only $(length(shells)) shells inside its provably-sufficient box; " *
        "this is an internal inconsistency, not a usage error."
    ))
    return radii[1:nshells], _sort_shells!(shells[1:nshells])
end

"""
    g_star(lattice, shell) -> Vector{SVector{D,Int}}

The star of reciprocal vectors in the given shell (`shell = 1` is the shortest non-zero shell):
six vectors on a triangular lattice's first shell, four on a square lattice's.
"""
g_star(lattice::BravaisLattice{D}, shell::Int) where {D} = g_shells(lattice, shell)[2][shell]




#############################################################
###### Plane-wave cells and generic continuum models    #####
#############################################################

"""
    plane_wave_cell(lattice; shells=nothing, gcut=nothing, gvectors=nothing, schema=nothing,
                    site_name=:origin, origin=nothing, origins=nothing) -> UnitCell{D}

A continuum (plane-wave) cell: one `:plane_wave` site at the origin carrying the reciprocal
vectors selected by exactly one of

- `shells = n`: `G = 0` plus the first `n` complete stars,
- `gcut = r`: `G = 0` plus every `|G| ≤ r` (cartesian), or
- `gvectors = [...]`: an explicit list of integer `G`s, taken verbatim -- the selector for sets
  no isotropic criterion produces, such as the `monolayer_bz_gvectors` of a `TwistedBilayer`
  (the moiré Gs that tile a monolayer Brillouin zone).

`schema` adds internal axes (layer, spin, sublattice ...) at construction; every plane wave is
replicated over them.

Momentum origins:
-----------------
The momentum of a state is `bᵀ(k + G - K_s)` with `K_s` the **state's** origin -- a valley, a
layer's Dirac point, a k·p expansion point. Declare it here, on the basis, rather than on each
term:

- `origin = K`: one cartesian origin for every state (a k·p expansion point), or
- `origins = :layer => [-1 => KM, +1 => KMp]`: per-level origins on one internal axis.

Origins are accepted **cartesian** and stored fractional. With per-level origins, a single
`dirac(v)` covers every layer at its own Dirac point -- no per-layer `k0=` + `project` pattern --
and a ΔG coupling across levels with different origins carries a *computable* physical transfer,
which `moire_tunnelling` solves for.

Notes:
------
The basis order is deterministic: `G = 0` first, then shell by shell, lexicographic within a
shell -- and an explicit `gvectors` list is kept in the caller's order. Determinism matters
because a compiled coupling list stores raw state indices, so two cells built the same way must
BE the same cell.

A plane wave has no centre: the site sits at the origin by construction, and the former
`position=` keyword is gone. A declared position `τ` enters the basis definition
`u_G(r) = e^{iG·(r-τ)}` through a per-harmonic matrix-element phase `e^{2πi ΔG·τ}` that the
compiled coupling never carried, so a non-zero `τ` made the documented basis and the emitted
Hamiltonian disagree by exactly that phase -- harmless for one site's spectrum (a rigid
translation of the potential) but a genuine relative phase, i.e. physics, between two
plane-wave sites. Forcing `τ = 0` makes the definition and the emitter agree identically.

Truncation and symmetry-protected degeneracies: convergence in `shells` is NOT uniform across
the spectrum, and protected crossings feel it first, because the finite G-window breaks the
protecting symmetry slightly. A moiré Dirac touching that is exact in the full model shows a
spurious ~0.05 meV gap at 3-4 shells that collapses by orders of magnitude at 5-6 (measured on
the BM model at realistic couplings). Before reading a small gap at a high-symmetry point as
physics, re-run with one more shell and watch whether it collapses.
"""
function plane_wave_cell(lattice::BravaisLattice{D};
                         shells::Union{Nothing,Int}=nothing,
                         gcut::Union{Nothing,Real}=nothing,
                         gvectors::Union{Nothing,AbstractVector}=nothing,
                         schema::Union{Nothing,InternalSchema}=nothing,
                         site_name::Symbol=:origin,
                         origin::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         origins::Union{Nothing,Pair}=nothing) where {D}
    count(!isnothing, (shells, gcut, gvectors)) == 1 || throw(ArgumentError(
        "Give exactly one of `shells` (a shell count), `gcut` (a cartesian cutoff), or " *
        "`gvectors` (an explicit list of integer G vectors)."
    ))

    Gs = SVector{D,Int}[]
    if !isnothing(gvectors)
        for g in gvectors
            length(g) == D || throw(DimensionMismatch(
                "Every entry of `gvectors` must have length $D; got one of length $(length(g))."))
            ### `canonicalize_int` rather than a silent `round`: a non-integer entry means the
            ### caller passed cartesian or fractional momenta, not reciprocal integer labels --
            ### and the refusal must NAME the keyword, not surface as a bare DomainError
            gi = try
                canonicalize_int(g)
            catch
                throw(ArgumentError(
                    "`gvectors` entry $(Tuple(g)) is not an integer vector. `gvectors` takes " *
                    "INTEGER reciprocal labels (Miller indices); passing cartesian momenta " *
                    "`B*g` or fractional momenta is the usual mistake -- convert with " *
                    "`round.(Int, reciprocal_to_fractional(g_cart, lattice))`."))
            end
            all(m -> gi[m] == 0, (nperiodic(lattice)+1):D) || throw(ArgumentError(
                "`gvectors` entry $(Tuple(gi)) is non-zero along an aperiodic direction of " *
                "this partially periodic lattice; there is no reciprocal lattice there."))
            push!(Gs, gi)
        end
        allunique(Gs) || throw(ArgumentError("`gvectors` contains duplicate entries."))
    elseif !isnothing(shells)
        shells >= 0 || throw(DomainError(shells,
            "The shell count must be non-negative (0 keeps only G = 0)."))
        push!(Gs, zero(SVector{D,Int}))
        if shells > 0
            _, sh = g_shells(lattice, shells)
            foreach(s -> append!(Gs, s), sh)
        end
    else
        gcut >= 0 || throw(DomainError(gcut, "The cutoff must be non-negative."))
        push!(Gs, zero(SVector{D,Int}))
        bmin, datol = _recip_scale(lattice)
        ### A cutoff is a RADIUS, so enumerate by radius -- O((gcut/bmin)^D), the size of the
        ### answer. The old code asked `g_shells` for a shell COUNT and doubled that count until
        ### the last resolved radius cleared the cutoff; since the number of distinct radii below
        ### `gcut` grows like (gcut/bmin)^D, the count converged on ~(gcut/bmin)^D and `g_shells`
        ### then enumerated a box of radius count*bmin ~ gcut^D, i.e. O(gcut^{D^2}) vectors to
        ### return O(gcut^D) of them (in 2D: a quartic search for a quadratic answer).
        ### The enumeration radius is stretched past `gcut` by the shell tolerance so that a
        ### shell whose REPRESENTATIVE clears the cutoff still arrives with every member, exactly
        ### as when it was carved out of a much larger table.
        radii, sh = _g_shells_within(lattice, (gcut + 2 * datol) * (1 + 4 * SHELL_RTOL) + datol)
        for s in eachindex(radii)
            radii[s] <= gcut + datol || continue
            append!(Gs, sort!(sh[s]; by = Tuple))
        end
    end

    sch = isnothing(schema) ? TRIVIAL_SCHEMA : schema
    k0_of = _resolve_origins(lattice, sch, origin, origins)

    ### STATE_ORDER: G outermost, then internal (one site) -- the same order the generative
    ### `UnitCell(...; G=Gs)` constructor produces, so the two spellings stay interchangeable.
    pos = zero(SVector{D,Float64})
    basis = BasisState{D}[]
    sizehint!(basis, length(Gs) * n_internal(sch))
    for g in Gs, internal in 1:n_internal(sch)
        push!(basis, BasisState{D}(pos, g, k0_of[internal], internal, sch))
    end
    ### The kind is DECLARED, not classified: a `shells=0` (k·p) cell has only G = 0 states and
    ### would otherwise read as :localized, refusing the very kernels it exists for.
    return UnitCell{D}(lattice, sch, [site_name], [pos], basis; site_kind=[:plane_wave])
end

"""
    _resolve_origins(lattice, schema, origin, origins) -> Vector{SVector{D,Float64}}

The per-internal-state momentum origins, fractional, indexed by interned internal state.
Cartesian input; every level of the named axis must be given (a partial table would silently
put the missing levels at Γ, which is exactly the class of silent default this package refuses).
"""
function _resolve_origins(lattice::BravaisLattice{D}, sch::InternalSchema,
                          origin::Union{Nothing,AbstractVector{<:Real}},
                          origins::Union{Nothing,Pair}) where {D}
    zero_k = zero(SVector{D,Float64})
    isnothing(origin) && isnothing(origins) && return fill(zero_k, n_internal(sch))
    (isnothing(origin) || isnothing(origins)) || throw(ArgumentError(
        "Give `origin` (one expansion point) or `origins` (per-level), not both."))

    if !isnothing(origin)
        length(origin) == D || throw(DimensionMismatch(
            "`origin` has length $(length(origin)) but the lattice is $D-dimensional."))
        kf = _conform_origin(reciprocal_to_fractional(origin, lattice), lattice)
        return fill(SVector{D,Float64}(kf), n_internal(sch))
    end

    ax = Symbol(origins.first)
    a = axis_position(sch, ax)
    a == 0 && throw(ArgumentError(
        "`origins` names axis `$ax`, but the schema declares $(axis_names(sch))."))
    table = Dict{Int,SVector{D,Float64}}()
    for p in origins.second
        p isa Pair || throw(ArgumentError(
            "`origins` must be `axis => [level => K_cartesian, ...]`; got a non-pair entry."))
        o = level_ordinal(sch, a, _as_level(p.first))
        o == 0 && throw(ArgumentError(
            "`origins` names level $(repr(p.first)) on axis `$ax`, which declares " *
            "$(levels(sch, a))."))
        length(p.second) == D || throw(DimensionMismatch(
            "`origins` entry for level $(repr(p.first)) has length $(length(p.second)); " *
            "expected a cartesian momentum of length $D."))
        haskey(table, o) && throw(ArgumentError(
            "`origins` names level $(repr(p.first)) twice; each level takes one origin."))
        table[o] = _conform_origin(
            SVector{D,Float64}(reciprocal_to_fractional(p.second, lattice)), lattice)
    end
    length(table) == n_levels(sch, a) || throw(ArgumentError(
        "`origins` covers $(length(table)) of the $(n_levels(sch, a)) levels of axis `$ax`; " *
        "every level needs an origin (use `origin=` for a single shared one)."))

    out = Vector{SVector{D,Float64}}(undef, n_internal(sch))
    for internal in 1:n_internal(sch)
        out[internal] = table[ordinal_at(sch, internal, a)]
    end
    return out
end

"""
    _conform_origin(kf, lattice) -> SVector{D,Float64}

On a partially periodic lattice, a momentum origin must have no weight along the aperiodic
directions: numerical residue below tolerance is zeroed exactly, anything larger is refused
by name (the cartesian origin genuinely points out of the periodic subspace).
"""
function _conform_origin(kf::SVector{D,Float64}, lattice::BravaisLattice{D}) where {D}
    P = nperiodic(lattice)
    P == D && return kf
    for m in (P+1):D
        abs(kf[m]) <= LATTICE_ATOL || throw(ArgumentError(
            "The momentum origin resolves to fractional $(Tuple(kf)), with weight along " *
            "aperiodic direction $m; a momentum origin must lie in the periodic subspace."))
    end
    return SVector{D,Float64}(ntuple(m -> m <= P ? kf[m] : 0.0, D))
end

"""
    free_electron_model(lattice; shells=2, mass=1.0, hbar=1.0, gauge=DEFAULT_GAUGE,
                        name="free_electron") -> HamiltonianModel{D}

The empty-lattice model on a plane-wave cell of the first `shells` stars: exactly one diagonal
entry per plane wave,

    E_G(k) = ħ² |bᵀ(k + G)|² / 2m,

i.e. the free parabola folded into the first Brillouin zone -- the standard starting point of
any continuum calculation, and the exactly-solvable anchor for tests. `shells = 0` keeps only
`G = 0` (the bare parabola); truncation converges from below at fixed `k`, since adding shells
only adds higher folded branches.
"""
function free_electron_model(lattice::BravaisLattice{D}; shells::Int=2,
                             mass::Real=1.0, hbar::Real=1.0,
                             gauge::Symbol=DEFAULT_GAUGE,
                             name::AbstractString="free_electron") where {D}
    cell = plane_wave_cell(lattice; shells=shells)
    return compile(cell, [kinetic(; mass=mass, hbar=hbar)]; gauge=gauge, name=String(name))
end
