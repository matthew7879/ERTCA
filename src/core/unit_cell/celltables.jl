##################################################################
###### export only the following functions from this module ######
##################################################################

export CellTables, cell_tables
export state_index, states_with_G, states_with_internal, states_with_level
export states_on_site_with_G




#####################################################################
###### Lookup tables for a unit cell                            #####
#####################################################################

"""
    CellTables{D}

Flat lookup tables for a `UnitCell`, built once and reused by every term during compilation.

Purpose:
--------
Answers the two questions the compiler asks over and over -- *which state carries `G + ΔG`?* and
*which states sit on this site?* -- without hashing a float and without scanning the basis.

Keying:
-------
Every key here is an exact integer: the plane-wave label `G`, the interned internal index, and the
site id. `BasisState` itself is deliberately **not** used as a key: its `hash` folds in
`r_frac::SVector{D,Float64}`, and rounding is not a stable equivalence relation, so a key derived
by arithmetic -- exactly what a ΔG lookup or a symmetry image is -- can land in a different bucket
than the state it denotes. `state_index` is the tolerant resolver for the rare case where a caller
genuinely holds a `BasisState`.

Fields:
-------
- `nstates::Int`
- `G::Vector{SVector{D,Int}}`, `tau::Vector{SVector{D,Float64}}`, `internal::Vector{Int}`
    Per-state columns, so an inner loop reads `t.internal[i]` rather than `cell.basis[i].internal`.
- `by_G::Dict{SVector{D,Int},Vector{Int}}`
    Drives the ΔG continuum couplings.
- `by_site_G::Dict{Tuple{Int,SVector{D,Int}},Vector{Int}}`
    The intersection of `site_states` and `by_G`, which every real-space emitter wants: a bond
    couples states that share a site *and* carry the same `G`. Materialising it costs one extra
    pass over the basis and removes an O(n_states) scan from the innermost emission loop -- see
    `states_on_site_with_G`.
- `by_internal::Vector{Vector{Int}}`
    Indexed *directly* by internal id -- an array, not a dictionary, since ids are dense in
    `1:n_internal`. This replaces the former `by_orbital` and `by_qnum` dictionaries: with orbital
    character now an axis like any other, one table serves both.
- `state_site::Vector{Int}`, `site_states::Vector{Vector{Int}}`,
  `site_positions::Vector{SVector{D,Float64}}`, `site_names::Vector{Symbol}`
    The site partition, copied from the cell rather than recomputed. Sites are a declared
    cell-level concept now, not something the tables derive.
- `schema::InternalSchema`
    Carried so that rules can be compiled against the same space the indices refer to.
- `atol::Float64`
    The cell's position tolerance, carried so `state_index` resolves positions at the same
    tolerance the cell's own site machinery used -- a cell built with a loose `atol` must not be
    silently queried at the default.
"""
struct CellTables{D}
    nstates::Int
    G::Vector{SVector{D,Int}}
    tau::Vector{SVector{D,Float64}}
    k0::Vector{SVector{D,Float64}}
    internal::Vector{Int}
    schema::InternalSchema
    by_G::Dict{SVector{D,Int},Vector{Int}}
    by_site_G::Dict{Tuple{Int,SVector{D,Int}},Vector{Int}}
    by_internal::Vector{Vector{Int}}
    state_site::Vector{Int}
    site_states::Vector{Vector{Int}}
    site_positions::Vector{SVector{D,Float64}}
    site_names::Vector{Symbol}
    site_kind::Vector{Symbol}
    site_period::Vector{SMatrix{D,D,Int}}
    atol::Float64
    nperiodic::Int      # the lattice's periodic rank, for periodic-aware position resolution
end

"""
    cell_tables(cell::UnitCell{D}) -> CellTables{D}

Build the lookup tables for `cell`.

Notes:
------
The site partition is taken from the cell verbatim. Because a site is a *position*, the two spin
states of one sublattice share a site, and a plane-wave cell whose states all sit at the origin has
exactly one site -- so the neighbour search runs over sites, not states, and costs nothing in the
continuum case.
"""
function cell_tables(cell::UnitCell{D})::CellTables{D} where {D}
    basis = cell.basis
    n = length(basis)

    Gs = Vector{SVector{D,Int}}(undef, n)
    tau = Vector{SVector{D,Float64}}(undef, n)
    k0s = Vector{SVector{D,Float64}}(undef, n)
    internal = Vector{Int}(undef, n)

    by_G = Dict{SVector{D,Int},Vector{Int}}()
    sizehint!(by_G, n)
    by_site_G = Dict{Tuple{Int,SVector{D,Int}},Vector{Int}}()
    sizehint!(by_site_G, n)
    by_internal = [Int[] for _ in 1:n_internal(cell.schema)]

    for (i, bs) in pairs(basis)
        Gs[i] = bs.G
        tau[i] = bs.r_frac
        k0s[i] = bs.k0
        internal[i] = bs.internal

        push!(get!(Vector{Int}, by_G, bs.G), i)
        push!(get!(Vector{Int}, by_site_G, (cell.state_site[i], bs.G)), i)
        push!(by_internal[bs.internal], i)
    end

    return CellTables{D}(n, Gs, tau, k0s, internal, cell.schema,
                         by_G, by_site_G, by_internal,
                         ### Shared, not copied. `UnitCell` documents itself as immutable once
                         ### constructed, and these were handed straight out to the term emitters
                         ### unprotected in any case -- so the copies defended against a mutation
                         ### the type's own contract already forbids, at the cost of `n_sites + 3`
                         ### allocations on every compilation.
                         cell.state_site, cell.site_states,
                         cell.site_positions, cell.site_names,
                         cell.site_kind, cell.site_period,
                         cell.atol, nperiodic(cell.lattice))
end




#####################################################################
###### Accessors                                                #####
#####################################################################

"""
    EMPTY_INDICES

Shared empty result for a lookup miss. Callers must not mutate it.
"""
const EMPTY_INDICES = Int[]

"""
    n_states(t::CellTables) -> Int
"""
@inline n_states(t::CellTables)::Int = t.nstates

"""
    n_sites(t::CellTables) -> Int
"""
@inline n_sites(t::CellTables)::Int = length(t.site_states)

"""
    states_with_G(t, G) -> Vector{Int}

The states carrying a given plane-wave label. Exact integer key.
"""
@inline function states_with_G(t::CellTables{D}, G::SVector{D,Int})::Vector{Int} where {D}
    return get(t.by_G, G, EMPTY_INDICES)
end
states_with_G(t::CellTables{D}, G::AbstractVector{<:Integer}) where {D} =
    states_with_G(t, SVector{D,Int}(G))

"""
    states_on_site_with_G(t, site, G) -> Vector{Int}

The states that sit on `site` **and** carry the plane-wave label `G`, ascending.

Notes:
------
This is the set every real-space emitter fans a bond out over: a hopping is G-diagonal and lands
on two declared sites, so its partners are exactly `site_states[site] ∩ by_G[G]`. Both factors
can be the whole basis in the two regimes the package cares about -- one site carrying hundreds
of plane waves (a moiré cell), or hundreds of sites all carrying `G = 0` (a supercell) -- so
intersecting them by scanning either factor costs O(n_states) per bond endpoint. Keyed directly,
it costs one hash.
"""
@inline function states_on_site_with_G(t::CellTables{D}, site::Integer,
                                       G::SVector{D,Int})::Vector{Int} where {D}
    return get(t.by_site_G, (Int(site), G), EMPTY_INDICES)
end

"""
    states_with_internal(t, internal) -> Vector{Int}

The states carrying a given interned internal state. Array-indexed: no hashing.
"""
@inline function states_with_internal(t::CellTables, internal::Integer)::Vector{Int}
    return 1 <= internal <= length(t.by_internal) ? t.by_internal[internal] : EMPTY_INDICES
end

"""
    states_with_level(t, key::Symbol, level) -> Vector{Int}

The states whose internal state takes `level` on axis `key`.

Notes:
------
Replaces the former `states_with_orbital` and `states_with_qnum`: orbital character is now an axis
like any other, so one accessor serves both.
"""
function states_with_level(t::CellTables, key::Symbol, level::QLevel)::Vector{Int}
    a = axis_position(t.schema, key)
    a == 0 && throw(ArgumentError(
        "The cell declares no internal axis `$key`; its axes are $(axis_names(t.schema))."
    ))
    o = level_ordinal(t.schema, a, level)
    o == 0 && return EMPTY_INDICES
    out = Int[]
    for internal in 1:n_internal(t.schema)
        ordinal_at(t.schema, internal, a) == o && append!(out, t.by_internal[internal])
    end
    return sort!(out)
end

"""
    state_index(t, bs::BasisState{D}; atol=t.atol) -> Union{Nothing,Int}

Resolve a `BasisState` to its index in the cell, or `nothing` when it is absent.

Notes:
------
Exact on `(internal, G)` -- both integers -- and tolerant only on `r_frac`. That split is the whole
point of the table: a position produced by arithmetic may differ from the stored one in its last
bits, but the labels never do.

The position compare defaults to the **cell's own** tolerance (carried in the table) and is
periodic: `0.0` and `1 - 1e-15` denote the same place, exactly as the site machinery treats them.
"""
function state_index(t::CellTables{D}, bs::BasisState{D};
                     atol::Float64=t.atol)::Union{Nothing,Int} where {D}
    bs.schema == t.schema || return nothing
    ### `by_G` crossed with the `internal` column replaces the former `(internal, G)` dictionary.
    ### Same complexity -- a G bucket holds one state per internal level -- with one fewer table to
    ### build, which mattered because `cell_tables` runs inside every `compile`.
    for i in get(t.by_G, bs.G, EMPTY_INDICES)
        t.internal[i] == bs.internal || continue
        _same_site(t.tau[i], bs.r_frac, atol, t.nperiodic) && return i
    end
    return nothing
end
