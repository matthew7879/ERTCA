##################################################################
###### export only the following functions from this module ######
##################################################################

export UnitCell, ndim
export n_states, state_positions, state_g_vectors, state_origins, state_internals, display_names
export n_sites, site_names, site_positions, site_of, site_id, states_on_site
export site_kinds, site_periods, is_folded
export same_sites, add_axis, add_plane_waves, delete_sites
export h5_write, h5_read




#############################
###### Main definition ######
#############################

### ------------------------------------------------------------------------------------
### The validation and conformance pass, shared by both `UnitCell` constructors
### ------------------------------------------------------------------------------------
###
### Extracted from the inner constructor on 2026-09-17 so that `new` is called in exactly ONE
### place while there are two ways to decide `K`: DERIVED, for a cell built from scratch, and
### SUPPLIED, for a transform whose signature promises to preserve its input's class. Every
### check below is unchanged; only where the result goes is new.
function _unitcell_fields(::Val{D}, lattice::BravaisLattice{D}, schema::InternalSchema,
                          site_names::Vector{Symbol},
                          site_positions::Vector{SVector{D,Float64}},
                          basis::Vector{BasisState{D}};
                          atol::Float64=LATTICE_ATOL, warn_empty_sites::Bool=true,
                          site_kind::Union{Nothing,Vector{Symbol}}=nothing,
                          site_period::Union{Nothing,AbstractVector{<:SMatrix{D,D,Int}}}=nothing,
                          state_site::Union{Nothing,Vector{Int}}=nothing) where {D}
    length(site_names) == length(site_positions) || throw(DimensionMismatch(
        "Got $(length(site_names)) site names but $(length(site_positions)) site positions."
    ))
    allunique(site_names) || throw(ArgumentError(
        "Site names must be unique; got $site_names."
    ))
    ### Copied, not aliased: `basis` is effectively copied below, and a caller holding the
    ### registry vectors could otherwise `push!` to one of them and desync the two lengths
    ### past every check above.
    names = copy(site_names)
    positions = copy(site_positions)

    ### DECLARED kinds resolve first: site identity is kind-aware -- a :plane_wave site is
    ### identified by its NAME, its position is gauge-only and may coincide with other
    ### sites' -- so the separation check and the state assignment below need the kinds.
    ### Undeclared kinds keep the legacy position-keyed path bit-for-bit (strict
    ### separation, positional assignment, classification from the states afterwards).
    declared_kinds = if isnothing(site_kind)
        nothing
    else
        length(site_kind) == length(positions) || throw(DimensionMismatch(
            "Got $(length(site_kind)) site kinds but $(length(positions)) sites."))
        for k in site_kind
            k in (:localized, :plane_wave) || throw(ArgumentError(
                "Unknown site kind `$k`; expected :localized or :plane_wave."))
        end
        copy(site_kind)
    end
    P = nperiodic(lattice)
    _check_site_separation(positions, declared_kinds; atol=atol, nperiodic=P)

    ### Conform every state onto the cell's schema object, so that `===` holds inside a cell.
    conformed = Vector{BasisState{D}}(undef, length(basis))
    for (i, b) in pairs(basis)
        b.schema == schema || throw(ArgumentError(
            "Basis state $i carries the schema $(b.schema), but the cell declares " *
            "$(schema). All states of a cell must share one schema."
        ))
        conformed[i] = b.schema === schema ? b :
            BasisState{D}(b.r_frac, b.G, b.k0, b.internal, schema; nperiodic=P)
    end

    ### On a partially periodic lattice the aperiodic components carry no translation
    ### symmetry: a plane-wave label or a momentum origin along them is not a meaningful
    ### quantum number, and letting one through would silently corrupt every Bloch phase.
    if P < D
        for (i, b) in pairs(conformed)
            all(m -> b.G[m] == 0, (P+1):D) || throw(ArgumentError(
                "Basis state $i carries G = $(Tuple(b.G)), non-zero along an APERIODIC " *
                "direction (the lattice is periodic only along its first $P direction" *
                "$(P == 1 ? "" : "s")). A plane wave along a non-periodic direction is " *
                "not a valid label."))
            all(m -> abs(b.k0[m]) <= LATTICE_ATOL, (P+1):D) || throw(ArgumentError(
                "Basis state $i carries a momentum origin k0 = $(Tuple(b.k0)) with a " *
                "non-zero component along an aperiodic direction. Momentum is not defined " *
                "there."))
        end
    end

    ### Sites are resolved BEFORE the duplicate check, and the check then runs on
    ### `(site, G, internal)`.
    ###
    ### Doing it the other way round -- as this constructor originally did, via
    ### `check_unrepeated` -- asks two different questions about the same thing: `BasisState`
    ### equality compares `r_frac` with *exact* float equality, while `_assign_sites` resolves
    ### positions to sites with a *tolerance*. Two states the cell then places on one site,
    ### carrying identical `G` and `internal`, sailed through the guard whose whole job is to
    ### reject them, and would silently double-count in the Hamiltonian.
    ###
    ### Keying on the resolved site asks the tolerance question once, in one place, and the key
    ### becomes three exact integers -- so the `Set` holds `isbits` tuples instead of hashing a
    ### `BasisState` (and its schema) per state.
    stsite = if isnothing(state_site)
        _assign_sites(conformed, positions, declared_kinds; atol=atol, nperiodic=P)
    else
        ### the explicit mapping: the disambiguation for coincident positions (several
        ### named :plane_wave families, or a localized orbital at the plane-wave site's
        ### position), validated against everything positions can still check
        length(state_site) == length(conformed) || throw(DimensionMismatch(
            "`state_site` has $(length(state_site)) entries for $(length(conformed)) " *
            "basis states."))
        for (i, s) in pairs(state_site)
            1 <= s <= length(positions) || throw(ArgumentError(
                "`state_site[$i]` = $s is not a site index in 1:$(length(positions))."))
            if isnothing(declared_kinds) || declared_kinds[s] === :localized
                _same_site(conformed[i].r_frac, positions[s], atol, P) || throw(ArgumentError(
                    "`state_site` puts basis state $i (at $(Tuple(conformed[i].r_frac))) " *
                    "on site `$(names[s])` at $(Tuple(positions[s])), but a localized " *
                    "site's position is physical and must match its states."))
            end
        end
        copy(state_site)
    end

    site_states = [Int[] for _ in positions]
    for v in site_states
        sizehint!(v, cld(length(conformed), max(length(positions), 1)))
    end
    for (i, s) in pairs(stsite)
        push!(site_states[s], i)
    end

    ### The site KIND declares what the basis function is (see `terms.jl` and the design
    ### notes): `:localized` -- a comb of Wannier-like orbitals, position physical, constant
    ### kernels only; `:plane_wave` -- a smooth `φ`, position gauge-only, any kernel. When not
    ### declared, a site carrying any non-zero `G` or momentum origin is classified
    ### `:plane_wave` -- the legacy G-stacked spellings keep working under that reading.
    ### `fold` declares `:localized` with `M_s ≠ I` explicitly.
    kinds = if isnothing(declared_kinds)
        derived = [any(i -> !iszero(conformed[i].G) || !iszero(conformed[i].k0), site_states[s]) ?
                       :plane_wave : :localized for s in eachindex(positions)]
        ### a plane wave has no centre: its declared position is GAUGE-ONLY and never enters
        ### a phase (see `_effective_positions`). Carrying one is a deprecated legacy
        ### spelling -- it used to leak into the cross-site ΔG Bloch phases, making spectra
        ### depend on a coordinate that has no physical meaning for a delocalized state.
        for s in eachindex(positions)
            if derived[s] === :plane_wave && !_same_site(positions[s], zero(positions[s]), atol, P)
                @warn "Site `$(names[s])` is classified :plane_wave but sits at " *
                      "$(Tuple(positions[s])). A plane wave has no centre: the position is " *
                      "gauge-only, never enters a phase, and is deprecated -- put " *
                      "plane-wave sites at the origin." maxlog = 1
            end
        end
        derived
    else
        declared_kinds
    end

    periods = if isnothing(site_period)
        fill(one(SMatrix{D,D,Int}), length(positions))
    else
        length(site_period) == length(positions) || throw(DimensionMismatch(
            "Got $(length(site_period)) site periods but $(length(positions)) sites."))
        collect(SMatrix{D,D,Int}, site_period)
    end
    detabs = [_int_det_abs(SMatrix{D,D,Int}(P)) for P in periods]
    ### `site_period` is cell-uniform on the folded sites by constraint: `fold` only ever
    ### produces uniform `M_s`, and unequal per-site periods leave cross-site translations and
    ### the congruence rule undefined (there is no spec for them yet).
    folded = unique(Tuple(periods[s]) for s in eachindex(periods)
                    if kinds[s] === :localized && detabs[s] != 1)
    length(folded) <= 1 || throw(ArgumentError(
        "Sites declare different non-trivial period matrices $(folded). The period lattice " *
        "must be shared by every folded site -- unequal per-site periods leave cross-site " *
        "couplings undefined."))
    for s in eachindex(periods)
        detabs[s] >= 1 || throw(ArgumentError(
            "Site `$(names[s])` declares a singular period matrix $(Tuple(periods[s]))."))
        kinds[s] === :plane_wave && detabs[s] != 1 && throw(ArgumentError(
            "Site `$(names[s])` is :plane_wave but declares a period matrix; a plane wave " *
            "has no period lattice (M_s is meaningful on :localized sites only)."))
    end

    ### Duplicate check. On an ordinary site the key is the exact triple (site, G, internal).
    ### On a FOLDED localized site (M_s ≠ I) two labels congruent mod M_s ℤᴰ denote the same
    ### comb up to a constant phase, so the key uses the exact integer coset label instead.
    seen = Set{Tuple{Int,SVector{D,Int},Int}}()
    sizehint!(seen, length(conformed))
    for (i, b) in pairs(conformed)
        s = stsite[i]
        gkey = if kinds[s] === :localized && detabs[s] != 1
            A = SMatrix{D,D,Int}(periods[s])
            _coset_label(_int_adjugate(A), detabs[s], b.G)
        else
            b.G
        end
        key = (s, gkey, b.internal)
        key in seen && throw(ArgumentError(
            "Basis state $i duplicates an earlier one: both sit on site " *
            "`$(names[s])`, carry G = $(Tuple(b.G)) " *
            (kinds[s] === :localized && detabs[s] != 1 ?
                 "(congruent mod the site's period lattice) " : "") *
            "and the same internal state. Two states agreeing on all three are the same state."
        ))
        push!(seen, key)
    end

    ### A declared site with no states is almost always a typo -- most often a mistyped level
    ### in `orbitals=`/`site_levels=`. It is also quadratically expensive downstream, because
    ### the neighbour search and the term emitters loop over site *pairs*.
    if warn_empty_sites
        empty_sites = [names[s] for s in eachindex(site_states) if isempty(site_states[s])]
        isempty(empty_sites) || @warn "Declared site(s) $empty_sites carry no basis states. " *
            "They still cost work in the neighbour search, which runs over site pairs."
    end

    return (lattice, schema, names, positions, conformed, stsite, site_states, kinds,
            periods, atol)
end


"""
    UnitCell{D,K}: A geometric unit cell.

Purely geometric: it carries no information about the Hamiltonian.

Three registries, three distinct jobs:
--------------------------------------
| handle  | names                                 | stored in                       |
|---------|---------------------------------------|---------------------------------|
| site    | a position (`:A`, `:M`)               | `site_names` / `site_positions` |
| orbital | an envelope character (`:pz`, `:dxy`) | an axis of `schema`             |
| qnum    | spin / valley / layer / Nambu         | an axis of `schema`             |

Each name is stored exactly once, for the *kind* it describes -- never once per state. A cell with
two sublattices and hundreds of plane waves holds two site names.

Fields:
-------
- `lattice::BravaisLattice{D}`
    The Bravais lattice.

- `schema::InternalSchema`
    The declared internal label space, shared by every basis state. The constructor conforms the
    states onto this exact object, so `b.schema === cell.schema` holds throughout.

- `site_names::Vector{Symbol}`, `site_positions::Vector{SVector{D,Float64}}`
    The site registry, indexed by site id. A site is a *position*: everything non-spatial is
    internal structure at that position, so `p_x` and `p_y` on one atom are one site with two
    internal states -- which is what a site-symmetry group means.

- `basis::Vector{BasisState{D}}`
    The basis states.

- `state_site::Vector{Int}`, `site_states::Vector{Vector{Int}}`
    The state/site correspondence, both directions. Stored rather than recomputed because sites are
    a declared cell-level concept and the "every state sits on a declared site" invariant is
    checked at construction anyway. `site_states[s]` is sorted ascending.

- `atol::Float64`
    The tolerance that resolved positions to sites. Stored because it is part of what makes the
    partition well defined: a cell built with a loose tolerance must keep it when `add_axis`,
    `add_plane_waves` or `supercell` rebuild it, or the derived cell would be re-matched against
    the registry at the default and could fail a check its parent passed.

!!! warning
    The vector fields are mutable, and the invariants above are established at construction and not
    maintained afterwards. `push!(uc.basis, bs)` desyncs `n_states` from `state_site` with no
    diagnostic. Treat a constructed `UnitCell` as immutable.

Notes:
------
Sites may be declared by name, or derived from the basis positions and named `:site1`, `:site2`, ...
There is no orbital auto-assignment: the label that used to be auto-assigned per *state* is now a
schema axis, and the site partition is derived per *position*. That removes a latent trap in which
an unlabelled plane-wave basis gave every `G` its own site.
"""
struct UnitCell{D,K<:KPeriodicity}
    lattice::BravaisLattice{D}
    schema::InternalSchema
    site_names::Vector{Symbol}
    site_positions::Vector{SVector{D,Float64}}
    basis::Vector{BasisState{D}}
    state_site::Vector{Int}
    site_states::Vector{Vector{Int}}
    site_kind::Vector{Symbol}                 # :localized | :plane_wave, per site
    site_period::Vector{SMatrix{D,D,Int}}     # M_s with a = M_s · a_s; I for an ordinary orbital
    atol::Float64

    ### The ONE `new`. Private by arity -- ten positional, already-validated fields, produced
    ### only by `_unitcell_fields`. Public construction goes through the two outer constructors
    ### below, which differ in nothing but how `K` is decided.
    ### the collection arguments are deliberately loose: `new` converts each to its declared
    ### field type exactly as it did when this was the validating inner constructor, and
    ### spelling `Vector{SMatrix{D,D,Int}}` here instead fails to match the concrete
    ### `Vector{SMatrix{2,2,Int64,4}}` that `_unitcell_fields` produces -- `Vector` is
    ### invariant and `SMatrix{D,D,Int}` leaves the length parameter free.
    UnitCell{D,K}(lattice::BravaisLattice{D}, schema::InternalSchema,
                  site_names::AbstractVector, site_positions::AbstractVector,
                  basis::AbstractVector, state_site::AbstractVector,
                  site_states::AbstractVector, site_kind::AbstractVector,
                  site_period::AbstractVector,
                  atol::Real) where {D,K<:KPeriodicity} =
        new{D,K}(lattice, schema, site_names, site_positions, basis, state_site, site_states,
                 site_kind, site_period, atol)
end

### ------------------------------------------------------------------------------------
### Reading the class back off the type
### ------------------------------------------------------------------------------------

"""
    kperiodicity(cell::UnitCell) -> KPeriodicity

How `H(k)` on this cell relates to the Brillouin-zone torus: `ExactlyPeriodic()`,
`TruncatedWindow()` or `NoZone()`, each documented on its own type.

Since 2026-09-17 this is a read of the cell's TYPE parameter, not a derivation: the class is
decided once, by whichever constructor built the cell, and cannot disagree with itself
afterwards. `kperiodicity(model)` and `kperiodicity(spectrum)` forward to the cell they hold.
"""
kperiodicity(::UnitCell{D,K}) where {D,K} = K()

### The class as a TYPE, for the `{D,K}` constructors of everything that holds a cell -- a
### model, a spectrum, a compile context. Spelled once so no caller writes
### `typeof(kperiodicity(cell))` by hand.
_kpar(::UnitCell{D,K}) where {D,K} = K

### The cell-level forward to the field-level derivation in `kperiodicity.jl`. Used to CHECK a
### cell (the tests do) and by the transforms that legitimately re-derive -- `add_plane_waves`
### changes the class by construction, so preserving it there would be wrong.
_derive_kperiodicity(cell::UnitCell)::KPeriodicity =
    _derive_kperiodicity(cell.site_kind, cell.site_states, cell.basis)

"""
    _check_supplied_kperiodicity(cell, source) -> Nothing

Refuse a cell whose supplied class cannot be reconciled with its contents. `source` names
where the value came from, so the refusal points at the file or the transform rather than at
this function.
"""
function _check_supplied_kperiodicity(cell::UnitCell, source::AbstractString)::Nothing
    stored  = kperiodicity(cell)
    derived = _derive_kperiodicity(cell)
    _kper_agrees(stored, derived) && return nothing
    throw(ArgumentError(
        "$source declares the k-periodicity class `$(repr(stored))`, but this cell's own " *
        "site kinds and `G` sets imply `$(repr(derived))`, and that pair cannot arise from " *
        "any transform this package performs. The ONE legitimate disagreement is a stored " *
        "`NoZone()` where the derivation says `TruncatedWindow()` -- a folded k·p frame, " *
        "whose `|det M|` coset labels read as a truncation window. Honouring this value " *
        "instead would re-enable, or wrongly refuse, the Brillouin-zone quantities the class " *
        "exists to gate: a cell wrongly marked `ExactlyPeriodic()` gets a Chern number " *
        "rather than a refusal. If the file was written by a newer version of " *
        "SingleParticleModels, update the package; its `format_version` is what should have " *
        "refused this file first."))
end

"""
    UnitCell{D}(lattice, schema, site_names, site_positions, basis; kwargs...) -> UnitCell{D,K}

Build a cell from scratch and DERIVE its k-periodicity class from the finished site kinds and
`G` sets. This is the right constructor for every cell that is being built rather than mapped:
a named-site cell, a plane-wave cell, a cell read back from a file.

A transform that must not change the class uses [`UnitCell{D,K}`](@ref) instead -- see the note
there, and `fold` for the case that motivated the distinction.
"""
function UnitCell{D}(lattice::BravaisLattice{D}, schema::InternalSchema,
                     site_names::Vector{Symbol}, site_positions::Vector{SVector{D,Float64}},
                     basis::Vector{BasisState{D}}; kwargs...) where {D}
    f = _unitcell_fields(Val(D), lattice, schema, site_names, site_positions, basis; kwargs...)
    K = typeof(_derive_kperiodicity(f[8], f[7], f[5]))
    return UnitCell{D,K}(f...)
end

"""
    UnitCell{D,K}(lattice, schema, site_names, site_positions, basis; kwargs...) -> UnitCell{D,K}

Build a cell with its k-periodicity class SUPPLIED rather than derived.

This exists because the derivation is right for every cell built from scratch and wrong for one
that came out of a basis change. Folding replicates a single plane-wave site's one `G` label
into `|det M|` coset representatives, which the derivation reads as a truncation window: a k·p
frame with no Brillouin-zone torus would come back as `TruncatedWindow()`, laundering every
refusal that protects it. Whether `H(k)` is BZ-periodic is a property of the physics, not of
which basis it is written in.

The point of putting `K` in the type is that a transform now states the invariant in its own
signature -- `fold(uc::UnitCell{D,K}) where {D,K} -> UnitCell{D,K}` -- so preserving the class
is not something a future transform can forget to do. It was previously carried by hand on the
compiled model, which left the invariant sometimes derived and sometimes carried
which left the invariant sometimes derived and sometimes carried.
"""
function UnitCell{D,K}(lattice::BravaisLattice{D}, schema::InternalSchema,
                       site_names::Vector{Symbol}, site_positions::Vector{SVector{D,Float64}},
                       basis::Vector{BasisState{D}}; kwargs...) where {D,K<:KPeriodicity}
    f = _unitcell_fields(Val(D), lattice, schema, site_names, site_positions, basis; kwargs...)
    return UnitCell{D,K}(f...)
end



# ----- Main constructor: named sites, the everyday spelling -----
"""
    UnitCell(lattice, :A => [1/3,2/3], :B => [2/3,1/3]; schema=..., G=..., site_levels=..., orbitals=...)

Build a cell generatively from named sites: the basis is the product of sites, internal states and
plane-wave labels.

Args:
-----
- `sites`: `name => fractional position` pairs
- `schema`: the internal label space; defaults to [`TRIVIAL_SCHEMA`](@ref)
- `G`: the plane-wave labels to replicate over; defaults to `[0]`
- `site_levels`: `site name => (axis => allowed levels)`, restricting which internal states occur
  on a given site
- `orbitals`: sugar for `site_levels` restricted to the `:orbital` axis

Returns:
--------
- `UnitCell{D}`

Notes:
------
`orbitals` is not a fourth mechanism -- it lowers to `site_levels`. Restricting levels per site is
what lets a metal carry d orbitals and an oxygen carry p orbitals from one shared `:orbital` axis:
the axis is the union across sites, and only the combinations that occur get states.
"""
function UnitCell(lattice::BravaisLattice{D}, sites::Pair{Symbol,<:AbstractVector}...;
                  schema::InternalSchema=TRIVIAL_SCHEMA,
                  G::AbstractVector=[zeros(Int, D)],
                  site_levels::Union{Nothing,AbstractDict}=nothing,
                  orbitals::Union{Nothing,AbstractDict}=nothing,
                  site_kind::Union{Nothing,Vector{Symbol}}=nothing,
                  atol::Float64=LATTICE_ATOL)::UnitCell{D} where {D}
    isempty(sites) && throw(ArgumentError("At least one site must be given."))

    names = Symbol[p.first for p in sites]
    pos = SVector{D,Float64}[canonicalize_fracpos(SVector{D,Float64}(p.second);
                                                  nperiodic=nperiodic(lattice))
                             for p in sites]

    ### `orbitals` is sugar over `site_levels`
    levels_by_site = Dict{Symbol,Dict{Symbol,Vector{QLevel}}}()
    if !isnothing(orbitals)
        for (s, vs) in orbitals
            levels_by_site[Symbol(s)] = Dict(:orbital => QLevel[_as_level(v) for v in vs])
        end
    end
    if !isnothing(site_levels)
        for (s, spec) in site_levels
            d = get!(Dict{Symbol,Vector{QLevel}}, levels_by_site, Symbol(s))
            for (ax, vs) in spec
                d[Symbol(ax)] = QLevel[_as_level(v) for v in vs]
            end
        end
    end
    for (s, restriction) in levels_by_site
        s in names || throw(ArgumentError(
            "`site_levels`/`orbitals` names site `$s`, which is not declared; sites are $names."
        ))
        _validate_restriction(schema, s, restriction)
    end

    Gs = SVector{D,Int}[canonicalize_int(g) for g in G]
    allunique(Gs) || throw(ArgumentError(
        "`G` contains repeated plane-wave labels; every (site, internal, G) combination must be " *
        "distinct. Got $(Tuple.(Gs))."
    ))

    ### Ordering: G outermost, then internal, then site. See `STATE_ORDER` -- this is what makes
    ### `UnitCell(lat, sites...; schema, G)` produce the *same basis order* as building it up with
    ### `add_plane_waves(add_axis(UnitCell(lat, sites...), ...), Gs)`.
    basis = BasisState{D}[]
    ssite = Int[]
    for g in Gs, internal in 1:n_internal(schema)
        for (sidx, sname) in pairs(names)
            _internal_allowed(schema, internal, get(levels_by_site, sname, nothing)) || continue
            push!(basis, BasisState{D}(pos[sidx], g, internal, schema;
                                       nperiodic=nperiodic(lattice)))
            push!(ssite, sidx)
        end
    end

    ### the generative constructor KNOWS the state -> site mapping, so it passes it explicitly:
    ### with declared :plane_wave kinds, several named sites may share a position (identity is
    ### the name) and the positional rule alone could not decide
    return UnitCell{D}(lattice, schema, names, pos, basis;
                       atol=atol, site_kind=site_kind, state_site=ssite)
end

"""
    UnitCell(lattice, sites::AbstractVector{<:Pair}; kwargs...) -> UnitCell{D}

Vector-of-pairs spelling of the generative constructor, so that `sites` may be written the same way
whether it is passed positionally or as the `sites=` keyword.
"""
function UnitCell(lattice::BravaisLattice{D}, sites::AbstractVector{<:Pair};
                  kwargs...)::UnitCell{D} where {D}
    return UnitCell(lattice, sites...; kwargs...)
end

# ----- Explicit-basis constructor -----
"""
    UnitCell(lattice, basis; schema=nothing, sites=nothing, atol=LATTICE_ATOL)

Build a unit cell from an explicit list of basis states.

Args:
-----
- `basis`: the basis states
- `schema`: the internal label space; inferred from the first state when omitted
- `sites`: a vector of `name => position` pairs. When omitted, sites are derived from the basis
  positions and named `:site1`, `:site2`, ...

Returns:
--------
- `UnitCell{D}`
"""
function UnitCell(lattice::BravaisLattice{D}, basis::Vector{BasisState{D}};
                  kwargs...)::UnitCell{D} where {D}
    sch, names, pos, rest = _cell_from_basis(lattice, basis; kwargs...)
    return UnitCell{D}(lattice, sch, names, pos, basis; rest...)
end

"""
    UnitCell{D,K}(lattice, basis::Vector{BasisState{D}}; kwargs...) -> UnitCell{D,K}

The class-PRESERVING form of the basis constructor, for a transform whose signature promises to
return its input's k-periodicity class. `fold` is the case that motivated it; see the note on
[`UnitCell{D,K}`](@ref)'s five-argument form.
"""
function UnitCell{D,K}(lattice::BravaisLattice{D}, basis::Vector{BasisState{D}};
                       kwargs...)::UnitCell{D,K} where {D,K<:KPeriodicity}
    sch, names, pos, rest = _cell_from_basis(lattice, basis; kwargs...)
    return UnitCell{D,K}(lattice, sch, names, pos, basis; rest...)
end

### The schema/site-name/position resolution the two constructors above share, so that deciding
### `K` is the ONLY thing that differs between them.
function _cell_from_basis(lattice::BravaisLattice{D}, basis::Vector{BasisState{D}};
                          schema::Union{Nothing,InternalSchema}=nothing,
                          sites::Union{Nothing,AbstractVector{<:Pair}}=nothing,
                          site_kind::Union{Nothing,Vector{Symbol}}=nothing,
                          site_period::Union{Nothing,AbstractVector{<:SMatrix{D,D,Int}}}=nothing,
                          state_site::Union{Nothing,Vector{Int}}=nothing,
                          atol::Float64=LATTICE_ATOL) where {D}
    sch = if !isnothing(schema)
        schema
    elseif isempty(basis)
        TRIVIAL_SCHEMA
    else
        basis[1].schema
    end

    if isnothing(sites)
        pos, _ = _derive_sites([b.r_frac for b in basis]; atol=atol,
                               nperiodic=nperiodic(lattice))
        names = [_auto_site_name(i) for i in eachindex(pos)]
    else
        names = Symbol[p.first for p in sites]
        pos = SVector{D,Float64}[canonicalize_fracpos(SVector{D,Float64}(p.second);
                                                      nperiodic=nperiodic(lattice))
                                 for p in sites]
    end
    return sch, names, pos, (atol=atol, site_kind=site_kind, site_period=site_period,
                             state_site=state_site)
end

"""
    UnitCell(lattice, basis::BasisState...; kwargs...)
"""
function UnitCell(lattice::BravaisLattice{D}, basis::BasisState{D}...; kwargs...) where {D}
    return UnitCell(lattice, collect(basis); kwargs...)
end

### Spelled out explicitly because the two varargs constructors above (named-site pairs, basis
### states) both match a call with no trailing arguments and neither is more specific -- so
### without this method `UnitCell(lattice)` raises an ambiguity error that names no cause. An
### empty cell is legitimate, and `UnitCell(lattice, BasisState{D}[])` already builds one.
"""
    UnitCell(lattice) -> UnitCell{D}

The empty cell: no basis states and no sites.
"""
function UnitCell(lattice::BravaisLattice{D})::UnitCell{D} where {D}
    return UnitCell(lattice, BasisState{D}[])
end

"""
    _internal_allowed(schema, internal, restriction) -> Bool

Whether an internal state is permitted on a site, given a per-axis level restriction.
"""
function _internal_allowed(schema::InternalSchema, internal::Int,
                           restriction::Union{Nothing,AbstractDict})::Bool
    isnothing(restriction) && return true
    for (ax, allowed) in restriction
        a = _restriction_axis(schema, ax)
        level_at(schema, internal, a) in allowed || return false
    end
    return true
end

"""
    _validate_restriction(schema, site, restriction) -> Nothing

Check that a per-site level restriction names a declared axis and declared levels.

Notes:
------
The level half matters as much as the axis half. Without it, `orbitals = Dict(:M => [:dxz])`
against an axis declaring `[:dxy, :dyz]` matches no internal state, so the site is built with
**zero** basis states and nothing is reported -- the same silent-empty failure the schema exists to
eliminate for quantum-number rules.
"""
function _validate_restriction(schema::InternalSchema, site::Symbol,
                               restriction::AbstractDict)::Nothing
    for (ax, allowed) in restriction
        a = _restriction_axis(schema, ax)
        isempty(allowed) && throw(ArgumentError(
            "Level restriction for site `$site` on axis `$(axis_names(schema)[a])` is empty, " *
            "which would leave the site with no basis states."
        ))
        for v in allowed
            level_ordinal(schema, a, v) == 0 && throw(ArgumentError(
                "Level restriction for site `$site` names $(repr(v)) on axis " *
                "`$(axis_names(schema)[a])`, which declares $(levels(schema, a)). No internal " *
                "state would match, so the site would be built with no basis states."
            ))
        end
    end
    return nothing
end

"""
    _restriction_axis(schema, ax) -> Int
"""
function _restriction_axis(schema::InternalSchema, ax)::Int
    a = axis_position(schema, Symbol(ax))
    a == 0 && throw(ArgumentError(
        "Level restriction names axis `$ax`, which the schema does not declare; " *
        "declared axes are $(axis_names(schema))."
    ))
    return a
end

"""
    _auto_site_name(i) -> Symbol

Name for a site the caller did not name.
"""
_auto_site_name(i::Integer)::Symbol = Symbol(AUTO_SITE_PREFIX, i)




#####################################################################
###### The site partition                                       #####
#####################################################################

"""
    _frac_delta(u, v) -> SVector{D,Float64}

Fractional separation reduced to the nearest periodic image.

Notes:
------
`d .- round.(d)` is what makes positions at `0.0` and `1 - 1e-15` the same site. Comparing raw
fractional coordinates puts them at opposite ends of a lexicographic sort, so they never merge and
one site is reported as two.
"""
@inline function _frac_delta(u::SVector{D,Float64}, v::SVector{D,Float64},
                             nperiodic::Int=D)::SVector{D,Float64} where {D}
    d = u .- v
    nperiodic == D && return d .- round.(d)
    ### only the periodic components have images: an aperiodic separation (a ribbon's
    ### transverse offset) is a real distance and must never be folded
    return SVector{D,Float64}(ntuple(m -> m <= nperiodic ? d[m] - round(d[m]) : d[m], D))
end

"""
    _same_site(u, v, atol[, nperiodic]) -> Bool
"""
@inline function _same_site(u::SVector{D,Float64}, v::SVector{D,Float64}, atol::Float64,
                            nperiodic::Int=D)::Bool where {D}
    return all(abs.(_frac_delta(u, v, nperiodic)) .<= atol)
end

"""
    _derive_sites(tau; atol) -> (site_positions, state_site)

Cluster fractional positions into sites when no registry was declared.

Notes:
------
Each position is matched against the *representative* of an existing cluster, not against its
predecessor in sort order. Comparing to the predecessor is not transitive: positions at `0`,
`0.9*atol` and `1.8*atol` chain into one cluster although the endpoints are further apart than
`atol`. Anchoring on the representative bounds every cluster by one ball of radius `atol`, and
`_check_site_separation` then rejects the residual ambiguity outright instead of resolving it
silently.
"""
function _derive_sites(tau::Vector{SVector{D,Float64}};
                       atol::Float64=LATTICE_ATOL, nperiodic::Int=D) where {D}
    n = length(tau)
    state_site = Vector{Int}(undef, n)
    site_tau = SVector{D,Float64}[]
    n == 0 && return site_tau, state_site

    order = sortperm(1:n; by = i -> Tuple(tau[i]))
    for i in order
        s = findfirst(r -> _same_site(tau[i], r, atol, nperiodic), site_tau)
        if isnothing(s)
            push!(site_tau, tau[i])
            s = length(site_tau)
        end
        state_site[i] = s
    end
    _check_site_separation(site_tau; atol=atol, nperiodic=nperiodic)
    return site_tau, state_site
end

"""
    _check_site_separation(site_positions; atol) -> Nothing

Reject a site registry whose members are closer than `2*atol`.

Notes:
------
Below that separation, "within `atol` of the same site" stops being an equivalence relation and the
partition would depend on the order the basis happens to be given in. Raising is better than
choosing one of the two answers.
"""
function _check_site_separation(site_tau::Vector{SVector{D,Float64}};
                                atol::Float64=LATTICE_ATOL, nperiodic::Int=D)::Nothing where {D}
    for a in eachindex(site_tau), b in (a+1):length(site_tau)
        _same_site(site_tau[a], site_tau[b], 2 * atol, nperiodic) && throw(ArgumentError(
            "Sites $a and $b sit at $(Tuple(site_tau[a])) and $(Tuple(site_tau[b])), separated by " *
            "less than 2*atol = $(2*atol). Site membership is then not transitive and the " *
            "partition would depend on basis ordering. Merge them, or lower `atol`."
        ))
    end
    return nothing
end

"""
    _check_site_separation(site_positions, kinds; atol) -> Nothing

The kind-aware separation check: a `:plane_wave` site is identified by its NAME -- its position
is gauge-only -- so only pairs of `:localized` sites must be positionally separated. Two named
plane-wave families at the origin, or a localized orbital coexisting at the plane-wave site's
position, are legitimate cells. `kinds === nothing` (undeclared, legacy path) falls back to the
strict all-pairs check above.
"""
function _check_site_separation(site_tau::Vector{SVector{D,Float64}},
                                kinds::Union{Nothing,Vector{Symbol}};
                                atol::Float64=LATTICE_ATOL, nperiodic::Int=D)::Nothing where {D}
    isnothing(kinds) && return _check_site_separation(site_tau; atol=atol, nperiodic=nperiodic)
    for a in eachindex(site_tau), b in (a+1):length(site_tau)
        (kinds[a] === :plane_wave || kinds[b] === :plane_wave) && continue
        _same_site(site_tau[a], site_tau[b], 2 * atol, nperiodic) && throw(ArgumentError(
            "Sites $a and $b sit at $(Tuple(site_tau[a])) and $(Tuple(site_tau[b])), separated by " *
            "less than 2*atol = $(2*atol). Site membership is then not transitive and the " *
            "partition would depend on basis ordering. Merge them, or lower `atol`."
        ))
    end
    return nothing
end

"""
    _assign_sites(tau, site_positions; atol) -> Vector{Int}

Resolve each state against a declared site registry.

Notes:
------
Fully order-independent: each position must match exactly one declared site, and both "matches
none" and "matches several" are errors.
"""
function _assign_sites(states, site_tau::Vector{SVector{D,Float64}};
                       atol::Float64=LATTICE_ATOL, nperiodic::Int=D)::Vector{Int} where {D}
    out = Vector{Int}(undef, length(states))
    for (i, st) in pairs(states)
        ### Accepts basis states or bare positions, so the caller need not materialize a whole
        ### temporary vector of `r_frac` just to have it scanned once.
        t = _position_of(st)
        ### Counted in place rather than with `findall`, which would allocate a vector per state --
        ### 244 of them for a moire cell, purely to check that exactly one matched.
        hit = 0
        nhits = 0
        for (s, r) in pairs(site_tau)
            if _same_site(t, r, atol, nperiodic)
                nhits += 1
                hit = s
            end
        end
        nhits == 1 || throw(ArgumentError(
            nhits == 0 ?
                "Basis state $i sits at $(Tuple(t)), which matches no declared site " *
                "(declared: $(Tuple.(site_tau)))." :
                "Basis state $i at $(Tuple(t)) matches $nhits declared sites."
        ))
        out[i] = hit
    end
    return out
end

"""
    _assign_sites(states, site_positions, kinds; atol) -> Vector{Int}

The kind-aware assignment: with declared kinds, positions may legitimately coincide
(`:plane_wave` sites are name-identified), so a state whose position matches SEVERAL sites is
an ambiguity the positional rule cannot decide -- refused with the disambiguation contract
named. `kinds === nothing` falls back to the strict positional method above.
"""
function _assign_sites(states, site_tau::Vector{SVector{D,Float64}},
                       kinds::Union{Nothing,Vector{Symbol}};
                       atol::Float64=LATTICE_ATOL, nperiodic::Int=D)::Vector{Int} where {D}
    isnothing(kinds) && return _assign_sites(states, site_tau; atol=atol, nperiodic=nperiodic)
    out = Vector{Int}(undef, length(states))
    for (i, st) in pairs(states)
        t = _position_of(st)
        hit = 0
        nhits = 0
        for (s, r) in pairs(site_tau)
            if _same_site(t, r, atol, nperiodic)
                nhits += 1
                hit = s
            end
        end
        nhits == 1 || throw(ArgumentError(
            nhits == 0 ?
                "Basis state $i sits at $(Tuple(t)), which matches no declared site " *
                "(declared: $(Tuple.(site_tau)))." :
                "Basis state $i at $(Tuple(t)) matches $nhits declared sites. Coincident " *
                "positions are legal when `:plane_wave` sites are involved (their identity is " *
                "the NAME, not the position), but then the positional rule cannot decide which " *
                "site each state belongs to: pass `state_site=` (one site index per state) to " *
                "the constructor, or build generatively from named sites, which knows the " *
                "mapping."
        ))
        out[i] = hit
    end
    return out
end

"""
    _position_of(x) -> SVector{D,Float64}

The fractional position of a basis state, or of a bare position.
"""
@inline _position_of(bs::BasisState{D}) where {D} = bs.r_frac
@inline _position_of(p::SVector{D,Float64}) where {D} = p




######################################################
###### Geometric equivalence between unit cells ######
######################################################

"""
    ==(uc1::UnitCell, uc2::UnitCell) -> Bool

Whether two unit cells are geometrically equivalent.

Equivalent when their Bravais lattices are equivalent, their internal schemas are structurally
equal, they hold the same number of basis states, and each state of one matches a state of the
other after transporting its position and plane-wave label through the lattice equivalence.

Notes:
------
The schema check comes first and is a single word, via the schema's cached hash. Without it the
comparison would be meaningless: internal index `3` denotes different states under different
schemas, so the transported states could match on integers alone.

**Site names are excluded**, mirroring `BravaisLattice`: `==` asks whether two cells describe the
same physical object, `same_sites` asks whether they describe it the same way.
"""
function Base.:(==)(uc1::UnitCell{D1}, uc2::UnitCell{D2})::Bool where {D1, D2}
    D1 == D2 || return false
    uc1.lattice == uc2.lattice || return false
    uc1.schema == uc2.schema || return false
    length(uc1.basis) == length(uc2.basis) || return false

    ### The kind/period registry must agree as a MULTISET (site order may differ between two
    ### equivalent descriptions, and `==` ignores names): a folded cell is not equal to its
    ### unfolded spelling even when the states happen to transport onto each other.
    sort(collect(zip(uc1.site_kind, abs.(det.(uc1.site_period))))) ==
        sort(collect(zip(uc2.site_kind, abs.(det.(uc2.site_period))))) || return false

    for b in uc1.basis
        # express the position in the other lattice
        real_pos = fractional_to_real(b.r_frac, uc1.lattice)
        new_pos = canonicalize_fracpos(real_to_fractional(real_pos, uc2.lattice);
                                       nperiodic=nperiodic(uc2.lattice))

        # express the plane-wave label in the other reciprocal lattice
        real_G = fractional_to_reciprocal(b.G, uc1.lattice)
        raw_G = reciprocal_to_fractional(real_G, uc2.lattice)
        ### `canonicalize_int` RAISES when the round trip does not land on integers, so calling it
        ### bare would make `==` throw instead of answering. A transported Miller index that is not
        ### integral simply means the cells are not equivalent.
        rounded_G = round.(raw_G)
        isapprox(raw_G, rounded_G; atol=INT_ROUND_ATOL) || return false
        new_G = SVector{D2,Int}(rounded_G)

        # the momentum origin transports exactly like a (fractional) momentum
        real_k0 = fractional_to_reciprocal(b.k0, uc1.lattice)
        new_k0 = reciprocal_to_fractional(real_k0, uc2.lattice)

        ### Exact on the integer labels, tolerant and PERIODIC on the transported position. The
        ### position is arithmetic-derived, and after grid rounding an exact float compare can
        ### miss by one grid step (the keying doctrine of `celltables.jl`); `_same_site` also
        ### folds the [0,1) boundary, where 0.0 and 1 - eps denote the same place.
        tol = max(uc1.atol, uc2.atol)
        any(b2 -> b2.internal == b.internal && b2.G == new_G &&
                  isapprox(b2.k0, new_k0; atol=tol) &&
                  _same_site(b2.r_frac, new_pos, tol, nperiodic(uc2.lattice)),
            uc2.basis) || return false
    end
    return true
end

function Base.:(!=)(uc1::UnitCell, uc2::UnitCell)::Bool
    return !(uc1 == uc2)
end

"""
    hash(uc::UnitCell, h::UInt)

Hash consistent with `==`.

Notes:
------
`==` is equivalence under a unimodular change of the lattice basis and a relabelling of the sites,
so a consistent hash may fold in only what survives both. Three things do: the dimension, the number
of basis states, and the internal schema -- all of which `==` compares directly. Positions, site
names and the primitive vectors do not, and none of them may be hashed here.

Coarse but exact, in the same spirit as the `BravaisLattice` method: `a == b => hash(a) == hash(b)`
is what the contract asks, and defining nothing at all -- as this type previously did -- let `==`
cells land in different `Dict` buckets.
"""
Base.hash(uc::UnitCell{D}, h::UInt) where {D} =
    hash(:UnitCell, hash(D, hash(length(uc.basis), hash(uc.schema, h))))

"""
    same_sites(uc1, uc2; atol=LATTICE_ATOL) -> Bool

Whether two cells declare the same site registry -- same names at the same positions, in any
order -- and not merely the same geometry.

`==` answers "the same lattice and basis?"; this answers "described the same way?".

Notes:
------
**Permutation-insensitive**: the registry is compared as a set of `name => position` associations,
so `[:A, :B]` and `[:B, :A]` at correspondingly swapped positions are the same registry. Site ids
are an implementation detail of the ordering, not part of what the registry *says*; two cells that
name the same places the same way describe the same thing regardless of the order they were listed
in. Use `uc1.site_names == uc2.site_names` directly if the order itself matters.

Names are unique within a cell, so the association is matched by name and compared on position with
`atol`.

Cells of different dimension return `false` rather than raising, matching `==`.
"""
same_sites(::UnitCell, ::UnitCell; kwargs...)::Bool = false

function same_sites(uc1::UnitCell{D}, uc2::UnitCell{D};
                    atol::Float64=LATTICE_ATOL)::Bool where {D}
    length(uc1.site_names) == length(uc2.site_names) || return false
    for (k, name) in pairs(uc1.site_names)
        s = findfirst(==(name), uc2.site_names)
        isnothing(s) && return false
        _same_site(uc1.site_positions[k], uc2.site_positions[s], atol,
                   nperiodic(uc1.lattice)) || return false
    end
    return true
end




#################################################
###### Accessors                           ######
#################################################

"""
    n_states(uc::UnitCell) -> Int

The number of basis states.
"""
@inline n_states(uc::UnitCell{D}) where {D} = length(uc.basis)

"""
    state_positions(uc::UnitCell) -> Vector{SVector{D,Float64}}

Fractional positions of all basis states, in basis order.
"""
state_positions(uc::UnitCell{D}) where {D} = [bs.r_frac for bs in uc.basis]

"""
    state_g_vectors(uc::UnitCell) -> Vector{SVector{D,Int}}

Plane-wave labels of all basis states, in basis order.
"""
state_g_vectors(uc::UnitCell{D}) where {D} = [bs.G for bs in uc.basis]

"""
    state_origins(uc::UnitCell) -> Vector{SVector{D,Float64}}

Momentum origins `K_s` of all basis states (fractional reciprocal coordinates), in basis order.
"""
state_origins(uc::UnitCell{D}) where {D} = [bs.k0 for bs in uc.basis]

"""
    state_internals(uc::UnitCell) -> Vector{Int}

Interned internal indices of all basis states, in basis order. Decode with `level_at` or `decode`.
"""
state_internals(uc::UnitCell{D}) where {D} = [bs.internal for bs in uc.basis]

"""
    n_sites(uc::UnitCell) -> Int
"""
@inline n_sites(uc::UnitCell) = length(uc.site_positions)

"""
    site_names(uc::UnitCell) -> Vector{Symbol}

The declared site names, indexed by site id. Returns a copy.
"""
site_names(uc::UnitCell)::Vector{Symbol} = copy(uc.site_names)

"""
    site_positions(uc::UnitCell) -> Vector{SVector{D,Float64}}

The declared site positions, indexed by site id. Returns a copy.
"""
site_positions(uc::UnitCell{D}) where {D} = copy(uc.site_positions)

"""
    site_kinds(uc::UnitCell) -> Vector{Symbol}

The site kinds (`:localized` | `:plane_wave`), indexed by site id. Returns a copy.

A `:localized` site is a comb of Wannier-like orbitals: its position is physical, its couplings
are given as bonds, and k-kernels are refused on it (`p` on a comb is meaningless). A
`:plane_wave` site holds smooth basis functions labelled by `G`: its position is gauge-only, and
any kernel applies. Kinds are classified at construction (any non-zero `G` or momentum origin on
a site makes it `:plane_wave`) unless declared explicitly, as `fold` does.
"""
site_kinds(uc::UnitCell)::Vector{Symbol} = copy(uc.site_kind)

"""
    site_periods(uc::UnitCell) -> Vector{SMatrix{D,D,Int}}

The per-site period matrices `M_s` (`a = M_s · a_s`), indexed by site id. Returns a copy.
`M_s = I` for an ordinary orbital; `fold(cell, M)` records `M_s = M` on localized sites, whose
admissible `G` labels are then one representative per coset of `M_s ℤᴰ`.
"""
site_periods(uc::UnitCell{D}) where {D} = copy(uc.site_period)

"""
    is_folded(uc::UnitCell) -> Bool

Whether any localized site carries a non-trivial period lattice (`|det M_s| > 1`), i.e. the cell
is a `fold` of a finer one.
"""
is_folded(uc::UnitCell)::Bool = any(s -> uc.site_kind[s] === :localized &&
                                         abs(det(uc.site_period[s])) != 1,
                                    eachindex(uc.site_kind))

"""
    site_of(uc::UnitCell, i) -> Int

The site id of basis state `i`.
"""
@inline site_of(uc::UnitCell, i::Integer)::Int = uc.state_site[i]

"""
    site_id(uc::UnitCell, name::Symbol) -> Int

The id of the site called `name`.
"""
function site_id(uc::UnitCell, name::Symbol)::Int
    s = findfirst(==(name), uc.site_names)
    isnothing(s) && throw(ArgumentError(
        "The cell declares no site `$name`; declared sites are $(uc.site_names)."
    ))
    return s
end

"""
    states_on_site(uc::UnitCell, name::Symbol) -> Vector{Int}
    states_on_site(uc::UnitCell, s::Integer) -> Vector{Int}

The basis-state indices sitting on a site, by name or by id, sorted ascending.

Notes:
------
Returns a copy, matching `site_names` and `site_positions`. Handing out the cell's own vector would
let a caller `push!` into it and corrupt the state/site correspondence.
"""
states_on_site(uc::UnitCell, name::Symbol)::Vector{Int} = copy(uc.site_states[site_id(uc, name)])

function states_on_site(uc::UnitCell, s::Integer)::Vector{Int}
    1 <= s <= n_sites(uc) || throw(ArgumentError(
        "Site id $s is out of range; the cell declares $(n_sites(uc)) sites $(uc.site_names)."
    ))
    return copy(uc.site_states[s])
end

"""
    ndim(uc::UnitCell) -> Int
"""
@inline ndim(uc::UnitCell{D}) where {D} = D

"""
    display_name(uc::UnitCell, i) -> String

A human-readable label for basis state `i`: its site name, its internal levels, and its plane-wave
label when non-zero -- for example `"A, up, G=(1, 0)"`.

Notes:
------
Synthesized from the cell's registries rather than stored per state. That is the point of holding
names once per kind: the label is always consistent with the structure, and it costs nothing until
it is asked for.
"""
function display_name(uc::UnitCell{D}, i::Integer)::String where {D}
    bs = uc.basis[i]
    parts = String[string(uc.site_names[uc.state_site[i]])]
    for a in 1:n_axes(uc.schema)
        push!(parts, string(level_at(uc.schema, bs.internal, a)))
    end
    iszero(bs.G) || push!(parts, "G=$(Tuple(bs.G))")
    return join(parts, ", ")
end

function display_name(uc::UnitCell{D}, bs::BasisState{D})::String where {D}
    ### `something(x, throw(y))` does NOT work here: Julia evaluates arguments eagerly, so the
    ### `throw` fires on every call regardless of whether the lookup succeeded. Resolve, then branch.
    ### A direct scan, not `state_index(cell_tables(uc), ...)`: building the full lookup tables to
    ### resolve a single state -- and discarding them immediately -- costs more than the O(n) walk,
    ### and this is a display helper. `_same_site` folds the [0,1) boundary, like the site machinery.
    i = findfirst(b -> b.internal == bs.internal && b.G == bs.G &&
                       _same_site(b.r_frac, bs.r_frac, uc.atol, nperiodic(uc.lattice)),
                  uc.basis)
    isnothing(i) && throw(ArgumentError(
        "That basis state is not in this cell."
    ))
    return display_name(uc, i)
end

"""
    display_names(uc::UnitCell) -> Vector{String}

`display_name` for every basis state, in basis order.
"""
display_names(uc::UnitCell{D}) where {D} = [display_name(uc, i) for i in eachindex(uc.basis)]

function Base.show(io::IO, ::MIME"text/plain", uc::UnitCell{D}) where {D}
    println(io, "UnitCell{$D} with ", n_states(uc), " states on ", n_sites(uc), " sites")
    for s in 1:n_sites(uc)
        println(io, "  site ", uc.site_names[s], " at ",
                    Tuple(round.(uc.site_positions[s], digits=DISPLAY_DIGITS)),
                    "  (", length(uc.site_states[s]), " states)")
    end
    if is_trivial(uc.schema)
        print(io, "  internal: trivial")
    else
        print(io, "  internal: ",
              join(("$(a) => $(levels(uc.schema, a))" for a in axis_names(uc.schema)), ", "))
    end
end

### the compact one-line form, so a cell nested inside another display stays one line
Base.show(io::IO, uc::UnitCell{D}) where {D} =
    print(io, "UnitCell{", D, "}(", n_states(uc), " states on ", n_sites(uc), " sites)")




#####################################################################
###### Generative constructors                                  #####
#####################################################################

"""
    STATE_ORDER

The basis ordering every generative path obeys: **the new dimension varies slowest.**

Adding a dimension of size `L` to a cell of `n` states puts the parent basis in the inner loop, so

    new_index(parent_index, k) = (k - 1) * n + parent_index        for k in 1:L

Two consequences worth relying on:

- **Parent indices are preserved.** The `k = 1` block *is* the parent basis, in order, so state `i`
  of the parent is state `i` of the child. `add_axis` additionally arranges the interned internal
  indices to match (`internal_new = internal_old + (k-1) * n_internal_old`).
- **The paths compose.** Building a cell in one call and building it up incrementally give the same
  basis in the same order:

      UnitCell(lat, sites...; schema=sch, G=Gs)
        == add_plane_waves(UnitCell(lat, sites...; schema=sch), Gs)

This is load-bearing rather than cosmetic: a compiled `CouplingList` stores raw state indices, so two
cells that differ only in basis order are not interchangeable, and a model compiled against one is
silently wrong against the other.

Obeyed by: the generative `UnitCell` constructor, [`add_axis`](@ref), [`add_plane_waves`](@ref),
`supercell` and `open_boundary`. `stack` follows the same discipline with B as the "new
dimension": the A block keeps its indices verbatim and state `m` of B becomes `n_states(A) + m`.
"""
const STATE_ORDER = nothing

"""
    add_axis(cell::UnitCell{D}, key::Symbol, levels) -> UnitCell{D}

Extend the cell's internal schema with a new axis, replicating the basis over its levels at the
same positions.

Args:
-----
- `key`: the new axis name, e.g. `:spin`
- `levels`: its levels, e.g. `[:up, :down]`

Returns:
--------
- `UnitCell{D}` with `n_states` multiplied by the number of levels and `n_sites` unchanged.

Notes:
------
The site registry is untouched: adding spin, valley, layer or Nambu changes *what lives on a site*,
never *where the sites are*. That invariance is the payoff of defining sites by position alone.

Because `add_axis` appends, the new axis has `stride = n_internal(old)` and states are emitted
level-outer, so `internal_new = internal_old + (l-1) * n_internal(old)`: every internal index of
the parent cell keeps its meaning in the child.

This is the basis half of adding a degree of freedom. Nambu doubling additionally requires the
Hamiltonian half -- conjugating the hole block -- which is not a basis operation; it is not built
yet (see the design note in `transforms/nambu_doubling.jl`).
"""
function add_axis(cell::UnitCell{D}, key::Symbol, lvls::AbstractVector)::UnitCell{D} where {D}
    sch = add_axis(cell.schema, key, lvls)
    nold = n_internal(cell.schema)
    L = n_levels(sch, n_axes(sch))

    basis = Vector{BasisState{D}}(undef, L * length(cell.basis))
    n = 0
    for l in 1:L, b in cell.basis
        basis[n += 1] = BasisState{D}(b.r_frac, b.G, b.k0, b.internal + (l - 1) * nold, sch;
                                      nperiodic=nperiodic(cell.lattice))
    end
    ### kinds and periods ride along: adding spin, valley, layer or Nambu changes what lives on a
    ### site, never what KIND of basis function it is. The state -> site mapping rides along
    ### too (level outermost, per STATE_ORDER), so name-identified coincident :plane_wave
    ### sites survive the transform without positional re-resolution.
    return UnitCell{D}(cell.lattice, sch, cell.site_names, cell.site_positions, basis;
                       atol=cell.atol, site_kind=cell.site_kind, site_period=cell.site_period,
                       state_site=repeat(cell.state_site, L))
end

"""
    add_plane_waves(cell::UnitCell{D}, Gs) -> UnitCell{D}

Replicate the basis over a set of plane-wave labels, at the same positions.

Args:
-----
- `Gs`: integer Miller-index vectors

Returns:
--------
- `UnitCell{D}` with `n_states` multiplied by `length(Gs)` and `n_sites` unchanged.

Notes:
------
The counterpart of [`add_axis`](@ref) for the plane-wave label. `G` is deliberately *not* a schema
axis: every quantum-number rule compiles to an `n_internal × n_internal` matrix, so putting a
several-hundred-element plane-wave shell in the schema would make those matrices enormous. A moiré
cell has `n_internal = 4` and hundreds of states.

Existing labels are added to, not replaced: the cell's current `G` on each state is offset by each
entry of `Gs`. Starting from a cell whose states all carry `G = 0`, that is just `Gs` itself.
"""
function add_plane_waves(cell::UnitCell{D}, Gs::AbstractVector)::UnitCell{D} where {D}
    isempty(Gs) && throw(ArgumentError("`Gs` must contain at least one plane-wave label."))
    ### On a FOLDED cell the admissible labels are exactly one representative per coset of the
    ### period lattice; extending them freely would break that contract. G-space extension of a
    ### localized basis is `fold`, which generates the cosets itself.
    is_folded(cell) && throw(ArgumentError(
        "`add_plane_waves` cannot extend a folded cell: its G labels are the cosets of the " *
        "period lattice, generated by `fold`. Fold with a larger matrix instead."))
    gs = SVector{D,Int}[canonicalize_int(g) for g in Gs]
    allunique(gs) || throw(ArgumentError("`Gs` contains repeated plane-wave labels."))

    basis = Vector{BasisState{D}}(undef, length(gs) * length(cell.basis))
    n = 0
    for g in gs, b in cell.basis
        basis[n += 1] = BasisState{D}(b.r_frac, b.G + g, b.k0, b.internal, cell.schema;
                                      nperiodic=nperiodic(cell.lattice))
    end
    ### kinds are re-derived: a site that now carries G ≠ 0 states reads as :plane_wave
    return UnitCell{D}(cell.lattice, cell.schema, cell.site_names, cell.site_positions, basis;
                       atol=cell.atol)
end

"""
    delete_sites(cell::UnitCell{D}, names) -> UnitCell{D}

A new cell with the named sites removed -- registry entries, kinds AND every basis state that
sat on them. `names` is a `Symbol` or a collection of `Symbol`s. The surviving sites and states
keep their relative order (indices are renumbered, of course: state indices are positions in
`cell.basis`).

This is the vacancy/defect entry point the `fold` docstring promises: build the supercell,
delete the vacancy site, recompile:

    sc  = supercell(cell, [3 0; 0 3])          # sites named :A_1..:A_9, :B_1..:B_9
    vac = delete_sites(sc, :A_5)               # a vacancy
    m   = compile(vac, [hopping(-1.0)])

Note that `spectral_weight` cannot unfold THIS model -- its state count no longer matches
`|det M|` copies of the parent, and the projection refuses by name. For an unfolding-compatible
spelling of the same defect, keep the site and pin it out of the spectrum with a large onsite
energy (`onsite(1e4; sites=[:A_5])` -- agrees with the deleted-site spectrum to O(t²/U)); see
`examples/hamiltonianconstruction/04_disorder_and_defects.jl` for the worked comparison.

Cell-level, deliberately:
-------------------------
`delete_sites` acts on the CELL only -- there is no model method. Recompile your terms on the
smaller cell, as above. A coupling-level deletion (dropping the compiled entries that touch the
deleted states) would silently discard real couplings: every bond into the vacancy simply
vanishes with no accounting, and amplitudes that were derived jointly (a callback that saw the
full geometry, a shell that renumbers without the site) would be silently stale. Recompiling
asks every term the question again on the geometry that actually remains.

Refusals, stated rather than fudged:
------------------------------------
- an unknown site name (the cell's declared sites are listed);
- deleting ALL sites (an empty cell has no states to model);
- folded cells (`is_folded`): a folded site is one coset representative of the fine lattice, so
  deleting it would remove EVERY fine-lattice image at once -- almost never the intended
  vacancy -- and the folded neighbour machinery assumes the coset structure is complete. Delete
  the site in the FINE cell and `fold` again instead.
"""
function delete_sites(cell::UnitCell{D}, names)::UnitCell{D} where {D}
    victims = names isa Symbol ? [names] : collect(Symbol, names)
    for s in victims
        s in cell.site_names || throw(ArgumentError(
            "`delete_sites` names the site `:$s`, but the cell declares no such site (its " *
            "sites are $(cell.site_names))."))
    end
    is_folded(cell) && throw(ArgumentError(
        "`delete_sites` does not support folded cells: a folded site stands for one coset " *
        "representative per fine-lattice cell, so deleting it would remove every fine-lattice " *
        "image at once, and the fold's neighbour bookkeeping assumes the coset structure is " *
        "complete. Delete the site in the fine (parent) cell and `fold` again."))
    gone = Set(victims)
    keep = [s for s in eachindex(cell.site_names) if !(cell.site_names[s] in gone)]
    isempty(keep) && throw(ArgumentError(
        "`delete_sites` would remove ALL $(n_sites(cell)) sites of the cell; an empty cell " *
        "has no basis left to model. Keep at least one site."))

    newid = zeros(Int, n_sites(cell))                 # old site id -> new site id (0 = deleted)
    for (new, old) in pairs(keep)
        newid[old] = new
    end
    kept_states = [i for i in eachindex(cell.basis) if newid[cell.state_site[i]] != 0]
    ### the explicit state -> site mapping is passed through, so name-identified coincident
    ### :plane_wave sites survive without positional re-resolution -- same policy as add_axis
    return UnitCell{D}(cell.lattice, cell.schema,
                       cell.site_names[keep], cell.site_positions[keep],
                       cell.basis[kept_states];
                       atol=cell.atol,
                       site_kind=cell.site_kind[keep],
                       site_period=cell.site_period[keep],
                       state_site=[newid[cell.state_site[i]] for i in kept_states])
end




### For fast repeated state lookup use `cell_tables(cell)` + `state_index`: exact on the integer
### labels `(internal, G)` and tolerant on the position. A `Dict{BasisState,Int}` is deliberately
### not offered -- see the keying doctrine in `celltables.jl`.
