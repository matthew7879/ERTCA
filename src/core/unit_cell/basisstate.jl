##################################################################
###### export only the following functions from this module ######
##################################################################

export BasisState, plane_wave_state, display_name




#####################################################################
###### Define the basis-state structure and minimal checks     ######
#####################################################################

"""
    BasisState{D}: A Bloch-periodic basis state.

Physical representation:
------------------------
A `BasisState` is decomposed into three pieces:
    1. A (Wannier-like) localized component `ϕ(r - R)` centered at fractional position
       `R = r_frac` within the unit cell. The localized function must be unit-cell periodic:
       `ϕ(r) = ϕ(r + a_j)` for any Bravais lattice vector `a_j`.
    2. A plane-wave component labeled by the integer Miller indices `G`, namely `e^{i (G⋅b) ⋅ r}`
       where `b` are the reciprocal lattice vectors.
    3. An internal state -- orbital character, spin, valley, layer, Nambu -- carried as a single
       interned index into an [`InternalSchema`](@ref).

The total is the Bloch-periodic wavefunction
    ψ(r) = ϕ(r - R) * e^{i (G⋅b) ⋅ (r - R)} * |internal⟩
where (ϕ, |internal⟩) are encoded in abstract manner through the "internal state" integer.

Notes:
------
1) For tight-binding models the localized components are usually treated as delta functions, so only
their centers matter, and the plane-wave component is zero (except in supercells).
For continuum / plane-wave models the localized component is the constant function and the
plane-wave component carries all the information.

Keeping the two separate is not redundant in practice: a localized function expanded in plane waves
needs very high Miller indices, so separating them is what makes truncation tractable.

2) This type deliberately carries **no** name and no per-state dictionary. A name describes a *kind*,
not an instance -- there are two sublattices but perhaps hundreds of plane-wave states -- so names
live once in cell-level registries:

| handle  | names                                  | lives in                       |
|---------|----------------------------------------|--------------------------------|
| site    | a position (`:A`, `:M`)                | the `UnitCell` site registry   |
| orbital | an envelope character (`:pz`, `:dxy`)  | an `InternalSchema` axis       |
| qnum    | spin / valley / layer / Nambu          | an `InternalSchema` axis       |

Use `display_name(cell, i)` for a human-readable label; it is synthesized from that structure.

Fields:
-------
- `r_frac::SVector{D,Float64}`
    Position in *fractional* coordinates of the real-space Bravais lattice `a`, canonicalized into
    `[0,1)^D`.

- `G::SVector{D,Int}`
    Integer Miller indices of the plane-wave component. The reciprocal vector is these integers
    times the reciprocal lattice vectors `b`.

- `k0::SVector{D,Float64}`
    The state's **momentum origin** `K_s`, in fractional reciprocal coordinates; zero for every
    tight-binding state. The physical momentum of the state is `bᵀ(k + G - k0)` -- a valley, a
    layer's Dirac point, a k·p expansion point are all per-state origins, not per-term ones.
    Cell builders accept origins in cartesian coordinates and store them fractional.

- `internal::Int`
    The interned internal state, in `1:n_internal(schema)`. Decode it with `level_at` or `decode`.

- `schema::InternalSchema`
    The declared internal label space. Shared -- every state of a `UnitCell` holds the *same*
    object, so `==` between states of one cell is a pointer compare. Carrying it here is what keeps
    a `BasisState` a self-contained value: `==` and `hash` stay well defined, and two states built
    against separately-constructed but structurally equal schemas compare equal.
"""
struct BasisState{D}
    r_frac::SVector{D,Float64}
    G::SVector{D,Int}
    k0::SVector{D,Float64}
    internal::Int
    schema::InternalSchema

    function BasisState{D}(r_frac::SVector{D,Float64}, G::SVector{D,Int},
                           k0::SVector{D,Float64}, internal::Int, schema::InternalSchema;
                           digits::Int=LATTICE_ROUND_DIGITS, nperiodic::Int=D) where {D}
        ### Out-of-domain positions are wrapped by `canonicalize_fracpos` below. That is the
        ### documented canonical behavior -- `BasisState([1.5, -0.25])` is a valid way to write
        ### `[0.5, 0.75]` -- so no warning fires here. On a partially periodic lattice pass its
        ### `nperiodic` so the trailing (aperiodic) components stay unwrapped -- the cell
        ### constructors do this for you.
        validate_internal(schema, internal)
        all(isfinite, k0) || throw(DomainError(k0, "The momentum origin k0 must be finite."))
        return new{D}(canonicalize_fracpos(r_frac; digits=digits, nperiodic=nperiodic),
                      G, k0, internal, schema)
    end
end

### Positional back-compat form: a state with no declared momentum origin sits at k0 = 0, which
### is every tight-binding state and every plane wave without a per-level expansion point.
BasisState{D}(r_frac::SVector{D,Float64}, G::SVector{D,Int}, internal::Int,
              schema::InternalSchema; kwargs...) where {D} =
    BasisState{D}(r_frac, G, zero(SVector{D,Float64}), internal, schema; kwargs...)


# ----- Main constructor -----
"""
    BasisState(r_frac, G; schema=TRIVIAL_SCHEMA, internal=nothing, digits=..., axis_levels...)

Construct a basis state.

Args:
-----
- `r_frac`: fractional position, length `D`
- `G`: integer Miller indices, length `D`
- `schema`: the internal label space; defaults to [`TRIVIAL_SCHEMA`](@ref)
- `internal`: an already-interned index, as an alternative to naming the levels
- any remaining keywords are read as **axis levels**, e.g. `orbital=:pz, spin=+1`

Returns:
--------
- `BasisState{D}`

Notes:
------
`schema`, `internal`, `G` and `digits` are consumed by their own keyword slots; everything else is
an axis. A mistyped axis name therefore raises, listing the declared axes, instead of silently
producing a state that no term will ever match.

Every declared axis must be given. A partial specification is an error rather than being defaulted:
a state that omits `:spin` would otherwise be indistinguishable from one that declares its default.
"""
function BasisState(r_frac::AbstractVector{<:Real},
                    G::AbstractVector{<:Integer};
                    schema::InternalSchema=TRIVIAL_SCHEMA,
                    internal::Union{Nothing,Integer}=nothing,
                    k0::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    digits::Int=LATTICE_ROUND_DIGITS,
                    axis_levels...)

    # --- check dims
    D = length(r_frac)
    length(G) == D || throw(DimensionMismatch(
        "r_frac and G must have same dimension D=$D, got $(length(r_frac)) and $(length(G))"
    ))
    k0v = isnothing(k0) ? zeros(Float64, D) : k0
    length(k0v) == D || throw(DimensionMismatch(
        "k0 must have dimension D=$D, got $(length(k0v)). It is a fractional reciprocal vector."
    ))

    # --- resolve the internal index
    idx = _resolve_internal(schema, internal, axis_levels)

    # --- canonicalize and hand to the strict constructor
    # `D` comes from a runtime length, so converting to `SVector{D}` here would be a type-unstable
    # construction and every downstream call would box. Going through `Val(D)` costs one dynamic
    # dispatch and makes everything after it concrete.
    return _build_basisstate(Val(D), r_frac, G, k0v, idx, schema, digits)
end

@inline function _build_basisstate(::Val{D}, r_frac::AbstractVector{<:Real},
                                   G::AbstractVector{<:Integer},
                                   k0::AbstractVector{<:Real}, idx::Int,
                                   schema::InternalSchema, digits::Int)::BasisState{D} where {D}
    return BasisState{D}(SVector{D,Float64}(r_frac), SVector{D,Int}(G),
                         SVector{D,Float64}(k0), idx, schema; digits=digits)
end

"""
    _resolve_internal(schema, internal, axis_levels) -> Int

Turn either an explicit interned index or a set of named levels into a validated index.
"""
function _resolve_internal(schema::InternalSchema, internal::Union{Nothing,Integer},
                           axis_levels)::Int
    if !isnothing(internal)
        isempty(axis_levels) || throw(ArgumentError(
            "Give either `internal` (an already-interned index) or the axis levels " *
            "($(collect(keys(axis_levels)))), not both."
        ))
        return validate_internal(schema, internal)
    end
    ### `axis_levels` is a `Base.Pairs`, which already iterates as `key => value`; converting it to
    ### a `NamedTuple` first would allocate for nothing.
    isempty(axis_levels) || return internal_index(schema, axis_levels)
    is_trivial(schema) || throw(ArgumentError(
        "The schema declares axes $(axis_names(schema)) but no internal state was given. " *
        "Pass the levels as keywords (e.g. $(axis_names(schema)[1])=...) or `internal=`."
    ))
    return 1
end

# ----- Convenience constructors -----
"""
    BasisState(r_frac; G=zeros(Int,D), kwargs...)

Position-only form; the plane-wave label defaults to zero.
"""
function BasisState(r_frac::AbstractVector{<:Real};
                    G::AbstractVector{<:Integer}=zeros(Int, length(r_frac)), kwargs...)
    return BasisState(r_frac, G; kwargs...)
end

"""
    plane_wave_state(G; r_frac=zeros(D), kwargs...)

A state labeled by its plane wave, sitting at the origin of the cell unless told otherwise.

Notes:
------
This replaces the former `BasisState(G::AbstractVector{<:Integer})` method, which was a trap:
`<:Integer` is more specific than `<:Real`, so `BasisState([0,0])` -- an integer-valued *position*
-- silently dispatched to it and became a plane-wave label instead.
"""
function plane_wave_state(G::AbstractVector{<:Integer};
                          r_frac::AbstractVector{<:Real}=zeros(Float64, length(G)), kwargs...)
    return BasisState(r_frac, G; kwargs...)
end




#####################################################################
###### Equality and hashing                                     #####
#####################################################################

"""
    ==(b1::BasisState{D}, b2::BasisState{D})

Equality on everything that identifies the state: position, plane wave, internal index and the
schema that gives that index meaning.

Notes:
------
The schema is compared **structurally**, not by identity, so two states built in different cells
against equivalent schemas compare equal. The check cannot be dropped: internal index `3` means
different things under different schemas, and comparing the bare integers would match states that
are not the same.

Comparisons run cheapest-and-most-discriminating first, and the schema comes last because states of
one cell share the object and hit the `===` fast path.
"""
function Base.:(==)(b1::BasisState{D}, b2::BasisState{D})::Bool where {D}
    b1.internal == b2.internal || return false
    b1.G == b2.G || return false
    b1.r_frac == b2.r_frac || return false
    b1.k0 == b2.k0 || return false
    return b1.schema == b2.schema
end

function Base.:(!=)(b1::BasisState{D}, b2::BasisState{D})::Bool where {D}
    return !(b1 == b2)
end

"""
    hash(b::BasisState{D}, h::UInt)

Hash consistent with `==`.

Notes:
------
`InternalSchema` caches its own hash, so this is O(1) regardless of how many axes are declared.
"""
function Base.hash(b::BasisState{D}, h::UInt)::UInt where {D}
    h = hash(D, h)
    h = hash(b.r_frac, h)
    h = hash(b.G, h)
    h = hash(b.k0, h)
    h = hash(b.internal, h)
    return hash(b.schema, h)
end




#####################################################################
###### Display                                                  #####
#####################################################################

"""
    display_name(bs::BasisState{D}) -> String

A human-readable label for a standalone state: its decoded internal levels, then its position and
plane wave.

Notes:
------
Only a `UnitCell` knows site *names*, so this bare form prints the position instead. Prefer
`display_name(cell, i)`, which resolves the site name from the cell's registry.
"""
function display_name(bs::BasisState{D})::String where {D}
    parts = String[]
    for a in 1:n_axes(bs.schema)
        push!(parts, string(level_at(bs.schema, bs.internal, a)))
    end
    push!(parts, "pos:$(Tuple(round.(bs.r_frac, digits=DISPLAY_DIGITS)))")
    iszero(bs.G) || push!(parts, "G:$(Tuple(bs.G))")
    iszero(bs.k0) || push!(parts, "k0:$(Tuple(round.(bs.k0, digits=DISPLAY_DIGITS)))")
    return join(parts, ", ")
end

function Base.show(io::IO, bs::BasisState{D}) where {D}
    print(io, "BasisState{$D}(", display_name(bs), ")")
end




#####################################################################
###### Lookup helpers built on equality and hashing             #####
#####################################################################

"""
    find_basisstate(bs, vec) -> Union{Nothing,Int}

Return the first index where `bs` appears in `vec`, or `nothing` if absent.
Uses `==` on `BasisState`. This is linear in the length of the vector.
"""
function find_basisstate(bs::BasisState, vec::AbstractVector{<:BasisState})::Union{Nothing,Int}
    return findfirst(isequal(bs), vec)
end

"""
    has_basisstate(bs, vec) -> Bool

Whether `bs` is present in `vec`. Linear in the length of the vector.
"""
has_basisstate(bs::BasisState, vec::AbstractVector{<:BasisState})::Bool =
    !isnothing(find_basisstate(bs, vec))

### There is deliberately NO Dict{BasisState,Int} index-map helper. `BasisState.hash` folds in
### `r_frac::SVector{D,Float64}`, and rounding is not a stable equivalence relation: a key derived
### by arithmetic (a ΔG lookup, a symmetry image, a transported position) can land in a different
### bucket than the state it denotes. Exact keying belongs to `CellTables` (integer `G` and
### interned `internal`); tolerant resolution to `state_index`. The linear helpers below use `==`
### on values the caller already holds, which is safe.

"""
    check_unrepeated(vec) -> Bool

Whether `vec` contains no repeated basis states.

Notes:
------
Uses a pre-sized `Set` and returns at the first repeat, rather than materializing the full set of
distinct states before comparing lengths -- a duplicate basis is exactly what a caller is checking
for, so the early exit is the common path when the answer is `false`.
"""
function check_unrepeated(vec::AbstractVector{T})::Bool where {T<:BasisState}
    seen = Set{T}()
    sizehint!(seen, length(vec))
    for bs in vec
        bs in seen && return false
        push!(seen, bs)
    end
    return true
end

