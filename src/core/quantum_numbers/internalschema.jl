##################################################################
###### export only the following functions from this module ######
##################################################################

export QLevel, InternalSchema, TRIVIAL_SCHEMA
export n_internal, n_axes, axis_names, axis_position, has_axis
export levels, n_levels, level_ordinal, level_at, ordinal_at
export internal_index, decode, validate_internal, is_trivial
export add_axis




#####################################################################
###### The declared internal label space                        #####
#####################################################################

"""
    QLevel = Union{Symbol,Int}

A single level of an internal axis.

`Symbol` when the name is the only meaningful label (`:dxy`, `:up`), `Int` when the value is used
arithmetically (`:valley => [+1,-1]`, where `τ` multiplies a momentum offset). The choice is made
per axis, not globally.

Notes:
------
A two-member `Union` is split by the compiler, so comparisons cost no dispatch and no allocation.
Levels are only touched when a state is constructed and when a rule is compiled to a matrix; after
interning, everything downstream is a plain `Int`.
"""
const QLevel = Union{Symbol,Int}

"""
    InternalSchema

The declared space of internal (non-spatial) labels shared by every basis state of a unit cell.

An internal state is one level from each axis. Those tuples are interned to a single `Int` in
`1:n_internal`, which is what a `BasisState` stores.

Physical meaning:
-----------------
An axis is any label that distinguishes states sitting at the same place: the orbital character of
the envelope function (`:orbital => [:dxy,:dyz]`), spin, valley, layer, Nambu. Position is *not* an
axis -- that is the site registry on `UnitCell` -- and neither is the plane-wave label `G`.

Fields:
-------
- `axes::Vector{Symbol}`
    Axis names, in declaration order.

- `levels::Vector{Vector{QLevel}}`
    `levels[a]` are the levels of axis `a`, in declaration order.
    **The ordering is semantic**: `:spin => [+1,-1]` and `:spin => [-1,+1]` are different schemas,
    because the interned index depends on it.

- `strides::Vector{Int}`
    Mixed-radix strides, `strides[a] = prod(length.(levels[1:a-1]))`. Axis 1 varies fastest.

- `ninternal::Int`
    `prod(length.(levels))`, and `1` when there are no axes.

- `hashval::UInt`
    Cached structural hash. `Base.hash` returns it directly, which is what keeps
    `hash(::BasisState)` O(1) -- the previous `Dict{Symbol,Int}` field hashed a dictionary per
    state.

Notes:
------
Interning uses mixed-radix arithmetic rather than a stored decode table, for a property worth
having deliberately: `add_axis` appends, so `strides[new] = n_internal(old)` and therefore

    internal_new = internal_old + (l - 1) * n_internal(old)

Every internal index of the parent schema stays valid in the extended one -- the `l = 1` block is
the parent verbatim. With a stored table that would be an accident rather than an identity.

There are no `Dict` fields. Axis and level lookups are linear scans, which beat hashing for the
handful of axes and levels a model declares (the crossover is around 30 entries), and they keep two
heap objects out of `==` and `hash`.

!!! warning
    An `InternalSchema` must never be mutated. `axes`, `levels` and `strides` are `Vector`s, so
    immutability is a convention rather than a guarantee, and mutating one would silently desync
    `hashval`. Making them `NTuple`s would push their length into the type parameter and cascade
    into `BasisState{D,N}`, exactly as `BravaisLattice` documents for `SMatrix{D,D,Float64,L}`.
    This matters most for [`TRIVIAL_SCHEMA`](@ref), which is shared globally.
"""
struct InternalSchema
    axes::Vector{Symbol}
    levels::Vector{Vector{QLevel}}
    strides::Vector{Int}
    ninternal::Int
    hashval::UInt

    function InternalSchema(axes::Vector{Symbol}, levels::Vector{Vector{QLevel}};
                            max_internal::Int=MAX_INTERNAL)
        length(axes) == length(levels) || throw(DimensionMismatch(
            "Got $(length(axes)) axis names but $(length(levels)) level lists."
        ))
        allunique(axes) || throw(ArgumentError(
            "Axis names must be unique; got $axes."
        ))
        ### Enforced at construction, not at first use: `internal_index` packs one ordinal per
        ### axis into a fixed-width integer, so a schema with more axes could be built but never
        ### interned by level names -- better to refuse it here.
        length(axes) <= 64 || throw(ArgumentError(
            "A schema may declare at most 64 axes; got $(length(axes))."
        ))
        for (a, ls) in pairs(levels)
            isempty(ls) && throw(ArgumentError(
                "Axis `$(axes[a])` declares no levels, which would make `n_internal` zero. " *
                "An axis with a single level is the way to declare a label that never varies."
            ))
            allunique(ls) || throw(ArgumentError(
                "Levels of axis `$(axes[a])` must be unique; got $ls."
            ))
        end

        ### mixed-radix strides, axis 1 fastest-varying
        strides = Vector{Int}(undef, length(axes))
        n = 1
        for a in eachindex(axes)
            strides[a] = n
            n *= length(levels[a])
        end
        n <= max_internal || throw(ArgumentError(
            "Schema declares n_internal = $n, above `max_internal = $max_internal`. " *
            "Every quantum-number rule compiles to an n_internal × n_internal matrix, so this " *
            "would need $(16 * n * n) bytes per rule. A bound this large usually means an index " *
            "landed in the schema that belongs elsewhere: plane waves belong in `G`, positions in " *
            "the site registry."
        ))

        ### cached structural hash; `Base.hash` just returns it
        h = hash(:InternalSchema)
        for a in eachindex(axes)
            h = hash(axes[a], h)
            for l in levels[a]
                h = hash(l, h)
            end
        end
        ### Copied, not aliased -- the same defensive policy as the `UnitCell` constructor: a
        ### caller holding the vectors could otherwise mutate them and silently desync the cached
        ### `strides`/`ninternal`/`hashval`.
        return new(copy(axes), [copy(l) for l in levels], strides, n, h)
    end
end


# ----- Main constructor -----
"""
    InternalSchema(pairs; max_internal=MAX_INTERNAL)

Build a schema from `axis => levels` pairs.

Args:
-----
- `pairs`: a vector of `Symbol => AbstractVector` pairs, in the order the axes should be declared

Returns:
--------
- `InternalSchema`
"""
function InternalSchema(axis_pairs::AbstractVector{<:Pair};
                        max_internal::Int=MAX_INTERNAL)::InternalSchema
    ### `<:Pair` rather than `<:Pair{Symbol,<:AbstractVector}`: axes whose levels have different
    ### element types (`[:top,:bot]` beside `[+1,-1]`) widen the collected eltype to `Pair{Symbol}`,
    ### which the tighter signature would not match. Validate the halves here instead.
    axes = Symbol[]
    lvls = Vector{QLevel}[]
    for p in axis_pairs
        p.first isa Symbol || throw(ArgumentError(
            "Axis names must be `Symbol`; got $(typeof(p.first)) ($(repr(p.first)))."
        ))
        p.second isa Union{AbstractVector,Tuple} || throw(ArgumentError(
            "Levels of axis `$(p.first)` must be given as a vector or tuple; got $(typeof(p.second))."
        ))
        push!(axes, p.first)
        push!(lvls, QLevel[_as_level(v) for v in p.second])
    end
    return InternalSchema(axes, lvls; max_internal=max_internal)
end

# ----- Convenience constructors -----
"""
    InternalSchema(:orbital => [:dxy,:dyz], :spin => [+1,-1])
    InternalSchema()

Varargs form, and the no-axis form used for a trivial internal space.
"""
InternalSchema(axis_pairs::Pair...; kwargs...)::InternalSchema =
    InternalSchema(collect(Pair, axis_pairs); kwargs...)

InternalSchema(; kwargs...)::InternalSchema =
    InternalSchema(Symbol[], Vector{QLevel}[]; kwargs...)

"""
    TRIVIAL_SCHEMA

The schema with no axes: `n_internal == 1`, every state has `internal == 1`.

Shared globally, so a model with no internal structure allocates no schema at all.
"""
const TRIVIAL_SCHEMA = InternalSchema()


"""
    _as_level(v) -> QLevel

Normalize a user-supplied level, rejecting anything that is neither a name nor an integer.
"""
_as_level(v::Symbol)::QLevel = v
_as_level(v::Integer)::QLevel = Int(v)
_as_level(v) = throw(ArgumentError(
    "Levels must be `Symbol` or `Int`; got $(typeof(v)) ($(repr(v))). Use a `Symbol` for a named " *
    "character (`:dxy`, `:up`) and an `Int` where the value is used arithmetically (`+1`)."
))




#####################################################################
###### Accessors                                                #####
#####################################################################

"""
    n_internal(schema) -> Int

The number of distinct internal states, `prod` of the axis sizes. `1` for a trivial schema.
"""
@inline function n_internal(s::InternalSchema)::Int
    return s.ninternal
end

"""
    n_axes(schema) -> Int

The number of declared axes.
"""
@inline function n_axes(s::InternalSchema)::Int
    return length(s.axes)
end

"""
    is_trivial(schema) -> Bool

Whether the schema declares no axes, so every state has `internal == 1`.
"""
@inline function is_trivial(s::InternalSchema)::Bool
    return isempty(s.axes)
end

"""
    axis_names(schema) -> Vector{Symbol}

The declared axis names, in order. Returns a copy; the schema must not be mutated.
"""
axis_names(s::InternalSchema)::Vector{Symbol} = copy(s.axes)

"""
    axis_position(schema, key) -> Int

The position of axis `key`, or `0` when the schema has no such axis.
"""
@inline function axis_position(s::InternalSchema, key::Symbol)::Int
    a = findfirst(==(key), s.axes)
    return isnothing(a) ? 0 : a
end

"""
    has_axis(schema, key) -> Bool
"""
@inline function has_axis(s::InternalSchema, key::Symbol)::Bool
    return axis_position(s, key) != 0
end

"""
    levels(schema, a) -> Vector{QLevel}
    levels(schema, key::Symbol) -> Vector{QLevel}

The levels of an axis, by position or by name. Returns a copy.
"""
levels(s::InternalSchema, a::Integer)::Vector{QLevel} = copy(s.levels[a])
levels(s::InternalSchema, key::Symbol)::Vector{QLevel} = copy(s.levels[_require_axis(s, key)])

"""
    n_levels(schema, a) -> Int
    n_levels(schema, key::Symbol) -> Int
"""
@inline n_levels(s::InternalSchema, a::Integer)::Int = length(s.levels[a])
n_levels(s::InternalSchema, key::Symbol)::Int = length(s.levels[_require_axis(s, key)])

"""
    level_ordinal(schema, a, v) -> Int

The position of level `v` within axis `a`, or `0` when that axis has no such level.
"""
@inline function level_ordinal(s::InternalSchema, a::Integer, v::QLevel)::Int
    o = findfirst(==(v), s.levels[a])
    return isnothing(o) ? 0 : o
end

"""
    _require_axis(schema, key) -> Int

Resolve an axis name, raising an informative error rather than returning a sentinel.
"""
function _require_axis(s::InternalSchema, key::Symbol)::Int
    a = axis_position(s, key)
    a == 0 && throw(ArgumentError(
        "Schema declares no axis `$key`; declared axes are $(s.axes)."
    ))
    return a
end




#####################################################################
###### Interning and decoding                                   #####
#####################################################################

"""
    ordinal_at(schema, internal, a) -> Int

The level ordinal of axis `a` within internal state `internal`.

Notes:
------
Pure integer arithmetic on the mixed-radix representation -- no table lookup, no allocation. This
is the primitive the quantum-number rules use when compiling themselves to a matrix.
"""
@inline function ordinal_at(s::InternalSchema, internal::Int, a::Integer)::Int
    return ((internal - 1) ÷ s.strides[a]) % length(s.levels[a]) + 1
end

"""
    level_at(schema, internal, a) -> QLevel
    level_at(schema, internal, key::Symbol) -> QLevel

The level that internal state `internal` carries on an axis.
"""
@inline function level_at(s::InternalSchema, internal::Int, a::Integer)::QLevel
    return @inbounds s.levels[a][ordinal_at(s, internal, a)]
end
level_at(s::InternalSchema, internal::Int, key::Symbol)::QLevel =
    level_at(s, internal, _require_axis(s, key))

"""
    decode(schema, internal) -> Vector{Pair{Symbol,QLevel}}

The full per-axis decomposition of an internal state, in axis order.

Notes:
------
Allocates. Intended for display, metadata and round-trip tests -- inside a loop use
`ordinal_at` or `level_at`, which do not.
"""
function decode(s::InternalSchema, internal::Integer)::Vector{Pair{Symbol,QLevel}}
    i = validate_internal(s, internal)
    return [s.axes[a] => level_at(s, i, a) for a in eachindex(s.axes)]
end

"""
    internal_index(schema, values) -> Int

Intern a set of per-axis levels into a single index.

Args:
-----
- `values`: anything `pairs` accepts -- a `NamedTuple`, a `Dict{Symbol,<:Any}`, or an iterable of
  `Symbol => level` pairs. **Every** axis must be given.

Returns:
--------
- `Int` in `1:n_internal(schema)`

Notes:
------
A partial specification is an error rather than being defaulted. Silently filling a missing axis is
exactly the failure mode this schema replaces: previously a state that simply omitted `:spin` was
indistinguishable from one that declared the default value, and a mistyped axis name produced a
state carrying a quantum number nothing would ever match.
"""
function internal_index(s::InternalSchema, values)::Int
    n_axes(s) <= 64 || throw(ArgumentError(
        "Schemas with more than 64 axes are not supported (this one has $(n_axes(s)))."
    ))
    ### The index is accumulated directly and the "which axes were given" bookkeeping is a bitmask,
    ### so the success path allocates nothing. Collecting ordinals into a scratch vector -- or a
    ### list of names purely to phrase a possible error -- costs a heap allocation on every state
    ### constructed, which is exactly what this schema exists to remove.
    idx = 1
    seen = zero(UInt64)
    nseen = 0
    for (key, v) in _axis_pairs(values)
        a = _require_axis(s, Symbol(key))
        bit = UInt64(1) << (a - 1)
        iszero(seen & bit) || throw(ArgumentError(
            "Axis `$(s.axes[a])` was given more than once."
        ))
        o = level_ordinal(s, a, _as_level(v))
        o == 0 && throw(ArgumentError(
            "Axis `$(s.axes[a])` declares no level $(repr(_as_level(v))); " *
            "its levels are $(s.levels[a])."
        ))
        seen |= bit
        nseen += 1
        idx += (o - 1) * s.strides[a]
    end
    nseen == n_axes(s) || throw(ArgumentError(
        "Incomplete internal state: axes " *
        "$([s.axes[a] for a in eachindex(s.axes) if iszero(seen & (UInt64(1) << (a - 1)))]) " *
        "were not given. Every declared axis must be specified."
    ))
    return idx
end

internal_index(s::InternalSchema, ::Nothing)::Int = internal_index(s, ())

"""
    _axis_pairs(values)

Normalize the accepted spellings of "per-axis levels" into an iterator of `key => level`.

Notes:
------
`pairs` is right for a `NamedTuple` or a `Dict`, but wrong for a vector of pairs: `pairs(v)` on an
array yields `index => element`, so `decode`'s own output would come back as `1 => (:spin => 1)`.
An iterable of pairs is therefore iterated directly.
"""
_axis_pairs(v::NamedTuple) = pairs(v)
_axis_pairs(v::AbstractDict) = pairs(v)
_axis_pairs(v) = v

"""
    _internal_from_ordinals(schema, ords) -> Int

Fold per-axis ordinals into the mixed-radix index.
"""
@inline function _internal_from_ordinals(s::InternalSchema, ords)::Int
    idx = 1
    @inbounds for a in eachindex(s.axes)
        idx += (ords[a] - 1) * s.strides[a]
    end
    return idx
end

"""
    validate_internal(schema, internal) -> Int

Check that `internal` is a legal index for `schema`, and return it as an `Int`.
"""
@inline function validate_internal(s::InternalSchema, internal::Integer)::Int
    1 <= internal <= s.ninternal || throw(ArgumentError(
        "Internal index $internal is out of range for a schema with n_internal = $(s.ninternal)."
    ))
    return Int(internal)
end




#####################################################################
###### Extension                                                #####
#####################################################################

"""
    add_axis(schema, key, levels) -> InternalSchema

Append an axis, returning a new schema.

Notes:
------
Appending (rather than inserting) is what preserves interned indices: the new axis gets
`stride = n_internal(old)`, so an internal state of the parent schema keeps its index in the
extended one, sitting in the block where the new axis takes its first level.

See also the `UnitCell` method, which additionally replicates the basis over the new levels.
"""
function add_axis(s::InternalSchema, key::Symbol, lvls::AbstractVector)::InternalSchema
    has_axis(s, key) && throw(ArgumentError(
        "Schema already declares an axis `$key` with levels $(s.levels[axis_position(s, key)])."
    ))
    axes = vcat(s.axes, key)
    ls = vcat(s.levels, [QLevel[_as_level(v) for v in lvls]])
    return InternalSchema(axes, ls)
end




#####################################################################
###### Equality, hashing and display                            #####
#####################################################################

"""
    ==(s1::InternalSchema, s2::InternalSchema)

Structural equality: the same axes, in the same order, with the same levels in the same order.

Notes:
------
Structural rather than identity-based, because it is what lets a `BasisState` stay a self-contained
value. Two states built against separately-constructed but equivalent schemas must compare equal --
and, just as importantly, internal index `3` under two *different* schemas must not.

The `===` fast path is the common case, since `UnitCell` conforms every state onto one schema
object. The cached hash then rejects unequal schemas in a single word before any vector compare.
"""
function Base.:(==)(s1::InternalSchema, s2::InternalSchema)::Bool
    s1 === s2 && return true
    s1.hashval == s2.hashval || return false
    s1.axes == s2.axes || return false
    return s1.levels == s2.levels
end

Base.:(!=)(s1::InternalSchema, s2::InternalSchema)::Bool = !(s1 == s2)

Base.hash(s::InternalSchema, h::UInt)::UInt = hash(s.hashval, h)

function Base.show(io::IO, ::MIME"text/plain", s::InternalSchema)
    if is_trivial(s)
        print(io, "InternalSchema (trivial: no axes, n_internal = 1)")
        return
    end
    println(io, "InternalSchema with n_internal = ", s.ninternal)
    for a in eachindex(s.axes)
        println(io, "  ", s.axes[a], " => ", s.levels[a],
                    "   (stride ", s.strides[a], ")")
    end
    print(io, "  axis 1 varies fastest")
end

function Base.show(io::IO, s::InternalSchema)
    if is_trivial(s)
        print(io, "InternalSchema()")
    else
        print(io, "InternalSchema(",
              join(("$(s.axes[a]) => $(s.levels[a])" for a in eachindex(s.axes)), ", "), ")")
    end
end
