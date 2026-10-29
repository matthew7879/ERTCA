# Paper-specific geometry routines.

const QLevel = Union{Symbol,Int}

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
        h = hash(:InternalSchema)
        for a in eachindex(axes)
            h = hash(axes[a], h)
            for l in levels[a]
                h = hash(l, h)
            end
        end
        return new(copy(axes), [copy(l) for l in levels], strides, n, h)
    end
end

function InternalSchema(axis_pairs::AbstractVector{<:Pair};
                        max_internal::Int=MAX_INTERNAL)::InternalSchema
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

InternalSchema(axis_pairs::Pair...; kwargs...)::InternalSchema =
    InternalSchema(collect(Pair, axis_pairs); kwargs...)

InternalSchema(; kwargs...)::InternalSchema =
    InternalSchema(Symbol[], Vector{QLevel}[]; kwargs...)

const TRIVIAL_SCHEMA = InternalSchema()

_as_level(v::Symbol)::QLevel = v

_as_level(v::Integer)::QLevel = Int(v)

_as_level(v) = throw(ArgumentError(
    "Levels must be `Symbol` or `Int`; got $(typeof(v)) ($(repr(v))). Use a `Symbol` for a named " *
    "character (`:dxy`, `:up`) and an `Int` where the value is used arithmetically (`+1`)."
))

@inline function n_internal(s::InternalSchema)::Int
    return s.ninternal
end

@inline function n_axes(s::InternalSchema)::Int
    return length(s.axes)
end

@inline function is_trivial(s::InternalSchema)::Bool
    return isempty(s.axes)
end

axis_names(s::InternalSchema)::Vector{Symbol} = copy(s.axes)

@inline function axis_position(s::InternalSchema, key::Symbol)::Int
    a = findfirst(==(key), s.axes)
    return isnothing(a) ? 0 : a
end

levels(s::InternalSchema, a::Integer)::Vector{QLevel} = copy(s.levels[a])

levels(s::InternalSchema, key::Symbol)::Vector{QLevel} = copy(s.levels[_require_axis(s, key)])

@inline n_levels(s::InternalSchema, a::Integer)::Int = length(s.levels[a])

n_levels(s::InternalSchema, key::Symbol)::Int = length(s.levels[_require_axis(s, key)])

@inline function level_ordinal(s::InternalSchema, a::Integer, v::QLevel)::Int
    o = findfirst(==(v), s.levels[a])
    return isnothing(o) ? 0 : o
end

function _require_axis(s::InternalSchema, key::Symbol)::Int
    a = axis_position(s, key)
    a == 0 && throw(ArgumentError(
        "Schema declares no axis `$key`; declared axes are $(s.axes)."
    ))
    return a
end

@inline function ordinal_at(s::InternalSchema, internal::Int, a::Integer)::Int
    return ((internal - 1) ÷ s.strides[a]) % length(s.levels[a]) + 1
end

@inline function level_at(s::InternalSchema, internal::Int, a::Integer)::QLevel
    return @inbounds s.levels[a][ordinal_at(s, internal, a)]
end

level_at(s::InternalSchema, internal::Int, key::Symbol)::QLevel =
    level_at(s, internal, _require_axis(s, key))

function internal_index(s::InternalSchema, values)::Int
    n_axes(s) <= 64 || throw(ArgumentError(
        "Schemas with more than 64 axes are not supported (this one has $(n_axes(s)))."
    ))
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

_axis_pairs(v::NamedTuple) = pairs(v)

_axis_pairs(v::AbstractDict) = pairs(v)

_axis_pairs(v) = v

@inline function validate_internal(s::InternalSchema, internal::Integer)::Int
    1 <= internal <= s.ninternal || throw(ArgumentError(
        "Internal index $internal is out of range for a schema with n_internal = $(s.ninternal)."
    ))
    return Int(internal)
end

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

abstract type KPeriodicity end

struct ExactlyPeriodic <: KPeriodicity end

struct TruncatedWindow <: KPeriodicity end

struct NoZone <: KPeriodicity end

Base.show(io::IO, ::ExactlyPeriodic) = print(io, "ExactlyPeriodic()")

Base.show(io::IO, ::TruncatedWindow) = print(io, "TruncatedWindow()")

Base.show(io::IO, ::NoZone) = print(io, "NoZone()")

function _derive_kperiodicity(site_kind::Vector{Symbol}, site_states::Vector{Vector{Int}},
                              basis::AbstractVector)::KPeriodicity
    out = ExactlyPeriodic()
    for s in eachindex(site_kind)
        site_kind[s] === :plane_wave || continue
        gset = unique(basis[i].G for i in site_states[s])
        if length(gset) <= 1
            return NoZone()                    # NoZone dominates
        else
            out = TruncatedWindow()
        end
    end
    return out
end

_require_zone(::KPeriodicity, ::AbstractString, ::AbstractString) = nothing

_require_zone(::NoZone, what::AbstractString, fix::AbstractString) = throw(ArgumentError(
    "$what, and this model has no Brillouin-zone torus: its k-periodicity is `NoZone()` -- a " *
    "k·p frame, whose lattice is a coordinate frame rather than a zone, so a fractional mesh " *
    "over it samples nothing physical. $fix"))

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

@inline _matrix_to_rows(m::AbstractMatrix, ::Val{D}) where {D} =
    ntuple(i -> SVector{D,Float64}(ntuple(j -> Float64(m[i, j]), D)), D)

@inline _rows_to_matrix(rows::NTuple{D,SVector{D,Float64}}) where {D} =
    SMatrix{D,D,Float64,D*D}(ntuple(k -> rows[1 + (k - 1) % D][1 + (k - 1) ÷ D], D * D))

@inline function Base.getproperty(lattice::BravaisLattice{D}, name::Symbol) where {D}
    name === :a && return _rows_to_matrix(getfield(lattice, :arows))
    name === :b && return _rows_to_matrix(getfield(lattice, :brows))
    return getfield(lattice, name)
end

Base.propertynames(::BravaisLattice) = (:a, :b, :arows, :brows, :nperiodic)

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

function BravaisLattice(lattice_vectors::AbstractVector{<:AbstractVector{<:Real}};
                        kwargs...)::BravaisLattice
    A = transpose(hcat(lattice_vectors...))
    return BravaisLattice(A; kwargs...)
end

function ndim(lattice::BravaisLattice{D})::Int where {D}
    return D
end

@inline nperiodic(lattice::BravaisLattice)::Int = getfield(lattice, :nperiodic)

function primitive_recip_vectors(lattice::BravaisLattice{D})::Vector{SVector{D,Float64}} where {D}
    return [lattice.b[i,:] for i in 1:D]
end

@doc (@doc primitive_vectors) primitive_recip_vectors

@inline function primitive_vector(lattice::BravaisLattice{D}, i::Integer)::SVector{D,Float64} where {D}
    1 <= i <= D || throw(ArgumentError("Primitive-vector index $i is out of range 1:$D."))
    return getfield(lattice, :arows)[i]
end

@inline function primitive_recip_vector(lattice::BravaisLattice{D}, i::Integer)::SVector{D,Float64} where {D}
    1 <= i <= D || throw(ArgumentError("Primitive-vector index $i is out of range 1:$D."))
    return getfield(lattice, :brows)[i]
end

@doc (@doc primitive_vector) primitive_recip_vector

@inline recip_matrix(lattice::BravaisLattice{D}) where {D} =
    SMatrix{D,D,Float64}(transpose(lattice.b))

@doc (@doc direct_matrix) recip_matrix

@inline metric(lattice::BravaisLattice{D}) where {D} =
    SMatrix{D,D,Float64}(lattice.a * transpose(lattice.a))

@inline function cell_volume(lattice::BravaisLattice{D})::Float64 where {D}
    return abs(det(lattice.a))
end

@inline function bz_volume(lattice::BravaisLattice{D})::Float64 where {D}
    return (2π)^nperiodic(lattice) / cell_volume(lattice)
end

@inline function length_scale(lattice::BravaisLattice{D})::Float64 where {D}
    P = nperiodic(lattice)
    P == 0 && return 1.0
    return cell_volume(lattice)^(1 / P)
end

@inline function distance_atol(lattice::BravaisLattice{D};
                               atol::Float64=LATTICE_ATOL)::Float64 where {D}
    return atol * length_scale(lattice)
end

reciprocal(lattice::BravaisLattice{D}) where {D} =
    BravaisLattice{D}(lattice.b, lattice.a; force=true, nperiodic=nperiodic(lattice))

function same_basis(l1::BravaisLattice{D1}, l2::BravaisLattice{D2};
                    atol::Float64=LATTICE_ATOL)::Bool where {D1,D2}
    D1 == D2 || return false
    return isapprox(l1.a, l2.a; atol=atol * length_scale(l1))
end

function lattice_parameters(lattice::BravaisLattice{D}; degrees::Bool=true) where {D}
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

function Base.:(==)(bl1::BravaisLattice{D1}, bl2::BravaisLattice{D2})::Bool where {D1, D2}
    if D1 != D2 || nperiodic(bl1) != nperiodic(bl2)
        return false
    end
    # Compute transformation matrix: M = inv(a₁) · a₂
    M = bl1.a * transpose(bl2.b) / (2π) # M = a₁ · inv(a₂) = a₁ · b₂.T / (2π)
    tol = INT_ROUND_ATOL * max(1.0, maximum(abs, M))
    if !isapprox(M, round.(M); atol=tol)
        return false
    end
    return abs(det(round.(Int, M))) == 1
end

function Base.:(!=)(bl1::BravaisLattice, bl2::BravaisLattice)::Bool
    return !(bl1 == bl2)
end

Base.hash(l::BravaisLattice{D}, h::UInt) where {D} =
    hash(:BravaisLattice, hash(D, hash(nperiodic(l), h)))

function fractional_to_real(r_frac::AbstractVector{<:Real},
                            lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(r_frac) == D || throw(DimensionMismatch(
        "Input dimension $(length(r_frac)) does not match lattice dimension $D"
    ))
    return transpose(lattice.a) * SVector{D,Float64}(r_frac)
end

function fractional_to_reciprocal(k_frac::AbstractVector{<:Real},
                                  lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(k_frac) == D || throw(DimensionMismatch(
        "Input dimension $(length(k_frac)) does not match lattice dimension $D"
    ))
    return transpose(lattice.b) * SVector{D,Float64}(k_frac)
end

function real_to_fractional(r_real::AbstractVector{<:Real},
                            lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(r_real) == D || throw(DimensionMismatch(
        "Input dimension $(length(r_real)) does not match lattice dimension $D"
    ))
    return (lattice.b * SVector{D,Float64}(r_real)) ./ (2π)
end

function reciprocal_to_fractional(k_cart::AbstractVector{<:Real},
                                  lattice::BravaisLattice{D})::SVector{D,Float64} where {D}
    length(k_cart) == D || throw(DimensionMismatch(
        "Input dimension $(length(k_cart)) does not match lattice dimension $D"
    ))
    return (lattice.a * SVector{D,Float64}(k_cart)) ./ (2π)
end

struct BasisState{D}
    r_frac::SVector{D,Float64}
    G::SVector{D,Int}
    k0::SVector{D,Float64}
    internal::Int
    schema::InternalSchema

    function BasisState{D}(r_frac::SVector{D,Float64}, G::SVector{D,Int},
                           k0::SVector{D,Float64}, internal::Int, schema::InternalSchema;
                           digits::Int=LATTICE_ROUND_DIGITS, nperiodic::Int=D) where {D}
        validate_internal(schema, internal)
        all(isfinite, k0) || throw(DomainError(k0, "The momentum origin k0 must be finite."))
        return new{D}(canonicalize_fracpos(r_frac; digits=digits, nperiodic=nperiodic),
                      G, k0, internal, schema)
    end
end

BasisState{D}(r_frac::SVector{D,Float64}, G::SVector{D,Int}, internal::Int,
              schema::InternalSchema; kwargs...) where {D} =
    BasisState{D}(r_frac, G, zero(SVector{D,Float64}), internal, schema; kwargs...)

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

function _resolve_internal(schema::InternalSchema, internal::Union{Nothing,Integer},
                           axis_levels)::Int
    if !isnothing(internal)
        isempty(axis_levels) || throw(ArgumentError(
            "Give either `internal` (an already-interned index) or the axis levels " *
            "($(collect(keys(axis_levels)))), not both."
        ))
        return validate_internal(schema, internal)
    end
    isempty(axis_levels) || return internal_index(schema, axis_levels)
    is_trivial(schema) || throw(ArgumentError(
        "The schema declares axes $(axis_names(schema)) but no internal state was given. " *
        "Pass the levels as keywords (e.g. $(axis_names(schema)[1])=...) or `internal=`."
    ))
    return 1
end

function BasisState(r_frac::AbstractVector{<:Real};
                    G::AbstractVector{<:Integer}=zeros(Int, length(r_frac)), kwargs...)
    return BasisState(r_frac, G; kwargs...)
end

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

function Base.hash(b::BasisState{D}, h::UInt)::UInt where {D}
    h = hash(D, h)
    h = hash(b.r_frac, h)
    h = hash(b.G, h)
    h = hash(b.k0, h)
    h = hash(b.internal, h)
    return hash(b.schema, h)
end

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
    names = copy(site_names)
    positions = copy(site_positions)
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
    conformed = Vector{BasisState{D}}(undef, length(basis))
    for (i, b) in pairs(basis)
        b.schema == schema || throw(ArgumentError(
            "Basis state $i carries the schema $(b.schema), but the cell declares " *
            "$(schema). All states of a cell must share one schema."
        ))
        conformed[i] = b.schema === schema ? b :
            BasisState{D}(b.r_frac, b.G, b.k0, b.internal, schema; nperiodic=P)
    end
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
    stsite = if isnothing(state_site)
        _assign_sites(conformed, positions, declared_kinds; atol=atol, nperiodic=P)
    else
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
    kinds = if isnothing(declared_kinds)
        derived = [any(i -> !iszero(conformed[i].G) || !iszero(conformed[i].k0), site_states[s]) ?
                       :plane_wave : :localized for s in eachindex(positions)]
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
    if warn_empty_sites
        empty_sites = [names[s] for s in eachindex(site_states) if isempty(site_states[s])]
        isempty(empty_sites) || @warn "Declared site(s) $empty_sites carry no basis states. " *
            "They still cost work in the neighbour search, which runs over site pairs."
    end

    return (lattice, schema, names, positions, conformed, stsite, site_states, kinds,
            periods, atol)
end

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
    UnitCell{D,K}(lattice::BravaisLattice{D}, schema::InternalSchema,
                  site_names::AbstractVector, site_positions::AbstractVector,
                  basis::AbstractVector, state_site::AbstractVector,
                  site_states::AbstractVector, site_kind::AbstractVector,
                  site_period::AbstractVector,
                  atol::Real) where {D,K<:KPeriodicity} =
        new{D,K}(lattice, schema, site_names, site_positions, basis, state_site, site_states,
                 site_kind, site_period, atol)
end

kperiodicity(::UnitCell{D,K}) where {D,K} = K()

_kpar(::UnitCell{D,K}) where {D,K} = K

_derive_kperiodicity(cell::UnitCell)::KPeriodicity =
    _derive_kperiodicity(cell.site_kind, cell.site_states, cell.basis)

function UnitCell{D}(lattice::BravaisLattice{D}, schema::InternalSchema,
                     site_names::Vector{Symbol}, site_positions::Vector{SVector{D,Float64}},
                     basis::Vector{BasisState{D}}; kwargs...) where {D}
    f = _unitcell_fields(Val(D), lattice, schema, site_names, site_positions, basis; kwargs...)
    K = typeof(_derive_kperiodicity(f[8], f[7], f[5]))
    return UnitCell{D,K}(f...)
end

function UnitCell{D,K}(lattice::BravaisLattice{D}, schema::InternalSchema,
                       site_names::Vector{Symbol}, site_positions::Vector{SVector{D,Float64}},
                       basis::Vector{BasisState{D}}; kwargs...) where {D,K<:KPeriodicity}
    f = _unitcell_fields(Val(D), lattice, schema, site_names, site_positions, basis; kwargs...)
    return UnitCell{D,K}(f...)
end

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
    return UnitCell{D}(lattice, schema, names, pos, basis;
                       atol=atol, site_kind=site_kind, state_site=ssite)
end

function UnitCell(lattice::BravaisLattice{D}, sites::AbstractVector{<:Pair};
                  kwargs...)::UnitCell{D} where {D}
    return UnitCell(lattice, sites...; kwargs...)
end

function UnitCell(lattice::BravaisLattice{D}, basis::Vector{BasisState{D}};
                  kwargs...)::UnitCell{D} where {D}
    sch, names, pos, rest = _cell_from_basis(lattice, basis; kwargs...)
    return UnitCell{D}(lattice, sch, names, pos, basis; rest...)
end

function UnitCell{D,K}(lattice::BravaisLattice{D}, basis::Vector{BasisState{D}};
                       kwargs...)::UnitCell{D,K} where {D,K<:KPeriodicity}
    sch, names, pos, rest = _cell_from_basis(lattice, basis; kwargs...)
    return UnitCell{D,K}(lattice, sch, names, pos, basis; rest...)
end

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

function UnitCell(lattice::BravaisLattice{D}, basis::BasisState{D}...; kwargs...) where {D}
    return UnitCell(lattice, collect(basis); kwargs...)
end

function UnitCell(lattice::BravaisLattice{D})::UnitCell{D} where {D}
    return UnitCell(lattice, BasisState{D}[])
end

function _internal_allowed(schema::InternalSchema, internal::Int,
                           restriction::Union{Nothing,AbstractDict})::Bool
    isnothing(restriction) && return true
    for (ax, allowed) in restriction
        a = _restriction_axis(schema, ax)
        level_at(schema, internal, a) in allowed || return false
    end
    return true
end

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

function _restriction_axis(schema::InternalSchema, ax)::Int
    a = axis_position(schema, Symbol(ax))
    a == 0 && throw(ArgumentError(
        "Level restriction names axis `$ax`, which the schema does not declare; " *
        "declared axes are $(axis_names(schema))."
    ))
    return a
end

_auto_site_name(i::Integer)::Symbol = Symbol(AUTO_SITE_PREFIX, i)

@inline function _frac_delta(u::SVector{D,Float64}, v::SVector{D,Float64},
                             nperiodic::Int=D)::SVector{D,Float64} where {D}
    d = u .- v
    nperiodic == D && return d .- round.(d)
    return SVector{D,Float64}(ntuple(m -> m <= nperiodic ? d[m] - round(d[m]) : d[m], D))
end

@inline function _same_site(u::SVector{D,Float64}, v::SVector{D,Float64}, atol::Float64,
                            nperiodic::Int=D)::Bool where {D}
    return all(abs.(_frac_delta(u, v, nperiodic)) .<= atol)
end

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

function _assign_sites(states, site_tau::Vector{SVector{D,Float64}};
                       atol::Float64=LATTICE_ATOL, nperiodic::Int=D)::Vector{Int} where {D}
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
                "Basis state $i at $(Tuple(t)) matches $nhits declared sites."
        ))
        out[i] = hit
    end
    return out
end

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

@inline _position_of(bs::BasisState{D}) where {D} = bs.r_frac

@inline _position_of(p::SVector{D,Float64}) where {D} = p

function Base.:(==)(uc1::UnitCell{D1}, uc2::UnitCell{D2})::Bool where {D1, D2}
    D1 == D2 || return false
    uc1.lattice == uc2.lattice || return false
    uc1.schema == uc2.schema || return false
    length(uc1.basis) == length(uc2.basis) || return false
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
        rounded_G = round.(raw_G)
        isapprox(raw_G, rounded_G; atol=INT_ROUND_ATOL) || return false
        new_G = SVector{D2,Int}(rounded_G)

        # the momentum origin transports exactly like a (fractional) momentum
        real_k0 = fractional_to_reciprocal(b.k0, uc1.lattice)
        new_k0 = reciprocal_to_fractional(real_k0, uc2.lattice)
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

Base.hash(uc::UnitCell{D}, h::UInt) where {D} =
    hash(:UnitCell, hash(D, hash(length(uc.basis), hash(uc.schema, h))))

@inline n_states(uc::UnitCell{D}) where {D} = length(uc.basis)

state_positions(uc::UnitCell{D}) where {D} = [bs.r_frac for bs in uc.basis]

state_g_vectors(uc::UnitCell{D}) where {D} = [bs.G for bs in uc.basis]

state_origins(uc::UnitCell{D}) where {D} = [bs.k0 for bs in uc.basis]

state_internals(uc::UnitCell{D}) where {D} = [bs.internal for bs in uc.basis]

@inline n_sites(uc::UnitCell) = length(uc.site_positions)

site_names(uc::UnitCell)::Vector{Symbol} = copy(uc.site_names)

site_positions(uc::UnitCell{D}) where {D} = copy(uc.site_positions)

site_kinds(uc::UnitCell)::Vector{Symbol} = copy(uc.site_kind)

is_folded(uc::UnitCell)::Bool = any(s -> uc.site_kind[s] === :localized &&
                                         abs(det(uc.site_period[s])) != 1,
                                    eachindex(uc.site_kind))

@inline ndim(uc::UnitCell{D}) where {D} = D

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
    i = findfirst(b -> b.internal == bs.internal && b.G == bs.G &&
                       _same_site(b.r_frac, bs.r_frac, uc.atol, nperiodic(uc.lattice)),
                  uc.basis)
    isnothing(i) && throw(ArgumentError(
        "That basis state is not in this cell."
    ))
    return display_name(uc, i)
end

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

Base.show(io::IO, uc::UnitCell{D}) where {D} =
    print(io, "UnitCell{", D, "}(", n_states(uc), " states on ", n_sites(uc), " sites)")

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
                         cell.state_site, cell.site_states,
                         cell.site_positions, cell.site_names,
                         cell.site_kind, cell.site_period,
                         cell.atol, nperiodic(cell.lattice))
end

const EMPTY_INDICES = Int[]

@inline n_states(t::CellTables)::Int = t.nstates

@inline n_sites(t::CellTables)::Int = length(t.site_states)

@inline function states_with_G(t::CellTables{D}, G::SVector{D,Int})::Vector{Int} where {D}
    return get(t.by_G, G, EMPTY_INDICES)
end

states_with_G(t::CellTables{D}, G::AbstractVector{<:Integer}) where {D} =
    states_with_G(t, SVector{D,Int}(G))

@inline function states_on_site_with_G(t::CellTables{D}, site::Integer,
                                       G::SVector{D,Int})::Vector{Int} where {D}
    return get(t.by_site_G, (Int(site), G), EMPTY_INDICES)
end

abstract type AbstractQnumRule end

function qnum_factor end

function _rule_axis(s::InternalSchema, key::Symbol, rule::AbstractQnumRule)::Int
    a = axis_position(s, key)
    a == 0 && throw(ArgumentError(
        "$(nameof(typeof(rule))) acts on the internal axis `$key`, which this cell does not " *
        "declare (its axes are $(axis_names(s))). The rule would reject every pair and the term " *
        "would silently emit nothing."
    ))
    return a
end

function _check_rule_levels(s::InternalSchema, a::Int, wanted, rule::AbstractQnumRule)::Nothing
    for v in wanted
        level_ordinal(s, a, v) == 0 && throw(ArgumentError(
            "$(nameof(typeof(rule))) on axis `$(s.axes[a])` names the level $(repr(v)), which the " *
            "axis does not declare (its levels are $(s.levels[a])). No pair would match, so the " *
            "term would silently emit nothing."
        ))
    end
    return nothing
end

struct AnyQnum <: AbstractQnumRule end

@inline qnum_factor(::AnyQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64 = one(ComplexF64)

struct ConserveQnum <: AbstractQnumRule
    keys::Vector{Symbol}

    function ConserveQnum(keys::Vector{Symbol})
        isempty(keys) && throw(ArgumentError("`ConserveQnum` requires at least one key; use `AnyQnum()` for no constraint."))
        return new(sort(unique(keys)))
    end
end

function qnum_factor(rule::ConserveQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    for k in rule.keys
        a = _rule_axis(s, k, rule)
        ordinal_at(s, i, a) == ordinal_at(s, j, a) || return zero(ComplexF64)
    end
    return one(ComplexF64)
end

struct DiagonalQnum <: AbstractQnumRule
    key::Symbol
    values::Dict{QLevel,ComplexF64}

    function DiagonalQnum(key::Symbol, values::AbstractDict)
        isempty(values) && throw(ArgumentError("`DiagonalQnum` for key `$key` has no values; it would reject every pair."))
        return new(key, Dict{QLevel,ComplexF64}(_as_level(k) => ComplexF64(v) for (k, v) in values))
    end
end

function qnum_factor(rule::DiagonalQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    oi = ordinal_at(s, i, a)
    oi == ordinal_at(s, j, a) || return zero(ComplexF64)
    return get(rule.values, s.levels[a][oi], zero(ComplexF64))
end

struct OffDiagonalQnum <: AbstractQnumRule
    key::Symbol
    pairs::Dict{Tuple{QLevel,QLevel},ComplexF64}

    function OffDiagonalQnum(key::Symbol, pairs::AbstractDict)
        isempty(pairs) && throw(ArgumentError("`OffDiagonalQnum` for key `$key` has no pairs; it would reject every pair."))
        return new(key, Dict{Tuple{QLevel,QLevel},ComplexF64}((_as_level(k[1]), _as_level(k[2])) => ComplexF64(v) for (k, v) in pairs))
    end
end

function qnum_factor(rule::OffDiagonalQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    key = (s.levels[a][ordinal_at(s, i, a)], s.levels[a][ordinal_at(s, j, a)])
    return get(rule.pairs, key, zero(ComplexF64))
end

struct MatrixQnum <: AbstractQnumRule
    key::Symbol
    M::Matrix{ComplexF64}

    function MatrixQnum(key::Symbol, M::AbstractMatrix{<:Number})
        size(M, 1) == size(M, 2) || throw(DimensionMismatch(
            "`MatrixQnum` for axis `$key` needs a square matrix; got $(size(M))."))
        all(isfinite, M) || throw(DomainError(M,
            "`MatrixQnum` for axis `$key` contains a non-finite entry."))
        return new(key, Matrix{ComplexF64}(M))
    end
end

function qnum_factor(rule::MatrixQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    return rule.M[ordinal_at(s, i, a), ordinal_at(s, j, a)]
end

struct TwoLevelMatrixQnum <: AbstractQnumRule
    key::Symbol
    M::Matrix{ComplexF64}
    up::Any                                 # nothing = resolve against the schema at compile
    down::Any

    function TwoLevelMatrixQnum(key::Symbol, M::AbstractMatrix{<:Number}, up, down)
        size(M) == (2, 2) || throw(DimensionMismatch(
            "`TwoLevelMatrixQnum` for axis `$key` needs a 2×2 matrix; got $(size(M))."))
        all(isfinite, M) || throw(DomainError(M,
            "`TwoLevelMatrixQnum` for axis `$key` contains a non-finite entry."))
        isnothing(up) == isnothing(down) || throw(ArgumentError(
            "Give both `up` and `down` levels of axis `$key`, or neither (auto-resolution)."))
        return new(key, Matrix{ComplexF64}(M), up, down)
    end
end

function _two_level_updown(rule::TwoLevelMatrixQnum, s::InternalSchema, a::Int)
    isnothing(rule.up) || return (rule.up, rule.down)
    lv = s.levels[a]
    if (isequal(lv[1], 1) && isequal(lv[2], -1)) || (isequal(lv[1], -1) && isequal(lv[2], 1))
        return (1, -1)
    end
    return (lv[1], lv[2])
end

function qnum_factor(rule::TwoLevelMatrixQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    up, _ = _two_level_updown(rule, s, a)
    lv = s.levels[a]
    pos(o) = isequal(lv[o], up) ? 1 : 2
    return rule.M[pos(ordinal_at(s, i, a)), pos(ordinal_at(s, j, a))]
end

abstract type LevelOrder end

struct DeclaredOrder <: LevelOrder end

struct TwoLevelOrder <: LevelOrder
    up::Any
    down::Any
end

TwoLevelOrder() = TwoLevelOrder(nothing, nothing)

struct NamedOrder <: LevelOrder
    levels::Vector{QLevel}

    function NamedOrder(levels::Vector{QLevel})
        isempty(levels) && throw(ArgumentError(
            "`NamedOrder` needs the level order of the rule's matrix; got an empty list."))
        allunique(levels) || throw(ArgumentError(
            "`NamedOrder` level order $(levels) repeats a level; each row/column of the " *
            "rule's matrix must map to one distinct level."))
        return new(levels)
    end
end

struct DirectionalQnum{L<:LevelOrder} <: AbstractQnumRule
    key::Symbol
    f::Function
    order::L
end

qnum_factor(rule::DirectionalQnum, ::InternalSchema, ::Int, ::Int) = throw(ArgumentError(
    "`directional_rule` on axis `$(rule.key)` has no per-pair factor without a bond " *
    "direction: its matrix is `f(d_cart)`, resolved per site pair when the term compiles. " *
    "Compile the term (or call `qnum_matrix` on its static factors only) instead of " *
    "evaluating the rule directly."))

rule_axes(rule::DirectionalQnum, ::InternalSchema) = (rule.key,)

function validate_rule(rule::DirectionalQnum{DeclaredOrder}, s::InternalSchema)::Nothing
    _rule_axis(s, rule.key, rule)
    return nothing
end

function validate_rule(rule::DirectionalQnum{TwoLevelOrder}, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    o = rule.order
    n_levels(s, a) == 2 || throw(DimensionMismatch(
        "A two-level directional rule on axis `$(rule.key)` needs exactly two levels; " *
        "the axis declares $(n_levels(s, a)) ($(levels(s, a)))."))
    isnothing(o.up) && return nothing
    _check_rule_levels(s, a, (o.up, o.down), rule)
    isequal(o.up, o.down) && throw(ArgumentError(
        "`up` and `down` of axis `$(rule.key)` must be the two DISTINCT levels; both " *
        "are $(o.up)."))
    return nothing
end

function validate_rule(rule::DirectionalQnum{NamedOrder}, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    for v in s.levels[a]
        any(l -> isequal(l, v), rule.order.levels) || throw(ArgumentError(
            "Directional rule on axis `$(rule.key)`: the axis declares the level " *
            "$(repr(v)), which is not one of the rule's labels $(rule.order.levels). The " *
            "rule cannot silently leave a level uncoupled -- declare the axis with these " *
            "labels only, or build the full block with `directional_rule`."))
    end
    return nothing
end

struct ProductQnumRule <: AbstractQnumRule
    rules::Vector{AbstractQnumRule}

    function ProductQnumRule(rules::Vector{<:AbstractQnumRule})
        isempty(rules) && throw(ArgumentError("`ProductQnumRule` requires at least one rule; use `AnyQnum()` for no constraint."))
        return new(collect(AbstractQnumRule, rules))
    end
end

function qnum_factor(rule::ProductQnumRule, s::InternalSchema, i::Int, j::Int)::ComplexF64
    acc = one(ComplexF64)
    for r in rule.rules
        f = qnum_factor(r, s, i, j)
        iszero(f) && return zero(ComplexF64)     # short-circuit: the product is already dead
        acc *= f
    end
    return acc
end

compose(a::AbstractQnumRule, b::AbstractQnumRule) = ProductQnumRule(AbstractQnumRule[a, b])

compose(::AnyQnum, b::AbstractQnumRule) = b

compose(a::AbstractQnumRule, ::AnyQnum) = a

compose(::AnyQnum, ::AnyQnum) = AnyQnum()

compose(a::ProductQnumRule, b::AbstractQnumRule) = ProductQnumRule(vcat(a.rules, AbstractQnumRule[b]))

compose(a::AbstractQnumRule, b::ProductQnumRule) = ProductQnumRule(vcat(AbstractQnumRule[a], b.rules))

compose(a::ProductQnumRule, b::ProductQnumRule) = ProductQnumRule(vcat(a.rules, b.rules))

compose(a::ProductQnumRule, ::AnyQnum) = a

compose(::AnyQnum, b::ProductQnumRule) = b

otimes(a::AbstractQnumRule, b::AbstractQnumRule) = compose(a, b)

const ⊗ = otimes

matrix_rule(key::Symbol, M::AbstractMatrix{<:Number}) = MatrixQnum(key, M)

project(key::Symbol, value::QLevel) = DiagonalQnum(key, Dict(value => one(ComplexF64)))

raise(key::Symbol; from::QLevel=-1, to::QLevel=1) =
    OffDiagonalQnum(key, Dict((to, from) => one(ComplexF64)))

lower(key::Symbol; from::QLevel=1, to::QLevel=-1) =
    OffDiagonalQnum(key, Dict((to, from) => one(ComplexF64)))

@doc (@doc raise) lower

function qnum_matrix(rule::AbstractQnumRule, s::InternalSchema)::Matrix{ComplexF64}
    validate_rule(rule, s)
    n = n_internal(s)
    touched = _touched_axes!(falses(n_axes(s)), rule, s)
    unnamed = findall(!, touched)
    M = Matrix{ComplexF64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        M[i, j] = _diagonal_on(s, i, j, unnamed) ? qnum_factor(rule, s, i, j) :
                                                   zero(ComplexF64)
    end
    return M
end

@inline function _diagonal_on(s::InternalSchema, i::Int, j::Int, axes)::Bool
    for a in axes
        ordinal_at(s, i, a) == ordinal_at(s, j, a) || return false
    end
    return true
end

rule_axes(rule::AbstractQnumRule, ::InternalSchema) = throw(ArgumentError(
    "Define `rule_axes(::$(typeof(rule)), ::InternalSchema)` returning the tuple of internal " *
    "axes the rule names: every axis the rule does not name is completed with a Kronecker " *
    "delta at compile, so the compiler must not guess."))

rule_axes(::AnyQnum, ::InternalSchema) = ()

rule_axes(rule::ConserveQnum, ::InternalSchema) = Tuple(rule.keys)

rule_axes(rule::DiagonalQnum, ::InternalSchema) = (rule.key,)

rule_axes(rule::OffDiagonalQnum, ::InternalSchema) = (rule.key,)

rule_axes(rule::MatrixQnum, ::InternalSchema) = (rule.key,)

rule_axes(rule::TwoLevelMatrixQnum, ::InternalSchema) = (rule.key,)

rule_axes(rule::ProductQnumRule, s::InternalSchema) =
    Tuple(unique(Symbol[k for r in rule.rules for k in rule_axes(r, s)]))

function _touched_axes!(mask::BitVector, rule::AbstractQnumRule, s::InternalSchema)
    for k in rule_axes(rule, s)
        mask[_rule_axis(s, k, rule)] = true
    end
    return mask
end

function validate_rule(rule::AbstractQnumRule, s::InternalSchema)::Nothing
    for k in rule_axes(rule, s)
        _rule_axis(s, k, rule)
    end
    return nothing
end

function validate_rule(rule::DiagonalQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    return _check_rule_levels(s, a, keys(rule.values), rule)
end

function validate_rule(rule::OffDiagonalQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    return _check_rule_levels(s, a, Iterators.flatten(keys(rule.pairs)), rule)
end

function validate_rule(rule::MatrixQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    nl = n_levels(s, a)
    size(rule.M, 1) == nl || throw(DimensionMismatch(
        "`matrix_rule(:$(rule.key), M)` carries a $(size(rule.M, 1))×$(size(rule.M, 2)) matrix, " *
        "but the axis declares $nl levels ($(levels(s, a)))."))
    return nothing
end

function validate_rule(rule::TwoLevelMatrixQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    n_levels(s, a) == 2 || throw(DimensionMismatch(
        "A two-level Pauli block on axis `$(rule.key)` needs exactly two levels; the axis " *
        "declares $(n_levels(s, a)) ($(levels(s, a)))."))
    isnothing(rule.up) && return nothing
    _check_rule_levels(s, a, (rule.up, rule.down), rule)
    isequal(rule.up, rule.down) && throw(ArgumentError(
        "`up` and `down` of axis `$(rule.key)` must be the two DISTINCT levels; both are " *
        "$(rule.up)."))
    return nothing
end

function validate_rule(rule::ProductQnumRule, s::InternalSchema)::Nothing
    for r in rule.rules
        validate_rule(r, s)
    end
    return nothing
end

function _lagrange_reduce(lat::BravaisLattice{2})
    a1 = primitive_vector(lat, 1)
    a2 = primitive_vector(lat, 2)
    U = SMatrix{2,2,Int}(1, 0, 0, 1)
    for _ in 1:64
        if norm(a1) > norm(a2)
            a1, a2 = a2, a1
            U = SMatrix{2,2,Int}(0, 1, 1, 0) * U
        end
        r = dot(a1, a2) / dot(a1, a1)
        m = round(Int, r - 1e-9)
        m == 0 && break
        a2 = a2 - m * a1
        U = SMatrix{2,2,Int}(1, -m, 0, 1) * U
    end
    if norm(a1) > norm(a2)                        # one final swap can be pending
        a1, a2 = a2, a1
        U = SMatrix{2,2,Int}(0, 1, 1, 0) * U
    end
    if dot(a1, a2) < -1e-12 * norm(a1) * norm(a2)  # angle into [60°, 90°]
        a2 = -a2
        U = SMatrix{2,2,Int}(1, 0, 0, -1) * U
    end
    return a1, a2, U
end

function _unimodular_matches(Gu::SMatrix{D,D,Float64}, Gc::SMatrix{D,D,Float64},
                             tol::Float64) where {D}
    out = SMatrix{D,D,Int}[]
    n = D * D
    pow3 = ntuple(j -> 3^(n - j), Val(n))
    for want_det in (1, -1)
        for code in 0:(3^n - 1)
            dig = ntuple(j -> (code ÷ pow3[j]) % 3 - 1, Val(n))
            W = SMatrix{D,D,Int}(ntuple(k -> dig[((k - 1) % D) * D + (k - 1) ÷ D + 1], Val(n)))
            _int_det(W) == want_det || continue
            isapprox(W * Gc * transpose(W), Gu; atol=tol) && push!(out, W)
        end
    end
    return out
end

function _metric_match(Gu::SMatrix{D,D,Float64}, Gc::SMatrix{D,D,Float64},
                       tol::Float64) where {D}
    isapprox(Gc, Gu; atol=tol) && return one(SMatrix{D,D,Int})
    ms = _unimodular_matches(Gu, Gc, tol)
    return isempty(ms) ? nothing : first(ms)
end

@inline _gram_of_rows(A::SMatrix{D,D,Float64}) where {D} = A * transpose(A)

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
        if n < bestn - 1e-12 * max(1.0, bestn)
            bestn = n
            best = x
        end
    end
    return best
end

function embed_into_higher_dimension(lat::BravaisLattice{D}, D_new::Int;
                                     periodic::Bool=false, padding::Real=1.0,
                                     kwargs...) where {D}
    if D_new < D
        throw(ArgumentError("New dimension must be greater than or equal to the current dimension."))
    end
    nperiodic(lat) == D || throw(ArgumentError(
        "The lattice is already partially periodic (nperiodic = $(nperiodic(lat)) < D = $D); " *
        "embedding it again is not supported -- build the embedding in one step."))
    padding > 0 || throw(ArgumentError(
        "`padding` must be a positive length; got $padding."))
    (periodic || padding == 1.0) || throw(ArgumentError(
        "`embed_into_higher_dimension`: `padding = $padding` with `periodic = false`. An " *
        "APERIODIC added row is a coordinate axis, not a translation, and must be a unit " *
        "vector: that contract is what keeps `cell_volume` equal to the true " *
        "$(nperiodic(lat))-volume and transverse fractional coordinates equal to cartesian " *
        "ones. A physical padding LENGTH describes a vacuum layer, which is a periodic " *
        "direction -- pass `periodic=true` alongside it."))
    incD = D_new - D
    newvecs = hcat(zeros(incD, D), diagm(fill(Float64(padding), incD)))
    a_embedded = vcat(hcat(lat.a, zeros(D, incD)), newvecs)
    P = periodic ? D_new : D
    return BravaisLattice(a_embedded; force=true, nperiodic=P, kwargs...)
end

function leading_block(lat::BravaisLattice{D}, P::Int;
                       atol::Float64=distance_atol(lat)) where {D}
    (1 <= P <= D) || throw(ArgumentError("`leading_block`: `P` must lie in 1:$D; got $P."))
    P == D && return lat
    nperiodic(lat) == D || throw(ArgumentError(
        "`leading_block`: this lattice is periodic along only $(nperiodic(lat)) of its $D " *
        "directions, so its trailing axes are padding rather than decoupled translations. " *
        "That is `periodic_block`'s case, not this one."))
    A = lat.a
    for m in (P + 1):D, l in 1:P
        (abs(A[m, l]) <= atol && abs(A[l, m]) <= atol) || throw(ArgumentError(
            "`leading_block`: row $m is not orthogonal to the leading $P × $P block " *
            "(a[$m,$l] = $(A[m, l]), a[$l,$m] = $(A[l, m]), tolerance $atol). Taking the " *
            "leading block would change the metric rather than restrict it -- re-express the " *
            "cell so the first $P rows span the periodic subspace and the rest are normal to " *
            "it."))
    end
    return BravaisLattice(SMatrix{P,P,Float64}(A[1:P, 1:P]))
end

@inline function _pad_components(v, ::Val{Dnew}, fill=0) where {Dnew}
    P = length(v)
    P <= Dnew || throw(ArgumentError(
        "`_pad_components`: cannot pad a length-$P vector down to $Dnew components."))
    T = promote_type(eltype(v), typeof(fill))
    return SVector{Dnew,T}(ntuple(j -> j <= P ? T(v[j]) : T(fill), Val(Dnew)))
end

@inline function _translation_range(bnorms::NTuple{D,Float64}, rcut::Float64,
                                    dtau::SVector{D,Float64}, nperiodic::Int=D) where {D}
    return ntuple(D) do m
        m > nperiodic && return 0:0
        c = bnorms[m] * rcut / (2π)
        lo = floor(Int, -dtau[m] - c - LATTICE_ATOL)
        hi = ceil(Int, -dtau[m] + c + LATTICE_ATOL)
        lo:hi
    end
end

@inline function _is_canonical_pair(si::Int, sj::Int, dR::SVector{D,Int})::Bool where {D}
    for m in 1:D
        dR[m] > 0 && return true
        dR[m] < 0 && return false
    end
    return si <= sj
end

function _group_shells(dists::Vector{Float64}, atol::Float64;
                       rtol::Float64=SHELL_RTOL)
    n = length(dists)
    assign = zeros(Int, n)
    radii = Float64[]
    n == 0 && return radii, assign
    for e in sortperm(dists)
        d = dists[e]
        d > atol || continue          # only reachable for a degenerate lattice
        if isempty(radii) || d - radii[end] > max(atol, rtol * d)
            push!(radii, d)
        end
        assign[e] = length(radii)
    end
    return radii, assign
end

function _warn_near_degenerate(radii::Vector{Float64}, atol::Float64;
                               rtol::Float64=SHELL_RTOL)
    for k in 2:length(radii)
        gap = radii[k] - radii[k-1]
        tol = max(atol, rtol * radii[k])
        if gap < SHELL_DEGENERACY_FACTOR * tol
            @warn "Neighbour shells $(k-1) and $k are nearly degenerate " *
                  "(radii $(radii[k-1]) and $(radii[k]), gap $gap ≈ tolerance $tol). " *
                  "Selecting by `shell=` is ambiguous here; select by explicit `distance=` instead."
        end
    end
    return nothing
end

function _order_entries(key::Vector{T}, ci::Vector{Int}, cj::Vector{Int},
                        cR::Vector{SVector{D,Int}}, keep::F)::Vector{Int} where {T,D,F}
    return sort(findall(keep, key); by = e -> (key[e], ci[e], cj[e], Tuple(cR[e])))
end

struct NeighborTable{D}
    rcut::Float64
    radii::Vector{Float64}
    site_i::Vector{Int}
    site_j::Vector{Int}
    dR::Vector{SVector{D,Int}}
    dist::Vector{Float64}
    shell::Vector{Int}
    offset::Vector{Int}
end

function neighbor_table(cell::UnitCell{D}, rcut::Real;
                        tables::CellTables{D}=cell_tables(cell),
                        rtol::Float64=SHELL_RTOL,
                        atol::Float64=LATTICE_ATOL,
                        warn_degenerate::Bool=true,
                        group_shells::Bool=true)::NeighborTable{D} where {D}
    datol = distance_atol(cell.lattice; atol=atol)
    rcut > 0 || throw(DomainError(rcut, "Neighbour cutoff radius must be positive."))
    rc = Float64(rcut)
    lat = cell.lattice
    aT = transpose(lat.a)
    bnorms = ntuple(m -> norm(lat.brows[m]), D)     # hoisted: constant over the site-pair loop
    ns = n_sites(tables)

    ci = Int[]; cj = Int[]; cR = SVector{D,Int}[]; cd = Float64[]
    for si in 1:ns, sj in si:ns
        if si != sj && (tables.site_kind[si] === :plane_wave ||
                        tables.site_kind[sj] === :plane_wave)
            continue
        end
        dtau = tables.site_positions[sj] - tables.site_positions[si]
        for n in Iterators.product(_translation_range(bnorms, rc, dtau,
                                                      nperiodic(lat))...)
            dR = SVector{D,Int}(n)
            si == sj && iszero(dR) && continue                 # the state itself, not a bond

            d = norm(aT * (SVector{D,Float64}(dR) + dtau))
            d <= rc + datol || continue
            if _is_canonical_pair(si, sj, dR)
                push!(ci, si); push!(cj, sj); push!(cR, dR); push!(cd, d)
            elseif si != sj
                push!(ci, sj); push!(cj, si); push!(cR, -dR); push!(cd, d)
            end
        end
    end

    if group_shells
        radii, shells = _group_shells(cd, datol; rtol=rtol)
        warn_degenerate && _warn_near_degenerate(radii, datol; rtol=rtol)
        order = _order_entries(shells, ci, cj, cR, !iszero)
    else
        radii = Float64[]
        shells = zeros(Int, length(cd))
        order = _order_entries(cd, ci, cj, cR, d -> d > datol)
    end

    site_i = ci[order]; site_j = cj[order]
    dRs = cR[order]; dist = cd[order]; shell = shells[order]

    # offset[s]:offset[s+1]-1 spans shell s, exploiting that `shell` is now sorted
    offset = Vector{Int}(undef, length(radii) + 1)
    pos = 1
    for s in 1:length(radii)
        offset[s] = pos
        while pos <= length(shell) && shell[pos] == s
            pos += 1
        end
    end
    offset[end] = pos

    return NeighborTable{D}(rc, radii, site_i, site_j, dRs, dist, shell, offset)
end

function neighbor_table_by_shell(cell::UnitCell{D}, nshells::Integer;
                                 tables::CellTables{D}=cell_tables(cell),
                                 rtol::Float64=SHELL_RTOL,
                                 atol::Float64=LATTICE_ATOL,
                        warn_degenerate::Bool=true)::NeighborTable{D} where {D}
    datol = distance_atol(cell.lattice; atol=atol)
    nshells >= 1 || throw(DomainError(nshells, "Requested shell count must be at least 1."))
    lat = cell.lattice
    if nperiodic(lat) == 0
        aT = transpose(lat.a)
        dmax = 0.0
        for si in 1:n_sites(tables), sj in (si+1):n_sites(tables)
            dmax = max(dmax, norm(aT * (tables.site_positions[sj] - tables.site_positions[si])))
        end
        dmax > datol || throw(ArgumentError(
            "This cell is a finite cluster (0 periodic directions) with " *
            "$(n_sites(tables) == 1 ? "a single site" : "all sites coincident") -- there are " *
            "no pair distances, so no neighbour shell exists."))
        nt = neighbor_table(cell, dmax + datol; tables=tables, rtol=rtol, atol=atol,
                            warn_degenerate=warn_degenerate)
        length(nt.radii) >= nshells || throw(ArgumentError(
            "This cell is a finite cluster (0 periodic directions) and realizes only " *
            "$(length(nt.radii)) distance shell$(length(nt.radii) == 1 ? "" : "s") in total; " *
            "shell $nshells does not exist. A cluster has finitely many pair distances -- " *
            "request a shallower shell, or address bonds with `explicit_bonds`."))
        return nt
    end
    P = nperiodic(lat)
    rmin = minimum(norm(lat.arows[m]) for m in 1:P)
    est = 2.5 * length_scale(lat) / Float64(n_sites(tables))^(1 / P)
    rcut = min(rmin, max(est, 4 * datol))
    extra = rcut < rmin ? ceil(Int, log2(rmin / rcut)) : 0
    tried = 0
    last = rcut
    for _ in 0:(SHELL_MAX_GROW + extra)
        last = rcut
        tried += 1
        nt = neighbor_table(cell, rcut; tables=tables, rtol=rtol, atol=atol, warn_degenerate=false)
        # Only shells strictly inside the searched radius are complete: the outermost one may be
        # missing neighbours that sit just beyond the cutoff.
        if length(nt.radii) > nshells
            _warn_near_degenerate(nt.radii, distance_atol(cell.lattice; atol=atol); rtol=rtol)
            return nt
        end
        rcut *= 2
    end
    throw(ArgumentError(
        "Could not reach $nshells neighbour shells after $tried searches, the largest with cutoff " *
        "$last. Either the lattice is extremely anisotropic or `nshells` is too large; " *
        "call `neighbor_table(cell, rcut)` with an explicit radius instead."
    ))
end

@inline n_shells(nt::NeighborTable)::Int = length(nt.radii)

function shell_entries(nt::NeighborTable, s::Integer)::UnitRange{Int}
    1 <= s <= length(nt.radii) || throw(BoundsError(nt.radii, s))
    return nt.offset[s]:(nt.offset[s+1] - 1)
end

function _fold_fine_neighbor_table(cell::UnitCell{D}, deepest::Int;
                                   rcut::Union{Nothing,Real}=nothing) where {D}
    s = findfirst(k -> cell.site_kind[k] === :localized &&
                       abs(det(cell.site_period[k])) != 1, eachindex(cell.site_kind))
    M = SMatrix{D,D,Int}(cell.site_period[s])
    fine_a = inv(SMatrix{D,D,Float64}(M)) * SMatrix{D,D,Float64}(cell.lattice.a)
    fine_lat = BravaisLattice(Matrix(fine_a); force=true)
    Mtf = SMatrix{D,D,Float64}(transpose(M))
    fine_pos = [canonicalize_fracpos(SVector{D,Float64}(Mtf * p)) for p in cell.site_positions]
    proxy_basis = [BasisState{D}(fine_pos[k], zero(SVector{D,Int}), 1, TRIVIAL_SCHEMA)
                   for k in eachindex(fine_pos)]
    proxy = UnitCell{D}(fine_lat, TRIVIAL_SCHEMA, copy(cell.site_names), fine_pos, proxy_basis;
                        atol=cell.atol, warn_empty_sites=false)
    return isnothing(rcut) ? neighbor_table_by_shell(proxy, deepest) :
                             neighbor_table(proxy, rcut)
end

function _fold_fine_wrapshifts(cell::UnitCell{D})::Vector{SVector{D,Int}} where {D}
    s = findfirst(k -> cell.site_kind[k] === :localized &&
                       abs(det(cell.site_period[k])) != 1, eachindex(cell.site_kind))
    M = SMatrix{D,D,Int}(cell.site_period[s])
    Mtf = SMatrix{D,D,Float64}(transpose(M))
    return [canonicalize_int(canonicalize_fracpos(SVector{D,Float64}(Mtf * p)) - Mtf * p)
            for p in cell.site_positions]
end

function _integer_expansion(a_new::SMatrix{D,D,Float64}, a_old::SMatrix{D,D,Float64};
                            atol::Float64=INT_ROUND_ATOL)::SMatrix{D,D,Int} where {D}
    Mf = a_new * inv(a_old)
    tol = atol * max(1.0, maximum(abs, Mf))
    isapprox(Mf, round.(Mf); atol=tol) || throw(DomainError(Mf,
        "The proposed cell is not an integer supercell of the layer lattice."
    ))
    return SMatrix{D,D,Int}(round.(Int, Mf))
end

function _coincidence_vectors(l1::BravaisLattice{D}, l2::BravaisLattice{D},
                              ranges::NTuple{D,UnitRange{Int}}, rcut::Float64;
                              atol::Float64=INT_ROUND_ATOL) where {D}
    Q = l2.b * transpose(l1.a) / (2π)
    aT = transpose(l1.a)
    found = Tuple{Float64,SVector{D,Int},SVector{D,Float64}}[]
    for n in Iterators.product(ranges...)
        nv = SVector{D,Int}(n)
        iszero(nv) && continue
        x = aT * SVector{D,Float64}(nv)
        d = norm(x)
        d <= rcut || continue
        f = Q * SVector{D,Float64}(nv)
        tol = atol * max(1.0, maximum(abs, f))
        isapprox(f, round.(f); atol=tol) || continue
        push!(found, (d, nv, x))
    end
    sort!(found; by = e -> (e[1], Tuple(e[2])))
    return found
end

function _shortest_basis(found::Vector{Tuple{Float64,SVector{D,Int},SVector{D,Float64}}},
                         l1::BravaisLattice{D}) where {D}
    isempty(found) && return nothing
    chosen = SVector{D,Float64}[]
    onb = SVector{D,Float64}[]                 # orthonormal basis of the span chosen so far
    for (d, _, x) in found
        r = x
        for u in onb
            r -= dot(r, u) * u
        end
        norm(r) / d > LINDEP_RTOL || continue
        push!(chosen, x)
        push!(onb, r / norm(r))
        length(chosen) == D && break
    end
    length(chosen) == D || return nothing
    det(_rows_to_matrix(ntuple(i -> chosen[i], Val(D)))) * det(l1.a) < 0 &&
        (chosen[D] = -chosen[D])
    for i in 1:(D - 1)
        u, v = chosen[i], chosen[D]
        if dot(u, v) < 0 && norm(u + v) <= norm(v) * (1 + LINDEP_RTOL)
            chosen[D] = u + v
        end
    end
    return ntuple(i -> chosen[i], Val(D))
end

function _coincidence_lattice(l1::BravaisLattice{D}, l2::BravaisLattice{D}, rcut0::Float64;
                              max_grow::Int=4, maxpoints::Int=4_000_000,
                              atol::Float64=INT_ROUND_ATOL,
                              known_commensurate::Bool=false)::BravaisLattice{D} where {D}
    bnorms = ntuple(m -> norm(primitive_recip_vector(l1, m)), D)
    rcut = rcut0
    budget_msg = known_commensurate ?
        "This twist was built from integer indices, which is commensurate whenever " *
        "a₁·a₂/|a₁|² is RATIONAL (always for triangular and square lattices). If it is, the " *
        "coincidence cell is just highly anisotropic and exceeds the search budget -- raise " *
        "`max_grow` (radius doublings) and, if needed, `maxpoints`. If a₁·a₂/|a₁|² is " *
        "irrational (a generic rhombic lattice), the two layers genuinely share no " *
        "coincidence lattice at any radius." :
        "The twist angle is likely incommensurate; construct from integer indices " *
        "`twisted_bilayer(lat, m, n)` (commensurate by construction), or raise " *
        "`max_grow`/`maxpoints` if the angle is genuinely commensurate with a very large cell."
    for _ in 0:max_grow
        ranges = _translation_range(bnorms, rcut, zero(SVector{D,Float64}))
        npts = prod(length, ranges)
        npts > maxpoints && throw(ArgumentError(
            "Coincidence search at radius $rcut needs $npts candidates, above the cap " *
            "`maxpoints = $maxpoints`. " * budget_msg))
        basis = _shortest_basis(_coincidence_vectors(l1, l2, ranges, rcut; atol=atol), l1)
        isnothing(basis) || return BravaisLattice(collect(basis))
        rcut *= 2
    end
    throw(ArgumentError(
        "No coincidence lattice found within radius $(rcut / 2) after $(max_grow + 1) " *
        "attempts. " * budget_msg))
end

@inline _rotation2(theta::Real) =
    SMatrix{2,2,Float64}(cos(theta), sin(theta), -sin(theta), cos(theta))

_rotated_lattice(lat::BravaisLattice{2}, theta::Real)::BravaisLattice{2} =
    BravaisLattice(lat.a * transpose(_rotation2(theta)))

struct TwistedBilayer
    monolayer::BravaisLattice{2}
    layers::NTuple{2,BravaisLattice{2}}
    moire::BravaisLattice{2}
    M::NTuple{2,SMatrix{2,2,Int,4}}
    theta::Float64
    indices::Union{Nothing,Tuple{Int,Int}}
end

function _twisted_bilayer(lat::BravaisLattice{2}, theta::Float64,
                          indices::Union{Nothing,Tuple{Int,Int}}, rcut0::Float64;
                          max_grow::Int, maxpoints::Int, atol::Float64)::TwistedBilayer
    l1 = _rotated_lattice(lat, -theta / 2)
    l2 = _rotated_lattice(lat, +theta / 2)
    moire = _coincidence_lattice(l1, l2, rcut0; max_grow=max_grow, maxpoints=maxpoints,
                                 atol=atol, known_commensurate=!isnothing(indices))
    M1 = _integer_expansion(moire.a, l1.a; atol=atol)
    M2 = _integer_expansion(moire.a, l2.a; atol=atol)
    abs(det(M1)) == abs(det(M2)) || throw(ErrorException(
        "The two layers decompose the moiré cell into different numbers of unit cells " *
        "($(abs(det(M1))) vs $(abs(det(M2)))); this is an internal inconsistency, not a usage error."
    ))
    return TwistedBilayer(lat, (l1, l2), moire, (M1, M2), theta, indices)
end

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
    isapprox(norm(a1), norm(a2); rtol=SHELL_RTOL) || throw(ArgumentError(
        "The (m, n) construction requires primitive vectors of equal length; this lattice has " *
        "|a₁| = $(norm(a1)) and |a₂| = $(norm(a2)). Give the twist angle directly: " *
        "`twisted_bilayer(lat, theta)`."))
    if dot(a1, a2) < 0
        a1, a2 = -a2, a1
    end

    v1 = m * a1 + n * a2
    v2 = n * a1 + m * a2
    theta = acos(clamp(dot(v1, v2) / (norm(v1) * norm(v2)), -1.0, 1.0))
    r0 = isnothing(rcut) ? 1.2 * norm(v1) : Float64(rcut)
    return _twisted_bilayer(lat, theta, (Int(m), Int(n)), r0;
                            max_grow=max_grow, maxpoints=maxpoints, atol=atol)
end

twisted_bilayer(lat::BravaisLattice{2}, mn::Tuple{Integer,Integer}; kwargs...) =
    twisted_bilayer(lat, mn[1], mn[2]; kwargs...)

function twisted_bilayer(lat::BravaisLattice{2}, theta::Real;
                         degrees::Bool=true, rcut::Union{Nothing,Real}=nothing,
                         max_grow::Int=4, maxpoints::Int=4_000_000,
                         atol::Float64=INT_ROUND_ATOL)::TwistedBilayer
    th = degrees ? deg2rad(float(theta)) : float(theta)
    0 < th < Float64(π) || throw(DomainError(theta,
        "The twist angle must lie strictly between 0 and 180 degrees."))
    lmax = maximum(norm(primitive_vector(lat, i)) for i in 1:2)
    r0 = isnothing(rcut) ? max(3 * lmax / (2 * sin(th / 2)), 2 * lmax) : Float64(rcut)
    return _twisted_bilayer(lat, th, nothing, r0;
                            max_grow=max_grow, maxpoints=maxpoints, atol=atol)
end

twist_angle(tb::TwistedBilayer; degrees::Bool=true)::Float64 =
    degrees ? rad2deg(tb.theta) : tb.theta

n_monolayer_cells(tb::TwistedBilayer)::Int = abs(det(tb.M[1]))

function layer_lattice(tb::TwistedBilayer, l::Integer)::BravaisLattice{2}
    l in (1, 2) || throw(ArgumentError("Layer index must be 1 or 2; got $l."))
    return tb.layers[l]
end

function supercell_matrix(tb::TwistedBilayer, l::Integer)::SMatrix{2,2,Int}
    l in (1, 2) || throw(ArgumentError("Layer index must be 1 or 2; got $l."))
    return tb.M[l]
end

function monolayer_bz_gvectors(tb::TwistedBilayer, layer::Integer)::Vector{SVector{2,Int}}
    M = supercell_matrix(tb, layer)
    N = n_monolayer_cells(tb)
    B = recip_matrix(tb.moire)
    bmin = minimum(norm(primitive_recip_vector(tb.moire, m)) for m in 1:2)
    datol = LATTICE_ATOL * bmin
    adjM = _int_adjugate(M)
    lay = layer_lattice(tb, layer)
    Rmax = sum(norm(primitive_recip_vector(lay, m)) for m in 1:2) / 2 + datol
    box = ntuple(m -> ceil(Int, Rmax * norm(primitive_vector(tb.moire, m)) / (2π)) + 1, 2)
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
