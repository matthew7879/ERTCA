##################################################################
###### export only the following functions from this module ######
##################################################################

export wrap01, wrap_lattice, canonicalize_pos, canonicalize_fracpos, canonicalize_int
export round_atol




#####################################################################
###### The rounding / tolerance relationship                    #####
#####################################################################

"""
    round_atol(digits) -> Float64

The comparison tolerance that belongs with a rounding granularity of `digits` decimal places.

Purpose:
--------
`atol = 10^(-digits + ROUND_SLACK_DECADES)`. Every place in the package that rounds to a grid and
then compares on that grid goes through this, so the two numbers cannot drift apart.

Notes:
------
Comparing at or below the rounding step is the mistake this exists to prevent: two values that round
to the same grid point can still differ by a full step, so a tolerance equal to the step rejects
values the rounding just declared identical. `LATTICE_ATOL` and `INT_ROUND_ATOL` are both defined as
`round_atol` of their own granularity.
"""
@inline round_atol(digits::Integer)::Float64 = exp10(-digits + ROUND_SLACK_DECADES)




#####################################################################
###### Wrapping into the fundamental domain                     #####
#####################################################################

"""
    wrap01(x; digits=LATTICE_ROUND_DIGITS) -> Real
    wrap01(v; digits=LATTICE_ROUND_DIGITS) -> SVector

Wrap a real, or each component of a vector, into `[0, 1)` and round to `digits` decimals.

Notes:
------
A value just below 1 -- which `mod` produces for any small negative input -- must come back as `0.0`
rather than `0.999...`, or the two ends of the interval would name the same point twice. The
threshold for "close enough to 1" is `round_atol(digits)`, the same relationship used everywhere
else, rather than a tolerance invented locally.
"""
function wrap01(x::Real; digits::Integer=LATTICE_ROUND_DIGITS)::Float64
    y = round(mod(float(x), 1.0), digits=digits)
    return isapprox(y, 1.0; atol=round_atol(digits)) ? 0.0 : y
end

@inline function wrap01(v::SVector{D,<:Real}; digits::Integer=LATTICE_ROUND_DIGITS)::SVector{D,Float64} where {D}
    return SVector{D,Float64}(wrap01.(v; digits=digits))
end

function wrap01(v::AbstractVector{<:Real}; kwargs...)
    return wrap01(SVector{length(v),Float64}(v); kwargs...)
end

"""
    wrap_lattice(v, S; digits=LATTICE_ROUND_DIGITS, atol=LATTICE_ATOL) -> SVector{D,Float64}

Wrap a cartesian vector `v` into the fundamental domain of the lattice whose vectors are the rows
of `S`, returning fractional coordinates.

Args:
-----
- `v`: a cartesian vector, length `D`
- `S`: a `D×D` matrix of lattice vectors
- `rtol`: relative tolerance for the linear-independence guard on `S`

Returns:
--------
- `SVector{D,Float64}` in `[0,1)^D`
"""
function wrap_lattice(v::SVector{D,Float64}, S::SMatrix{D,D,Float64};
                      digits::Integer=LATTICE_ROUND_DIGITS,
                      rtol::Float64=LINDEP_RTOL)::SVector{D,Float64} where {D}
    ### Scale-free guard, matching the `BravaisLattice` constructor: `det(S)` carries units of
    ### length^D, so comparing it against a fixed absolute tolerance is unit-system dependent
    ### (a valid nm-scale lattice expressed in metres has det ~ 1e-20 and would be rejected).
    abs(det(S)) / prod(norm(S[i, :]) for i in 1:D) < rtol && throw(DomainError(S,
        "Bravais lattice vectors not linearly independent, det(S)=$(det(S))"
    ))
    coords = inv(S) * v
    return canonicalize_fracpos(coords; digits=digits)
end

function wrap_lattice(v::AbstractVector{<:Real}, S::AbstractMatrix{<:Real}; kwargs...)
    D = length(v)
    size(S) == (D, D) || throw(DimensionMismatch("S must be square of size D×D"))
    return wrap_lattice(SVector{D,Float64}(v), SMatrix{D,D,Float64}(S); kwargs...)
end




#####################################################################
###### Canonicalizing onto the rounding grid                    #####
#####################################################################

# `SVector{length(v),...}` builds a type from a *runtime* length, which is type-unstable and boxes
# the result. The statically-sized methods below are what callers holding an `SVector` -- every hot
# path -- dispatch to; the `AbstractVector` methods keep the convenient entry point and pay a single
# dynamic dispatch instead of one per element access.

"""
    canonicalize_pos(v; digits=LATTICE_ROUND_DIGITS) -> SVector{D,Float64}

Round a position onto the rounding grid, without wrapping.

Notes:
------
Rounding is what makes exact `==` on a position meaningful at all: two positions computed by
different routes agree bit-for-bit only after being put on a common grid. Use
[`canonicalize_fracpos`](@ref) for fractional coordinates, which must also be wrapped.
"""
@inline function canonicalize_pos(v::SVector{D,<:Real};
                                  digits::Integer=LATTICE_ROUND_DIGITS)::SVector{D,Float64} where {D}
    return SVector{D,Float64}(round.(v, digits=digits))
end

function canonicalize_pos(v::AbstractVector{<:Real}; kwargs...)
    return canonicalize_pos(SVector{length(v),Float64}(v); kwargs...)
end

"""
    canonicalize_fracpos(v; digits=LATTICE_ROUND_DIGITS, nperiodic=length(v)) -> SVector{D,Float64}

Wrap the first `nperiodic` components of a fractional position into `[0,1)` and round every
component onto the rounding grid.

Notes:
------
This is the canonical form every `BasisState` position is stored in, so that positions produced by
different arithmetic routes compare equal. On a partially periodic lattice (`nperiodic < D`) the
trailing components are APERIODIC coordinates -- a ribbon's transverse position, a layer's height:
wrapping them would silently identify physically distinct points, so they are only rounded.
"""
@inline function canonicalize_fracpos(v::SVector{D,<:Real};
                                      digits::Integer=LATTICE_ROUND_DIGITS,
                                      nperiodic::Integer=D)::SVector{D,Float64} where {D}
    nperiodic == D && return canonicalize_pos(wrap01(v; digits=digits); digits=digits)
    ### `nperiodic == 0` is a Γ-only finite cluster (every direction aperiodic): nothing wraps,
    ### every component is a real coordinate and is only rounded
    (0 <= nperiodic <= D) || throw(ArgumentError(
        "`nperiodic` must lie in 0:$D; got $nperiodic."))
    return SVector{D,Float64}(ntuple(
        m -> m <= nperiodic ? wrap01(v[m]; digits=digits) : round(Float64(v[m]); digits=digits),
        D))
end

function canonicalize_fracpos(v::AbstractVector{<:Real}; kwargs...)
    return canonicalize_fracpos(SVector{length(v),Float64}(v); kwargs...)
end

"""
    canonicalize_int(v; atol=INT_ROUND_ATOL) -> SVector{D,Int64}

Convert a vector that should be integral into exact integers, raising if it is not.

Args:
-----
- `atol`: how far from an integer each component may be

Returns:
--------
- `SVector{D,Int64}`

Notes:
------
Raises `DomainError` rather than rounding silently: a Miller index or a supercell matrix that does
not come back integral means the transformation was not a lattice operation, and rounding would
hide that.

`atol` defaults to `round_atol(INT_ROUND_DIGITS)`, i.e. two decades of slack above the integer
rounding step. That slack is load-bearing here -- a Miller index of magnitude ~30 transported
through two matrix products accumulates error well above a single step, so a tighter tolerance
rejects labels that are integers in every sense that matters.
"""
@inline function canonicalize_int(v::SVector{D,<:Real};
                                  atol::Float64=INT_ROUND_ATOL)::SVector{D,Int64} where {D}
    rounded = round.(v)
    isapprox(v, rounded; atol=atol) || throw(DomainError(v,
        "Cannot convert to integer with tolerance atol=$atol; nearest integers are $rounded."
    ))
    return SVector{D,Int64}(rounded)
end

### An already-integer input has nothing to check; `atol` is accepted and ignored so that callers
### forwarding `atol` generically do not have to know which method they will hit.
@inline function canonicalize_int(v::SVector{D,<:Integer};
                                  atol::Float64=INT_ROUND_ATOL)::SVector{D,Int64} where {D}
    return SVector{D,Int64}(v)
end

function canonicalize_int(v::AbstractVector{<:Real}; kwargs...)
    return canonicalize_int(SVector{length(v),Float64}(v); kwargs...)
end

function canonicalize_int(v::AbstractVector{<:Integer}; kwargs...)
    return SVector{length(v),Int64}(v)
end
