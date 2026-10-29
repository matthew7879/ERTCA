# Coincidence (moiré) lattices: the sublattice two lattices share, at any dimension.
#
# This is generic math that lived inside `twisted_bilayer.jl` and was written at D = 2 because
# that is where it was first needed. The TWIST is genuinely two-dimensional -- a rotation by one
# angle, the `(m, n)` indexing, the γ = 60° convention -- but "which vectors do these two
# lattices share, and what is the primitive cell of that sublattice" is not. Separating them is
# what lets the twist file keep only what is really planar.

"""
    _integer_expansion(a_new, a_old; atol=INT_ROUND_ATOL) -> SMatrix{D,D,Int}

The integer matrix `M` with `a_new = M * a_old`, raising if no such integer relation holds.

Notes:
------
Integrality is tested RELATIVE to the size of `M`, the same policy as `==` on `BravaisLattice`:
the float error of `a_new * inv(a_old)` grows with the magnitude of its entries, and small-angle
twists legitimately produce entries of order tens to hundreds. A fixed absolute tolerance rejects
exactly the supercells this file exists to build.
"""
function _integer_expansion(a_new::SMatrix{D,D,Float64}, a_old::SMatrix{D,D,Float64};
                            atol::Float64=INT_ROUND_ATOL)::SMatrix{D,D,Int} where {D}
    Mf = a_new * inv(a_old)
    tol = atol * max(1.0, maximum(abs, Mf))
    isapprox(Mf, round.(Mf); atol=tol) || throw(DomainError(Mf,
        "The proposed cell is not an integer supercell of the layer lattice."
    ))
    return SMatrix{D,D,Int}(round.(Int, Mf))
end

"""
    _coincidence_vectors(l1, l2, ranges, rcut; atol) -> Vector{(dist, n, x_cart)}

Every non-zero vector of lattice `l1` within `rcut` that is ALSO a lattice vector of `l2`,
sorted by length (ties broken lexicographically on the integer coordinates, for determinism).

Notes:
------
A point of `l1` with integer coordinates `n` sits at `x = a₁ᵀ n`; its fractional coordinates in
`l2` are `f = b₂ x / 2π = Q n` with `Q = b₂ a₁ᵀ / 2π`. The point is common to both lattices iff
`f` is integral. Integrality is tested relative to the size of `f` (the same policy as
`_integer_expansion`): `|f|` grows with the search radius, and so does its float error.
"""
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

"""
    _shortest_basis(found, l1) -> Union{Nothing, NTuple{D,SVector{D,Float64}}}

The `D` shortest linearly independent coincidence vectors, as cartesian rows of the moiré cell,
or `nothing` if fewer than `D` independent vectors were found.

Notes:
------
Through `D = 4` the successive minima of a lattice are always attained by a basis (Minkowski), so
"shortest, then shortest independent, ..." over an exhaustively searched ball IS a primitive
coincidence cell -- no reduction pass is needed afterwards. This is what makes the construction
return the PRIMITIVE cell even when the input indices describe a non-primitive one (the classic
example: hexagonal `(m, n)` with `3 | m - n`, whose primitive moiré cell is 3× smaller than the
`m² + mn + n²` formula cell). Beyond `D = 4` the claim fails and this would need
`_greedy_reduce` afterwards; `BravaisLattice` refuses those dimensions anyway.

Independence is the residual of a candidate after projecting out the span already chosen,
relative to the candidate's own length. At `D = 2` that residual is `|x₁ × x₂| / |x₂|`, so the
test is `|sin θ| > LINDEP_RTOL` -- exactly the scalar cross product this replaced, computed by
projection instead of by a formula that only exists in the plane.

The last vector is flipped if needed so the moiré cell has the same orientation (sign of
`det a`) as the layers, and then replaced by `xᵢ + x_D` when that is no longer -- so an
equal-length obtuse partner (the 120° choice a triangular lattice always offers) is traded for
the acute one, and the moiré cell of a triangular monolayer comes out in the same γ = 60°
convention. `det [x₁; x₁ + x₂] = det [x₁; x₂]`, so this keeps both primitivity and orientation.
"""
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

"""
    _coincidence_lattice(l1, l2, rcut0; max_grow, maxpoints, atol) -> BravaisLattice{D}

The coincidence (moiré) lattice `l1 ∩ l2`, searched inside a ball of radius `rcut0` that is
doubled up to `max_grow` times if no `D` independent coincidence vectors are found.

Notes:
------
The growth loop mirrors `neighbor_table_by_shell`: the right radius is not knowable in advance
(a coincidence cell can be much larger than the naive `a/θ` moiré scale), so the search grows
until it succeeds or exhausts its budget. `maxpoints` caps the enumeration box per attempt --
an incommensurate angle would otherwise grow forever, since it has NO coincidence lattice at any
radius. Hitting either limit raises with the incommensurability diagnosis rather than looping.
"""
function _coincidence_lattice(l1::BravaisLattice{D}, l2::BravaisLattice{D}, rcut0::Float64;
                              max_grow::Int=4, maxpoints::Int=4_000_000,
                              atol::Float64=INT_ROUND_ATOL,
                              known_commensurate::Bool=false)::BravaisLattice{D} where {D}
    bnorms = ntuple(m -> norm(primitive_recip_vector(l1, m)), D)
    rcut = rcut0
    ### the honest diagnosis depends on what the caller knows: the (m, n) constructor is
    ### commensurate BY CONSTRUCTION, so a failed search there means an exhausted budget on a
    ### highly anisotropic coincidence cell (a low-symmetry rhombic lattice can need a second
    ### basis vector tens of times longer than the first) -- never incommensurability
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
