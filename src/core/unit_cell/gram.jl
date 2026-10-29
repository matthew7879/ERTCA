# The Gram-automorphism search: every unimodular basis change carrying one reduced metric onto
# another. One search, one tolerance, at any dimension.
#
# It has exactly two consumers and they want different halves of it. `identify_bravais` takes the
# FIRST match, to map a lattice onto a tabulated type; `lattice_point_group` takes the WHOLE set
# at `Gu == Gc`, which is the lattice's own holohedry. Two searches with two tolerances would be
# two answers to "does this lattice have a C3", which is why this is a file and not a private
# helper inside either caller.

"""
    _unimodular_matches(Gu, Gc, tol) -> Vector{SMatrix{D,D,Int}}

EVERY unimodular `W` with `W Gc Wᵀ = Gu`, proper (`det +1`) ones first.

Entries in `-1:1` suffice because both metrics are REDUCED: a reduced basis is a set of
successive minima, and no automorphism of it can carry a basis vector onto an integer
combination with a coefficient of two or more without lengthening it. On a non-reduced basis
that is false, and a caller must reduce and conjugate back (see [`lattice_point_group`](@ref)).
Measured on this package's own `bravais_lattice` templates: after
[`_greedy_reduce`](@ref) the holohedry order comes out exact for all 14 three-dimensional
Bravais types, with `max |W_ij| = 1` even when the search is widened to `-3:3`.

The enumeration is `3^(D²)` candidates -- 81 at `D = 2`, 19,683 at `D = 3`, microseconds either
way -- and runs in ROW-major digit order with the last entry varying fastest. That order is not
decoration: `_metric_match` returns the FIRST match and `identify_bravais` carries tabulated
high-symmetry points through it, so a different (equally valid) `W` would silently relabel them.
The order is exactly the one the hand-unrolled `D = 2` loop had.
"""
function _unimodular_matches(Gu::SMatrix{D,D,Float64}, Gc::SMatrix{D,D,Float64},
                             tol::Float64) where {D}
    out = SMatrix{D,D,Int}[]
    n = D * D
    pow3 = ntuple(j -> 3^(n - j), Val(n))
    ### proper first: a symmetric lattice admits improper matches too, and preferring proper
    ### keeps the identity case mapping to the identity -- an improper W would silently relabel
    ### chirality-sensitive points (K vs K') even when nothing needed relabeling
    for want_det in (1, -1)
        for code in 0:(3^n - 1)
            ### `dig[j]` is the j-th entry of W read ROW-major (j = 1 most significant), and the
            ### index arithmetic below places it: an `SMatrix` fills COLUMN-major, so linear
            ### index `k` is row `(k-1) % D + 1`, column `(k-1) ÷ D + 1`, whose row-major
            ### position is `(row - 1) * D + column`.
            dig = ntuple(j -> (code ÷ pow3[j]) % 3 - 1, Val(n))
            W = SMatrix{D,D,Int}(ntuple(k -> dig[((k - 1) % D) * D + (k - 1) ÷ D + 1], Val(n)))
            _int_det(W) == want_det || continue
            isapprox(W * Gc * transpose(W), Gu; atol=tol) && push!(out, W)
        end
    end
    return out
end

### The first match, or `nothing`. The identity short-circuit is kept: it guarantees that equal
### metrics map through the identity rather than through whichever proper W the loop reaches
### first, which is what keeps `identify_bravais` from relabeling a lattice onto itself.
function _metric_match(Gu::SMatrix{D,D,Float64}, Gc::SMatrix{D,D,Float64},
                       tol::Float64) where {D}
    isapprox(Gc, Gu; atol=tol) && return one(SMatrix{D,D,Int})
    ms = _unimodular_matches(Gu, Gc, tol)
    return isempty(ms) ? nothing : first(ms)
end

"""
    _gram_of_rows(A::SMatrix{D,D,Float64}) -> SMatrix{D,D,Float64}

The metric `G[i,j] = aᵢ·aⱼ` of a basis whose vectors are the ROWS of `A`, i.e. `A Aᵀ`.

Trivial, and it exists because the row convention is the package's one real spelling trap: the
classifier and the holohedry search both built this by hand, entry by entry, and a transposed
metric is wrong only on a lattice with a non-symmetric automorphism -- which is to say, exactly
on the triangular one.
"""
@inline _gram_of_rows(A::SMatrix{D,D,Float64}) where {D} = A * transpose(A)

"""
    _basis_from_gram(G) -> SMatrix{P,P,Float64}
    _basis_from_gram(G, ::Val{P}) -> SMatrix{P,P,Float64}

The inverse of [`_gram_of_rows`](@ref): the lower-triangular, positive-diagonal Cartesian basis
whose ROWS have metric `G`, by Cholesky factorization.

Canonical rather than arbitrary -- of the infinitely many bases with a given metric (all related
by a rigid rotation) this picks the one with `a₁` along the first axis -- which is what makes it
usable as a frame: `periodic_block` builds a slab's in-plane lattice with it, and the moire
archive reconstructs its Cartesian frame with it. Both hand-wrote the 2x2 closed form.

The second form takes the rank explicitly, for a caller holding a dynamically sized or
non-`Float64` metric (the archive stores exact `Rational` Gram matrices).
"""
function _basis_from_gram(G::SMatrix{P,P,Float64}) where {P}
    F = cholesky(Symmetric(G); check=false)
    issuccess(F) || throw(ArgumentError(
        "`_basis_from_gram`: the matrix is not positive definite, so it is not the metric of " *
        "any basis:\n$G"))
    return SMatrix{P,P,Float64}(F.L)
end

_basis_from_gram(G::AbstractMatrix{<:Real}, ::Val{P}) where {P} =
    _basis_from_gram(SMatrix{P,P,Float64}(G))
