##################################################################
###### export only the following functions from this module ######
##################################################################

export supercell, supercell_real, supercell_reciprocal, coset_representatives


####################
###### Helper ######
####################


"""
    _latticepoints_in_supercell(M::AbstractMatrix{<:Integer}) -> Vector{SVector{D,Float64}}

For a supercell a_new = M * a_old, return the position of the Bravais lattice points (n1 * a_old[1] + n2 * a_old[2] + ...) that are inequivalent in the supercell.
The positions are returned in fractional coordinates of the new lattice; there are abs(det(M)) of them (i.e., the absolute value of the determinant).

Notes:
------
Two integer vectors `n`, `n'` name the same supercell point iff `n - n' ∈ Mᵀ ℤᴰ`, so the job is
to enumerate the cosets of `ℤᴰ / Mᵀ ℤᴰ`. With `H` the column-style Hermite normal form of `Mᵀ`
(lower triangular, positive diagonal, same column lattice), a complete set of representatives is
the exact integer box `0 ≤ xᵢ < H[i,i]` -- `prod(diag(H)) = |det M|` points, no search, no
dedup. The previous implementation scanned a `(|det M|+1)ᴰ` hypercube and deduplicated wrapped
FLOATS, which was both `O(det²)`-ish in 2D and carried the rounding-boundary hazard every float
key carries.
"""
function _latticepoints_in_supercell(M::SMatrix{D,D,<:Integer}; digits::Int=LATTICE_ROUND_DIGITS)::Vector{SVector{D,Float64}} where {D}
    numpts = _int_det_abs(SMatrix{D,D,Int}(M))
    numpts == 0 && throw(ArgumentError(
        "Supercell transformation matrix must be invertible."
    ))
    H = _hnf_cols(SMatrix{D,D,Int}(transpose(M)))
    MinvT = transpose(inv(SMatrix{D,D,Float64}(M)))
    points = Vector{SVector{D,Float64}}(undef, numpts)
    idx = 1
    for n in Iterators.product(ntuple(a -> 0:(H[a, a] - 1), D)...)
        points[idx] = canonicalize_fracpos(MinvT * SVector{D,Float64}(n); digits=digits)
        idx += 1
    end
    return sort!(points, by = x -> Tuple(x))
end

### `_hnf_cols` and `_int_adjugate` live in `utils/intlinalg.jl`, shared with the congruence
### duplicate check of folded cells.
function _latticepoints_in_supercell(M::AbstractMatrix{<:Integer}; digits::Int=LATTICE_ROUND_DIGITS)
    D = size(M, 1)
    size(M) == (D, D) || throw(DimensionMismatch("Supercell transformation matrix must be square."))
    return _latticepoints_in_supercell(SMatrix{D,D}(M); digits=digits)
end

"""
    _copy_site_namer(parent_names::AbstractVector{Symbol}, ncopies::Integer) -> (k, l) -> Symbol

The copy-suffix naming policy of every transform that replaces each parent site by `ncopies`
copies at distinct positions -- `supercell`'s `|det M|` coset images and `open_boundary`'s
`ncells` repeats. Returns the namer `(k, l) -> name of copy l of parent site k`; `ncopies == 1`
keeps the parent names untouched.

Notes:
------
The obvious `Symbol(name, "_", l)` is unique *within* the generated registry, but it is not
injective against the PARENT's own names: a cell carrying both `:A` and `:A_1` (a Wannier90
import, a hand-built layer registry) supercells into `[:A_1, :A_2, :A_1_1, :A_1_2]`, where
`:A_1` no longer means what it meant on the parent -- it names the first copy of `:A`. No name
is duplicated, so the `UnitCell` uniqueness check never fires, and a site-selective term
recompiled on the transformed cell (`onsite(...; sites=[:A_1])`) silently lands on a different
site. Only the RE-compiled term is wrong; the transported couplings are right, which is what
made this invisible.

The separator is therefore lengthened (`_` -> `__` -> ...) until no generated name is one the
parent already declares, in the spirit of `_spliced_names`' repeated prefixing in
[`direct_sum`](@ref) -- and the escape is announced, because the old name then resolves to
nothing and a recompiled term must be refused by name rather than redirected. A single global
separator (rather than a per-name escape) keeps the scheme predictable: every copy on a given
cell is suffixed the same way. Termination is immediate -- the generated names grow by one
character per round while the parent registry is fixed and finite.
"""
function _copy_site_namer(parent_names::AbstractVector{Symbol}, ncopies::Integer)
    return _copy_site_namer(parent_names, [string(l) for l in 1:Int(ncopies)])
end

### The same policy where the copy index is not a single integer. `open_boundary` opening
### several directions at once labels a repeat by its MULTI-index (`:A_2_3`), so it cannot pass
### a count -- but it must not grow a second naming policy either, because the escaping is the
### whole point and a second copy of it is a second thing to forget. Both callers hand over the
### exact tails they will generate.
function _copy_site_namer(parent_names::AbstractVector{Symbol}, tails::AbstractVector{<:AbstractString})
    length(tails) <= 1 && return (k::Int, l::Int) -> parent_names[k]
    sep = "_"
    shadowed = _shadowed_by_copy_names(parent_names, sep, tails)
    escaped = shadowed
    while !isempty(shadowed)
        sep *= "_"
        shadowed = _shadowed_by_copy_names(parent_names, sep, tails)
    end
    isempty(escaped) || @warn "The copies of each site are named `<site>$(sep)<copy index>`, " *
        "not the usual `<site>_<copy index>`: the parent registry already declares " *
        "$(join(("`:" * String(s) * "`" for s in escaped), ", ")), which the single-underscore " *
        "suffix would have shadowed -- `:$(String(escaped[1]))` would have named a copy of a " *
        "DIFFERENT site, and a term recompiled with that name would have landed on it silently. " *
        "Site-selective terms written against the parent must be recompiled against the new " *
        "names; `site_names(cell)` lists them."
    return (k::Int, l::Int) -> Symbol(parent_names[k], sep, tails[l])
end

"""
    _shadowed_by_copy_names(parent_names, sep, ncopies) -> Vector{Symbol}

The parent names that the copy suffix `sep` would collide with, i.e. the names of the form
`<another parent name><sep><index>` with `1 <= index <= ncopies`. Tested against the parent
registry (`O(n_sites)` matches) rather than by generating all `n_sites × ncopies` candidate
names, which on a large supercell is the whole output.

The split is unambiguous even though the head is greedy: the index is printed with `string(l)`,
so it is a digit run with no leading zero and cannot swallow a separator -- there is exactly one
way to read a name as `head * sep * index`. Two GENERATED names likewise cannot collide with
each other for the same reason (unique parent names in, unique names out), so disjointness from
the parent registry is the whole check.
"""
function _shadowed_by_copy_names(parent_names::AbstractVector{Symbol}, sep::AbstractString,
                                 tails::AbstractVector{<:AbstractString})::Vector{Symbol}
    nameset = Set{Symbol}(parent_names)
    tailset = Set{String}(String.(tails))
    ### `s` (dotall) and byte-safe regex matching: site names may carry non-ASCII characters,
    ### where indexing backwards from `end` by a character count is not valid. The tail is a
    ### digit run, or digit runs joined by the separator on a multi-direction opening, so the
    ### greedy head still leaves exactly one way to read `head * sep * tail` for a tail we emit.
    pat = Regex("^(.*?)" * sep * "([1-9][0-9]*(?:" * sep * "[1-9][0-9]*)*)\$", "s")
    out = Symbol[]
    for nm in parent_names
        m = match(pat, String(nm))
        m === nothing && continue
        m.captures[2] in tailset || continue
        Symbol(m.captures[1]) in nameset && push!(out, nm)
    end
    return out
end

"""
    coset_representatives(M; space=:real) -> Vector{SVector{D,Float64}}

The `|det M|` coset representatives of a supercell matrix `M` (`a_new = M a_old`), in the NEW
cell's fractional coordinates -- the public form of the enumeration every supercell/fold
consumer needs:

- `space = :real`: the real-space representatives `wrap(M⁻ᵀ ℤᴰ)` -- where the copies of a
  parent site sit in the supercell; the `lp` of `supercell`'s coupling map.
- `space = :reciprocal`: the reciprocal representatives `wrap(M⁻¹ ℤᴰ)` -- the BRANCH SET of the
  momentum mapping: the spectrum of the big model (supercell or fold) at `k_new` is the union
  of parent spectra over `k_old(c) = M⁻¹ k_new + g_c` with `g_c` these representatives. The
  `kpoints` of [`spectral_weight`](@ref) are exactly `M⁻¹ k_new .+ coset_representatives(M;
  space=:reciprocal)`.

Deterministic order (sorted by components), exact integer enumeration underneath.
"""
function coset_representatives(M::AbstractMatrix{<:Integer}; space::Symbol=:real)
    space === :real && return _latticepoints_in_supercell(M)
    space === :reciprocal && return _latticepoints_in_supercell(collect(transpose(M)))
    throw(ArgumentError("`space` must be :real or :reciprocal; got :$space."))
end


############################################################
###### Supercell transformations for Bravais lattices ######
############################################################

"""
    supercell(lat::BravaisLattice{D}, M::AbstractMatrix{<:Integer}) -> BravaisLattice{D}

Return the real-space supercell lattice defined by the integer transformation
matrix `M`, where the new lattice vectors are `a_new = M * a_old`.
"""
function supercell(lat::BravaisLattice{D}, M::SMatrix{D,D,Int})::BravaisLattice{D} where {D}
    P = nperiodic(lat)
    if P < D
        ### expansion may only act on the periodic block: the padding rows are coordinate
        ### axes, and mixing them into a primitive vector (or "expanding" along them) is
        ### meaningless. This funnel serves every supercell/fold entry point.
        (all(iszero, M[1:P, (P+1):D]) && all(iszero, M[(P+1):D, 1:P]) &&
         M[(P+1):D, (P+1):D] == one(SMatrix{D-P,D-P,Int})) || throw(ArgumentError(
            "This lattice is periodic only along its first $P direction$(P == 1 ? "" : "s"); " *
            "a supercell matrix must be blockdiag(M_P, I) on it. Got M = $M."))
    end
    a_new = M * lat.a
    return BravaisLattice(a_new; force=true, nperiodic=P)
end
function supercell(lat::BravaisLattice{D}, M::AbstractMatrix{<:Integer})::BravaisLattice{D} where {D}
    size(M) == (D, D) || throw(DimensionMismatch("Supercell transformation matrix must be $D×$D"))
    return supercell(lat::BravaisLattice{D}, SMatrix{D,D,Int}(M))
end

"""
    supercell_real(lat::BravaisLattice{D}, new_a::AbstractMatrix{<:Real}; atol::Float64=INT_ROUND_ATOL) -> BravaisLattice{D}

Construct a real-space supercell from explicit new DIRECT lattice vectors `new_a`, one per row.

The integer matrix `M` with `a_new = M a_old` is recovered from them and must exist: a `new_a`
that is not an integer superlattice of `lat` is refused. `atol` is the integer-snap tolerance
(`INT_ROUND_ATOL` by contract, not the fractional `LATTICE_ATOL`).

The reciprocal-lattice counterpart is [`supercell_reciprocal`](@ref); see [`supercell`](@ref)
for the whole family in one table.
"""
function supercell_real(lat::BravaisLattice{D}, new_a::AbstractMatrix{<:Real}; atol::Float64=INT_ROUND_ATOL)::BravaisLattice{D} where {D}
    M = _supercell_integer_matrix(new_a, lat.a; atol=atol)
    return supercell(lat, M)
end

"""
    supercell_reciprocal(lat::BravaisLattice{D}, new_b::AbstractMatrix{<:Real}; atol::Float64=INT_ROUND_ATOL) -> BravaisLattice{D}

Construct a lattice from explicit new RECIPROCAL lattice vectors `new_b`, one per row: the
direct vectors are recovered as `a_new = 2π (new_b)^{-T}` and handed to
[`supercell_real`](@ref).

This is the reciprocal-space member of the [`supercell`](@ref) family, and the lattice half of
what [`fold`](@ref) does to a cell. It was reachable only as
`supercell(lat, A; mode = :bz_fold)` before 2026-09-17 and is exported under its own name now.
"""
function supercell_reciprocal(lat::BravaisLattice{D}, new_b::AbstractMatrix{<:Real}; atol::Float64=INT_ROUND_ATOL)::BravaisLattice{D} where {D}
    size(new_b) == (D, D) || throw(DimensionMismatch("New reciprocal lattice vectors must form DxD matrix."))
    ### scale-free linear-independence guard: det(new_b) is dimensionful, so an absolute
    ### tolerance would make this check unit-system dependent
    abs(det(new_b)) / prod(norm(new_b[i, :]) for i in 1:D) < LINDEP_RTOL &&
        throw(DomainError("Reciprocal lattice vectors are linearly dependent."))
    new_a = 2π * inv(transpose(new_b))
    return supercell_real(lat, new_a; atol=atol)
end

### A `supercell(lat, A; mode = :unit_expand | :bz_fold)` wrapper sat here until 2026-09-17 and
### dispatched to one of the two functions above on the keyword. Removing it was right -- the
### two are different operations, and the same argument meant DIRECT vectors under one value
### and RECIPROCAL vectors under the other, which is a name and a flag saying opposite things.
###
### But the commit that removed it justified itself by saying both alternatives were already
### public, and they were NOT exported. So the deletion also removed the only exported route to
### a reciprocal-lattice supercell. Both are exported now, and `supercell(lat, new_a)` below is
### documented as the direct-lattice member of the family rather than as an anonymous wrapper.
"""
    supercell(lat::BravaisLattice{D}, new_a::AbstractMatrix{<:Real}; atol) -> BravaisLattice{D}

Construct a real-space supercell from explicit new DIRECT lattice vectors, one per row: the
convenience spelling of [`supercell_real`](@ref), which it forwards to unchanged.

The family, so that the choice is made by NAME and never by a keyword:

| you have | call |
|---|---|
| an integer matrix `M`, `a_new = M a_old` | [`supercell`](@ref)`(lat, M)` |
| the new DIRECT vectors | [`supercell`](@ref)`(lat, new_a)` or [`supercell_real`](@ref) |
| the new RECIPROCAL vectors | [`supercell_reciprocal`](@ref)`(lat, new_b)` |
| a cell or model, real-space | [`supercell`](@ref)`(cell, M)` |
| a cell, momentum-space (a FOLD) | [`fold`](@ref)`(cell, M)` or [`fold`](@ref)`(cell, new_b)` |

The last row is the distinction the removed keyword blurred: a supercell multiplies the SITE
count and moves positions, a fold multiplies the STATE count and relabels plane waves. They are
different operations on different arguments, and each has its own name.
"""
function supercell(lat::BravaisLattice{D}, new_a::AbstractMatrix{<:Real};
                   atol::Float64=INT_ROUND_ATOL)::BravaisLattice{D} where {D}
    return supercell_real(lat, new_a; atol=atol)
end


#######################################################
###### Supercell transformations for BasisStates ######
#######################################################

### The one place a transformed basis state is BUILT: the canonicalization, the momentum
### origin's transformation and the `nperiodic` threading are identical for both transforms
### below, and were identical inside the two branches of the `mode` keyword this replaced.
"""
    supercell(bs::Vector{BasisState{D}}, M::AbstractMatrix{<:Integer}; mode::Symbol) -> Vector{BasisState{D}}

Transform basis states under a real-space supercell defined by integer matrix `M`.
The new basis states are the translated copies of the original basis states within
the expanded unit cell.

Every site of the parent reappears once per real-space coset representative, so the state count
multiplies by `|det M|`. The momentum-space counterpart -- relabelling plane waves onto the
reciprocal cosets rather than translating positions -- is `fold`, whose basis half is
`_fold_basis` below; the two were one function with a `mode` keyword until 2026-09-17.
"""
@inline function _push_transformed_state!(out::Vector{BasisState{D}}, b::BasisState{D},
                                          r_frac, G, M::AbstractMatrix{<:Integer};
                                          digits::Int, atol::Float64,
                                          nperiodic::Int) where {D}
    push!(out, BasisState{D}(
        canonicalize_fracpos(SVector{D,Float64}(r_frac); digits=digits, nperiodic=nperiodic),
        canonicalize_int(SVector{D,Float64}(G); atol=atol),
        SVector{D,Float64}(M * b.k0),
        b.internal, b.schema; nperiodic=nperiodic))
    return nothing
end

function supercell(bs::Vector{BasisState{D}}, M::AbstractMatrix{<:Integer};
                   digits::Int=LATTICE_ROUND_DIGITS, atol::Float64=INT_ROUND_ATOL,
                   nperiodic::Int=D)::Vector{BasisState{D}} where {D}
    size(M) == (D, D) || throw(DimensionMismatch("Supercell transformation matrix must be $D×$D"))
    ### real-space coset representatives: wrap(M⁻ᵀ n), the images of the old lattice vectors in
    ### new fractional coordinates
    lattice_points = _latticepoints_in_supercell(M; digits=digits)
    Minv = inv(Matrix{Float64}(M))
    new_bs = BasisState{D}[]
    ### Lattice point OUTER, parent basis inner -- the `STATE_ORDER` convention shared with
    ### `add_axis` and `add_plane_waves`, so the parent basis is reproduced in order as the
    ### first `length(bs)` states and `new_index(i, l) = (l-1)*length(bs) + i`.
    for lp in lattice_points, b in bs
        ### Positions transform with M⁻ᵀ and plane-wave labels with M: `a_new = M a_old` forces
        ### `b_new = M⁻ᵀ b_old`, so keeping the physical G_cart = Σᵢ Gᵢ bᵢ fixed requires
        ### G_new = M G_old. (Mᵀ here is indistinguishable for the symmetric matrices tests
        ### favour, and silently relabels momenta for every other M.)
        _push_transformed_state!(new_bs, b, transpose(Minv) * b.r_frac .+ lp, M * b.G, M;
                                 digits=digits, atol=atol, nperiodic=nperiodic)
    end
    return new_bs
end

"""
    _fold_basis(bs, M; digits, atol, nperiodic) -> Vector{BasisState{D}}

The basis half of [`fold`](@ref): one copy of each parent state per RECIPROCAL coset
representative, positions left where they are, plane-wave labels relabelled onto the cosets.

A fold re-labels momenta, it does not move places -- which is the whole reason this is not a
mode of `supercell` above. It was one until 2026-09-17, and what that cost is on the record:
the shared body had nowhere to put the fine period lattice that a fold has to record on its
localized sites, so `supercell(cell, M; mode=:bz_fold)` never recorded it and real-space terms
compiled wrongly on the folded cell for as long as the operation was a keyword. The feature
became writable the moment the two were separate functions.
"""
function _fold_basis(bs::Vector{BasisState{D}}, M::AbstractMatrix{<:Integer};
                     digits::Int=LATTICE_ROUND_DIGITS, atol::Float64=INT_ROUND_ATOL,
                     nperiodic::Int=D)::Vector{BasisState{D}} where {D}
    size(M) == (D, D) || throw(DimensionMismatch("Fold matrix must be $D×$D"))
    ### RECIPROCAL cosets: representatives of wrap(M⁻¹ n) = `_latticepoints_in_supercell(Mᵀ)`
    lattice_points = _latticepoints_in_supercell(collect(transpose(M)); digits=digits)
    Minv = inv(Matrix{Float64}(M))
    new_bs = BasisState{D}[]
    for lp in lattice_points, b in bs
        ### M(G + lp) = M G + n with n integer, by construction of the representatives
        _push_transformed_state!(new_bs, b, transpose(Minv) * b.r_frac, M * (b.G .+ lp), M;
                                 digits=digits, atol=atol, nperiodic=nperiodic)
    end
    return new_bs
end


################################################################
###### Combine for supercell transformations for UnitCell ######
################################################################

"""
    supercell(uc::UnitCell{D}, M::AbstractMatrix{<:Integer}; kwargs...) -> UnitCell{D}

Return a new `UnitCell` whose lattice has been expanded by the integer
transformation matrix `M` and whose basis states have been translated into
the expanded unit cell.

The momentum-space counterpart is [`fold`](@ref). It was reachable here as
`mode = :bz_fold` until 2026-09-17, deprecated since the day `fold` was written.
"""
function supercell(uc::UnitCell{D}, M::AbstractMatrix{<:Integer}; kwargs...)::UnitCell{D} where {D}
    ### expanding a FOLDED cell has no consistent per-site description: the |det M| copies of a
    ### folded site are fine-lattice images of one another, so "one comb per site" breaks down.
    ### Unfold first (or take the real-space supercell of the parent, then fold).
    is_folded(uc) && throw(ArgumentError(
        "`supercell(cell, M)` cannot expand a folded cell: the copies of a " *
        "folded site are fine-lattice images of one another. Supercell the parent first, then " *
        "`fold`."))
    ### the same refusal the MODEL-level transform already makes: a plane wave has no centre
    ### to copy into |det M| positions -- compiling on position-copied plane-wave sites gives
    ### |det M|-fold degenerate copies of the same dispersion, silently
    for s in 1:n_sites(uc)
        uc.site_kind[s] === :plane_wave && throw(ArgumentError(
            "`supercell(cell, M)` cannot expand a cell with :plane_wave " *
            "sites (site `$(uc.site_names[s])`): a plane wave has no centre to copy into " *
            "|det M| positions. The momentum-space supercell of a plane-wave basis is " *
            "`fold(cell, M)`."))
    end
    new_lat = supercell(uc.lattice, M)
    new_basis = supercell(uc.basis, M; nperiodic=nperiodic(uc.lattice), kwargs...)
    ### kinds survive the expansion (each site reappears once per coset with the SAME kind);
    ### periods stay trivial -- each copy's comb is Bloch-summed over the NEW lattice, so its
    ### period lattice IS the new lattice. Dropping the kinds would silently reclassify.
    ### The coset count IS |det M| -- no need to enumerate the representatives to count them.
    nlp = _int_det_abs(SMatrix{D,D,Int}(M))
    kinds = repeat(uc.site_kind; inner=nlp)
    ### state -> site in CLOSED FORM: states go lattice-point-outer, parent-inner (STATE_ORDER,
    ### see `supercell(::Vector{BasisState}, M)`), sites go parent-site-outer, point-inner
    ### (`_supercell_sites`). Passing it skips the constructor's O(n_states × n_sites)
    ### tolerance re-derivation -- quadratic in |det M|, the dominant cost of big supercells.
    nold = length(uc.basis)
    stsite = Vector{Int}(undef, nold * nlp)
    for l in 1:nlp, i in 1:nold
        stsite[(l - 1) * nold + i] = (uc.state_site[i] - 1) * nlp + l
    end
    return UnitCell(new_lat, new_basis;
                    schema=uc.schema, sites=_supercell_sites(uc, M; kwargs...),
                    site_kind=kinds, state_site=stsite,
                    atol=uc.atol)
end

"""
    _supercell_sites(uc, M; digits) -> Vector{Pair{Symbol,SVector{D,Float64}}}

The site registry of a supercell.

Notes:
------
`:unit_expand` multiplies the number of sites by `|det M|`: each site of the parent reappears once
per coset representative. The images are named `:A_1, :A_2, ...` from the parent's name, so the
provenance of a site survives the transform instead of degenerating into `:site7` -- with the
separator lengthened (and the escape announced) if that would shadow a name the parent already
declares; see [`_copy_site_namer`](@ref).

A fold, by contrast, leaves every position alone -- it re-labels plane waves, not places -- so
the parent registry carries over unchanged apart from the change of fractional basis. That is
[`_fold_sites`](@ref), not a mode of this function.
"""
function _supercell_sites(uc::UnitCell{D}, M::AbstractMatrix{<:Integer};
                          digits::Int=LATTICE_ROUND_DIGITS, kwargs...) where {D}
    P = nperiodic(uc.lattice)
    mapped = _rebased_site_positions(uc, M; digits=digits)
    lattice_points = _latticepoints_in_supercell(M; digits=digits)
    ### one shared copy-suffix policy (see `_copy_site_namer`): the suffix must not collide with
    ### a name the PARENT already declares, or a recompiled site-selective term silently changes
    ### which site it addresses
    namer = _copy_site_namer(uc.site_names, length(lattice_points))
    out = Pair{Symbol,SVector{D,Float64}}[]
    for k in eachindex(mapped), (l, lp) in pairs(lattice_points)
        push!(out, namer(k, l) => canonicalize_fracpos(mapped[k] .+ lp; digits=digits, nperiodic=P))
    end
    return out
end

"""
    _rebased_site_positions(uc, M; digits) -> Vector{SVector{D,Float64}}

The parent's site positions in the new fractional basis, `M⁻ᵀ τ`, canonicalized. Shared by
[`_supercell_sites`](@ref) and [`_fold_sites`](@ref), which differ in what they do NEXT.
"""
_rebased_site_positions(uc::UnitCell{D}, M::AbstractMatrix{<:Integer};
                        digits::Int=LATTICE_ROUND_DIGITS) where {D} =
    [canonicalize_fracpos(SVector{D,Float64}(transpose(inv(Matrix{Float64}(M))) * p);
                          digits=digits, nperiodic=nperiodic(uc.lattice))
     for p in uc.site_positions]

"""
    _fold_sites(uc, M; digits) -> Vector{Pair{Symbol,SVector{D,Float64}}}

The site registry of a folded cell: the parent's names and positions, rebased into the new
fractional coordinates and otherwise untouched. A fold multiplies the STATE count by
`|det M|` and leaves the SITE count alone, which is what distinguishes it from a supercell.
"""
_fold_sites(uc::UnitCell{D}, M::AbstractMatrix{<:Integer};
            digits::Int=LATTICE_ROUND_DIGITS) where {D} =
    [uc.site_names[k] => p
     for (k, p) in pairs(_rebased_site_positions(uc, M; digits=digits))]

"""
    supercell(uc::UnitCell{D}, new_a::AbstractMatrix{<:Real}; atol::Float64=LATTICE_ATOL) -> UnitCell{D}

Construct a real-space supercell `UnitCell` from explicit new lattice vectors
`new_a`. The new lattice is accepted only if it is an integer multiple of the
original lattice vectors.
"""
function supercell(uc::UnitCell{D}, new_A::AbstractMatrix{<:Real};
                   atol::Float64=INT_ROUND_ATOL,
                   digits::Int=LATTICE_ROUND_DIGITS)::UnitCell{D} where {D}
    ### Keywords are routed explicitly: `atol` is the integer-snap tolerance (consumed by
    ### `_supercell_integer_matrix`, which does not accept `digits`), `digits` the position
    ### rounding grid. Forwarding a blind `kwargs...` into both legs raised a MethodError.
    new_lat = supercell_real(uc.lattice, new_A; atol=atol)
    M = _supercell_integer_matrix(new_lat.a, uc.lattice.a; atol=atol)
    ### same integer-M path as the method above, kinds threaded, folded cells refused
    return supercell(uc, Matrix{Int}(M); digits=digits)
end


##################################################
###### Supercell transformations for Models ######
##################################################

"""
    supercell(model::HamiltonianModel{D}, M; digits, atol) -> HamiltonianModel{D}

Map a Hamiltonian onto a real-space supercell defined by the integer matrix `M`.

How the couplings map:
----------------------
This transforms the **compiled bond list** rather than recompiling the terms. That is the point of
keeping the Hamiltonian as a flat `(i, j, ΔR)` array: a supercell is then an index remapping plus a
translation remapping, with no geometry search, and it works for models built from imported or
hand-written bonds just as well as for ones built from distance shells.

The old lattice vectors `R` split as `R = R_super + t`, with `t` one of the `|det M|` coset
representatives that `_latticepoints_in_supercell` returns. In fractional coordinates of the new
lattice an old vector maps as `R -> Mᵀ⁻¹ R`, so the copy of old state `i` sitting at representative
`lp` occupies new position `Mᵀ⁻¹ τᵢ + lp`. A bond `(i, j, ΔR)` then relates that copy to whichever
copy of `j` lands at

    u = lp + Mᵀ⁻¹ ΔR   =   lp' + ΔR_new

split into a representative `lp'` and an integer translation of the new lattice.

Args:
-----
- `model`: the Hamiltonian to transform
- `M`: integer supercell matrix, `a_new = M * a_old`
- `digits`, `atol`: tolerances for matching wrapped positions to coset representatives

Returns:
--------
- `HamiltonianModel{D}` on the supercell, in the same gauge (the couplings are mapped, not
  re-derived from terms).

Notes:
------
Only `mode = :unit_expand` is meaningful here: `fold` is the momentum-space counterpart.

Every kernel maps: `q -> M q` (fractional momenta transform with `M`, exactly like the `G`
labels), the index/translation bookkeeping is the same coset arithmetic for every entry, and
the kernels ride along -- so a continuum model's kinetic energy supercells like any bond. What
is refused instead is a KIND-domain violation: a `:plane_wave` site cannot be position-copied
into `|det M|` sites (`:unit_expand` semantics are meaningless for a basis function with no
centre; `fold` is its supercell), and a folded model (rational `ΔR`) has no integer coset
arithmetic to ride on -- supercell before folding.
"""
function supercell(model::HamiltonianModel{D}, M::AbstractMatrix{<:Integer};
                   digits::Int=LATTICE_ROUND_DIGITS,
                   atol::Float64=LATTICE_ATOL)::HamiltonianModel{D} where {D}
    size(M) == (D, D) || throw(DimensionMismatch("Supercell transformation matrix must be $D×$D"))
    for s in 1:n_sites(model.cell)
        model.cell.site_kind[s] === :plane_wave && throw(ArgumentError(
            "`supercell` cannot :unit_expand a model with :plane_wave sites (site " *
            "`$(model.cell.site_names[s])`): a plane wave has no centre to copy into |det M| " *
            "positions. The momentum-space supercell of a plane-wave basis is `fold`."
        ))
    end
    is_folded(model.cell) && throw(ArgumentError(
        "`supercell` cannot expand a folded model: its translations are rational and the exact " *
        "integer coset arithmetic does not apply. Take the real-space supercell first, then fold."
    ))

    Ms = SMatrix{D,D,Int}(M)
    new_cell = supercell(model.cell, Ms; digits=digits)
    lattice_points = _latticepoints_in_supercell(Ms; digits=digits)
    nlp = length(lattice_points)
    nold = n_states(model.cell)
    MinvT = transpose(inv(Matrix{Float64}(Ms)))
    Mf = SMatrix{D,D,Float64}(Ms)

    ### `supercell(::Vector{BasisState}, M)` pushes lattice-point-outer, parent-inner, per the
    ### `STATE_ORDER` convention -- so the parent basis is the first `nold` states, in order.
    new_index(iold::Int, ilp::Int) = (ilp - 1) * nold + iold

    ### Coset bookkeeping in EXACT integers: representative `lp` translated by `ΔR` lands in the
    ### coset labeled `adj(Mᵀ)·(v + ΔR) mod |det M|`, with `v` the integer vector of `lp` -- an
    ### O(1) dictionary hit per (coupling, lattice point) instead of a tolerance-based
    ### `findfirst` over all `|det M|` representatives, and immune to the rounding-boundary
    ### hazard of float coset keys (the class of bug fixed in `monolayer_bz_gvectors`).
    A = SMatrix{D,D,Int}(transpose(Ms))
    adjA = _int_adjugate(A)
    ### integer snapping uses the integer contract constant: `A * lp` accumulates float error
    ### proportional to the entries of M, and the 1e-12 fractional tolerance can throw on a
    ### perfectly valid large-M matrix
    vint = [canonicalize_int(A * lattice_points[l]; atol=INT_ROUND_ATOL) for l in 1:nlp]
    coset = Dict{SVector{D,Int},Int}(_coset_label(adjA, nlp, vint[l]) => l for l in 1:nlp)
    length(coset) == nlp || throw(ErrorException(
        "Supercell mapping failed: the $nlp coset representatives of $Ms do not carry distinct " *
        "integer labels. This is an internal inconsistency, not a usage error."
    ))

    ### The cell stores positions WRAPPED into [0,1)ᴰ, but the coset arithmetic below works with
    ### the unwrapped copy positions `M⁻ᵀτ + lp`. Each copy therefore carries an integer wrap
    ### shift `σ = stored − unwrapped` that must be added back into the mapped translation:
    ### without it the coupling list and the stored positions disagree by exactly that shift.
    ### Bands survive the disagreement (it is a k-dependent diagonal unitary), which is why no
    ### spectrum test can see it -- but every position-dependent quantity (`spectral_weight`
    ### first, the physical bond displacements `ΔR + τ_j − τ_i` themselves) is silently wrong.
    old_pos = state_positions(model.cell)
    new_pos = state_positions(new_cell)
    sigma = Matrix{SVector{D,Int}}(undef, nold, nlp)
    for ilp in 1:nlp, iold in 1:nold
        raw = MinvT * old_pos[iold] + lattice_points[ilp]
        sigma[iold, ilp] = canonicalize_int(new_pos[new_index(iold, ilp)] - raw;
                                            atol=INT_ROUND_ATOL)
    end

    c = model.couplings
    buf, kids = adopted_buffer(c; per_canonical=nlp)
    ### The source list stores both directions; `each_canonical` yields only the canonical
    ### halves, and the outer constructor's finalize re-materializes every Hermitian partner on
    ### the new cell. Do not "optimize" by mapping both directions, which would coalesce each
    ### coupling with its own image and double every amplitude.
    for cpl in each_canonical(c)
        i, j = cpl.i, cpl.j
        dR = SVector{D,Int}(Int.(numerator.(cpl.dR)))    # denominator 1 guaranteed above
        shift = MinvT * SVector{D,Float64}(dR)
        ### fractional momenta transform with M: bᵀ_new(k_new + Mq) = bᵀ_old(k_old + q), and the
        ### phase (Mk + Mq)·M⁻ᵀδ = (k + q)·δ is preserved simultaneously
        q_new = SVector{D,Float64}(Mf * cpl.q)
        for ilp in 1:nlp
            jlp = coset[_coset_label(adjA, nlp, vint[ilp] + dR)]
            u = lattice_points[ilp] + shift
            ### integer snapping uses the integer contract constant, not the (10× tighter)
            ### fractional tolerance the coset match used to need; the σ terms re-express the
            ### translation against the WRAPPED stored positions (see above)
            dR_new = canonicalize_int(u - lattice_points[jlp]; atol=INT_ROUND_ATOL) +
                     sigma[i, ilp] - sigma[j, jlp]
            push_coupling!(buf, new_index(i, ilp), new_index(j, jlp), dR_new, q_new, cpl.w,
                           kids[cpl.kernel], cpl.term)
        end
    end

    return HamiltonianModel(new_cell, buf; gauge=model.gauge, name=model.name)
end


