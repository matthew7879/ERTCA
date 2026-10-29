##################################################################
###### export only the following functions from this module ######
##################################################################

export RegistryHarmonics, registry_harmonics, registry_shifts




#############################################################
###### The registry harmonics container                 #####
#############################################################

"""
    HarmonicBlock

One matrix element of one registry harmonic, as an EXACT finite Bloch sum: the pairs
`(delta[m], amp[m])` represent `W_{ββ'}(k̄; Ḡ) = Σₘ amp[m] · e^{2πi k̄·delta[m]}`, with the
separations in monolayer-fractional coordinates. Exact at every k̄ -- inside or outside the
Brillouin zone; the atomic-gauge quasi-periodicity is automatic, never imposed.
"""
struct HarmonicBlock
    delta::Vector{SVector{2,Float64}}
    amp::Vector{ComplexF64}
end

Base.isempty(b::HarmonicBlock) = isempty(b.amp)

"""
    RegistryHarmonics

The k-dependent registry harmonics `W_{ββ'}(k̄; Ḡ)` extracted from a [`RegistryScan`](@ref) --
the complete parametrization of the ERTCA continuum model, twist-independent (the same
harmonics realize any angle through [`ertca_model`](@ref)).

Fields:
-------
- `lattice::BravaisLattice{2}`     the in-plane monolayer lattice
- `labels`, `shifted`, `layer_names`  the orbital bookkeeping, inherited from the scan
- `tau::Vector{SVector{2,Float64}}`  the reference orbital positions at d = 0
  (monolayer-fractional, in-plane) -- the comb offsets the folded-channel phases need
- `gvecs::Vector{SVector{2,Int}}`  the retained harmonics, CANONICAL HALF only of each
  ±Ḡ pair (first nonzero component positive; `Ḡ = 0` first). The −Ḡ half is never stored:
  `W_{β'β}(k̄; −Ḡ) = conj(W_{ββ'}(k̄; Ḡ))` regenerates it exactly, which is also what keeps
  the realized model Hermitian by construction.
- `blocks::Array{HarmonicBlock,3}` indexed `(iG, β, β')` -- the exact Bloch-sum data
- `cond::Float64`                  condition number of the solved `e^{−2πi Ḡ·d}` system
- `hermiticity::Float64`           the solve's Hermiticity-pairing residual BEFORE
  enforcement (exactly zero data-side; a large value flags an ill-sampled stacking set)
- `d::Vector{SVector{2,Float64}}`  the stacking shifts the harmonics were solved from
- `name::String`

Construct through [`registry_harmonics`](@ref).
"""
struct RegistryHarmonics
    lattice::BravaisLattice{2}
    labels::Vector{Symbol}
    shifted::BitVector
    tau::Vector{SVector{2,Float64}}
    layer_names::NTuple{2,Symbol}
    gvecs::Vector{SVector{2,Int}}
    blocks::Array{HarmonicBlock,3}
    cond::Float64
    hermiticity::Float64
    d::Vector{SVector{2,Float64}}
    name::String
end

function Base.show(io::IO, h::RegistryHarmonics)
    print(io, "RegistryHarmonics(\"", h.name, "\": ", length(h.gvecs),
          " canonical harmonics up to Ḡ = ", Tuple(h.gvecs[end]), ", ",
          length(h.labels), " orbitals, ", length(h.d), " stackings, cond = ",
          round(h.cond; sigdigits=3), ")")
end
Base.show(io::IO, ::MIME"text/plain", h::RegistryHarmonics) = show(io, h)

### the canonical member of a ±Ḡ pair: first nonzero component positive; Ḡ = 0 is its own
### partner (the same rule as `is_canonical_coupling`, on the harmonic label)
@inline function _g_canonical(g::SVector{2,Int})::Bool
    g[1] > 0 && return true
    g[1] < 0 && return false
    return g[2] >= 0
end

"""
    _harmonic_value(h, iG, i, j, k) -> ComplexF64

`W_{ij}(k̄; Ḡ_{iG})` from the stored Bloch sum, `k` in monolayer-fractional coordinates.
The −Ḡ half is `conj(_harmonic_value(h, iG, j, i, k))`.
"""
function _harmonic_value(h::RegistryHarmonics, iG::Int, i::Int, j::Int,
                         k::SVector{2,Float64})::ComplexF64
    b = h.blocks[iG, i, j]
    s = zero(ComplexF64)
    @inbounds for m in eachindex(b.amp)
        s += b.amp[m] * cispi(2 * dot(k, b.delta[m]))
    end
    return s
end

### the reconstructed commensurate Bloch Hamiltonian W(k̄, d) = Σ_Ḡ e^{−2πi Ḡ·d} W(k̄; Ḡ),
### the −Ḡ half regenerated from the canonical one
function _reconstruct_W(h::RegistryHarmonics, d::SVector{2,Float64},
                        k::SVector{2,Float64})::Matrix{ComplexF64}
    nb = length(h.labels)
    W = zeros(ComplexF64, nb, nb)
    for (iG, g) in pairs(h.gvecs)
        ph = cispi(-2 * dot(g, d))
        for j in 1:nb, i in 1:nb
            W[i, j] += ph * _harmonic_value(h, iG, i, j, k)
            iszero(g) && continue
            W[i, j] += conj(ph) * conj(_harmonic_value(h, iG, j, i, k))
        end
    end
    return W
end




#############################################################
###### The retained harmonic set                        #####
#############################################################

### the retained set F: ±closed, deduplicated, Ḡ = 0 included, sorted by (|Ḡ|, lex)
function _retained_set(lattice::BravaisLattice{2},
                       gshells::Union{Nothing,Int},
                       gvectors::Union{Nothing,AbstractVector})::Vector{SVector{2,Int}}
    (isnothing(gshells) == isnothing(gvectors)) && throw(ArgumentError(
        "Pass exactly one of `gshells=` (retain the harmonics of the first n reciprocal " *
        "shells) or `gvectors=` (an explicit ±closed set of integer harmonic labels)."))
    F = if !isnothing(gshells)
        gshells >= 0 || throw(DomainError(gshells, "`gshells` must be non-negative."))
        gs = gshells == 0 ? SVector{2,Int}[] :
             reduce(vcat, g_shells(lattice, gshells)[2][1:gshells])
        vcat([zero(SVector{2,Int})], gs)
    else
        raw = SVector{2,Int}[]
        for (m, g) in pairs(gvectors)
            (g isa AbstractVector && length(g) == 2 && all(x -> isinteger(x), g)) ||
                throw(ArgumentError(
                    "`gvectors[$m]` = $g is not an integer 2-vector; harmonics are labeled " *
                    "by monolayer reciprocal-lattice integer coordinates."))
            push!(raw, SVector{2,Int}(g))
        end
        unique(raw)
    end
    any(iszero, F) || throw(ArgumentError(
        "The retained harmonic set omits Ḡ = 0 -- the (d-averaged) dispersion itself. " *
        "Solving without it attributes the largest component of the data to the moiré " *
        "harmonics; include [0, 0]."))
    for g in F
        any(==(-g), F) || throw(ArgumentError(
            "The retained harmonic set is not closed under Ḡ → −Ḡ (it holds $(Tuple(g)) " *
            "without $(Tuple(-g))). Hermiticity pairs the two -- and the least-squares " *
            "solution only satisfies `W(−Ḡ) = W(Ḡ)†` on a ±closed set."))
    end
    return sort(F; by=g -> (norm(recip_matrix_times(lattice, g)), g[1], g[2]))
end

### cartesian reciprocal vector of an integer label (rows of `b` are the primitive
### reciprocal vectors, so Ḡ_cart = bᵀ·n̄)
@inline recip_matrix_times(lattice::BravaisLattice{2}, n::SVector{2,Int}) =
    transpose(lattice.b) * SVector{2,Float64}(n)




#############################################################
###### Proposing the stacking shifts                    #####
#############################################################

"""
    registry_shifts(lattice; gshells=nothing, gvectors=nothing) -> Vector{SVector{2,Float64}}

The stacking shifts to COMPUTE, chosen so the harmonic solve is exact: for the retained
harmonic set `F` (same keywords as [`registry_harmonics`](@ref)), returns `S = |F|`
fractional shifts on which the matrix `E[s, m] = e^{−2πi Ḡₘ·d_s}` is unitary (times √S) --
the linear inversion degenerates to a discrete Fourier transform, machine-exact on `F`.

On a triangular lattice: `gshells = 1` → 7 shifts (constant + first shell), `gshells = 2` →
13, one per resolvable harmonic. Run this BEFORE the DFT campaign; each returned `d` is one
commensurate-bilayer calculation.

The shifts are the unit-cell coset representatives of an index-`S` sublattice chosen so that
`F` is a complete residue system modulo it (verified exactly on the integers), preferring
the most isotropic such grid. If no such sublattice exists for an irregular `F`, a
deterministic low-discrepancy set is returned instead with a `@warn` reporting its condition
number -- the solve still works, just not at machine precision.
"""
function registry_shifts(lattice::BravaisLattice{2};
                         gshells::Union{Nothing,Int}=nothing,
                         gvectors::Union{Nothing,AbstractVector}=nothing
                         )::Vector{SVector{2,Float64}}
    F = _retained_set(lattice, gshells, gvectors)
    S = length(F)
    S == 1 && return [SVector(0.0, 0.0)]

    best_ds = nothing
    best_score = -Inf
    for a in 1:S
        S % a == 0 || continue
        c = S ÷ a
        for b in 0:(a-1)
            M = SMatrix{2,2,Int}(a, 0, b, c)        # columns (a, 0), (b, c): index-S sublattice
            _complete_residues(F, M) || continue
            ds = [SVector{2,Float64}(d) for d in coset_representatives(M; space=:real)]
            ### prefer the most isotropic sampling: maximize the minimal wrapped distance
            score = minimum(_min_wrapped_norm(ds[s] - ds[t])
                            for s in 1:S for t in (s+1):S; init=Inf)
            if score > best_score + 1e-12
                best_score = score
                best_ds = ds
            end
        end
    end
    isnothing(best_ds) || return best_ds

    ### No complete-residue sublattice: a deterministic Kronecker (low-discrepancy) set. The
    ### two generators must be irrational AND rationally independent of each other. The golden
    ### ratio is the classic 1D choice but its square is NOT an independent second generator --
    ### φ² = 1 − φ exactly, so (sφ, sφ²) lies on the anti-diagonal d₁ + d₂ ∈ ℤ for every s, the
    ### "2D" set collapses to a line, and E[s,m] depends only on n₁ − n₂: harmonics like (0,0)
    ### and (1,1) alias EXACTLY. Use the plastic number ρ (ρ³ = ρ + 1) instead -- the R2
    ### sequence α = (1/ρ, 1/ρ²), whose components are genuinely independent.
    rho = 1.3247179572447460      # the plastic number, real root of x³ = x + 1
    a1, a2 = 1 / rho, 1 / rho^2
    ds = [SVector(mod(s * a1, 1.0), mod(s * a2, 1.0)) for s in 0:(S-1)]
    E = [cispi(-2 * dot(F[m], ds[s])) for s in 1:S, m in 1:S]
    sv = svdvals(E)
    ### apply the SAME rank test `registry_harmonics` will apply, and refuse here: a set that
    ### cannot be inverted must not be handed out as a DFT campaign to run
    if sv[end] <= 1e-10 * sv[1]
        pair = _aliased_pair(E, F)
        throw(ArgumentError(
            "`registry_shifts`: no index-$S sublattice makes the retained set a complete " *
            "residue system, and the low-discrepancy fallback is rank-deficient on it " *
            "(cond = $(round(sv[1] / max(sv[end], eps() * sv[1]); sigdigits=3)))" *
            (isnothing(pair) ? "" :
             ": harmonics $(Tuple(pair[1])) and $(Tuple(pair[2])) are indistinguishable there") *
            ". Retain a different harmonic set (a full shell rather than a partial one is " *
            "usually the fix), or supply your own stacking shifts."))
    end
    @warn "registry_shifts: no index-$S sublattice makes the retained set a complete " *
          "residue system; returning a low-discrepancy stacking set with cond(E) = " *
          "$(round(sv[1] / sv[end]; sigdigits=3)). The harmonic solve is well-posed but " *
          "not a plain DFT, so out-of-set truncation leakage is amplified by that factor."
    return ds
end

### F is a complete residue system modulo the sublattice Mℤ² ⟺ pairwise differences never
### land in Mℤ² -- checked exactly on the integers: x ∈ Mℤ² ⟺ adj(M)·x ≡ 0 (mod det M)
function _complete_residues(F::Vector{SVector{2,Int}}, M::SMatrix{2,2,Int})::Bool
    detM, adjM = _int_det(M), _int_adjugate(M)
    ### `x ∈ Mℤ²` is exactly `_int_solve`'s solvability, so the membership test is the solver's
    ### and not a second spelling of `adj(M)·x ≡ 0 (mod det M)` living beside it.
    for a in eachindex(F), b in (a+1):length(F)
        isnothing(_int_solve(adjM, detM, F[a] - F[b])) || return false
    end
    return true
end

@inline function _min_wrapped_norm(v::SVector{2,Float64})::Float64
    w = v - round.(v)
    return norm(w)
end




#############################################################
###### The harmonic solve                               #####
#############################################################

"""
    registry_harmonics(scan; gshells=nothing, gvectors=nothing, enforce_hermiticity=true,
                       atol=1e-12, name=scan.name) -> RegistryHarmonics

Solve the registry scan for the k-dependent harmonics `W_{ββ'}(k̄; Ḡ)` -- stage two of the
ERTCA pipeline.

With finitely many computed stackings the extraction is a LINEAR INVERSION, not a full
Fourier transform: for the retained set `F` (exactly one of `gshells=` / `gvectors=`; must
be ±closed and contain `Ḡ = 0`), the sampled Bloch Hamiltonians satisfy

    W(k̄, d_s) = Σ_{Ḡ∈F} e^{−2πi Ḡ·d_s} W(k̄; Ḡ),

solved by the pseudo-inverse of `E[s, m] = e^{−2πi Ḡₘ·d_s}` -- which requires
`S ≥ |F|` stackings (7 for the first shell of a triangular lattice, 13 for two shells; see
[`registry_shifts`](@ref) for the exact-solve sampling, on which this degenerates to the
plain discrete Fourier transform; a fine regular d-grid works the same way). The solve
coefficients are k̄-independent, so it acts on the imported hopping lists DIRECTLY: each
harmonic comes out as an exact finite Bloch sum (a `HarmonicBlock`) -- no k-grid, no
interpolation, no Brillouin-zone wrapping, valid at every momentum.

`enforce_hermiticity` (default true) symmetrizes the solve coefficients over each ±Ḡ pair
so `W_{β'β}(−Ḡ) = W_{ββ'}(Ḡ)†` holds bit-exactly (the residual it removes -- zero for exact
sampling, roundoff otherwise -- is recorded in the result's `hermiticity` field). Harmonics
OUTSIDE the retained set contaminate the solved ones (truncation leakage / aliasing);
measure it with [`reconstruction_residual`](@ref) on held-out stackings and the decay of
[`harmonic_decay`](@ref). Accumulated Bloch-sum entries below `atol` are pruned.
"""
function registry_harmonics(scan::RegistryScan{D};
                            gshells::Union{Nothing,Int}=nothing,
                            gvectors::Union{Nothing,AbstractVector}=nothing,
                            enforce_hermiticity::Bool=true,
                            atol::Float64=1e-12,
                            name::AbstractString=scan.name)::RegistryHarmonics where {D}
    F = _retained_set(scan.lattice, gshells, gvectors)
    S = n_stackings(scan)
    nb = n_orbitals(scan)
    S >= length(F) || throw(ArgumentError(
        "The retained set holds $(length(F)) harmonics but the scan samples only $S " *
        "stacking(s); the inversion needs S ≥ $(length(F)). Compute more stackings " *
        "(`registry_shifts` proposes an exact-solve set) or retain fewer shells."))

    ### the design matrix and its pseudo-inverse (the solve coefficients are k̄-independent)
    E = [cispi(-2 * dot(F[m], scan.d[s])) for s in 1:S, m in eachindex(F)]
    sv = svdvals(E)
    condE = sv[1] / max(sv[end], eps() * sv[1])
    if sv[end] <= 1e-10 * sv[1]
        pair = _aliased_pair(E, F)
        throw(ArgumentError(
            "The harmonic system is rank-deficient on the sampled stackings" *
            (isnothing(pair) ? "" :
             ": harmonics $(Tuple(pair[1])) and $(Tuple(pair[2])) are indistinguishable " *
             "(aliased) there") *
            ". Sample stackings that separate the retained harmonics -- " *
            "`registry_shifts` constructs such a set -- or drop the aliased harmonic."))
    end
    ### an ill-conditioned (but not singular) system amplifies out-of-set truncation leakage
    ### by ~cond(E) -- typically a NEAR-duplicate stacking (rows nearly equal; the aliased-
    ### pair message above cannot see it, that check compares columns)
    if condE > 50
        dmin, smin, tmin = Inf, 0, 0
        for a in 1:S, b in (a+1):S
            v = _min_wrapped_norm(scan.d[a] - scan.d[b])
            v < dmin && ((dmin, smin, tmin) = (v, a, b))
        end
        @warn "registry_harmonics: the stacking system is ill-conditioned " *
              "(cond = $(round(condE; sigdigits=3))); harmonics outside the retained set " *
              "leak into the solved ones amplified by that factor. The closest stacking " *
              "pair is ($smin, $tmin) at wrapped distance $(round(dmin; sigdigits=3)) -- " *
              "a near-duplicate d is the usual cause. Check " *
              "`reconstruction_residual` on held-out stackings, or drop/recompute one of " *
              "the pair."
    end
    C = pinv(E)                                       # |F| × S

    ### Hermiticity pairing at the COEFFICIENT level: the −Ḡ row must be the conjugate of
    ### the +Ḡ row (exact in exact arithmetic; roundoff otherwise). Enforcing it here makes
    ### `W(−Ḡ) = W(Ḡ)†` hold bit-exactly data-side, because the scan's coupling lists are
    ### Hermitian-paired by construction.
    neg = [findfirst(==(-g), F)::Int for g in F]
    herm_res = maximum(abs(C[m, s] - conj(C[neg[m], s])) for m in eachindex(F), s in 1:S)
    if enforce_hermiticity
        C = [(C[m, s] + conj(C[neg[m], s])) / 2 for m in eachindex(F), s in 1:S]
    end

    ### canonical half of the retained set (Ḡ = 0 stays first: F is sorted by |Ḡ|)
    canon = [m for m in eachindex(F) if _g_canonical(F[m])]
    gvecs = F[canon]

    ### accumulate the Bloch-sum data straight off the hopping lists: coupling (i, j, ΔR, w)
    ### of stacking s contributes C[m, s]·w at the atomic separation δ = ΔR + τ_j − τ_i
    acc = [Dict{SVector{2,Float64},ComplexF64}() for _ in canon, _ in 1:nb, _ in 1:nb]
    for s in 1:S
        model = scan.models[s]
        perm = scan.permutations[s]
        c = model.couplings
        pos = state_positions(model.cell)
        for e in eachindex(c.i)
            w = c.w[e]
            iszero(w) && continue
            i0, j0 = c.i[e], c.j[e]
            ### the rounded key merges float-cancellation noise (~1e-16) across stackings
            ### while perturbing stored separations by ≤ 5e-13 -- phase-invisible. `_pz`
            ### normalizes −0.0: Dict lookups (and the ± pairing downstream) use `isequal`,
            ### which distinguishes the two zeros.
            dl = SVector{D,Float64}(c.dR[e]) + pos[j0] - pos[i0]
            key = SVector(_pz(round(dl[1]; digits=12)), _pz(round(dl[2]; digits=12)))
            i, j = perm[i0], perm[j0]
            for (mi, m) in pairs(canon)
                coef = C[m, s]
                iszero(coef) && continue
                d = acc[mi, i, j]
                d[key] = get(d, key, zero(ComplexF64)) + coef * w
            end
        end
    end

    blocks = Array{HarmonicBlock,3}(undef, length(canon), nb, nb)
    for idx in eachindex(IndexCartesian(), acc)
        entries = sort!([(k, v) for (k, v) in acc[idx] if abs(v) > atol];
                        by=kv -> (kv[1][1], kv[1][2]))
        blocks[idx] = HarmonicBlock([kv[1] for kv in entries], [kv[2] for kv in entries])
    end

    ### the reference orbital positions at d = 0 (in-plane, monolayer-fractional): the comb
    ### offsets the folded-channel phases need. Reference orbital β sits, in stacking 1, at
    ### the state its permutation maps to β; shifted orbitals subtract the declared d₁.
    pos1 = state_positions(scan.models[1].cell)
    ip1 = invperm(scan.permutations[1])
    tau = map(1:nb) do b
        p2 = SVector(pos1[ip1[b]][1], pos1[ip1[b]][2])
        t = scan.shifted[b] ? p2 - scan.d[1] : p2
        SVector(_pz(t[1] - floor(t[1])), _pz(t[2] - floor(t[2])))
    end

    return RegistryHarmonics(scan.lattice, copy(scan.labels), copy(scan.shifted), tau,
                             scan.layer_names, gvecs, blocks, condE, herm_res,
                             copy(scan.d), String(name))
end

"""
    _solve_coefficients(h) -> (F, rows, C)

The harmonics' full retained set `F` (canonical ∪ negatives), the map `rows[m] = (iG, sgn)`
into the canonical storage, and the solve coefficients `C = pinv(E)` with the Hermiticity
averaging applied -- rebuilt deterministically from the object (used by the folded-channel
emission to weigh alias branches: `α_m(Ḡ) = Σ_s C[m,s]e^{−2πiḠ·d_s}`, unimodular exactly
when the stacking set cannot distinguish Ḡ from the retained representative `m`).
"""
function _solve_coefficients(h::RegistryHarmonics)
    F = SVector{2,Int}[]
    rows = Tuple{Int,Int}[]
    for (iG, g) in pairs(h.gvecs)
        push!(F, g)
        push!(rows, (iG, 1))
        if !iszero(g)
            push!(F, -g)
            push!(rows, (iG, -1))
        end
    end
    E = [cispi(-2 * dot(F[m], h.d[s])) for s in eachindex(h.d), m in eachindex(F)]
    C = pinv(E)
    neg = [findfirst(==(-g), F)::Int for g in F]
    C = [(C[m, s] + conj(C[neg[m], s])) / 2 for m in eachindex(F), s in eachindex(h.d)]
    return F, rows, C
end

### −0.0 → 0.0 (isequal-based containers distinguish the two zeros)
@inline _pz(x::Float64) = x == 0.0 ? 0.0 : x

### name one aliased pair on rank deficiency: two columns equal up to a phase
function _aliased_pair(E::Matrix{ComplexF64}, F::Vector{SVector{2,Int}})
    S = size(E, 1)
    for a in axes(E, 2), b in (a+1):size(E, 2)
        abs(abs(dot(view(E, :, a), view(E, :, b))) - S) <= 1e-8 * S && return (F[a], F[b])
    end
    return nothing
end




#############################################################
###### HDF5 round trip                                  #####
#############################################################

"""
    h5_write(filename, h::RegistryHarmonics; note="") -> filename

Write the harmonics to HDF5 (`/registry_harmonics` group + the standard `/producer` stamp).
The stored Bloch-sum numbers ARE the harmonics, so the round trip through
`h5_read(filename, RegistryHarmonics)` is exact.
"""
function h5_write(filename::AbstractString, h::RegistryHarmonics;
                  note::AbstractString="")::String
    h5open(filename, "w") do file
        _h5_write_producer!(file, note)
        g = create_group(file, "registry_harmonics")
        write(g, "format_version", 1)
        write(g, "name", h.name)
        write(g, "lattice_a", Matrix{Float64}(h.lattice.a))
        write(g, "labels", [String(l) for l in h.labels])
        write(g, "shifted", Int.(collect(h.shifted)))
        write(g, "tau", reduce(hcat, h.tau; init=zeros(Float64, 2, 0)))
        write(g, "layer_names", [String(h.layer_names[1]), String(h.layer_names[2])])
        write(g, "gvecs", reduce(hcat, h.gvecs; init=zeros(Int, 2, 0)))
        write(g, "d", reduce(hcat, h.d; init=zeros(Float64, 2, 0)))
        write(g, "cond", h.cond)
        write(g, "hermiticity", h.hermiticity)
        ### one flat entry table: (iG, i, j) per Bloch-sum entry, plus δ and the amplitude
        iGs = Int[]; is = Int[]; js = Int[]
        deltas = SVector{2,Float64}[]; amps = ComplexF64[]
        for iG in axes(h.blocks, 1), i in axes(h.blocks, 2), j in axes(h.blocks, 3)
            b = h.blocks[iG, i, j]
            for m in eachindex(b.amp)
                push!(iGs, iG); push!(is, i); push!(js, j)
                push!(deltas, b.delta[m]); push!(amps, b.amp[m])
            end
        end
        write(g, "entry_g", iGs)
        write(g, "entry_i", is)
        write(g, "entry_j", js)
        write(g, "entry_delta", reduce(hcat, deltas; init=zeros(Float64, 2, 0)))
        write(g, "entry_amp_re", real.(amps))
        write(g, "entry_amp_im", imag.(amps))
    end
    return String(filename)
end

"""
    h5_read(filename, RegistryHarmonics) -> RegistryHarmonics

Read harmonics written by `h5_write(filename, ::RegistryHarmonics)`.
"""
function h5_read(filename::AbstractString, ::Type{RegistryHarmonics})::RegistryHarmonics
    h5open(filename, "r") do file
        haskey(file, "registry_harmonics") || throw(ArgumentError(
            "`$filename` holds no `/registry_harmonics` group; it is not a harmonics file."))
        g = file["registry_harmonics"]
        v = read(g["format_version"])
        v <= 1 || throw(ArgumentError(
            "`$filename` uses registry-harmonics format v$v; this build reads up to v1 -- " *
            "update the package."))
        lattice = BravaisLattice(Float64.(read(g["lattice_a"])))
        labels = Symbol.(read(g["labels"]))
        shifted = BitVector(read(g["shifted"]) .!= 0)
        taum = Float64.(read(g["tau"]))
        tau = [SVector{2,Float64}(taum[1, m], taum[2, m]) for m in axes(taum, 2)]
        ln = read(g["layer_names"])
        gm = Int.(read(g["gvecs"]))
        gvecs = [SVector{2,Int}(gm[1, m], gm[2, m]) for m in axes(gm, 2)]
        dm = Float64.(read(g["d"]))
        dvec = [SVector{2,Float64}(dm[1, m], dm[2, m]) for m in axes(dm, 2)]
        iGs = Int.(read(g["entry_g"])); is = Int.(read(g["entry_i"])); js = Int.(read(g["entry_j"]))
        deltam = Float64.(read(g["entry_delta"]))
        amps = complex.(Float64.(read(g["entry_amp_re"])), Float64.(read(g["entry_amp_im"])))
        nb = length(labels)
        blocks = [HarmonicBlock(SVector{2,Float64}[], ComplexF64[])
                  for _ in eachindex(gvecs), _ in 1:nb, _ in 1:nb]
        for m in eachindex(iGs)
            b = blocks[iGs[m], is[m], js[m]]
            push!(b.delta, SVector{2,Float64}(deltam[1, m], deltam[2, m]))
            push!(b.amp, amps[m])
        end
        RegistryHarmonics(lattice, labels, shifted, tau, (Symbol(ln[1]), Symbol(ln[2])),
                          gvecs, blocks, Float64(read(g["cond"])),
                          Float64(read(g["hermiticity"])), dvec, String(read(g["name"])))
    end
end
