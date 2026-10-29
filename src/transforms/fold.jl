##################################################################
###### export only the following functions from this module ######
##################################################################

export fold, spectral_weight


#############################################################
###### Brillouin-zone folding                           #####
#############################################################

### `K` is threaded through the SIGNATURE, which is the whole point of putting the class in
### the cell's type. Folding replicates a single plane-wave site's one `G` label into `|det M|`
### coset representatives, and the derivation reads that as a truncation window -- so a k·p
### frame with no Brillouin-zone torus came back as `TruncatedWindow()`, laundering every
### refusal that protects it. Whether `H(k)` is BZ-periodic is a property of the physics and
### not of which basis it is written in, so the parent's answer is the right one.
###
### Until 2026-09-17 this was patched one level up, on the compiled model, by rebuilding
### through the private inner constructor with the parent's flag, which left the invariant
### sometimes derived and sometimes carried. That left the CELL still wrong --
### `kperiodicity(fold(kp_cell))`
### answered `TruncatedWindow()` -- which is the latent second bug the ledger predicted and
### nothing had ever reported, because only the model was ever asked. Here it cannot recur: a
### transform that returns `UnitCell{D,K}` from `UnitCell{D,K}` has nothing left to forget.
"""
    fold(cell::UnitCell{D}, M) -> UnitCell{D}

Fold the Brillouin zone: enlarge the lattice to `a_new = M a_old` while keeping every position
where it is, giving each state `|det M|` copies labelled by the reciprocal coset offsets.

This is the completed form of what was once `supercell(cell, M; mode=:bz_fold)`: in addition to
re-labelling the plane waves it **records the period lattice** `M_s -> M · M_s` on every
localized site, so a term compiled on the folded cell knows that neighbours live on the *fine*
lattice and that its couplings carry `G` in their phase. `hopping(-1.0; shell=1)` on
`fold(graphene, 2I)` therefore compiles the parent's true bonds (at fractional translations of
the new cell) and reproduces the parent's bands exactly, block-diagonal over the cosets --
where the old spelling silently compiled a wrong dimer model.

For plane-wave sites the fold is exactly the old relabelling (the added cosets are new plane
waves); the period stays trivial.

The unfolded and folded models describe the same physics in two bases:
`fold(model, M)` maps a compiled model exactly, and [`spectral_weight`](@ref) computes the
unfolding weights of a (possibly perturbed) real-space supercell against the parent's momenta.
"""
function fold(uc::UnitCell{D,K}, M::AbstractMatrix{<:Integer};
              digits::Int=LATTICE_ROUND_DIGITS,
              atol::Float64=INT_ROUND_ATOL)::UnitCell{D,K} where {D,K}
    size(M) == (D, D) || throw(DimensionMismatch("Fold matrix must be $D×$D."))
    Ms = SMatrix{D,D,Int}(M)
    new_lat = supercell(uc.lattice, Ms)
    ### `nperiodic=` is load-bearing, not decoration: without it the basis rebuild defaults to
    ### D and wraps the APERIODIC coordinate into [0, 1), while the very next line rebases the
    ### sites with the lattice's true count. The two then disagree and the `UnitCell`
    ### constructor refuses the cell -- so `fold` was unusable on every ribbon and slab, while
    ### `supercell` on the same cell worked, because it passes this (supercell.jl:278).
    new_basis = _fold_basis(uc.basis, Ms; digits=digits, atol=atol,
                            nperiodic=nperiodic(uc.lattice))
    sites = _fold_sites(uc, Ms; digits=digits)
    kinds = copy(uc.site_kind)
    periods = SMatrix{D,D,Int}[uc.site_kind[s] === :localized ?
                                   SMatrix{D,D,Int}(Ms * uc.site_period[s]) :
                                   SMatrix{D,D,Int}(uc.site_period[s])
                               for s in eachindex(uc.site_kind)]
    return UnitCell{D,K}(new_lat, new_basis; schema=uc.schema, sites=sites,
                         site_kind=kinds, site_period=periods, atol=uc.atol)
end

### `UnitCell{D,K}` and not `UnitCell{D}`: they denote the same set, but Julia cannot rank
### them against each other, so the integer method above would be AMBIGUOUS for a `Matrix{Int}`
### rather than more specific. Spelling the first argument identically in both leaves the
### ranking to argument two, where `<:Integer` is strictly narrower than `<:Real`.
"""
    fold(uc::UnitCell{D,K}, new_b::AbstractMatrix{<:Real}; kwargs...) -> UnitCell{D,K}

Name the fold by the new RECIPROCAL basis instead of by the integer matrix: `new_b` must be an
integer sublattice of `uc.lattice.b`, and the integer `M` is recovered from it.

This is the reciprocal-space counterpart of `supercell(cell, new_a)`, which names a real-space
supercell by its new DIRECT vectors. It is a method rather than a mode because a fold and a
supercell are different operations -- and it only became writable once they were separate
functions: as `supercell(cell, new_A; mode=:bz_fold)` the same call had to route a reciprocal
basis through a function whose other branch read the argument as a direct one.
"""
function fold(uc::UnitCell{D,K}, new_b::AbstractMatrix{<:Real};
              atol::Float64=INT_ROUND_ATOL, kwargs...) where {D,K}
    new_lat = supercell_reciprocal(uc.lattice, new_b; atol=atol)
    M = _supercell_integer_matrix(new_lat.a, uc.lattice.a; atol=atol)
    return fold(uc, Matrix{Int}(M); atol=atol, kwargs...)
end

"""
    fold(model::HamiltonianModel{D}, M) -> HamiltonianModel{D}

Fold a compiled model: each coupling becomes one copy per reciprocal coset `c`,

    (i, j, ΔR, q, w, κ)  ->  (i_c, j_c, M⁻ᵀ ΔR, g_c + M q, w, κ)

with `g_c` the integer coset offset and `new_index(i, c) = (c-1)·n + i` (`STATE_ORDER`). The
mapping is exact: the parent phase `e^{2πi(k+q)·δ}` at `k = M⁻¹(k' + g_c)` equals
`e^{2πi(k'+g_c+Mq)·M⁻ᵀδ}` identically, and the kernel argument `bᵀ(k+q)` is preserved -- so
`H_folded(k')` is block-diagonal over the cosets, block `c` being the parent's `H(M⁻¹(k'+g_c))`
in relabelled indices. The `bands`-equality against the parent is the acceptance test.

The user-facing momentum mapping, in one line (identical for `fold` and `supercell`; getting
it wrong fails silently as a physics error, so here it is explicitly): fractional momenta
transform as `k_new = M * k_old`, and

    bands(fold(m, M), k_new) == sort(vcat([bands(m, inv(M) * k_new .+ g)
        for g in coset_representatives(M; space=:reciprocal)]...))
"""
function fold(model::HamiltonianModel{D}, M::AbstractMatrix{<:Integer};
              digits::Int=LATTICE_ROUND_DIGITS,
              atol::Float64=INT_ROUND_ATOL)::HamiltonianModel{D} where {D}
    size(M) == (D, D) || throw(DimensionMismatch("Fold matrix must be $D×$D."))
    Ms = SMatrix{D,D,Int}(M)
    new_cell = fold(model.cell, Ms; digits=digits, atol=atol)

    ### the reciprocal coset representatives, in the SAME order `supercell(basis, :bz_fold)`
    ### replicated the states in -- the index bookkeeping below depends on that
    lp = _latticepoints_in_supercell(collect(transpose(Ms)); digits=digits)
    nc = length(lp)
    nold = n_states(model.cell)
    Mf = SMatrix{D,D,Float64}(Ms)
    MinvT = _int_invT(Ms)
    gs = [canonicalize_int(Mf * lp[c]; atol=atol) for c in 1:nc]

    new_index(iold::Int, c::Int) = (c - 1) * nold + iold

    ### The folded cell stores positions WRAPPED into [0,1)ᴰ while `M⁻ᵀ ΔR` below maps the
    ### translations against the unwrapped rebase `M⁻ᵀτ`. Each state's integer wrap shift
    ### `σ = stored − M⁻ᵀτ` must be added back so the list stays consistent with the stored
    ### positions (bands cannot see the mismatch -- a k-dependent diagonal unitary -- but the
    ### physical displacements `ΔR + τ_j − τ_i` and everything built on them can).
    old_pos = state_positions(model.cell)
    new_pos = state_positions(new_cell)
    MinvTf = SMatrix{D,D,Float64}(MinvT)
    sigma = [canonicalize_int(new_pos[s] - MinvTf * old_pos[s]; atol=atol) for s in 1:nold]

    c0 = model.couplings
    buf, kids = adopted_buffer(c0; per_canonical=nc)
    ### canonical halves only; the outer constructor's finalize re-materializes the partners
    ### (mapping both directions would coalesce each coupling with its own image)
    for cpl in each_canonical(c0)
        dR_new = SVector{D,Rational{Int}}(MinvT * cpl.dR) + sigma[cpl.i] - sigma[cpl.j]
        for c in 1:nc
            q_new = SVector{D,Float64}(gs[c]) + Mf * cpl.q
            push_coupling!(buf, new_index(cpl.i, c), new_index(cpl.j, c), dR_new, q_new,
                           cpl.w, kids[cpl.kernel], cpl.term)
        end
    end

    ### The parent's k-periodicity, PROPAGATED rather than re-derived. A fold is a change of
    ### basis: it cannot make a Hamiltonian Brillouin-zone periodic that was not. Re-deriving
    ### from the folded cell gets a k·p frame wrong -- its single plane-wave site acquires
    ### |det M| coset labels, which the derivation reads as a truncation window and reports as
    ### `:truncated`. `chern_number` then answers 0 for a model with no torus at all instead of
    ### refusing, and `density_of_states` and `spectrum_along_line` are laundered the same way.
    ### The parent's k-periodicity, PROPAGATED rather than re-derived. A fold is a change of
    ### basis: it cannot make a Hamiltonian Brillouin-zone periodic that was not. Re-deriving
    ### from the folded cell gets a k·p frame wrong -- its single plane-wave site acquires
    ### |det M| coset labels, which `_derive_kperiodicity` reads as a truncation window and
    ### reports as `:truncated`. `chern_number` then answers 0 for a model with no torus at all
    ### instead of refusing, and `density_of_states` and `spectrum_along_line` are laundered
    ### the same way.
    ###
    ### No correction step any more: `new_cell` came from `fold(uc::UnitCell{D,K})`, which
    ### preserves `K` by signature, and the model reads its class off the cell's type. This is
    ### where the compiled model used to be rebuilt through the private inner constructor to
    ### overwrite a flag the derivation had got wrong.
    return HamiltonianModel(new_cell, buf; gauge=model.gauge, name=model.name)
end


#############################################################
###### Spectral unfolding                               #####
#############################################################

"""
    spectral_weight(sc::HamiltonianModel{D}, parent::UnitCell{D}, M, k_new)
        -> (energies, weights, kpoints)

Unfold a real-space supercell's spectrum at folded momentum `k_new` onto the parent's momenta:
the standard band-unfolding weights

    W[c, n] = Σ_i | ⟨ i, k_old(c) | ψ_n(k_new) ⟩ |² ,     k_old(c) = M⁻¹ k_new + lp_c ,

where `sc` is a model on `supercell(parent_cell, M)` (possibly perturbed -- a defect, disorder),
`ψ_n` its eigenvectors at `k_new`, and `lp_c` the reciprocal coset representatives. For an
unperturbed supercell each eigenvector projects entirely onto one coset (`W` is 0/1 up to
degeneracies) and `energies` reproduce the parent bands at the `kpoints`; a perturbation spreads
the weight -- the unfolded band structure.

Returns:
--------
- `energies::Vector{Float64}`: the `n_states(sc)` eigenvalues at `k_new`, ascending
- `weights::Matrix{Float64}`: `|det M| × n_states(sc)`; column `n` sums to 1
- `kpoints::Vector{SVector{D,Float64}}`: the parent fractional momenta `k_old(c)`

Notes:
------
The projection uses the atomic-gauge Bloch sums (positions attached to the orbitals), so the
model is regauged if needed. The parent must be a localized (`G ≡ 0`) cell -- tight-binding
unfolding; the supercell's state ordering must be the `STATE_ORDER` one `supercell` produces
(`new_index(i, l) = (l-1)·n_parent + i`).
"""
function spectral_weight(sc::HamiltonianModel{D}, parent::UnitCell{D},
                         M::AbstractMatrix{<:Integer},
                         k_new::AbstractVector{<:Real}) where {D}
    Ms = SMatrix{D,D,Int}(M)
    all(iszero(b.G) for b in parent.basis) || throw(ArgumentError(
        "`spectral_weight` unfolds tight-binding supercells: the parent cell must carry G ≡ 0."))
    ### a FOLDED model passes the state-count check but cannot be unfolded this way: each folded
    ### eigenvector lives on one reciprocal-coset block, so the real-space coset projection below
    ### cannot distinguish branches and returns flat 1/|det M| weights -- silently meaningless
    (is_folded(sc.cell) || any(!iszero(b.G) for b in sc.cell.basis)) && throw(ArgumentError(
        "`spectral_weight` unfolds REAL-SPACE supercells (`supercell(model, M)`); this model is " *
        "a BZ fold (folded cell / non-zero G labels), whose eigenstates live on single " *
        "reciprocal-coset blocks and carry no unfolding information here. A folded model's " *
        "spectrum at k' is exactly the union of parent spectra over the coset branches -- " *
        "unfold by evaluating the parent, or pass the real-space `supercell` model instead."))
    ### M must actually BE the supercell relation between the two lattices: any matrix with the
    ### right |det| passes the state-count check below but projects onto the wrong cosets,
    ### returning column-normalized garbage weights with no error
    isapprox(Matrix(sc.cell.lattice.a), Ms * Matrix(parent.lattice.a);
             atol=LATTICE_ATOL * length_scale(parent.lattice)) || throw(ArgumentError(
        "`spectral_weight`: the supercell model's lattice is not `M * parent.lattice` for the " *
        "given M = $M. The projection onto parent momenta is only meaningful for the actual " *
        "supercell matrix (the one passed to `supercell(model, M)`)."))
    lattice_points = _latticepoints_in_supercell(Ms)
    lp_recip = _latticepoints_in_supercell(collect(transpose(Ms)))
    N = length(lattice_points)
    nold = n_states(parent)
    n_states(sc.cell) == nold * N || throw(DimensionMismatch(
        "The supercell model has $(n_states(sc.cell)) states; a |det M| = $N supercell of the " *
        "$nold-state parent needs $(nold * N)."))

    A = SMatrix{D,D,Int}(transpose(Ms))
    v = [canonicalize_int(A * lattice_points[l]; atol=INT_ROUND_ATOL) for l in 1:N]

    scm = sc.gauge === :atomic ? sc : change_gauge(sc, :atomic)
    ### the coset projection reads the supercell-position content of the atomic eigenvectors
    ### (the per-orbital phase cancels in |·|², the per-copy one does not): with every localized
    ### state at the origin that content is absent and the weights are convention artifacts --
    ### the observable-layer relocation of the Wannier90 importer's missing-centres diagnostic
    if _all_localized_at_origin(scm.cell)
        @warn "Every localized state of the supercell model sits at the origin, so its atomic " *
              "assembly coincides with the zero-position (periodic) convention and the " *
              "unfolding weights carry no position information -- they are convention " *
              "artifacts, not spectral weights. A genuine `supercell(model, M)` spreads its " *
              "states over the parent copies; for a Wannier90 import, re-import with " *
              "`centres=` so the states sit at the true Wannier centres."
    end
    F = eigen(Hermitian(bloch_hamiltonian(scm, k_new)))
    E = F.values
    ψ = F.vectors

    kf = SVector{D,Float64}(k_new)
    Minv = inv(SMatrix{D,D,Float64}(Ms))
    kpoints = [SVector{D,Float64}(Minv * kf) + lp_recip[c] for c in 1:N]

    ### The projector phase factorizes, cispi(−2 lp_c·(v_l + τ_i)) = α_{c,l} · β_{c,i}, and the
    ### per-orbital factor β_{c,i} cancels inside |·|² -- so the former 4-deep loop (which
    ### evaluated nold²·N³ transcendentals for N² distinct phases) reduces to ONE BLAS-3
    ### product of the eigenvector block against the N × N coset-phase matrix.
    ntot = nold * N
    alpha = ComplexF64[cispi(-2 * dot(lp_recip[c], SVector{D,Float64}(v[l])))
                       for c in 1:N, l in 1:N]
    ### ψ[(l−1)·nold + i, n] reshaped to Ψ[i, l, n], then permuted so the l-contraction is a
    ### plain matrix product over the trailing index
    psi_p = reshape(permutedims(reshape(ψ, nold, N, ntot), (1, 3, 2)), nold * ntot, N)
    proj = psi_p * transpose(alpha)                    # (i, n) rows × coset columns
    W = Matrix{Float64}(undef, N, ntot)
    for c in 1:N
        Pc = reshape(view(proj, :, c), nold, ntot)
        for n in 1:ntot
            acc = 0.0
            @inbounds for i in 1:nold
                acc += abs2(Pc[i, n])
            end
            W[c, n] = acc / N
        end
    end
    return (energies=E, weights=W, kpoints=kpoints)
end
