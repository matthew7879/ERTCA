# =====================================================================================
#  Analysis utilities for the ERTCA moiré METAL examples
#
#  A moiré metal has no gap to hide behind: the Fermi level sits in the middle of a
#  densely folded manifold (4N bands for N moiré cells per layer), so a plain band plot
#  is a wall of ink. The right observable is the UNFOLDED spectral function
#
#      A_ℓ(κ, E) = Σ_n Σ_{α ∈ ℓ} |⟨κ, (α,ℓ) | ψ_n(k)⟩|²  δ(E − E_n(k)),
#
#  resolved on the momentum κ of the UNTWISTED MONOLAYER Brillouin zone. The ERTCA basis
#  makes this exact and free: its states ARE monolayer Bloch waves |k + G, (α,ℓ)⟩, so the
#  eigenvector amplitude on one basis state IS the unfolding weight — no projection
#  integral, no supercell overlap. Two facts organize the bookkeeping:
#
#    * κ = b_mᵀ(k + G) is the monolayer momentum of basis state (α, ℓ, G), in the LAB
#      frame (what ARPES sees). The layer's own frame is L_ℓ⁻¹κ.
#    * plane-wave labels of layer ℓ are physical only modulo that layer's reciprocal
#      lattice, the integer sublattice M_ℓℤ² of the moiré labels (the momentum COMB of
#      Eq. (4)). `monolayer_bz_gvectors(tb, ℓ)` holds exactly one representative per
#      coset, so each κ has exactly ONE carrier state per orbital — found here by an
#      exact integer coset label, never by float comparison.
#
#  Two complementary sweeps, both provided:
#    forward  (`unfolded_fermi_map`)  loop the moiré k-grid, harvest all N·n_k² momenta κ
#                                     at once — the cheap way to a 2D map at fixed energy;
#    reverse  (`unfolded_spectral_cut`) pick κ, solve the one moiré k that carries it —
#                                     the cheap way to A(κ, E) along a path.
#
#  `include` this after `loader.jl`.
# =====================================================================================

using SingleParticleModels
using LinearAlgebra
using StaticArrays
import SingleParticleModels: _harmonic_value, _rotation2, threaded_chunks, HarmonicBlock

### ---------------------------------------------------------------------------------------
### The uniform-registry (Ḡ = 0) reference bilayer
### ---------------------------------------------------------------------------------------

"""
    average_bilayer(h) -> (k_frac -> 4x4 Hermitian)

The Ḡ = 0 registry harmonic as a Bloch Hamiltonian: the d-AVERAGED commensurate bilayer,
`W(k̄; Ḡ=0) = ∫ d²d/A_c W(k̄, d)`. This is the two-centre baseline the ERTCA improves on --
truncating Eq. (9) to `g = 0` leaves exactly this -- and it is the reference every "what did
the moiré do?" comparison in these examples is made against.
"""
function average_bilayer(h::RegistryHarmonics)
    iG0 = findfirst(iszero, h.gvecs)
    isnothing(iG0) && error("the harmonics carry no Ḡ = 0 term")
    nb = length(h.labels)
    return k -> Hermitian([_harmonic_value(h, iG0, i, j, SVector{2,Float64}(k))
                           for i in 1:nb, j in 1:nb])
end

"""
    zero_interlayer(h) -> RegistryHarmonics

The same harmonics with every INTERLAYER block emptied — two layers that still carry their
own (rotated) dispersion and their own intralayer moiré potential, but do not talk to each
other. Realizing this and then ablating the `Ḡ ≠ 0` shells leaves the cleanest possible
starting point: two rigidly rotated, completely decoupled monolayers, whose Fermi surfaces
cross somewhere in the untwisted zone with nothing to gap them.

The layer split is the harmonics' own `shifted` flag, so this needs no geometry.
"""
function zero_interlayer(h::RegistryHarmonics)
    nb = length(h.labels)
    blocks = similar(h.blocks)
    for iG in axes(h.blocks, 1), i in 1:nb, j in 1:nb
        blocks[iG, i, j] = h.shifted[i] == h.shifted[j] ? h.blocks[iG, i, j] :
                           HarmonicBlock(SVector{2,Float64}[], ComplexF64[])
    end
    return RegistryHarmonics(h.lattice, copy(h.labels), copy(h.shifted), copy(h.tau),
                             h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                             copy(h.d), h.name * " (layers decoupled)")
end

"""
    spin_align(h, kbar) -> (h_aligned, info)

Rotate each layer's CO-CENTRED Wannier pair into the eigenbasis of its own `Ḡ = 0` block at
`kbar`, ordered by energy. At the K point of a TMD the only spin-orbit term is Ising (σ_z), so
S_z is a good quantum number there and this rotation is exactly "label the two Wannier
functions spin-up and spin-down".

One CONSTANT unitary is applied to every harmonic, `W → U† W U`, not a per-stacking fit. That
is legitimate here and cheaper than refitting each Wannierization: the per-d gauge is already
consistent (checked separately — the interlayer tripod at K agrees in phase to <1°, which a
drifting per-d phase would destroy), so a single rotation aligns the whole family. `U` is
block-diagonal within each co-centred pair, so it mixes only functions at identical τ̄ and
leaves every separation δ — hence every Bloch phase — untouched.

`info` reports the rotation angle per layer and the residual spin-flip element, so the
approximation is inspectable rather than assumed.
"""
function spin_align(h::RegistryHarmonics, kbar)
    nb = length(h.labels)
    iG0 = findfirst(iszero, h.gvecs)
    W0 = [_harmonic_value(h, iG0, i, j, SVector{2,Float64}(kbar)) for i in 1:nb, j in 1:nb]
    U = zeros(ComplexF64, nb, nb)
    info = NamedTuple[]
    for lay in (false, true)
        idx = findall(==(lay), h.shifted)
        B = Hermitian(W0[idx, idx])
        F = eigen(B)
        V = F.vectors
        ### fix each column's phase on its largest component, so U is deterministic
        for c in axes(V, 2)
            V[:, c] .*= cis(-angle(V[argmax(abs.(V[:, c])), c]))
        end
        U[idx, idx] = V
        push!(info, (layer=lay ? h.layer_names[2] : h.layer_names[1],
                     mixing_deg=rad2deg(atan(abs(V[2, 2]), abs(V[1, 2]))),
                     splitting_meV=1000 * (F.values[end] - F.values[1]),
                     spinflip_meV=1000 * abs(W0[idx, idx][1, 2])))
    end
    ### W → U† W U on every stored harmonic. The Bloch sums share their δ list, so the rotation
    ### is a linear recombination of AMPLITUDES at fixed separations -- no new δ appear.
    blocks = similar(h.blocks)
    for iG in axes(h.blocks, 1)
        ### collect the union of separations for this harmonic, then rotate amplitude vectors
        deltas = SVector{2,Float64}[]
        seen = Dict{SVector{2,Float64},Int}()
        for i in 1:nb, j in 1:nb, d in h.blocks[iG, i, j].delta
            haskey(seen, d) && continue
            push!(deltas, d); seen[d] = length(deltas)
        end
        A = zeros(ComplexF64, length(deltas), nb, nb)
        for i in 1:nb, j in 1:nb
            b = h.blocks[iG, i, j]
            for m in eachindex(b.amp)
                A[seen[b.delta[m]], i, j] += b.amp[m]
            end
        end
        for i in 1:nb, j in 1:nb
            amp = ComplexF64[]
            dl = SVector{2,Float64}[]
            for n in eachindex(deltas)
                v = zero(ComplexF64)
                for p in 1:nb, q in 1:nb
                    v += conj(U[p, i]) * A[n, p, q] * U[q, j]
                end
                abs(v) > 1e-14 && (push!(dl, deltas[n]); push!(amp, v))
            end
            blocks[iG, i, j] = HarmonicBlock(dl, amp)
        end
    end
    hh = RegistryHarmonics(h.lattice, copy(h.labels), copy(h.shifted), copy(h.tau),
                           h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                           copy(h.d), h.name * " (spin-aligned)")
    return hh, info
end

"""
    select_orbitals(h, idx) -> RegistryHarmonics

Keep only the orbitals `idx`, dropping their couplings to the rest. Valid when the dropped
sector is far in energy: after [`spin_align`](@ref) at K the two spin sectors of NbSe₂ are
151 meV apart while the largest coupling between them is ≲6.5 meV, so the induced error is
~6.5²/151 ≈ 0.3 meV. Use it to reduce a spinful 4-orbital family to the ONE spin sector that
a spinless Wu-MacDonald model describes.
"""
function select_orbitals(h::RegistryHarmonics, idx::AbstractVector{Int})
    blocks = [h.blocks[iG, i, j] for iG in axes(h.blocks, 1), i in idx, j in idx]
    return RegistryHarmonics(h.lattice, h.labels[idx], h.shifted[idx], h.tau[idx],
                             h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                             copy(h.d), h.name * " (orbitals $(Tuple(idx)))")
end

"""
    scale_harmonics(h; interlayer_g0=1.0, moire=1.0) -> RegistryHarmonics

Rescale parts of the harmonic set, so the model can be turned on continuously from nothing.

Three groups, separated because they are three different physical terms:

| group | what it is | kept at 1 by default |
|---|---|---|
| `Ḡ = 0`, intralayer | the bare ROTATED monolayer dispersion of each layer | always |
| `Ḡ = 0`, interlayer | the d-averaged uniform tunnelling — the only interlayer term a two-centre model at `g = 0` has | `interlayer_g0` |
| `Ḡ ≠ 0`, all | the moiré harmonics, intra- and interlayer | `moire` |

The `Ḡ = 0` intralayer group is never scaled: switching it off would delete the kinetic energy,
since in this construction there is no separate kinetic term. So `interlayer_g0 = moire = 0`
leaves exactly **two rigidly rotated, decoupled monolayers** — the natural λ = 0 of a sweep —
and `= 1` restores the full model. λ is of course not physical; it exists so the reconstruction
can be watched developing instead of met all at once.
"""
function scale_harmonics(h::RegistryHarmonics; interlayer_g0::Real=1.0, moire::Real=1.0)
    nb = length(h.labels)
    blocks = similar(h.blocks)
    for iG in axes(h.blocks, 1), i in 1:nb, j in 1:nb
        inter = h.shifted[i] != h.shifted[j]
        s = iszero(h.gvecs[iG]) ? (inter ? Float64(interlayer_g0) : 1.0) : Float64(moire)
        b = h.blocks[iG, i, j]
        blocks[iG, i, j] = s == 1 ? b : HarmonicBlock(copy(b.delta), s .* b.amp)
    end
    return RegistryHarmonics(h.lattice, copy(h.labels), copy(h.shifted), copy(h.tau),
                             h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                             copy(h.d), h.name * " (λ_inter0=$interlayer_g0, λ_moire=$moire)")
end

"""
    freeze_couplings(h, kbar; interlayer_only=false) -> RegistryHarmonics

Replace retained coupling harmonics W(k̄; Ḡ) by their complex values W(kbar; Ḡ).
`kbar` is in monolayer fractional coordinates: use `V.kbar` for the K valley,
not the Cartesian plane-wave origin `V.expansion`.

By default freeze all interlayer harmonics (including Ḡ = 0) and the nonzero-Ḡ
intralayer moiré potentials. With `interlayer_only=true`, freeze only tunnelling
between layers, leaving every intralayer harmonic unchanged.

Always preserve the intralayer Ḡ = 0 dispersion, including any trigonal warping.
The original harmonic set is not modified. Realize the returned harmonics with
the same twist, plane-wave cutoff and expansion as the original valley model.
"""
function freeze_couplings(h::RegistryHarmonics, kbar; interlayer_only::Bool=false)
    k = SVector{2,Float64}(kbar)
    all(isfinite, k) || throw(ArgumentError("The freezing momentum must be finite."))
    nb = length(h.labels)
    blocks = similar(h.blocks)
    for iG in axes(h.blocks, 1), i in 1:nb, j in 1:nb
        b = h.blocks[iG, i, j]
        inter = h.shifted[i] != h.shifted[j]
        freeze = inter || (!interlayer_only && !iszero(h.gvecs[iG]))
        blocks[iG, i, j] = isempty(b) || !freeze ? b :
            HarmonicBlock([SVector(0.0, 0.0)], [_harmonic_value(h, iG, i, j, k)])
    end
    scope = interlayer_only ? "interlayer tunnelling" : "couplings"
    return RegistryHarmonics(h.lattice, copy(h.labels), copy(h.shifted), copy(h.tau),
                             h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                             copy(h.d), h.name * " ($scope frozen at k̄=$(Tuple(k)))")
end

"""
    layer_dispersion(h, theta_top) -> (layer, kappa_cart) -> Vector{Float64}

The bare dispersion of one ROTATED layer at a LAB-frame cartesian momentum, straight from the
`Ḡ = 0` harmonic: `ε_ℓ(κ) = eig W(b̄⁻ᵀ L_ℓ⁻¹ κ; 0)|_ℓ`. No continuum model and no plane-wave
basis is involved, so this is the exact reference for "two rotated layers, no coupling" and
costs a 2×2 diagonalization per momentum — cheap enough for a fine map.

`layer` is 1 for the unshifted layer (rotated by −θ/2, so `L₁⁻¹ = R(+θ/2)`) and 2 for the
shifted one. The two curves are DEGENERATE wherever κ lies on a mirror line of the monolayer
(Γ→M and Γ→K): there `R(±θ/2)κ` are mirror images and the dispersion cannot tell them apart.
Those are exactly the momenta where the two rotated Fermi surfaces cross.
"""
function layer_dispersion(h::RegistryHarmonics, theta_top::Real)
    W0 = average_bilayer(h)
    binv = inv(transpose(h.lattice.b))
    idx = (findall(.!h.shifted), findall(h.shifted))
    rot = (_rotation2(+Float64(theta_top)), _rotation2(-Float64(theta_top)))
    return (l, kc) -> begin
        kf = binv * (rot[l] * SVector{2,Float64}(kc))
        eigvals(Hermitian(Matrix(W0(kf))[idx[l], idx[l]]))
    end
end

"""
    fermi_level(h; nk=180, filling) -> Float64

The Fermi level of the uniform-registry bilayer at the given filling of the Wannier
manifold, from the electron count alone -- nothing is fitted. `filling = 1/2` puts two
electrons in the four NbSe₂ spinor states (Nb⁴⁺ is d¹, two Nb per bilayer cell).
"""
function fermi_level(h::RegistryHarmonics; nk::Int=180, filling::Real)
    W0 = average_bilayer(h)
    nb = length(h.labels)
    Es = Float64[]
    for i in 0:nk-1, j in 0:nk-1
        append!(Es, eigvals(W0(SVector((i + 0.5) / nk, (j + 0.5) / nk))))
    end
    sort!(Es)
    m = clamp(round(Int, filling * length(Es)), 1, length(Es) - 1)
    return (Es[m] + Es[m+1]) / 2
end

### ---------------------------------------------------------------------------------------
### The unfolding map: monolayer momentum κ  <->  (moiré k, plane-wave label G)
### ---------------------------------------------------------------------------------------

"""
    UnfoldMap(h, tb, model; top=2)

The exact κ ↔ (k, G) dictionary for an ERTCA commensurate model. Fields used by the sweeps:
`bmT`/`bminv` (moiré reciprocal and its inverse), `layer_of[β]`, `kappa_of[state]` offsets,
and `state_of[(β, G)]`. Coset membership is resolved on the INTEGERS
(`adj(M_ℓ)·G mod N`), the same exact label `monolayer_bz_gvectors` uses.
"""
struct UnfoldMap
    bmT::SMatrix{2,2,Float64}
    bminv::SMatrix{2,2,Float64}
    nb::Int
    layer_of::Vector{Int}                                   # orbital -> 1 (unshifted) | 2 (shifted)
    rot_of::NTuple{2,SMatrix{2,2,Float64}}                  # L_ℓ⁻¹ per layer
    coset::Vector{Dict{Tuple{Int,Int},SVector{2,Int}}}      # per layer: label -> representative G
    Mdata::Vector{Tuple{SMatrix{2,2,Int},Int}}              # per layer: (adj M_ℓ, N)
    state_of::Dict{Tuple{Int,Tuple{Int,Int}},Int}           # (orbital, G) -> state index
    gsets::Vector{Vector{SVector{2,Int}}}
end

function UnfoldMap(h::RegistryHarmonics, tb::TwistedBilayer, model::HamiltonianModel{2};
                   top::Int=2)
    nb = length(h.labels)
    ### `ertca_model(h, tb; top)` puts the harmonics' SHIFTED layer on tb layer `top`
    lay_shift = top
    lay_fixed = top == 2 ? 1 : 2
    layer_of = [h.shifted[b] ? lay_shift : lay_fixed for b in 1:nb]
    ### L_ℓ⁻¹: tb layer 1 is rotated by −θ/2, layer 2 by +θ/2
    rot_of = (_rotation2(+tb.theta / 2), _rotation2(-tb.theta / 2))
    coset = [Dict{Tuple{Int,Int},SVector{2,Int}}() for _ in 1:2]
    Mdata = Vector{Tuple{SMatrix{2,2,Int},Int}}(undef, 2)
    gsets = [monolayer_bz_gvectors(tb, l) for l in 1:2]
    for l in 1:2
        M = supercell_matrix(tb, l)
        N = abs(M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        adj = SMatrix{2,2,Int}(M[2, 2], -M[2, 1], -M[1, 2], M[1, 1])
        Mdata[l] = (adj, N)
        for g in gsets[l]
            x = adj * g
            coset[l][(mod(x[1], N), mod(x[2], N))] = g
        end
        length(coset[l]) == N || error(
            "layer $l's tiling set is not one representative per coset ($(length(coset[l])) of $N)")
    end
    state_of = Dict{Tuple{Int,Tuple{Int,Int}},Int}()
    GS = state_g_vectors(model.cell)
    INT = state_internals(model.cell)
    for a in eachindex(GS)
        state_of[(INT[a], (GS[a][1], GS[a][2]))] = a
    end
    return UnfoldMap(SMatrix{2,2,Float64}(transpose(tb.moire.b)),
                     SMatrix{2,2,Float64}(inv(transpose(tb.moire.b))),
                     nb, layer_of, rot_of, coset, Mdata, state_of, gsets)
end

"""
    kappa_of(U, k_frac, G) -> SVector{2,Float64}

The LAB-frame monolayer momentum `κ = b_mᵀ(k + G)` of a plane-wave basis state (cartesian).
"""
@inline kappa_of(U::UnfoldMap, k::SVector{2,Float64}, G::SVector{2,Int}) =
    U.bmT * (k + SVector{2,Float64}(G))

"""
    carriers(U, kappa) -> (k_frac, state_indices)

The one moiré momentum `k` that carries the monolayer momentum `kappa`, and the single basis
state index per orbital that sits at it. This is the REVERSE unfolding map: `k` is `κ` wrapped
into the moiré zone, and each orbital's carrier is the tiling representative CONGRUENT to the
leftover label modulo that layer's reciprocal sublattice -- the comb equivalence of Eq. (4).
"""
function carriers(U::UnfoldMap, kappa::SVector{2,Float64})
    v = U.bminv * kappa
    G0 = floor.(Int, v)
    k = v - SVector{2,Float64}(G0)
    idx = Vector{Int}(undef, U.nb)
    for b in 1:U.nb
        l = U.layer_of[b]
        adj, N = U.Mdata[l]
        x = adj * SVector{2,Int}(G0)
        g = U.coset[l][(mod(x[1], N), mod(x[2], N))]
        idx[b] = U.state_of[(b, (g[1], g[2]))]
    end
    return k, idx
end

### ---------------------------------------------------------------------------------------
### Forward sweep: a 2D map of A(κ, E) at fixed energy -- the unfolded Fermi surface
### ---------------------------------------------------------------------------------------

"""
    unfolded_fermi_map(model, U, E0; nk=16, sigma=0.010, layers=(1,2), kmax=Inf)
        -> (kx, ky, weight, layer)

The unfolded spectral weight at energy `E0`, harvested the CHEAP way: every moiré `k` on an
`nk × nk` grid contributes all of its `N` plane-wave labels at once, so one diagonalization
yields `N` monolayer momenta per orbital. Returns flat vectors -- one entry per distinct
`(κ, layer)` visited -- ready for a scatter/heatmap; `κ` lands on the exact triangular
lattice `b_mᵀ ℤ²/nk`, so the points tile.

`sigma` is the Gaussian energy window (eV) around `E0`; `kmax` trims |κ| for plotting.
"""
function unfolded_fermi_map(model::HamiltonianModel{2}, U::UnfoldMap, E0::Real;
                            nk::Int=16, sigma::Real=0.010, kmax::Real=Inf)
    ns = n_states(model.cell)
    GS = state_g_vectors(model.cell)
    INT = state_internals(model.cell)
    kl = [(i, j) for i in 0:nk-1, j in 0:nk-1]
    acc = [Dict{Tuple{Int,Int},Float64}() for _ in 1:2]
    lk = ReentrantLock()
    threaded_chunks(length(kl)) do chunk
        H = zeros(ComplexF64, ns, ns)
        loc = [Dict{Tuple{Int,Int},Float64}() for _ in 1:2]
        for c in chunk
            (i, j) = kl[c]
            kf = SVector((i + 0.5) / nk, (j + 0.5) / nk)
            bloch_hamiltonian!(H, model, kf)
            F = eigen(Hermitian(H))
            for n in eachindex(F.values)
                w = exp(-((F.values[n] - E0) / sigma)^2)
                w < 1e-5 && continue
                for a in 1:ns
                    p = abs2(F.vectors[a, n])
                    p < 1e-8 && continue
                    l = U.layer_of[INT[a]]
                    key = (round(Int, nk * (kf[1] + GS[a][1]) - 0.5),
                           round(Int, nk * (kf[2] + GS[a][2]) - 0.5))
                    loc[l][key] = get(loc[l], key, 0.0) + w * p
                end
            end
        end
        lock(lk) do
            for l in 1:2, (key, v) in loc[l]
                acc[l][key] = get(acc[l], key, 0.0) + v
            end
        end
    end
    kx = Float64[]; ky = Float64[]; wt = Float64[]; lay = Int[]
    for l in 1:2, ((a, b), v) in acc[l]
        kc = U.bmT * SVector((a + 0.5) / nk, (b + 0.5) / nk)
        norm(kc) <= kmax || continue
        push!(kx, kc[1]); push!(ky, kc[2]); push!(wt, v); push!(lay, l)
    end
    return (kx=kx, ky=ky, weight=wt, layer=lay, nk=nk,
            spacing=norm(U.bmT * SVector(1.0, 0.0)) / nk)
end

### ---------------------------------------------------------------------------------------
### Reverse sweep: A(κ, E) along a κ path -- the unfolded band structure
### ---------------------------------------------------------------------------------------

"""
    unfolded_levels(model, U, kappas; layers=:both, wmin=1e-6)
        -> Vector of Vector{Tuple{Float64,Float64}}

Per LAB-frame monolayer momentum κ, the list of `(energy, unfolding weight)` pairs above
`wmin` — one diagonalization per κ, the REVERSE unfolding map. This is the raw material both
[`spectral_image`](@ref) (the A(κ,E) heatmap) and [`avoided_crossings`](@ref) (the gap
measurement) are built from, so a cut is diagonalized once and used for both.
`layers` is `:both`, `1`, or `2` to resolve which layer's weight is counted.
"""
function unfolded_levels(model::HamiltonianModel{2}, U::UnfoldMap, kappas::AbstractVector;
                         layers=:both, wmin::Real=1e-6)
    ns = n_states(model.cell)
    orbs = layers === :both ? collect(1:U.nb) : [b for b in 1:U.nb if U.layer_of[b] == layers]
    out = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(kappas))
    threaded_chunks(length(kappas)) do chunk
        H = zeros(ComplexF64, ns, ns)
        for c in chunk
            k, idx = carriers(U, SVector{2,Float64}(kappas[c]))
            bloch_hamiltonian!(H, model, k)
            F = eigen(Hermitian(H))
            rows = Tuple{Float64,Float64}[]
            for n in eachindex(F.values)
                p = 0.0
                for b in orbs
                    p += abs2(F.vectors[idx[b], n])
                end
                p >= wmin && push!(rows, (F.values[n], p))
            end
            out[c] = rows
        end
    end
    return out
end

"""
    spectral_image(levels, egrid; sigma=0.006) -> Matrix (length(egrid) × length(levels))

Gaussian-broaden [`unfolded_levels`](@ref) onto an energy grid: the `A(κ, E)` heatmap.
"""
function spectral_image(levels::AbstractVector, egrid::AbstractVector; sigma::Real=0.006)
    A = zeros(length(egrid), length(levels))
    for c in eachindex(levels), (e, p) in levels[c]
        for (m, x) in pairs(egrid)
            d = (x - e) / sigma
            abs(d) > 4 && continue
            A[m, c] += p * exp(-d^2)
        end
    end
    return A
end

"""
    unfolded_spectral_cut(model, U, kappas, egrid; sigma=0.006, layers=:both) -> Matrix

`A(κ, E)` in one call — [`unfolded_levels`](@ref) followed by [`spectral_image`](@ref).
"""
unfolded_spectral_cut(model::HamiltonianModel{2}, U::UnfoldMap, kappas::AbstractVector,
                      egrid::AbstractVector; sigma::Real=0.006, layers=:both) =
    spectral_image(unfolded_levels(model, U, kappas; layers=layers), egrid; sigma=sigma)

"""
    avoided_crossings(levels; wmin=0.02, window=(-Inf, Inf)) -> Vector{NamedTuple}

Locate the moiré MINIGAPS in an unfolded cut and measure them, instead of reading them off a
picture. A gap is an avoided crossing: two adjacent BRIGHT levels (unfolding weight above
`wmin` — dim levels are the dense folded background, not branches) whose separation has a
local minimum along the cut. Returns one row `(index, energy, gap)` per crossing, sorted by
gap, with energies at the midpoint of the pair.

Only bright-bright pairs count: a branch crossing the folded continuum is not a minigap.
"""
function avoided_crossings(levels::AbstractVector; wmin::Real=0.02,
                           window::Tuple{Real,Real}=(-Inf, Inf))
    ### the bright levels at each κ, energy-ordered
    bright = [sort([e for (e, p) in L if p >= wmin && window[1] <= e <= window[2]])
              for L in levels]
    rows = NamedTuple[]
    for c in 2:length(bright)-1
        nb = min(length(bright[c-1]), length(bright[c]), length(bright[c+1]))
        for n in 1:nb-1
            s0 = bright[c-1][n+1] - bright[c-1][n]
            s1 = bright[c][n+1] - bright[c][n]
            s2 = bright[c+1][n+1] - bright[c+1][n]
            ### a strict local minimum in the separation: the signature of hybridization
            (s1 < s0 && s1 < s2) || continue
            push!(rows, (index=c, energy=(bright[c][n] + bright[c][n+1]) / 2, gap=s1))
        end
    end
    return sort(rows; by=r -> -r.gap)
end

"""
    kappa_path(lattice, nodes; npoints=200) -> (kappas, dist, ticks)

A LAB-frame cartesian κ path through fractional monolayer nodes -- the untwisted monolayer
Brillouin zone the spectral function is unfolded onto.
"""
function kappa_path(lattice::BravaisLattice{2}, nodes; npoints::Int=200)
    bT = transpose(lattice.b)
    ks = SVector{2,Float64}[]; dist = Float64[]; ticks = Float64[0.0]
    dcur = 0.0
    for s in 1:length(nodes)-1
        a = SVector{2,Float64}(nodes[s]); b = SVector{2,Float64}(nodes[s+1])
        seg = norm(bT * (b - a))
        for t in range(0, 1; length=npoints)[(s == 1 ? 1 : 2):end]
            push!(ks, bT * (a + t * (b - a)))
            push!(dist, dcur + t * seg)
        end
        dcur += seg
        push!(ticks, dcur)
    end
    return (kappas=ks, dist=dist, ticks=ticks)
end

### ---------------------------------------------------------------------------------------
### Brillouin-zone furniture
### ---------------------------------------------------------------------------------------

### ---------------------------------------------------------------------------------------
### Rendering A(κ, ·): how to make a spectral function look like what it is
### ---------------------------------------------------------------------------------------
###
### Three things blur an unfolded Fermi surface, and only one of them is the moiré:
###
###  1. the ENERGY WINDOW. A Gaussian of width σ smears a sheet of velocity v over σ/v in
###     momentum. Keep σ/v below the κ spacing (here σ = 4 meV, v ≈ 1.6 eV·Å → 0.0025 Å⁻¹,
###     against Δκ = |b_m|/n_k) or the line is wider than the sampling and nothing is gained
###     by refining the grid.
###  2. the κ SAMPLING. Δκ = |b_m|/n_k; markers must be sized to tile it exactly, or the map
###     reads as either a dotted line (too small) or a fat one (too large).
###  3. the fact that NbSe₂ HAS FOUR SHEETS at E_F — two layers, each spin-split by Ising
###     spin-orbit coupling, the layers additionally rotated by ±θ/2 relative to each other.
###     Even the Ḡ = 0 reference is therefore four nearby contours, not one line. That is
###     physics and no amount of resolution removes it; `layers=` resolves it if wanted.
###
### On top of that, a linear colour scale cannot show a bright branch and a faint replica at
### once — the replicas that ARE the moiré signal are 1-2 orders down. So each map is drawn
### twice: linear (sharp lines, saturating at a high quantile) and logarithmic (replicas).
"""
    bin_to_grid(FS, xlims, ylims; bins=nothing, layers=(1, 2)) -> (xs, ys, Z)

Resample a forward-sweep map onto a regular CARTESIAN grid for `heatmap`.

Do not draw these maps as a marker scatter. The samples sit on a triangular lattice of
spacing `|b_m|/n_k`, so square markers have to be sized to tile it, and a marker even
slightly too large makes the lattice rows merge into long straight edges — a staircase that
reads as a physical feature and is nothing of the kind. (A marker too small gives a dotted
line instead.) Binning to a grid and letting `heatmap` interpolate removes the whole failure
mode: the rendering primitive no longer encodes a length scale.

`bins` defaults to a cell about 1.6× the sample spacing, which keeps essentially every cell
occupied for a triangular lattice; cells that still come up empty are filled once from their
4-neighbourhood, so a bin grid slightly finer than the sampling does not punch holes. Empty
cells with no occupied neighbour stay `0.0`.
"""
function bin_to_grid(FS, xlims::Tuple{Real,Real}, ylims::Tuple{Real,Real};
                     bins::Union{Nothing,Int}=nothing, layers=(1, 2))
    span = max(xlims[2] - xlims[1], ylims[2] - ylims[1])
    nb = isnothing(bins) ? max(24, round(Int, span / (1.6 * FS.spacing))) : bins
    xs = range(xlims[1], xlims[2]; length=nb)
    ys = range(ylims[1], ylims[2]; length=nb)
    dx = (xlims[2] - xlims[1]) / (nb - 1)
    dy = (ylims[2] - ylims[1]) / (nb - 1)
    acc = zeros(nb, nb)
    cnt = zeros(Int, nb, nb)
    for n in eachindex(FS.kx)
        FS.layer[n] in layers || continue
        ix = round(Int, (FS.kx[n] - xlims[1]) / dx) + 1
        iy = round(Int, (FS.ky[n] - ylims[1]) / dy) + 1
        (1 <= ix <= nb && 1 <= iy <= nb) || continue
        acc[iy, ix] += FS.weight[n]
        cnt[iy, ix] += 1
    end
    Z = [cnt[i, j] > 0 ? acc[i, j] / cnt[i, j] : 0.0 for i in 1:nb, j in 1:nb]
    ### one neighbour-fill pass so a slightly-too-fine bin grid shows no holes
    empty = [cnt[i, j] == 0 for i in 1:nb, j in 1:nb]
    Zf = copy(Z)
    for i in 1:nb, j in 1:nb
        empty[i, j] || continue
        s = 0.0; c = 0
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            a, b = i + di, j + dj
            (1 <= a <= nb && 1 <= b <= nb && !empty[a, b]) || continue
            s += Z[a, b]; c += 1
        end
        c > 0 && (Zf[i, j] = s / c)
    end
    return xs, ys, Zf
end

"""
    weight_clims(w; q=0.995) -> (0.0, hi)

Colour limits for a spectral map: saturate at a high quantile of the NONZERO weight rather
than at the maximum, so one hot pixel does not compress the whole scale.
"""
function weight_clims(w::AbstractVector; q::Real=0.995)
    nz = [x for x in w if x > 0]
    isempty(nz) && return (0.0, 1.0)
    s = sort(nz)
    return (0.0, s[clamp(ceil(Int, q * length(s)), 1, length(s))])
end

"""
    bz_hexagon(lattice) -> (xs, ys)

The first Brillouin-zone boundary of a triangular lattice, closed, cartesian.
"""
function bz_hexagon(lattice::BravaisLattice{2})
    bT = transpose(lattice.b)
    b1 = bT * SVector(1.0, 0.0); b2 = bT * SVector(0.0, 1.0)
    K = [(b1 + b2) / 3, (2b2 - b1) / 3, (b2 - 2b1) / 3,
         -(b1 + b2) / 3, (b1 - 2b2) / 3, (2b1 - b2) / 3]
    push!(K, K[1])
    return [k[1] for k in K], [k[2] for k in K]
end

"""
    bz_corners(lattice) -> Vector{SVector{2,Float64}}

The six K points of a triangular lattice, cartesian.
"""
function bz_corners(lattice::BravaisLattice{2})
    bT = transpose(lattice.b)
    b1 = bT * SVector(1.0, 0.0); b2 = bT * SVector(0.0, 1.0)
    return [(b1 + b2) / 3, (2b2 - b1) / 3, (b2 - 2b1) / 3,
            -(b1 + b2) / 3, (b1 - 2b2) / 3, (2b1 - b2) / 3]
end

### ---------------------------------------------------------------------------------------
### Density of states, on a shared k-grid so the comparison is apples to apples
### ---------------------------------------------------------------------------------------

"""
    gaussian_dos(levels, centre, halfwidth; nE=601, sigma=0.008, nk=1) -> (E_meV, dos)

A Gaussian-broadened DOS from a flat list of eigenvalues, in states/eV per moiré cell.
Energies are returned in meV RELATIVE to `centre`. Two models compared this way must be
sampled on the SAME `k`-grid: at the k-densities a 4N-band metal allows, the discrete-k
ripple is larger than the moiré structure, and it only cancels between curves that share
their momenta.
"""
function gaussian_dos(levels::AbstractVector{<:Real}, centre::Real, halfwidth::Real;
                      nE::Int=601, sigma::Real=0.008, nk::Int=1)
    grid = range(centre - halfwidth, centre + halfwidth; length=nE)
    g = zeros(nE)
    for e in levels
        (grid[1] - 5sigma <= e <= grid[end] + 5sigma) || continue
        for (n, x) in pairs(grid)
            d = (x - e) / sigma
            abs(d) > 5 && continue
            g[n] += exp(-d^2)
        end
    end
    return collect((grid .- centre) .* 1000), g ./ (sigma * sqrt(pi) * nk)
end
