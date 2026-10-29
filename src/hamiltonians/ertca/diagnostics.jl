##################################################################
###### export only the following functions from this module ######
##################################################################

export harmonic_decay, hermiticity_residual, reconstruction_residual




#############################################################
###### Harmonic diagnostics                             #####
#############################################################

"""
    harmonic_decay(h::RegistryHarmonics; kpoints=25) -> Vector{NamedTuple}

The magnitude of each retained harmonic: one row `(G, absG, wmax, wmax_intra, wmax_inter)`
per canonical Ḡ, sorted by `|Ḡ|` (cartesian). `wmax` is `max_{ββ', k̄} |W_{ββ'}(k̄; Ḡ)|`
with the Bloch sums EVALUATED (an amplitude sup bound would be blind to the phase
cancellations that carry the whole decay for interlayer blocks) on a deterministic
off-symmetry grid of `kpoints` momenta in the NEAR ZONE `|k̄| ≤ 1/4` (fractional). The near
zone is where the decay law is physical: an interlayer harmonic behaves as the tunnelling
form factor at `k̄ + Ḡ`, so momenta approaching `−Ḡ` re-inflate every harmonic to the Ḡ = 0
scale and a full-zone sweep would read as stalled on perfectly healthy data. Pass an
explicit `kpoints` vector to probe other momenta. The intra/inter columns split the maximum
by layer pair.

This is the paper's decay law made measurable (registry harmonics fall off exponentially in
`|Ḡ|` on the interlayer-distance scale) -- and the pipeline's gauge-health dashboard: a
family of Wannierizations that does NOT share one smooth gauge shows up as a decay that
stalls instead of falling.
"""
function harmonic_decay(h::RegistryHarmonics;
                        kpoints::Union{Int,AbstractVector}=25)::Vector{<:NamedTuple}
    nb = length(h.labels)
    ks = kpoints isa Int ? _nearzone_kgrid(kpoints) :
         [SVector{2,Float64}(k) for k in kpoints]
    rows = map(eachindex(h.gvecs)) do iG
        wintra = 0.0
        winter = 0.0
        for i in 1:nb, j in 1:nb, k in ks
            a = abs(_harmonic_value(h, iG, i, j, k))
            if h.shifted[i] == h.shifted[j]
                wintra = max(wintra, a)
            else
                winter = max(winter, a)
            end
        end
        (G=h.gvecs[iG], absG=norm(recip_matrix_times(h.lattice, h.gvecs[iG])),
         wmax=max(wintra, winter), wmax_intra=wintra, wmax_inter=winter)
    end
    return sort(rows; by=r -> r.absG)
end

"""
    hermiticity_residual(h::RegistryHarmonics) -> Float64

The solve's Hermiticity-pairing residual BEFORE enforcement (see
[`registry_harmonics`](@ref)): how far the ±Ḡ solve coefficients were from conjugate pairs.
Exactly zero for an exact-solve stacking set; roundoff-sized otherwise. A large value means
the stackings sample the retained harmonics badly -- check `h.cond` and the stacking set.
"""
hermiticity_residual(h::RegistryHarmonics)::Float64 = h.hermiticity

"""
    reconstruction_residual(h, scan; kpoints=25) -> NamedTuple

How well the retained harmonics reconstruct a scan's Bloch Hamiltonians:

    max / mean over stackings s and momenta k̄ of  | Σ_Ḡ e^{−2πi Ḡ·d_s} W(k̄; Ḡ) − W(k̄, d_s) |

in the scan's energy units. On the stackings the harmonics were SOLVED from, this is zero
by construction whenever `S = |F|` (the inversion interpolates its own samples) -- so
measure it on HELD-OUT stackings: pass a scan holding members that did not enter
[`registry_harmonics`](@ref). The residual is the truncation-leakage + gauge-consistency
dashboard: harmonics outside the retained set, and any gauge jump between Wannierizations,
land here.

`kpoints` is either the number of momenta of a deterministic off-symmetry sampling grid, or
an explicit vector of fractional 2-vectors. Returns `(max, mean, kpoints, stackings)`.
"""
function reconstruction_residual(h::RegistryHarmonics, scan::RegistryScan{D};
                                 kpoints::Union{Int,AbstractVector}=25) where {D}
    length(scan.labels) == length(h.labels) || throw(ArgumentError(
        "`reconstruction_residual`: the scan holds $(length(scan.labels)) orbitals but the " *
        "harmonics hold $(length(h.labels)); they do not describe the same family."))
    ks = kpoints isa Int ? _offsymmetry_kgrid(kpoints) :
         [SVector{2,Float64}(k) for k in kpoints]
    isempty(ks) && throw(ArgumentError("`reconstruction_residual` needs at least one momentum."))
    maxres = 0.0
    sumres = 0.0
    nres = 0
    for (s, model) in pairs(scan.models)
        ip = invperm(scan.permutations[s])
        for k in ks
            Wmodel = bloch_hamiltonian(model, _pad_k(k, Val(D)))[ip, ip]
            Wrec = _reconstruct_W(h, scan.d[s], k)
            r = maximum(abs, Wrec - Wmodel)
            maxres = max(maxres, r)
            sumres += r
            nres += 1
        end
    end
    return (max=maxres, mean=sumres / nres, kpoints=length(ks), stackings=length(scan.models))
end

### a deterministic, symmetry-avoiding fractional k-grid (no Γ, no zone-boundary points)
function _offsymmetry_kgrid(n::Int)::Vector{SVector{2,Float64}}
    n >= 1 || throw(DomainError(n, "`kpoints` must be positive."))
    side = max(1, ceil(Int, sqrt(n)))
    ks = [SVector((a + 0.31) / side, (b + 0.17) / side) for a in 0:(side-1), b in 0:(side-1)]
    return vec(ks)[1:n]
end

### the near-zone variant (|k̄| ≤ 1/4 fractional, Γ avoided): where the interlayer decay law
### is physical -- see harmonic_decay
function _nearzone_kgrid(n::Int)::Vector{SVector{2,Float64}}
    return [SVector(k[1] / 2 - 0.25, k[2] / 2 - 0.25) for k in _offsymmetry_kgrid(n)]
end

### the in-plane momentum, zero-padded to the DFT model's own dimension
@inline _pad_k(k::SVector{2,Float64}, ::Val{D}) where {D} = _pad_components(k, Val(D))
