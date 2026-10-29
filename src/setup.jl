# =====================================================================================
#  Entry point for paper scripts: loads the helpers and defines shared parameters.
#
#  From a script in this folder: include(joinpath(@__DIR__, "setup.jl"))
#  Loading this file does not run a registry scan or generate a figure.
# =====================================================================================

include(joinpath(@__DIR__, "loader.jl"))
include(joinpath(@__DIR__, "analysis.jl"))

using Printf

const PAPER_FIGURES = normpath(joinpath(@__DIR__, "..", "Figures"))

### FOUR retained shells (|F| = 31), not two. Held-out stackings that never entered the solve
### are reproduced to 11.5 meV at |F| = 31, against 15.8 at |F| = 13 and 12.2 at |F| = 19 — and
### it degrades again by |F| = 121, so this is a measured optimum, not "more is better". A wider
### retained set also covers more of the residue classes the folded-channel gate needs.
const GSHELLS = 4

### Bloch-sum pruning. Harmonics below this weight are dropped; 1e-6 costs ~0.6 meV against the
### method's own ~11 meV truncation, and shortens the sums fourfold.
const HARM_ATOL = 1e-6

### The DEFAULT twist, shared by the figures so they can be read against one another: (6,7) is
### θ = 5.0858°, N = 127 monolayer cells per layer. Small enough to diagonalize on a k-grid,
### small enough in angle that |b_m| = 0.187 Å⁻¹ is a tenth of the monolayer zone and the
### reconstruction is visible.
###
### It is a default, not a property of the pipeline: `ertca_context(mat; mn=(m, n))` takes any
### coprime pair, and `04_commensuration_classes.jl` walks several of them precisely because the
### (n, n+1) family this default belongs to is the DEGENERATE case of the comb bookkeeping
### (det 𝓝 = ±1 there, so every moiré label transfer has an integer parent harmonic).
const MN = (6, 7)
const HBAR2_2M = 3.80998212     # ħ²/(2mₑ) in eV·Å²

"""
    ertca_context(mat; mn=MN, gshells=GSHELLS, atol=HARM_ATOL, verbose=true) -> NamedTuple

Ingest the 144-stacking family of `mat`, solve the harmonics, and return everything the
scripts share: `(h, tb, EF, lat, bT, binv, moire, bmT, bminv, theta)`. Prints one
provenance line.

`mn` is the commensurate twist as a coprime index pair; it defaults to [`MN`](@ref) so the
figures share an angle, and the moiré-lattice and reciprocal fields are rebuilt from whatever
pair is passed.

`EF` is `nothing` for a material whose `filling` this repository has not pinned (WSe₂):
an unpinned Fermi level is reported as absent rather than silently inheriting a number
that belongs to a different electron count.
"""
function ertca_context(mat::ERTCAMaterial; mn::Tuple{Int,Int}=MN, gshells::Int=GSHELLS,
                       atol::Real=HARM_ATOL, verbose::Bool=true)
    (scan, nwarn) = quiet(() -> ertca_scan(mat))
    h = registry_harmonics(scan; gshells=gshells, atol=atol)
    tb = twisted_bilayer(h.lattice, mn...)
    EF = isnothing(mat.filling) ? nothing : fermi_level(h; filling=mat.filling)
    lat = h.lattice
    bT = transpose(lat.b)
    Lmat = SingleParticleModels._ertca_L(tb.theta / 2)
    moire = BravaisLattice(Matrix(lat.a * transpose(inv(Lmat))))
    if verbose
        @printf("""
        %s ERTCA — %d shifted-bilayer Wannierizations, %d retained harmonics (|F| = %d),
        cond(E) = %.8f, Hermiticity residual %.1e, %d drift warnings (max %.1e, relaxation).
        θ = %.4f°, N = %d cells/layer, |b̄| = %.4f Å⁻¹, |b_m| = %.4f Å⁻¹, E_F = %s.
        """, mat.name, n_stackings(scan), length(h.gvecs), 2 * length(h.gvecs) - 1,
                 h.cond, h.hermiticity, nwarn,
                 maximum(scan.drift), twist_angle(tb), n_monolayer_cells(tb),
                 norm(bT * SVector(1.0, 0.0)),
                 norm(transpose(moire.b) * SVector(1.0, 0.0)),
                 isnothing(EF) ? "not pinned (filling undeclared)" : @sprintf("%.5f eV", EF))
    end
    return (h=h, tb=tb, EF=EF, lat=lat, bT=bT, binv=inv(bT), moire=moire,
            bmT=transpose(moire.b), bminv=inv(transpose(moire.b)), theta=tb.theta)
end

"""
    valley_setup(ctx, which) -> NamedTuple

Everything a single-valley reduction needs: the spin-aligned one-sector harmonics, the valley
momentum, the per-layer mini-zone origins, and the expansion point that puts them there.

`which` is `:K` or `:Gamma`.

- At **K** the two layers' extrema are at `K_ℓ = L_ℓ K̄`, which differ by a mini-zone corner
  vector, so they land on the two INEQUIVALENT corners κ, κ′ — the Wu-Lovorn-MacDonald setting.
  Both must be C₃-registered (`C₃κ ≡ κ mod ℤ²`) or the tunnelling tripod is not integral, which
  is why they are pinned to (2/3,2/3) and (1/3,1/3) rather than placed symmetrically.
- At **Γ** the valley momentum is invariant under the twist, so BOTH layers' extrema sit at the
  same mini-zone point γ. There is no tripod: the interlayer coupling has a uniform `ΔG = 0`
  part and a first-shell part, and the moiré enters the intralayer potential.
"""
function valley_setup(ctx, which::Symbol)
    which in (:K, :Gamma) || throw(ArgumentError("`which` is :K or :Gamma; got :$which"))
    kbar = which === :K ? SVector(1 / 3, 1 / 3) : SVector(0.0, 0.0)
    kc = ctx.bT * kbar
    ### The spin basis is ALWAYS fixed at K, whichever valley is being analysed. If the only
    ### spin-orbit term is Ising, H = h₀σ₀ + h_z(k)σ_z is spin-diagonal at every k, so the σ_z
    ### basis is global -- but h_z VANISHES at Γ by symmetry, leaving the pair exactly
    ### degenerate there and its eigenvectors arbitrary. Diagonalizing at Γ therefore returns a
    ### meaningless rotation (measured: 71.65° on one layer, 82.98° on the other -- and a
    ### DIFFERENT arbitrary one per layer, which makes the interlayer amplitude w₀
    ### basis-dependent by cos of their difference). At K the splitting is 151 meV and the
    ### eigenvectors are sharp, so fix the gauge there and carry it everywhere.
    hal, info = spin_align(ctx.h, SVector(1 / 3, 1 / 3))
    h1 = select_orbitals(hal, [2, 4])
    th = ctx.theta
    Kb = SingleParticleModels._rotation2(-th / 2) * kc
    Kt = SingleParticleModels._rotation2(+th / 2) * kc
    if which === :K
        kap_b = SVector(2 / 3, 2 / 3)
        expansion = Kb - ctx.bmT * kap_b
        kap_t = ctx.bminv * (Kt - expansion)
    else
        kap_b = SVector(0.0, 0.0)
        expansion = Kb                      # = Kt = Γ: the twist leaves Γ fixed
        kap_t = ctx.bminv * (Kt - expansion)
    end
    return (h=h1, spin=info, kbar=kbar, kc=kc, expansion=expansion,
            kap_b=kap_b, kap_t=kap_t, Kb=Kb, Kt=Kt)
end

"""
    parabolicity(h1, kbar, lat, bm; radii=(0.5, 1.0, 2.0)) -> Vector{NamedTuple}

How parabolic the valley band actually is on the MINI-ZONE scale, which is the only scale a
continuum model has to get right. For each radius (in units of `bm = |b_m|`) this angle-
averages a least-squares fit `ε − ε₀ = −ħ²k²/2m*` and reports the fitted mass alongside the
worst deviation of the true band from that parabola. A single mass cannot serve a band whose
deviation over `|k| ≲ |b_m|` already exceeds the features being modelled.
"""
function parabolicity(h1::RegistryHarmonics, kbar, lat::BravaisLattice{2}, bm::Real;
                      radii=(0.5, 1.0, 2.0))
    binv = inv(transpose(lat.b))
    bT = transpose(lat.b)
    iG0 = findfirst(iszero, h1.gvecs)
    eps(kc) = real(_harmonic_value(h1, iG0, 1, 1, binv * SVector{2,Float64}(kc)))
    kc0 = bT * SVector{2,Float64}(kbar)
    e0 = eps(kc0)
    out = NamedTuple[]
    for rf in radii
        R = rf * bm
        pts = [(r, th) for r in range(R / 12, R; length=12)
               for th in range(0, 2pi; length=13)[1:12]]
        num = sum((eps(kc0 + r * SVector(cos(t), sin(t))) - e0) * r^2 for (r, t) in pts)
        den = sum(r^4 for (r, _) in pts)
        c = -num / den
        mx = maximum(abs((eps(kc0 + r * SVector(cos(t), sin(t))) - e0) + c * r^2)
                     for (r, t) in pts)
        push!(out, (radius_bm=rf, mstar=HBAR2_2M / c, maxdev_meV=1000mx))
    end
    return out
end

"""
    interlayer_kdependence(h, kbar, lat, bm; radius=1.0) -> NamedTuple

How much the `Ḡ = 0` interlayer amplitude VARIES across the mini-zone. A continuum model takes
it as a constant `w₀`; ERTCA keeps its full momentum dependence, so the spread measured here is
a floor on the error of any constant-`w` parametrization — and where `w₀` is large, a small
relative spread is a large absolute one.
"""
function interlayer_kdependence(h::RegistryHarmonics, kbar, lat::BravaisLattice{2}, bm::Real;
                                radius::Real=1.0)
    binv = inv(transpose(lat.b))
    bT = transpose(lat.b)
    iG0 = findfirst(iszero, h.gvecs)
    nb = length(h.labels)
    pair = first((i, j) for i in 1:nb, j in 1:nb if h.shifted[i] != h.shifted[j])
    kc0 = bT * SVector{2,Float64}(kbar)
    vals = Float64[]
    for r in range(0.0, radius * bm; length=7), t in range(0, 2pi; length=13)[1:12]
        kf = binv * (kc0 + r * SVector(cos(t), sin(t)))
        push!(vals, abs(_harmonic_value(h, iG0, pair[1], pair[2], SVector{2,Float64}(kf))))
    end
    w_at = abs(_harmonic_value(h, iG0, pair[1], pair[2], SVector{2,Float64}(kbar)))
    return (w0=w_at, min=minimum(vals), max=maximum(vals),
            spread_meV=1000 * (maximum(vals) - minimum(vals)),
            spread_frac=(maximum(vals) - minimum(vals)) / w_at)
end

"""
    valley_parameters(h1, kbar, lat) -> NamedTuple

Read the continuum-model parameters off the harmonics at `kbar`: the band edge, the effective
mass (and its anisotropy), the first-shell intralayer `V, ψ`, and the interlayer amplitudes
`Ḡ = 0` and first shell. Nothing is fitted.
"""
function valley_parameters(h1::RegistryHarmonics, kbar, lat::BravaisLattice{2})
    binv = inv(transpose(lat.b))
    iG0 = findfirst(iszero, h1.gvecs)
    Wv(iG, i, j, k) = _harmonic_value(h1, iG, i, j, SVector{2,Float64}(k))
    eps1(kf) = real(Wv(iG0, 1, 1, kf))
    dk = 2e-3
    Hs = zeros(2, 2)
    for a in 1:2, b in 1:2
        ea = SVector(a == 1 ? dk : 0.0, a == 2 ? dk : 0.0)
        eb = SVector(b == 1 ? dk : 0.0, b == 2 ? dk : 0.0)
        Hs[a, b] = (eps1(kbar .+ binv * (ea + eb)) - eps1(kbar .+ binv * (ea - eb))
                    - eps1(kbar .+ binv * (-ea + eb)) + eps1(kbar .+ binv * (-ea - eb))) / (4dk^2)
    end
    ev = eigvals(Symmetric(Hs))
    mstar = -2HBAR2_2M / (sum(ev) / 2)
    star1 = SingleParticleModels.g_star(lat, 1)
    C3 = SingleParticleModels._integer_c3(lat)
    reps = SingleParticleModels._c3_closed_reps(star1, C3)
    function amp(i, j, g)
        hit = SingleParticleModels._harmonic_match(h1, g)
        isnothing(hit) && return nothing
        iG, sg = hit
        sg > 0 ? Wv(iG, i, j, kbar) : conj(Wv(iG, j, i, kbar))
    end
    intra = [amp(1, 1, g) for g in reps]
    V = sum(abs, intra) / length(intra)
    psi = angle(sum(cis(angle(v)) for v in intra) / length(intra))
    w0 = amp(1, 2, SVector(0, 0))
    inter1 = [(g=g, w=amp(1, 2, g)) for g in star1 if !isnothing(amp(1, 2, g))]
    return (edge=eps1(kbar), mstar=mstar, hess=ev,
            anisotropy=abs(ev[1] - ev[2]) / abs(sum(ev) / 2),
            reps=reps, intra=intra, V=V, psi=psi, w0=w0, inter1=inter1)
end
