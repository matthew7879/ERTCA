##################################################################
###### export only the following functions from this module ######
##################################################################

export ertca_model




#############################################################
###### Geometry: rotations and the disregistry lattice  #####
#############################################################


### 𝓛 = L_b⁻¹ − L_t⁻¹ with L_t = R(θ_top), L_b = R(−θ_top): the disregistry map d(r) = 𝓛r
@inline _ertca_L(theta_top::Float64) = _rotation2(theta_top) - _rotation2(-theta_top)

### the mean-orientation rotation L_{ℓℓ'}⁻¹ applied to the midpoint momentum: the layer's own
### inverse rotation intralayer, the identity interlayer (the only layer-exchange-symmetric
### choice, and the paper's registry hypothesis)
@inline function _ertca_rot(shifted_i::Bool, shifted_j::Bool, theta_top::Float64)
    shifted_i && shifted_j && return _rotation2(-theta_top)      # both on the top layer
    !shifted_i && !shifted_j && return _rotation2(+theta_top)    # both on the bottom layer
    return one(SMatrix{2,2,Float64})                        # interlayer
end




#############################################################
###### Realizing the continuum model                    #####
#############################################################

"""
    ertca_model(h::RegistryHarmonics, tb::TwistedBilayer; top=2,
                gvectors=monolayer_bz_gvectors(tb), tol=1e-10, gauge=DEFAULT_GAUGE,
                name="ertca") -> HamiltonianModel{2}

    ertca_model(h::RegistryHarmonics, theta::Real; degrees=true,
                shells=nothing, gcut=nothing, gvectors=nothing, expansion=[0, 0],
                tol=1e-10, gauge=DEFAULT_GAUGE, name="ertca") -> HamiltonianModel{2}

Realize the ERTCA continuum model from registry harmonics -- stage three of the pipeline.
The master formula

    T^{ββ'}_{k,k'} = Σ_{g ∈ Λ*_m} W_{ββ'}(k*; Ḡ_g) δ_{k'−k, g},   k* = L_{ℓℓ'}⁻¹ (k+k')/2,

becomes one pair of [`k_envelope`](@ref) terms (real and imaginary part of the exact
Bloch-sum harmonic, coefficients `1` and `im` -- the stored-once Hermitization then rebuilds
`conj(W)` on the partner direction exactly) per retained harmonic and orbital channel, on a
plane-wave cell whose internal axis `:wannier` carries the harmonics' orbital labels. The
`Ḡ = 0` harmonic enters as the G-diagonal exact Wannier dispersion: there is NO separate
kinetic term and nothing is fitted.

The two methods choose the plane-wave basis:

- **Commensurate** (`tb::TwistedBilayer`): by default each orbital carries ITS layer's
  tiling set `monolayer_bz_gvectors(tb, layer)` -- the N moiré coset representatives tiling
  that layer's monolayer Brillouin zone, the natural exact truncation with no duplicated
  monolayer states (the model has exactly N bands per orbital, the commensurate supercell's
  count). Pass a flat `gvectors=` to override with one shared set -- it must stay
  one-representative-per-coset for each layer (an over-complete set duplicates monolayer
  bands), and it disables the folded channels below. `top=` names which `tb` layer
  (1 = rotated by −θ/2, 2 = by +θ/2, the default) is the harmonics' SHIFTED layer;
  flipping it is the same model at −θ.

  In this mode the model also carries the COMB channels (`folding=:auto`, the default).
  Plane-wave labels of a layer are physical only modulo that layer's reciprocal lattice, so
  a retained harmonic couples not only the pairs whose label transfer has it as a parent, but
  also the pairs the wrap sends elsewhere. Two families, both from the same stored Bloch sums:

  - INTERLAYER (folded / Umklapp), `ertca_fold<s>_re/_im`: evaluated at the unwrapped MEETING
    momentum where the two layers' extended zones coincide, with the comb wrap phases. Emitted
    only inside the stacking set's aliasing radius (`kappa_cut=`, default half the shortest
    alias-lattice vector; smooth roll-off `fold_width=`, default one moiré reciprocal length),
    carried by windowed [`BlochSumEnvelope`](@ref)s, verified against brute-force commensurate
    supercells to the alias floor.
  - INTRALAYER (wrapped), `ertca_wrap<s>_re/_im`: the same-layer counterpart, for transfers
    whose target leaves the fundamental domain and is represented by its coset rep. This
    identity is EXACT -- no window, no branch weighting -- because the wrap `m` depends only on
    the harmonic and the transfer, not on which label is being wrapped.

  Both need the per-layer coset structure, so a flat `gvectors=` override disables them, as
  does `folding=:off`, which leaves the bare retained master-formula channels. Folding also
  needs a stacking set with class structure (a `registry_shifts` coset set or a regular
  d-grid); scattered sets skip it with a named warning, and a set that is a regular grid much
  denser than the retained shells warns that its branch gate is dropping channels (retain more
  shells: the gate only passes branches the sampling cannot distinguish from a retained one).
- **Any angle** (`theta::Real`, degrees by default): the basis is a shell truncation
  (exactly one of `shells=`/`gcut=`/`gvectors=`, as [`plane_wave_cell`](@ref)) of the
  disregistry moiré lattice `Λ_m = 𝓛⁻¹Λ`, built directly from the twist -- commensurate or
  not. `expansion` is a cartesian momentum: the model's `k = 0` is placed there (a k·p
  window around one point of the monolayer zone; the basis covers `expansion + ` the chosen
  g-shell). A negative `theta` swaps the layer roles.

Every emitted term is named `ertca_shell<s>_re` / `_im` by harmonic shell, so
`reweight(model, "ertca_shell2_re" => 0, "ertca_shell2_im" => 0)` ablates a shell without
recompiling. Harmonic channels whose Bloch-sum weight is below `tol` are skipped.
"""
function ertca_model(h::RegistryHarmonics, tb::TwistedBilayer; top::Int=2,
                     gvectors::Union{Nothing,AbstractVector}=nothing,
                     folding::Symbol=:auto,
                     kappa_cut::Union{Nothing,Real}=nothing,
                     fold_width::Union{Nothing,Real}=nothing,
                     tol::Float64=1e-10, gauge::Symbol=DEFAULT_GAUGE,
                     name::AbstractString="ertca")::HamiltonianModel{2}
    top in (1, 2) || throw(ArgumentError(
        "`ertca_model`: `top` names a TwistedBilayer layer, 1 (rotated by −θ/2) or " *
        "2 (+θ/2); got $top."))
    folding in (:auto, :off) || throw(ArgumentError(
        "`ertca_model`: `folding` is :auto (emit the in-window folded interlayer channels) " *
        "or :off; got :$folding."))
    same_basis(h.lattice, tb.monolayer) || throw(ArgumentError(
        "`ertca_model`: the harmonics live on a different monolayer lattice than the " *
        "TwistedBilayer (`same_basis` fails); build both from the same `BravaisLattice`."))
    theta_top = (top == 2 ? +1.0 : -1.0) * tb.theta / 2
    Gsets = if isnothing(gvectors)
        ### each orbital tiles ITS OWN layer's monolayer BZ: exactly N plane waves per
        ### orbital, no duplicated monolayer states
        tile = (monolayer_bz_gvectors(tb, top == 2 ? 2 : 1),
                monolayer_bz_gvectors(tb, top == 2 ? 1 : 2))
        [h.shifted[b] ? tile[1] : tile[2] for b in eachindex(h.shifted)]
    else
        gs = map(collect(gvectors)) do g
            (length(g) == 2 && all(x -> isinteger(x), g)) || throw(ArgumentError(
                "`ertca_model`: `gvectors` entry $(Tuple(g)) is not an integer 2-vector. " *
                "`gvectors` takes INTEGER moiré reciprocal labels (Miller indices); " *
                "passing cartesian momenta is the usual mistake. The override must also " *
                "stay one-representative-per-coset for each layer, or monolayer bands " *
                "duplicate -- prefer the default per-layer tiling sets."))
            SVector{2,Int}(g)
        end
        [gs for _ in eachindex(h.shifted)]
    end
    ### the folded (Umklapp) channels need the per-layer tiling coset structure and the
    ### exact integer supercell matrices; a flat gvectors= override loses the former
    foldspec = if folding === :auto && isnothing(gvectors)
        Mtop = supercell_matrix(tb, top == 2 ? 2 : 1)
        Mbot = supercell_matrix(tb, top == 2 ? 1 : 2)
        _ertca_foldspec(h, Mbot, Mtop, kappa_cut, fold_width)
    else
        nothing
    end
    ### the parent-harmonic map is EXACT on the integers in this mode: 𝓛ᵀb̄ᵀḠ = b_mᵀ(M_bot−M_top)Ḡ,
    ### so the parent of a transfer ΔG solves (M_bot−M_top)·Ḡ = −ΔG in ℤ². The float round trip
    ### b̄⁻ᵀ𝓛⁻ᵀb_mᵀ is only NUMERICALLY unimodular here (the moiré basis comes from the coincidence
    ### search, and `twisted_bilayer`'s θ = acos(…) loses ~ε/θ² to cancellation), so an absolute
    ### integrality tolerance would reject genuine channels at small twist -- silently, and
    ### non-monotonically in θ. Hand the integer route the supercell matrices.
    Msh = SMatrix{2,2,Int}(supercell_matrix(tb, top == 2 ? 2 : 1))   # the shifted layer
    Mfx = SMatrix{2,2,Int}(supercell_matrix(tb, top == 2 ? 1 : 2))   # the other one
    Nmat = Mfx - Msh
    ### the comb-wrapped INTRALAYER channels are comb physics, exactly like the folded
    ### interlayer ones, so they ride the same `folding` switch: `folding=:off` means the bare
    ### retained master formula, with no wraps of either kind. They need the per-layer coset
    ### structure, so a flat `gvectors=` override disables them too.
    layerM = if folding === :auto && isnothing(gvectors)
        [h.shifted[b] ? Msh : Mfx for b in eachindex(h.shifted)]
    else
        nothing
    end
    return _ertca_realize(h, tb.moire, theta_top, Gsets,
                          SVector(0.0, 0.0), tol, gauge, String(name);
                          foldspec=foldspec, parent=Nmat, layerM=layerM)
end

function ertca_model(h::RegistryHarmonics, theta::Real; degrees::Bool=true,
                     shells::Union{Nothing,Int}=nothing,
                     gcut::Union{Nothing,Real}=nothing,
                     gvectors::Union{Nothing,AbstractVector}=nothing,
                     expansion::AbstractVector{<:Real}=[0.0, 0.0],
                     tol::Float64=1e-10, gauge::Symbol=DEFAULT_GAUGE,
                     name::AbstractString="ertca")::HamiltonianModel{2}
    th = degrees ? Float64(theta) * pi / 180 : Float64(theta)
    abs(sin(th / 2)) > 1e-12 || throw(ArgumentError(
        "`ertca_model`: the twist angle is zero (or a multiple of 2π); the disregistry map " *
        "𝓛 is singular and there is no moiré lattice. For the untwisted bilayer at fixed " *
        "registry, use the registry scan's own models."))
    length(expansion) == 2 || throw(DimensionMismatch(
        "`ertca_model`: `expansion` is a cartesian in-plane momentum with 2 components."))
    Lmat = _ertca_L(th / 2)
    moire = BravaisLattice(Matrix(h.lattice.a * transpose(inv(Lmat))))
    ### resolve the shared g-set once (the refusal logic is plane_wave_cell's) and give it to
    ### every orbital
    probe = plane_wave_cell(moire; shells=shells, gcut=gcut, gvectors=gvectors)
    gs = unique(state_g_vectors(probe))
    ### the monolayer-BZ capacity: one layer holds exactly 1/(4 sin²(θ/2)) distinct
    ### monolayer momenta per moiré zone. A window with more labels than that necessarily
    ### represents some monolayer state twice -- exact, mutually UNCOUPLED duplicates (this
    ### mode carries no folded channels to glue them) that pollute the spectrum, at large
    ### angles down to the band bottom.
    capacity = 1 / (4 * sin(th / 2)^2)
    if length(gs) > capacity + 1e-9
        @warn "ertca_model: the k·p window holds $(length(gs)) plane waves per orbital " *
              "but the monolayer zone at θ = $(round(abs(theta); sigdigits=4))" *
              (degrees ? "°" : " rad") * " has room for only $(round(capacity; sigdigits=4)) " *
              "distinct momenta -- the excess states are exact, uncoupled DUPLICATES of " *
              "monolayer states and corrupt the spectrum (multiplicities, even the band " *
              "bottom at large angles). Shrink `shells=`/`gcut=` below the capacity, or " *
              "use the commensurate mode, whose per-layer tiling basis is duplicate-free " *
              "by construction."
    end
    Gsets = [gs for _ in eachindex(h.shifted)]
    ### no folded channels here: in this mode the moiré lattice is the disregistry lattice
    ### itself, every transfer is its own parent (bmᵀ = 𝓛ᵀb̄ᵀ exactly), and the plane waves
    ### carry no comb redundancy
    return _ertca_realize(h, moire, th / 2, Gsets, SVector{2,Float64}(expansion), tol,
                          gauge, String(name); foldspec=nothing)
end

function _ertca_realize(h::RegistryHarmonics, moire::BravaisLattice{2},
                        theta_top::Float64, Gsets::Vector{<:AbstractVector{SVector{2,Int}}},
                        kappa0::SVector{2,Float64},
                        tol::Float64, gauge::Symbol, name::String;
                        foldspec::Union{Nothing,NamedTuple}=nothing,
                        parent::Union{Nothing,SMatrix{2,2,Int}}=nothing,
                        layerM::Union{Nothing,Vector{SMatrix{2,2,Int,4}}}=nothing
                        )::HamiltonianModel{2}
    nb = length(h.labels)
    schema = InternalSchema(:wannier => h.labels)
    ### the expansion point lives ENTIRELY in the envelopes (their affine offset): the cell
    ### carries no momentum origins, so the kernels receive the plain moiré-scale midpoint
    ### bᵀ(k + (G_i+G_j)/2), and each envelope maps it to the monolayer momentum κ0 + p
    ### before the L_{ℓℓ'}⁻¹ rotation and lookup
    cell = _ertca_cell(moire, schema, Gsets)

    binvT = inv(transpose(h.lattice.b))              # cartesian k̄ → monolayer fractional
    bmT = transpose(moire.b)
    Lmat = _ertca_L(theta_top)
    shell_of = _shell_ids(h)

    ### THE TRANSFER SET IS THE BASIS'S, NOT THE HARMONICS': every difference ΔG of two
    ### basis plane waves is a candidate channel, with parent harmonic Ḡ_Δ = 𝓛⁻ᵀ·Δ (integer
    ### on the monolayer reciprocal whenever the channel can carry ERTCA weight). This loop
    ### emits the master formula's channels -- parent equals a retained ± harmonic,
    ### amplitude evaluated at the label-midpoint momentum (the paper's convention). The
    ### interlayer channels whose parent lies BEYOND the retained shells (the folded /
    ### Umklapp instances of the same stored harmonics, reached through the layers' comb
    ### equivalence) are emitted by `_emit_folded!` below when `foldspec` is set.
    transfers = Set{SVector{2,Int}}()
    for i in 1:nb, j in 1:nb, ga in Gsets[i], gb in Gsets[j]
        dG = ga - gb
        push!(transfers, _g_canonical(dG) ? dG : -dG)
    end
    binvbarT = inv(transpose(h.lattice.b))
    LinvT = inv(transpose(Lmat))

    ### the exact integer parent route, when the caller supplied M_bot − M_top
    adjP, detP = isnothing(parent) ? (nothing, 0) : (_int_adjugate(parent), _int_det(parent))

    terms = AbstractHamiltonianTerm[]
    for dGc in sort!(collect(transfers); by=g -> (g[1], g[2]))
        ### parent monolayer harmonic of this transfer (the transfer is k'−k = −bᵀ·dG for an
        ### emitted coupling targeting G+dG from G, hence the sign)
        gbar = if isnothing(adjP)
            ### k·p mode: `moire` IS built from 𝓛, so this round trip cancels exactly
            gbar_fl = binvbarT * (LinvT * (bmT * SVector{2,Float64}(-dGc)))
            gbar_i = round.(Int, gbar_fl)
            maximum(abs, gbar_fl - gbar_i) <= INT_ROUND_ATOL || continue
            SVector{2,Int}(gbar_i)
        else
            ### commensurate mode: solve (M_bot − M_top)·Ḡ = −ΔG exactly; `nothing` means the
            ### transfer genuinely carries no ERTCA parent, with no dependence on roundoff
            detP == 0 && continue
            gp = _int_solve(adjP, detP, -dGc)
            isnothing(gp) && continue
            gp
        end
        hit = _harmonic_match(h, gbar)
        isnothing(hit) && continue
        iG, sgn = hit
        dG = iszero(dGc) ? nothing : Vector{Int}(dGc)
        for i in 1:nb, j in 1:nb
            iszero(dGc) && i > j && continue     # canonical half of the true on-site block
            ### the ±class: W(k̄; −Ḡ)ᵢⱼ = conj(W(k̄; Ḡ)ⱼᵢ) regenerates the negative half
            block = sgn > 0 ? h.blocks[iG, i, j] : _conj_block(h.blocks[iG, j, i])
            (isempty(block) || sum(abs, block.amp) <= tol) && continue
            rot = _ertca_rot(h.shifted[i], h.shifted[j], theta_top)
            Q = binvT * rot
            p0 = Q * kappa0
            rule = i == j ? project(:wannier, h.labels[i]) :
                   matrix_rule(:wannier, _one_sided(nb, i, j))
            nm = "ertca_shell$(shell_of[iG])"
            selfpaired = iszero(dGc) && i == j
            if selfpaired && !_conjugate_paired(block, tol)
                throw(ArgumentError(
                    "`ertca_model`: the on-site diagonal harmonic of orbital " *
                    "`$(h.labels[i])` is not a real function (its Bloch sum is not " *
                    "conjugate-paired); the input harmonics violate Hermiticity. Rebuild " *
                    "them with `enforce_hermiticity=true`."))
            end
            push!(terms,
                  k_envelope(BlochSumEnvelope(Q, p0, block.delta, block.amp, :re);
                             dG=dG, coefficient=1.0, name=nm * "_re") ⊗ rule)
            ### the imaginary part vanishes identically on a conjugate-paired block -- the
            ### real-valued channels (every self-paired one, and any block that happens to
            ### be a real function) emit one term, not a zero-valued second kernel
            if !selfpaired && !_conjugate_paired(block, tol)
                push!(terms,
                      k_envelope(BlochSumEnvelope(Q, p0, block.delta, block.amp, :im);
                                 dG=dG, coefficient=im, name=nm * "_im") ⊗ rule)
            end
        end
    end
    isnothing(parent) ||
        _emit_intra_wrapped!(terms, h, parent, layerM, Gsets, moire, theta_top, tol, shell_of)
    isnothing(foldspec) || _emit_folded!(terms, h, foldspec, Gsets, moire, tol, shell_of)
    isempty(terms) && throw(ArgumentError(
        "`ertca_model`: every harmonic channel fell below `tol = $tol`; nothing to build."))
    return compile(cell, terms; gauge=gauge, name=name)
end

#############################################################
###### Folded (Umklapp) interlayer channels             #####
#############################################################

### Plane-wave labels of a layer are physical only modulo THAT LAYER's reciprocal lattice
### (the integer sublattice M_ℓℤ² of the moiré labels): same-layer labels differing by a
### layer reciprocal are one Wannier comb, up to the phase e^{2πih·τ̄}. The one-representative
### -per-coset tiling basis therefore carries, besides the label-midpoint channels of the
### retained harmonics, their FOLDED instances: channels whose representative-label transfer
### has a large parent, but whose amplitude is the SAME stored harmonic evaluated at the
### unwrapped MEETING momentum κ_meet -- where the two layers' extended zones actually meet
### (κ_bra + G_bottom = κ_ket + G_top). Verified against brute-force commensurate supercells
### to the sampling-alias floor (3.9e-5 on the (6,7) two-center toy at κ_cut = 12).
###
### The emission is trustworthy only inside the stacking set's ALIASING RADIUS: beyond
### |κ_meet| ≈ half the shortest alias-lattice vector, congruent branches of the class data
### bleed into each other (no window: 0.317 dev on the same toy). The window rides on the
### envelope itself, so pairs of one transfer are kept or dropped by their own κ_meet.
###
### Intralayer blocks are NEVER folded: their Bloch sums are exactly periodic in k̄ (integer
### separations, blind to d), so they have no branch localization -- a folded intralayer
### channel would leak the dispersion into transfers the true layer Hamiltonian leaves empty.

### the fold prerequisites: the exact integer supercell matrices, the alias radius of the
### stacking set, and the solve coefficients for the branch-weight (α) gate
function _ertca_foldspec(h::RegistryHarmonics, Mbot::AbstractMatrix{<:Integer},
                         Mtop::AbstractMatrix{<:Integer},
                         kappa_cut::Union{Nothing,Real},
                         fold_width::Union{Nothing,Real})::Union{Nothing,NamedTuple}
    amin = _alias_min_norm(h)
    if isnothing(amin)
        @warn "ertca_model: no alias lattice was found for this stacking set within a search " *
              "half-width of $(_alias_box(h.d)) (sized from the shifts' own denominators). " *
              "Either the set is genuinely scattered / least-squares, or it is a coset grid " *
              "perturbed beyond ~1e-9, or its denominators exceed the search box. Folded " *
              "interlayer channels cannot be attributed to sampling classes and are NOT " *
              "emitted. Use a `registry_shifts` coset set (a constant offset is fine) or a " *
              "regular d-grid, or pass `folding=:off` to silence this."
        return nothing
    end
    kcut = isnothing(kappa_cut) ? amin / 2 : Float64(kappa_cut)
    kcut > 0 || throw(DomainError(kcut, "`ertca_model`: `kappa_cut` must be positive."))
    if kcut > amin / 2 + 1e-9
        @warn "ertca_model: kappa_cut = $(round(kcut; sigdigits=4)) exceeds the stacking " *
              "set's aliasing radius $(round(amin / 2; sigdigits=4)) (half its shortest " *
              "alias-lattice vector). Beyond that radius congruent branches of the sampled " *
              "harmonics bleed into each other and the folded channels DEGRADE the model; " *
              "densify the d-sampling instead of widening the window."
    end
    F, rows, C = _solve_coefficients(h)
    return (Mbot=SMatrix{2,2,Int}(Mbot), Mtop=SMatrix{2,2,Int}(Mtop), F=F, rows=rows, C=C,
            kcut=kcut, width=(isnothing(fold_width) ? nothing : Float64(fold_width)))
end

### the shortest cartesian vector of the alias lattice {Δ ∈ ℤ² : Δ·(d_s − d₁) ∈ ℤ ∀s} -- the
### harmonics the stacking set cannot tell apart (equal characters up to ONE overall phase,
### which the α gate compensates; testing differences rather than the d's themselves keeps
### offset coset grids first-class). `nothing` when no such vector lives inside the search
### box (a scattered set). The detection must be FAIL-SAFE under jitter: a d-set that is a
### slightly perturbed coset grid would pass the strict test only on an accidental sparser
### sublattice, silently inflating the trusted radius -- so a strictly-shorter LOOSE-tolerance
### near-alias demotes the whole set (folding off, with the named warning upstream).
### the alias-lattice search half-width: lcm of the denominators the stacking shifts actually
### use (so a regular N×N grid is always reachable), floored at 24 and capped for cost
function _alias_box(ds::Vector{SVector{2,Float64}})::Int
    den = 1
    for d in ds, c in (d[1], d[2])
        x = c - floor(c)
        q = findfirst(n -> abs(x * n - round(x * n)) < 1e-7 * max(1, n), 1:512)
        isnothing(q) && continue
        den = lcm(den, q)
        den > 512 && return 512
    end
    return clamp(den, 24, 512)
end

function _alias_min_norm(h::RegistryHarmonics)::Union{Nothing,Float64}
    ### the tolerance scales with ‖Δ‖₁: stacking shifts written with ~6 printed digits (real
    ### input decks) perturb Δ·d by ≲ ‖Δ‖₁·5e-7, which must still read as the same lattice,
    ### while genuine scatter (≳1e-3) must not
    ### The box must reach the alias lattice, whose shortest vector is N for a regular N×N
    ### d-grid -- a fixed ±24 silently reports "scattered" for any grid finer than 24×24 and
    ### then advises the user to supply the regular grid they already supplied. Size it from
    ### the shift denominators actually present: reconstruct each d component's denominator by
    ### continued-fraction (rational) approximation and take their lcm, with a floor of 24 and
    ### a ceiling that keeps the O(box²·S) scan cheap.
    ###
    ### A wide box alone is NOT enough to trust the radius: any set of shifts written with a
    ### few decimals has SOME alias lattice (2-decimal shifts give 100ℤ²), and reading a huge
    ### aliasing radius off it would declare a handful of scattered stackings trustworthy
    ### everywhere. The alias-lattice argument presumes the stackings are a COMPLETE coset
    ### system of that lattice, so require exactly that: index(alias lattice) == S. A regular
    ### N×N grid gives index N² = S; a `registry_shifts` coset set gives index |F| = S; a
    ### scattered set gives an index far larger than S and is demoted.
    nmax = _alias_box(h.d)
    S = length(h.d)
    vecs(tol0) = begin
        out = SVector{2,Int}[]
        for n1 in -nmax:nmax, n2 in -nmax:nmax
            (n1 == 0 && n2 == 0) && continue
            delta = SVector(n1, n2)
            tol = tol0 * max(1, abs(n1) + abs(n2))
            all(s -> (x = dot(delta, h.d[s] - h.d[1]); abs(x - round(x)) < tol),
                eachindex(h.d)) && push!(out, delta)
        end
        out
    end
    ### successive minima give a basis of a 2D lattice, hence its index
    index_of(vs) = begin
        isempty(vs) && return 0
        nrm(v) = norm(recip_matrix_times(h.lattice, v))
        v1 = argmin(nrm, vs)
        cross(v) = v1[1] * v[2] - v1[2] * v[1]
        indep = [v for v in vs if cross(v) != 0]
        isempty(indep) && return 0
        v2 = argmin(nrm, indep)
        abs(cross(v2))
    end
    minnorm(vs) = isempty(vs) ? nothing :
                  minimum(v -> norm(recip_matrix_times(h.lattice, v)), vs)
    vs_strict = vecs(1e-6)
    vs_loose = vecs(1e-4)
    strict = minnorm(vs_strict)
    loose = minnorm(vs_loose)
    isnothing(loose) && return nothing
    ### a near-alias the strict test missed (or beat) means jittered stackings beyond
    ### rounding noise: refuse the radius rather than over-trust it
    (isnothing(strict) || loose < strict - 1e-9) && return nothing
    ### and the stackings must be a COMPLETE coset system of the lattice they alias by, or the
    ### radius is read off a structure the sampling does not actually realize
    index_of(vs_strict) == S || return nothing
    return strict
end

function _emit_folded!(terms::Vector{AbstractHamiltonianTerm}, h::RegistryHarmonics,
                       fs::NamedTuple, Gsets::Vector{<:AbstractVector{SVector{2,Int}}},
                       moire::BravaisLattice{2}, tol::Float64,
                       shell_of::Vector{Int})::Nothing
    nb = length(h.labels)
    bbarT = transpose(h.lattice.b)
    binvT = inv(bbarT)
    bmT = transpose(moire.b)
    Nmat = fs.Mbot - fs.Mtop
    adjN, detN = _int_adjugate(Nmat), _int_det(Nmat)
    adjT, detT = _int_adjugate(fs.Mtop), _int_det(fs.Mtop)
    width = isnothing(fs.width) ?
            min(norm(bmT * SVector(1.0, 0.0)), norm(bmT * SVector(0.0, 1.0))) : fs.width
    ### how far the basis momenta reach (plus one moiré cell of Bloch-k excursion): families
    ### whose meeting momentum no pair can bring inside the window are skipped outright
    kreach = maximum(norm(bmT * SVector{2,Float64}(g)) for gs in Gsets for g in gs) +
             norm(bmT * SVector(0.5, 0.5))
    bmin = minimum(svdvals(Matrix(bbarT)))
    nbox_want = ceil(Int, (fs.kcut + width + kreach) / bmin) + 1
    nbox = min(12, nbox_want)
    if nbox < nbox_want
        @warn "ertca_model: the Umklapp-meeting enumeration is capped at ±$nbox layer " *
              "reciprocals but the window (κ_cut + roll-off + basis reach) needs " *
              "±$nbox_want; in-window folded meetings beyond the cap are NOT emitted. This " *
              "happens on dense stacking sets, whose large aliasing radius widens the " *
              "trusted window. Lower `kappa_cut` to match the cap, or accept the truncation " *
              "knowing which side it errs on (missing weight, never spurious weight)."
    end

    ### ordered transfers (both signs) between unshifted-bra and shifted-ket orbitals; the
    ### finalize step regenerates every (shifted, unshifted) partner block, so this
    ### orientation covers the full Hermitian interlayer structure
    transfers = Set{SVector{2,Int}}()
    for i in 1:nb, j in 1:nb
        (!h.shifted[i] && h.shifted[j]) || continue
        for ga in Gsets[i], gb in Gsets[j]
            push!(transfers, ga - gb)
        end
    end

    ### The branch-weight gate, cached per parent: the stored row that carries the parent, at
    ### unit weight. α is a pure PHASE only for a SQUARE solve (S = |F|, C = E⁻¹), where every
    ### branch is congruent to exactly one retained label modulo the alias lattice. On an
    ### OVERDETERMINED regular grid the columns of E are orthogonal characters, pinv(E) = E†/S,
    ### so α is exactly 1 on the retained labels and exactly 0 on every other branch -- the gate
    ### then drops all non-congruent families, and the fold content SHRINKS as the d-sampling is
    ### refined. Retaining more shells is the cure (it enlarges the covered residue classes);
    ### the rejection tally below exists so that can never happen silently.
    alpha_cache = Dict{SVector{2,Int},Union{Nothing,Tuple{Int,Int,ComplexF64}}}()
    rejected = Set{SVector{2,Int}}()
    function alpha_row(gp::SVector{2,Int})
        get!(alpha_cache, gp) do
            m_best = 0
            alpha = zero(ComplexF64)
            for m in eachindex(fs.F)
                a = sum(fs.C[m, s] * cispi(-2 * dot(gp, h.d[s])) for s in eachindex(h.d))
                if abs(a) > abs(alpha)
                    alpha = a
                    m_best = m
                end
            end
            if !(m_best != 0 && abs(abs(alpha) - 1) <= 1e-6)
                push!(rejected, gp)
                return nothing
            end
            (fs.rows[m_best][1], fs.rows[m_best][2], alpha)
        end
    end

    nfam = 0
    for dG in sort!(collect(transfers); by=g -> (g[1], g[2]))
        dGcart = bmT * SVector{2,Float64}(dG)
        ### the diagonal parent, when it exists AND is retained, is the master formula's own
        ### label-midpoint channel -- its diagonal meeting must not be re-emitted here
        gp_diag = detN == 0 ? nothing : _int_solve(adjN, detN, -dG)
        diag_retained = !isnothing(gp_diag) && !isnothing(_harmonic_match(h, gp_diag))
        ### EVERY integer meeting M_bot·n_b − M_top·n_t = −dG is a physical channel (the
        ### solutions form a coset of the layers' common reciprocal lattice; on cells with
        ### |det(M_bot − M_top)| > 1 most transfers have NO diagonal parent yet carry real
        ### Umklapp weight through independent meetings -- and exact score ties on Voronoi
        ### facets are symmetry partners that must BOTH be emitted). Each in-reach meeting
        ### becomes its own family; its window keeps or drops every pair by |κ_meet|.
        for m1 in -nbox:nbox, m2 in -nbox:nbox
            nbv = SVector(m1, m2)
            ntv = _int_solve(adjT, detT, fs.Mbot * nbv + dG)
            isnothing(ntv) && continue
            diag_retained && nbv == gp_diag && ntv == gp_diag && continue
            off = bmT * SVector{2,Float64}(fs.Mbot * nbv)    # κ_meet = κ_bra + off
            kmeet_c = off + dGcart / 2                       # the family's κ_meet centroid
            norm(kmeet_c) - kreach > fs.kcut + width && continue     # never in-window
            ### the canonical branch index: the harmonic nearest the meeting momentum (any
            ### congruent choice gives the same channel -- the α phase and the comb phases
            ### compensate -- so pick the one whose branch the evaluation reads directly)
            gp = SVector{2,Int}(round.(Int, binvT * kmeet_c))
            hit = alpha_row(gp)
            isnothing(hit) && continue
            iG, sgn, alpha = hit
            ### the channel: value = Ŵ_row(u)·phase, u = b̄⁻ᵀκ_meet − Ḡp; the envelope
            ### receives the representative midpoint p = b_mᵀ(k + (G_i+G_j)/2), and
            ### κ_bra = p + b_mᵀdG/2
            Q = SMatrix{2,2,Float64}(binvT)
            p0 = SVector{2,Float64}(binvT * (bmT * (SVector{2,Float64}(dG) / 2) + off) -
                                    SVector{2,Float64}(gp))
            wshift = SVector{2,Float64}(bbarT * SVector{2,Float64}(gp))
            for i in 1:nb, j in 1:nb
                (!h.shifted[i] && h.shifted[j]) || continue
                block = sgn > 0 ? h.blocks[iG, i, j] : _conj_block(h.blocks[iG, j, i])
                (isempty(block) || sum(abs, block.amp) <= tol) && continue
                ### comb wrap phases (τ̄ in monolayer-fractional), and the branch weight's
                ### inverse (a pure phase) from the α gate
                phase = cispi(2 * (dot(SVector{2,Float64}(gp - ntv), h.tau[j]) +
                                   dot(SVector{2,Float64}(nbv - gp), h.tau[i]))) * conj(alpha)
                nfam += 1
                ### name by HARMONIC SHELL, not by a running counter: a per-term counter
                ### makes every folded channel its own name, so `reweight` cannot ablate
                ### them as a group and the names are unstable under any change of basis
                nm = "ertca_fold$(shell_of[iG])"
                rule = matrix_rule(:wannier, _one_sided(nb, i, j))
                env_re = BlochSumEnvelope(Q, p0, block.delta, block.amp, :re;
                                          wmap=bbarT, wshift=wshift, wradius=fs.kcut,
                                          wwidth=width)
                push!(terms, k_envelope(env_re; dG=Vector{Int}(dG), coefficient=phase,
                                        name=nm * "_re") ⊗ rule)
                if !_conjugate_paired(block, tol)
                    env_im = BlochSumEnvelope(Q, p0, block.delta, block.amp, :im;
                                              wmap=bbarT, wshift=wshift, wradius=fs.kcut,
                                              wwidth=width)
                    push!(terms, k_envelope(env_im; dG=Vector{Int}(dG),
                                            coefficient=im * phase, name=nm * "_im") ⊗ rule)
                end
            end
        end
    end
    ### the gate above accepts a branch only when the stacking set cannot distinguish it from a
    ### retained harmonic. A refined d-grid distinguishes MORE, so it accepts FEWER -- the one
    ### place in the pipeline where better data yields a thinner model. Never let that be silent.
    if !isempty(rejected)
        keep = length(alpha_cache) - length(rejected)
        @warn "ertca_model: $(length(rejected)) of $(length(alpha_cache)) folded-channel " *
              "branches were dropped because their branch harmonic is neither retained nor " *
              "aliased onto a retained one by this stacking set (kept $keep). The gate accepts " *
              "a branch only when the d-sampling cannot TELL IT APART from a retained " *
              "harmonic, so a denser registry grid accepts fewer: an S ≫ |F| regular grid " *
              "keeps only the |F| retained residue classes, and the model loses real Umklapp " *
              "weight. Retain more shells (raise `gshells`) so the retained set covers the " *
              "branches the basis actually reaches, or solve on a `registry_shifts` coset set " *
              "of exactly |F| stackings. Dropped branch labels (up to 12): " *
              "$(Tuple.(first(sort!(collect(rejected); by=g->(abs(g[1])+abs(g[2]), g[1], g[2])), 12)))"
    end
    return nothing
end

#############################################################
###### Comb-wrapped INTRALAYER channels                 #####
#############################################################

### The interlayer counterpart of this pass is `_emit_folded!`; the same comb equivalence acts
### on SAME-LAYER pairs and was long assumed not to, on the grounds that intralayer Bloch sums
### are periodic in k̄. Periodicity is not the issue -- REPRESENTABILITY is. A physical
### intralayer channel connects extended-zone momenta κ and κ + 𝓛ᵀḠ; when the target leaves the
### fundamental domain the one-representative-per-coset basis stands in for it with the coset
### rep, and the LABEL transfer to that rep has a parent far outside the retained shells. The
### master loop then reads it as "not an ERTCA channel" and emits exactly zero, losing the
### channels whose labels sit within one moiré vector of the monolayer zone boundary.
###
### The identity is exact (no O(θ²) enters L_{ℓℓ}). With κ = b_mᵀ(k+G), a ket label G_b, and
### the bra standing in for the wrapped target:
###
###     G_a = G_b + M_ℓ m − 𝓝 Ḡ ,   𝓝 = M_fixed − M_shifted ,   dG = G_a − G_b
###     ⟨a,i|H|b,j⟩ = e^{−2πi m·τ̄_j} · W_{ij}(k*; Ḡ) ,   k* = b̄⁻ᵀ L_ℓ⁻¹ (κ_a + κ_b + G_B)/2
###
### with G_B = b_mᵀ M_ℓ m the layer reciprocal the wrap removed. Crucially m = M_ℓ⁻¹(dG + 𝓝Ḡ)
### depends only on (Ḡ, dG) and NOT on G_b, so every basis pair a transfer selects shares one
### comb phase and one momentum offset: an ordinary `dG` term carries the whole channel, with no
### window and no branch/α machinery. m = 0 is the master loop's own channel and is skipped.
function _emit_intra_wrapped!(terms::Vector{AbstractHamiltonianTerm}, h::RegistryHarmonics,
                              Nmat::SMatrix{2,2,Int},
                              layerM::Union{Nothing,Vector{SMatrix{2,2,Int,4}}},
                              Gsets::Vector{<:AbstractVector{SVector{2,Int}}},
                              moire::BravaisLattice{2}, theta_top::Float64,
                              tol::Float64, shell_of::Vector{Int})::Nothing
    isnothing(layerM) && return nothing
    nb = length(h.labels)
    binvT = inv(transpose(h.lattice.b))
    bmT = transpose(moire.b)

    ### coset representative lookup per orbital, on the INTEGERS (adj(M)·G mod det M)
    reps = map(eachindex(Gsets)) do b
        M = layerM[b]
        adjM, nd = _int_adjugate(M), _int_det_abs(M)
        t = Dict{SVector{2,Int},SVector{2,Int}}()
        for g in Gsets[b]
            t[_coset_label(adjM, nd, g)] = g
        end
        t
    end
    repof(b, g) = begin
        M = layerM[b]
        get(reps[b], _coset_label(_int_adjugate(M), _int_det_abs(M), g), nothing)
    end

    ### (Ḡ, m) channels, keyed by (dG, i, j): one term each
    nemit = 0
    for i in 1:nb, j in 1:nb
        h.shifted[i] == h.shifted[j] || continue
        M = layerM[j]
        adjM, detM = _int_adjugate(M), _int_det(M)
        rot = _ertca_rot(h.shifted[i], h.shifted[j], theta_top)     # L_ℓ⁻¹
        Q = binvT * rot
        ### m depends only on (Ḡ, dG), so all ket labels sharing a transfer are ONE channel
        seen = Set{Tuple{Int,Int,Int,Int}}()
        for (iG, gc) in pairs(h.gvecs), sgn in (1, -1)
            (sgn < 0 && iszero(gc)) && continue
            gbar = sgn > 0 ? gc : -gc
            for gb in Gsets[j]
                gt = gb - Nmat * gbar                       # the UNWRAPPED bra label
                ga = repof(i, gt)
                isnothing(ga) && continue
                mv = _int_solve(adjM, detM, ga - gt)        # the wrap, in layer reciprocals
                (isnothing(mv) || iszero(mv)) && continue   # m = 0 is the master loop's
                dGc = ga - gb
                _g_canonical(dGc) || continue               # one representative per ± pair
                (iszero(dGc) && i > j) && continue
                key = (dGc[1], dGc[2], iG, sgn)
                key in seen && continue
                push!(seen, key)
                block = sgn > 0 ? h.blocks[iG, i, j] : _conj_block(h.blocks[iG, j, i])
                (isempty(block) || sum(abs, block.amp) <= tol) && continue
                GB = bmT * SVector{2,Float64}(M * mv)
                p0 = Q * (GB / 2)
                phase = cispi(-2 * dot(SVector{2,Float64}(mv), h.tau[j]))
                rule = i == j ? project(:wannier, h.labels[i]) :
                       matrix_rule(:wannier, _one_sided(nb, i, j))
                nm = "ertca_wrap$(shell_of[iG])"
                push!(terms,
                      k_envelope(BlochSumEnvelope(Q, p0, block.delta, block.amp, :re);
                                 dG=iszero(dGc) ? nothing : Vector{Int}(dGc),
                                 coefficient=phase, name=nm * "_re") ⊗ rule)
                push!(terms,
                      k_envelope(BlochSumEnvelope(Q, p0, block.delta, block.amp, :im);
                                 dG=iszero(dGc) ? nothing : Vector{Int}(dGc),
                                 coefficient=im * phase, name=nm * "_im") ⊗ rule)
                nemit += 1
            end
        end
    end
    return nothing
end

### the exactly-retained representative: iG and sign with gbar == ±gvecs[iG]
function _harmonic_match(h::RegistryHarmonics,
                         gbar::SVector{2,Int})::Union{Nothing,Tuple{Int,Int}}
    for (iG, g) in pairs(h.gvecs)
        gbar == g && return (iG, 1)
        gbar == -g && return (iG, -1)
    end
    return nothing
end

### W(k̄; −Ḡ)ᵢⱼ = conj(W(k̄; Ḡ)ⱼᵢ): conjugate amplitudes on negated separations. `_pz` is
### load-bearing: negating a zero component gives −0.0, and `_conjugate_paired` looks partners
### up in an `isequal`-keyed Dict, where −0.0 ≠ +0.0 -- so without it EVERY negative-harmonic
### channel misclassifies as complex and emits a second, identically-zero `_im` kernel.
_conj_block(b::HarmonicBlock) =
    HarmonicBlock([SVector(_pz(-d[1]), _pz(-d[2])) for d in b.delta], conj.(b.amp))

### shell index per canonical harmonic: 0 for Ḡ = 0, then ranked by |Ḡ| (cartesian)
function _shell_ids(h::RegistryHarmonics)::Vector{Int}
    norms = [norm(recip_matrix_times(h.lattice, g)) for g in h.gvecs]
    order = sortperm(norms)
    ids = zeros(Int, length(norms))
    shell = -1
    last = -Inf
    for m in order
        if norms[m] - last > SHELL_RTOL * max(norms[m], 1e-12) + 1e-12
            shell += 1
            last = norms[m]
        end
        ids[m] = shell
    end
    return ids
end

@inline function _one_sided(nb::Int, i::Int, j::Int)::Matrix{ComplexF64}
    E = zeros(ComplexF64, nb, nb)
    E[i, j] = 1
    return E
end

### the plane-wave cell with PER-ORBITAL G sets: the plane_wave_cell state order (G outermost,
### internal innermost, over the union) with the (G, β) pairs an orbital's set does not carry
### simply absent -- each orbital tiles its own zone
function _ertca_cell(moire::BravaisLattice{2}, schema::InternalSchema,
                     Gsets::Vector{<:AbstractVector{SVector{2,Int}}})::UnitCell{2}
    length(Gsets) == n_internal(schema) || throw(DimensionMismatch(
        "`ertca_model`: $(length(Gsets)) plane-wave sets for $(n_internal(schema)) orbitals."))
    for (b, gs) in pairs(Gsets)
        isempty(gs) && throw(ArgumentError(
            "`ertca_model`: orbital $b carries an empty plane-wave set."))
        allunique(gs) || throw(ArgumentError(
            "`ertca_model`: orbital $b's plane-wave set contains duplicate G vectors."))
    end
    union_G = SVector{2,Int}[]
    seen = Set{SVector{2,Int}}()
    for gs in Gsets, g in gs
        g in seen && continue
        push!(seen, g)
        push!(union_G, g)
    end
    member = [Set(gs) for gs in Gsets]
    pos = zero(SVector{2,Float64})
    basis = BasisState{2}[]
    for g in union_G, b in 1:n_internal(schema)
        g in member[b] || continue
        push!(basis, BasisState{2}(pos, g, zero(SVector{2,Float64}), b, schema))
    end
    return UnitCell{2}(moire, schema, [:origin], [pos], basis; site_kind=[:plane_wave])
end

### a Bloch sum is a REAL function iff its entries pair as (δ, A) / (−δ, conj A)
function _conjugate_paired(b::HarmonicBlock, tol::Float64)::Bool
    lookup = Dict(b.delta[m] => b.amp[m] for m in eachindex(b.amp))
    scale = max(sum(abs, b.amp; init=0.0), 1.0)
    for m in eachindex(b.amp)
        neg = SVector(_pz(-b.delta[m][1]), _pz(-b.delta[m][2]))
        partner = get(lookup, neg, nothing)
        (isnothing(partner) || abs(partner - conj(b.amp[m])) > tol * scale) && return false
    end
    return true
end
