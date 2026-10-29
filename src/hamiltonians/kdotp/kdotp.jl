##################################################################
###### export only the following functions from this module ######
##################################################################

export KPolynomialTerm
export continuum_cell, k_polynomial, angular_harmonic, discretize, dirac_cone




#############################################################
###### Lattice-free k·p cells                           #####
#############################################################

"""
    continuum_cell(D; schema=nothing, origin=nothing, site_name=:origin) -> UnitCell{D}

A lattice-free k·p cell: one `:plane_wave` site with `G`-set `{0}` on the identity frame
(`a = I`, so `k_cart = 2π k_frac`) -- the lattice is only a coordinate system.

    c = continuum_cell(2; schema=InternalSchema(:band => [:c, :v]))
    bhz = compile(c, [onsite(M)        ⊗ pauli_z(:band; up=:c, down=:v),
                      k_quadratic(-B)  ⊗ pauli_z(:band; up=:c, down=:v),
                      k_linear([A, 0]) ⊗ pauli_x(:band; up=:c, down=:v),
                      k_linear([0, A]) ⊗ pauli_y(:band; up=:c, down=:v)])
    bands(bhz, [0.3, 0.0]; cartesian=true)

`origin` is the expansion point, cartesian, stored on the states (see `plane_wave_cell`). The
resulting models are not k-periodic (`kperiodicity(model) == :none`): analysis functions take an
explicit `kregion`, `chern_number` refuses in favour of [`berry_flux`](@ref) (a bare Dirac cone
honestly integrates to ±½), and [`discretize`](@ref) is the route to a genuine lattice model
with exactly integer topology.
"""
function continuum_cell(D::Integer;
                        schema::Union{Nothing,InternalSchema}=nothing,
                        origin::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        site_name::Symbol=:origin)
    D >= 1 || throw(DomainError(D, "The dimension must be at least 1."))
    frame = BravaisLattice(Matrix{Float64}(LinearAlgebra.I, D, D))
    return plane_wave_cell(frame; shells=0, schema=schema, origin=origin, site_name=site_name)
end




"""
    dirac_cone(v; axis, chirality=+1, hbar=1.0, sites=Symbol[], name="dirac_cone")
        -> Vector{KLinearTerm}

Both components of a 2D Dirac cone in one call:

    ħv (χ p_x σ_x + p_y σ_y)        on the two-level internal axis `axis`,

with `χ = chirality = ±1` selecting the valley convention (the other valley of graphene is
`χ = -1`; state the convention instead of silently assuming it). Pseudospin-up is resolved by
LEVEL VALUES where they speak: on an axis whose two levels are `{+1, -1}` (either declaration
order), up = `+1` -- the same convention as `pauli_x/y/z` and `raise`/`lower`, so the builder
and the `dirac ⊗ pauli_*` spelling always agree. On any other two-level axis
(`:band => [:c, :v]`) the FIRST declared level is up. Override explicitly with
`up=`/`down=` when matching an external convention.

The cone sits at each state's own momentum origin: with per-level `origins=` on the cell, ONE
`dirac_cone` covers every layer at its own Dirac point -- no per-layer `k0=` + `project`
duplication. The momenta are `p = bᵀ(k + G - K_s)` throughout.

2D only, by construction (the two emitted `k_linear` coefficients are 2-vectors): compiling
against a cell of any other dimension raises a `DimensionMismatch` naming the term. Being a
k-kernel, it is refused on `:localized` sites at compile, with the sites named.
"""
function dirac_cone(v::Real; axis::Symbol, chirality::Integer=+1, hbar::Real=1.0,
                    up=nothing, down=nothing,
                    sites=Symbol[], name::AbstractString="dirac_cone")
    abs(chirality) == 1 || throw(DomainError(chirality,
        "`chirality` selects the valley convention and must be ±1."))
    return [k_linear([Float64(chirality * hbar * v), 0.0]; sites=sites,
                     name=String(name) * "_x") ⊗ TwoLevelMatrixQnum(axis, _SIGMA_X, up, down),
            k_linear([0.0, Float64(hbar * v)]; sites=sites,
                     name=String(name) * "_y") ⊗ TwoLevelMatrixQnum(axis, _SIGMA_Y, up, down)]
end




#############################################################
###### Momentum polynomials beyond quadratic order      #####
#############################################################

"""
    KPolynomialTerm

A term whose amplitude is a real momentum polynomial:

    coefficient * Σ_m  c_m  Π_d  p_d^{α_md}          (c_m real)

subsuming `KLinearTerm`/`KQuadraticTerm` (which stay, as the fast common cases) and covering
cubic and higher orders -- TMD conduction-band corrections, trigonal warping. The package
invariant is preserved: THE k-DEPENDENT SCALAR IS REAL, every phase lives in the quantum-number
rule or the complex `coefficient`. This loses no generality -- any Hermitian k·p block
decomposes as Σ (real polynomial) × (Hermitian rule matrix); [`angular_harmonic`](@ref) does
that decomposition for the common `(k_x ± i k_y)^n` blocks.

Build with [`k_polynomial`](@ref). The exponent table fixes the dimension, checked against the
cell at compile.
"""
struct KPolynomialTerm <: KDependentTerm
    c::Vector{Float64}
    alpha::Matrix{Int}                  # n_monomials × D exponent table
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KPolynomialTerm(c::Vector{Float64}, alpha::Matrix{Int}, k0::Vector{Float64},
                             selector::AbstractBondSelector, rule::AbstractQnumRule,
                             coefficient::ComplexF64, name::String)
        length(c) == size(alpha, 1) || throw(DimensionMismatch(
            "`KPolynomialTerm` `$name`: $(length(c)) coefficients for $(size(alpha, 1)) monomials."))
        isempty(c) && throw(ArgumentError(
            "`KPolynomialTerm` `$name` has no monomials; it would emit nothing."))
        all(isfinite, c) || throw(DomainError(c,
            "`KPolynomialTerm` `$name`: a monomial coefficient is not finite."))
        all(>=(0), alpha) || throw(DomainError(alpha,
            "`KPolynomialTerm` `$name`: exponents must be non-negative integers."))
        all(isfinite, k0) || throw(DomainError(k0,
            "`KPolynomialTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KPolynomialTerm` `$name`: the prefactor is not finite."))
        return new(c, alpha, k0, selector, rule, coefficient, name)
    end
end

is_serializable(::KPolynomialTerm) = true
@inline _k0_raw(term::KPolynomialTerm) = term.k0_frac
@inline _kcoefficient(term::KPolynomialTerm) = term.coefficient
@inline _kselector(term::KPolynomialTerm) = term.selector
@inline _krule(term::KPolynomialTerm) = term.rule

otimes(term::KPolynomialTerm, rule::AbstractQnumRule) =
    KPolynomialTerm(term.c, term.alpha, term.k0_frac, term.selector,
                    compose(term.rule, rule), term.coefficient, term.name)

### `KPolynomialKernel` and its `_kvalue` live with the other kernels in
### `core/models/terms.jl`; this file holds the term type and the builders.

function _make_kernel(term::KPolynomialTerm, ::Val{D}) where {D}
    size(term.alpha, 2) == D || throw(DimensionMismatch(
        "Term `$(term.name)`: the exponent table covers $(size(term.alpha, 2)) momentum " *
        "components but the cell is $D-dimensional."))
    nm = length(term.c)
    return KPolynomialKernel(SVector{nm,Float64}(term.c),
                             SMatrix{nm,D,Int}(term.alpha))
end

### parity under p -> -p: definite when every monomial's total degree has the same parity; a
### mixed polynomial must be split before `nambu_doubling` can use it
function _kernel_parity(kern::KPolynomialKernel{D,NM}) where {D,NM}
    degs = [sum(kern.alpha[m, :]) for m in 1:NM]
    if all(iseven, degs)
        return 1
    elseif all(isodd, degs)
        return -1
    end
    throw(ArgumentError(
        "`nambu_doubling` needs a definite parity under p -> -p, but this momentum polynomial " *
        "mixes even and odd monomials (total degrees $degs). Split it into its even and odd " *
        "parts first."))
end

"""
    k_polynomial(monomials; coefficient=1.0, dG=nothing, k0=Float64[], sites=Symbol[],
                 sites_from=Symbol[], sites_to=Symbol[], name="k_polynomial") -> KPolynomialTerm

The general momentum polynomial `coefficient * Σ_m c_m Π_d p_d^α`, with `monomials` an iterable
of `exponents => c` pairs (real `c`):

    k_polynomial([[2, 0] => 1.0, [0, 2] => -1.0])       # k_x² - k_y²
    k_polynomial([[3, 0] => 1.0, [1, 2] => -3.0])       # Re k_+³: trigonal warping

The `dG`/`k0` keywords work exactly as in [`k_linear`](@ref). For `(k_x ± i k_y)^n` blocks use
[`angular_harmonic`](@ref), which does the real-polynomial × Hermitian-rule decomposition for
you.
"""
function k_polynomial(monomials; coefficient::Number=1.0,
                      dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                      k0::AbstractVector{<:Real}=Float64[],
                      sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                      name::AbstractString="k_polynomial")
    ms = collect(monomials)
    isempty(ms) && throw(ArgumentError("`k_polynomial` needs at least one monomial."))
    Dm = length(first(ms).first)
    alpha = Matrix{Int}(undef, length(ms), Dm)
    c = Vector{Float64}(undef, length(ms))
    for (m, pr) in pairs(ms)
        pr isa Pair || throw(ArgumentError(
            "`k_polynomial` monomials are `exponents => coefficient` pairs; got $(typeof(pr))."))
        length(pr.first) == Dm || throw(DimensionMismatch(
            "`k_polynomial` monomial exponent lists must share one length; got " *
            "$(length(pr.first)) and $Dm."))
        imag(ComplexF64(pr.second)) == 0 || throw(ArgumentError(
            "`k_polynomial` monomial coefficients must be REAL (`$(pr.second)` given): every " *
            "phase belongs in `coefficient` or in the quantum-number rule -- that is what keeps " *
            "H(k) Hermitian by construction."))
        alpha[m, :] = Int.(pr.first)
        c[m] = Float64(real(pr.second))
    end
    sel = isnothing(dG) ? OnSiteSelector(_site_list(sites)) :
          DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to))
    return KPolynomialTerm(c, alpha, Vector{Float64}(k0), sel, AnyQnum(),
                           ComplexF64(coefficient), String(name))
end

"""
    angular_harmonic(w, n; axis, sign=+1, dG=nothing, k0=Float64[], sites=Symbol[],
                     sites_from=Symbol[], sites_to=Symbol[], name="angular_harmonic")
        -> Vector{KPolynomialTerm}

The Hermitian block `w · (k_x + sign·i·k_y)^n · σ₊(axis) + h.c.`, decomposed automatically into
its real-polynomial × Hermitian-rule components:

    w k_±ⁿ σ₊ + h.c.  =  Re[w k_±ⁿ] σ_x  −  Im[w k_±ⁿ] σ_y

so trigonal warping is one call instead of a by-hand decomposition with a sign error waiting in
it:

    angular_harmonic(λ, 2; axis=:band)     # λ(k_x² − k_y²)σ_x − 2λ k_x k_y σ_y   (k_+² σ_+ + h.c.)

`w` may be complex (its phase folds into the polynomials); 2D only.
"""
function angular_harmonic(w::Number, n::Integer; axis::Symbol, sign::Integer=+1,
                          up=nothing, down=nothing,
                          dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          k0::AbstractVector{<:Real}=Float64[],
                          sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                          name::AbstractString="angular_harmonic")
    n >= 1 || throw(DomainError(n, "The harmonic order must be at least 1."))
    abs(sign) == 1 || throw(DomainError(sign, "`sign` selects k₊ (+1) or k₋ (-1)."))
    ### the σ₊ + h.c. = Re⊗σ_x − Im⊗σ_y decomposition holds only when the element and its h.c.
    ### live on the SAME entries; at ΔG ≠ 0 the h.c. sits at −ΔG and the decomposition would
    ### smuggle in a spurious σ₋ channel at +ΔG
    isnothing(dG) || iszero(dG) || throw(ArgumentError(
        "`angular_harmonic` is an ON-SITE block (its σ₊ + h.c. decomposition needs the " *
        "conjugate on the same entries); at ΔG ≠ 0 build the real and imaginary polynomial " *
        "parts explicitly with `k_polynomial(...; dG=...)` and one-sided ladder rules."))
    wr, wi = real(ComplexF64(w)), imag(ComplexF64(w))

    ### (k_x + i s k_y)^n = Σ_j binom(n, j) (i s)^j k_x^{n-j} k_y^j = P + iQ
    P = zeros(Float64, n + 1)                  # coefficient of k_x^{n-j} k_y^j
    Q = zeros(Float64, n + 1)
    for j in 0:n
        z = binomial(n, j) * (im * sign)^j
        P[j + 1] = real(z)
        Q[j + 1] = imag(z)
    end

    kwargs = (dG=dG, k0=k0, sites=sites, sites_from=sites_from, sites_to=sites_to)
    out = KPolynomialTerm[]
    cx = wr .* P .- wi .* Q                    # Re[w k_±ⁿ]
    cy = .-(wr .* Q .+ wi .* P)                # −Im[w k_±ⁿ]
    ### value-resolving two-level rules: {+1,-1} axes take up = +1 (the pauli_* convention,
    ### whatever the declaration order), named axes like (:c, :v) keep declaration order --
    ### see `TwoLevelMatrixQnum`; a wider axis raises at compile with the axis named
    if any(!iszero, cx)
        ms = [[n - j, j] => cx[j + 1] for j in 0:n if !iszero(cx[j + 1])]
        push!(out, k_polynomial(ms; name=String(name) * "_x", kwargs...) ⊗
                   TwoLevelMatrixQnum(axis, _SIGMA_X, up, down))
    end
    if any(!iszero, cy)
        ms = [[n - j, j] => cy[j + 1] for j in 0:n if !iszero(cy[j + 1])]
        push!(out, k_polynomial(ms; name=String(name) * "_y", kwargs...) ⊗
                   TwoLevelMatrixQnum(axis, _SIGMA_Y, up, down))
    end
    return out
end




#############################################################
###### Discretization onto a lattice                    #####
#############################################################

"""
    discretize(model, astep) -> HamiltonianModel{D}

Regularize a k·p model (`NoZone()`, `G`-set `{0}`) on a hypercubic lattice of
constant `astep`: every plane-wave state becomes a localized orbital, and every kernel entry is
rewritten into constant couplings by the standard finite-difference stencils

    c·p        ->  Σ_μ c_μ sin(p_μ a)/a               entries at ±e_μ, coefficient ∓i c_μ/(2a)
    p_μ²       ->  (2 − 2cos p_μ a)/a²                2/a² on site, −1/a² at ±e_μ
    p_μ p_ν    ->  sin(p_μ a) sin(p_ν a)/a²           −s_μ s_ν /(4a²) at ΔR = s_μ e_μ + s_ν e_ν

exact to `O(a²)` around each state's expansion point. A parent momentum origin `K` is fully
consumed here: the k-independent phase `e^{2πi q·δ}` of every stencil entry (with `q = −K`) is
baked into the stored amplitude and the output carries `q ≡ 0`, `k0 ≡ 0` -- so the result
really is an ordinary tight-binding model (`open_boundary` accepts it, and terms compiled
later on the cell inherit no hidden origin): exactly k-periodic, integer Chern numbers,
`open_boundary`, disorder --

    bhz_lat = discretize(bhz, 0.25)
    chern_number(bhz_lat, 1; nk=30)                   # exactly ±1 or 0
    ribbon  = open_boundary(bhz_lat.cell, 2; ncells=60)

Supported kernels: constant, linear, quadratic. A polynomial or callback kernel has no unique
finite-difference stencil and is refused.

ONE site, by construction: the stencil is a single hypercubic star, and a plane-wave site's
declared position is gauge-only (it never enters a Bloch phase) while the discretized orbitals
are `:localized` and theirs do. A multi-site continuum cell -- what `direct_sum` of two k·p
blocks produces -- is therefore refused by name; carry the extra blocks as INTERNAL levels
(`continuum_cell(D; schema=InternalSchema(:block => [:a, :b]))`) instead of separate sites.
"""
function discretize(model::HamiltonianModel{D}, astep::Real)::HamiltonianModel{D} where {D}
    astep > 0 || throw(DomainError(astep, "The lattice constant must be positive."))
    kperiodicity(model) isa NoZone || throw(ArgumentError(
        "`discretize` regularizes k·p frames -- `NoZone()`, a G-set of {0} -- onto a real " *
        "lattice; this model is already `$(repr(kperiodicity(model)))`, so it already has a " *
        "Brillouin zone to be periodic over."))
    nperiodic(model.cell.lattice) == D || throw(ArgumentError(
        "`discretize` builds a hypercubic stencil along every direction, which a partially " *
        "periodic frame does not support; discretize the fully periodic continuum model, then " *
        "`open_boundary` the result."))
    ### the k·p frame must be a scaled identity: a hypercubic stencil needs axis-aligned
    ### momenta. LATTICE_ATOL is the package's position-grid tolerance -- the frame is
    ### constructed exactly, so anything beyond rounding noise means a different lattice.
    aM = Matrix(model.cell.lattice.a)
    s0 = aM[1, 1]
    isapprox(aM, s0 * Matrix{Float64}(LinearAlgebra.I, D, D); atol=LATTICE_ATOL) || throw(ArgumentError(
        "`discretize` needs the identity-frame lattice of `continuum_cell`; this cell's frame " *
        "is not a scaled identity."))
    ### ONE stencil, one site. A plane-wave site has no centre: its declared position is
    ### gauge-only and `_effective_positions` zeroes it, so the parent's phase displacement is
    ### δ = ΔR and the stencil algebra below (which assumes exactly that) is exact. The output
    ### states are :localized, and a localized position DOES enter every phase as δ = ΔR + τ_j
    ### − τ_i. With two plane-wave sites there is no honest output geometry: coincident (the
    ### `direct_sum` default) is not a legal localized registry at all, and separated by a
    ### `shift_frac` would promote a gauge-only offset into a physical e^{2πi k·(τ_j−τ_i)} the
    ### continuum model never carried. Refuse rather than pick one -- and refuse HERE, where
    ### the physics can be named, instead of leaving the `UnitCell` constructor to throw a
    ### bare DimensionMismatch about site-kind lengths.
    ns = n_sites(model.cell)
    ns == 1 || throw(ArgumentError(
        "`discretize` regularizes a ONE-SITE continuum cell onto a single hypercubic stencil, " *
        "but this cell carries $ns sites ($(join(model.cell.site_names, ", "))). A plane-wave " *
        "site has no centre -- its position is gauge-only and never enters a Bloch phase -- " *
        "whereas the discretized orbitals are `:localized` and their positions enter every " *
        "phase, so there is no position this function could hand a second site without " *
        "inventing a hopping geometry the k·p model never had (and two of them at the same " *
        "position is not a valid localized site registry at all). Carry the extra blocks as " *
        "INTERNAL levels rather than sites -- " *
        "`continuum_cell($D; schema=InternalSchema(:block => [:a, :b]))`, with each block's " *
        "terms tagged `⊗ project(:block, :a)` -- and discretize that cell; a `direct_sum` of " *
        "two continuum models is precisely this cell in the shape `discretize` cannot use."))

    a = Float64(astep)
    ### output cell: same states, localized, on the hypercubic lattice of constant `astep`.
    ### The momentum origin is CONSUMED below (its k-independent phase baked into the stored
    ### amplitudes), so the output states carry k0 = 0 -- a true TB cell that `open_boundary`
    ### accepts and on which later-compiled terms inherit no hidden origin.
    scale = a / s0
    new_lat = BravaisLattice(a * Matrix{Float64}(LinearAlgebra.I, D, D))
    sch = model.cell.schema
    basis = [BasisState{D}(b.r_frac, zero(SVector{D,Int}), zero(SVector{D,Float64}),
                           b.internal, sch) for b in model.cell.basis]
    ### the kind vector is DERIVED from the site registry, never spelled out: `site_names` and
    ### `site_positions` are copied at the parent's full length, and a literal `[:localized]`
    ### alongside them is a second source of truth for the same count that can only ever drift
    new_cell = UnitCell{D}(new_lat, sch, copy(model.cell.site_names),
                           copy(model.cell.site_positions), basis;
                           atol=model.cell.atol, site_kind=fill(:localized, ns))

    c0 = model.couplings
    buf = CouplingBuffer{D}()
    ### provenance rides along: every stencil entry descends from exactly one parent term, so
    ### mass/velocity sweeps on the DISCRETIZED model reweight without re-discretizing
    adopt_provenance!(buf, c0)
    zeroR = zero(SVector{D,Int})
    for κ in eachindex(c0.kernels)
        isempty(c0.groups[κ]) && continue
        kern = c0.kernels[κ]
        kern isa Union{ConstantKernel,KLinearKernel,KQuadraticKernel} || throw(ArgumentError(
            "`discretize` supports constant, linear and quadratic kernels; the model carries " *
            "a $(typeof(kern)). Rewrite the term with `k_linear`/`k_quadratic`, or keep it " *
            "continuum."))
        for e in c0.groups[κ]
            i, j, dR, q, w = c0.i[e], c0.j[e], c0.dR[e], c0.q[e], c0.w[e]
            tid = c0.term[e]
            iszero(dR) || throw(ArgumentError(
                "`discretize` expects a pure k·p model; entry ($i, $j) carries a translation " *
                "$(Tuple(dR))."))
            is_canonical_coupling(i, j, dR) || continue
            ### the momentum origin's phase e^{2πi q·δ} is k-INDEPENDENT per entry, so it is
            ### CONSUMED here -- baked into the stored amplitude, with the output carrying
            ### q ≡ 0 (and k0 ≡ 0 on the states): the result is a true TB model. δ = ΔR here,
            ### since every parent state sits on the single plane-wave site at the origin.
            qn = SVector{D,Float64}(scale * q)
            bake(dRvec) = cispi(2 * dot(qn, SVector{D,Float64}(dRvec)))
            zeroq = zero(SVector{D,Float64})
            diag = (i == j)
            if kern isa ConstantKernel
                push_coupling!(buf, i, j, zeroR, zeroq, w, 1, tid)
            elseif kern isa KLinearKernel
                ### physical momentum p = (2π/s0)(k+q)_frame; sin(p_μ a) = sin(2π(k+q)_out,μ)
                for μ in 1:D
                    cμ = kern.c[μ]
                    iszero(cμ) && continue
                    eμ = SVector{D,Int}(ntuple(d -> d == μ ? 1 : 0, D))
                    push_coupling!(buf, i, j, eμ, zeroq, w * cμ * (-im / (2a)) * bake(eμ), 1, tid)
                    ### for a diagonal parent entry the ∓e_μ pieces are each other's Hermitian
                    ### partner: push the canonical half only, finalize supplies the other
                    diag || push_coupling!(buf, i, j, -eμ, zeroq,
                                           w * cμ * (+im / (2a)) * bake(-eμ), 1, tid)
                end
            else
                M = kern.M
                for μ in 1:D
                    Mμμ = M[μ, μ]
                    eμ = SVector{D,Int}(ntuple(d -> d == μ ? 1 : 0, D))
                    if !iszero(Mμμ)
                        push_coupling!(buf, i, j, zeroR, zeroq, w * Mμμ * (2 / a^2), 1, tid)
                        push_coupling!(buf, i, j, eμ, zeroq,
                                       w * Mμμ * (-1 / a^2) * bake(eμ), 1, tid)
                        diag || push_coupling!(buf, i, j, -eμ, zeroq,
                                               w * Mμμ * (-1 / a^2) * bake(-eμ), 1, tid)
                    end
                    for ν in (μ + 1):D
                        Mμν = M[μ, ν] + M[ν, μ]           # both off-diagonal monomial copies
                        iszero(Mμν) && continue
                        eν = SVector{D,Int}(ntuple(d -> d == ν ? 1 : 0, D))
                        for sμ in (1, -1), sν in (1, -1)
                            dRs = sμ * eμ + sν * eν
                            ### canonical halves only on the diagonal, as above
                            diag && !is_canonical_coupling(i, j, SVector{D,Rational{Int}}(dRs)) && continue
                            push_coupling!(buf, i, j, dRs, zeroq,
                                           w * Mμν * (-sμ * sν / (4 * a^2)) * bake(dRs), 1, tid)
                        end
                    end
                end
            end
        end
    end

    ### the outer constructor derives kperiodicity from the cell: every output site is
    ### :localized, so the discretized model is :exact -- a true TB model
    name = isempty(model.name) ? "discretized" : model.name * " (discretized)"
    return HamiltonianModel(new_cell, buf; gauge=model.gauge, name=name)
end
