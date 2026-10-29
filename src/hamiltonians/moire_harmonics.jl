##################################################################
###### export only the following functions from this module ######
##################################################################

export harmonic_potential, moire_tunnelling




#############################################################
###### Shared 2×2 blocks of the two-level builders      #####
#############################################################

### The Pauli blocks every two-level builder assembles (`moire_tunnelling` here,
### `dirac_cone`/`angular_harmonic` in kdotp.jl, which is included after this file). Defined
### once so a transcription typo in one builder cannot silently disagree with the others;
### `TwoLevelMatrixQnum` copies its matrix, so sharing the constants is safe. WHICH axis level
### is row 1 ("up") is not decided here -- that is the value-resolution rule of
### `TwoLevelMatrixQnum` (up = +1 on `{+1,-1}` axes, declaration order on named axes,
### `up=`/`down=` overriding).
const _SIGMA_X = ComplexF64[0 1; 1 0]
const _SIGMA_Y = ComplexF64[0 -im; im 0]
const _SIGMA_PLUS = ComplexF64[0 1; 0 0]        # σ₊ = |up⟩⟨down|
const _SIGMA_MINUS = ComplexF64[0 0; 1 0]       # σ₋ = |down⟩⟨up|

"""
    _require_g_sites(cell, builder) -> Nothing

Refuse, at build time, a cell on which a ΔG harmonic cannot mean anything: every site the
emitted `delta_g` terms would address must carry a plane-wave label -- `:plane_wave` sites, or
the coset labels of a folded cell. This is the same rule `compile` enforces per term
(`_validate_kinds`); applying it here as well makes the refusal name the BUILDER the user
called instead of one of its emitted terms.
"""
function _require_g_sites(cell::UnitCell, builder::AbstractString)::Nothing
    for s in eachindex(cell.site_kind)
        cell.site_kind[s] === :plane_wave && continue
        ### folded localized sites carry exact coset labels; ΔG harmonics on them are the
        ### moiré-tight-binding coupling and are accepted, exactly as in `_validate_kinds`
        (cell.site_kind[s] === :localized && abs(det(cell.site_period[s])) != 1) && continue
        throw(ArgumentError(
            "`$builder` emits ΔG (plane-wave) harmonics, but site `$(cell.site_names[s])` of " *
            "the cell is :localized -- a real-space orbital carries no plane-wave label to " *
            "shift. Build the model on a `plane_wave_cell` (or a folded cell); for " *
            "tight-binding couplings use `hopping`/`translation_hopping` instead."))
    end
    return nothing
end




#############################################################
###### Moiré potential harmonics                        #####
#############################################################

"""
    harmonic_potential(lattice_or_cell, V, psi; shell=1, name="moire_potential")
        -> Vector{HoppingTerm}

The shell-`s` harmonic of a real periodic potential, as a group of `delta_g` terms:

    Δ(r) = 2V Σ_{pairs j} cos(g_j ⋅ r + ψ)

with the sum over the `±` pairs of the shell's star (three pairs on a triangular lattice's first
shell, two on a square lattice's). On the triangular lattice this is exactly the standard
first-harmonic moiré potential of the TMD continuum models (Wu-MacDonald), with `ψ` the phase
that interpolates between honeycomb (`ψ = 0`) and triangular-well stackings.

Conjugate pairing -- why only half the star is emitted:
-------------------------------------------------------
A real potential fixes `V_{-g} = conj(V_g)`, and the compiled coupling list *already* enforces
exactly that: `finalize_couplings` materializes the Hermitian partner of every emitted coupling. So this
builder emits **one representative per ± pair** with amplitude `V e^{iψ}` (the coupling
`⟨G + g_j | Δ | G⟩`), and the `-g` couplings appear with amplitude `V e^{-iψ}` automatically.
Emitting the full star by hand -- the natural mistake this wrapper exists to prevent --
double-counts every element.

WHICH half carries `V e^{+iψ}` matters when matching a literature convention (relative to the
interlayer transfers it decides between a model and its C -> -C mirror): on a C₃ lattice the
representatives are a C₃-CLOSED orbit `{g, C₃g, C₃²g}` (the Wu-MacDonald `g₁, g₃, g₅`), sorted
counter-clockwise, and each returned term's NAME spells out its integer `g` --
`moire_potential[-1,0]` etc. -- so the chosen orbit is always inspectable on the terms
themselves.

Whether the lattice HAS that C₃ is decided by [`identify_bravais`](@ref) (the 2D hexagonal net
`:hp` is the only 2D class with a three-fold axis), at the one tolerance the package uses for
"how close to symmetric counts as symmetric" -- so a hexagonal lattice read from a file with
nine significant digits takes the same branch as an exactly-built one. On a lattice with NO
three-fold axis (square, oblique, and every dimension but 2) there is no orbit to close and the
representatives are the canonical-sign half of the star: the potential is still real and still
the shell-`s` harmonic, but `ψ` is then this builder's phase convention, NOT the Wu-MacDonald
phase.

Composing:
----------
The returned vector goes straight into `compile` (which flattens nested term collections), and
rules attach per term:

    compile(cell, [kinetic(mass=m), harmonic_potential(cell, V, ψ)])
    [t ⊗ project(:layer, :bot) for t in harmonic_potential(cell, V, ψ)]   # one layer only

See also [`moire_tunnelling`](@ref) for the interlayer counterpart.
"""
function harmonic_potential(lattice::BravaisLattice{D}, V::Real, psi::Real;
                            shell::Int=1, name::AbstractString="moire_potential") where {D}
    star = g_star(lattice, shell)

    ### One representative per ± pair -- and on a C₃-symmetric lattice the representatives are
    ### chosen CLOSED UNDER C₃ ({g, C₃g, C₃²g}), which is what makes ψ exactly the
    ### Wu-MacDonald phase. The canonical-sign choice (first non-zero component positive) picks
    ### 60°-spaced vectors on the triangular first shell, and attaching equal phases to THOSE is
    ### a translated WM potential at phase ψ/3 -- silently the wrong well lattice for every ψ a
    ### user takes from the literature. Lattices with NO three-fold symmetry at all (square,
    ### oblique, and every D ≠ 2) fall back to the canonical-sign reps, where no WM
    ### identification is claimed; a lattice that HAS a C₃ never reaches that fallback --
    ### `_integer_c3` refuses by name rather than returning `nothing` when it cannot represent
    ### one, and a C₃ lattice whose star does not decompose into C₃ orbits is an internal
    ### inconsistency, not a case to paper over.
    C3 = D == 2 ? _integer_c3(lattice) : nothing
    reps = if isnothing(C3)
        filter(_g_canonical_sign, star)
    else
        orbits = _c3_closed_reps(star, C3)
        ### NOT an internal error, though it looks like one: a C₃ really does permute every
        ### shell, so on an exactly hexagonal lattice this cannot happen. It happens when the
        ### lattice is hexagonal only to WITHIN A TOLERANCE, and the two tolerances involved
        ### disagree about it -- `identify_bravais`'s `rtol` says `hp`, while `g_shells`'
        ### `SHELL_RTOL` still resolves the near-degenerate radii into separate shells, so a
        ### star arrives already split and no C₃ can close it. Measured on
        ### `BravaisLattice([a 0; -a(1+ε)/2 a(1+ε)√3/2])` with a = 2.4675: ε = 1.0e-8 gives a
        ### six-member star and three orbits, ε = 1.2e-8 gives a four-member star and lands
        ### here, ε = 1.5e-8 is no longer `hp` at all and takes the non-C₃ branch. The caller
        ### supplied the lattice, so the caller is told what is wrong with it.
        isnothing(orbits) && throw(ArgumentError(
            "Shell $shell of this lattice did not decompose into C₃ orbits {g, C₃g, C₃²g}, " *
            "even though `identify_bravais` classifies it as hexagonal. A three-fold rotation " *
            "permutes every shell of an exactly hexagonal reciprocal lattice, so this means " *
            "the lattice is hexagonal only to within the classification tolerance: its " *
            "shell radii are split by more than `SHELL_RTOL` while its metric is hexagonal to " *
            "within `identify_bravais`'s `rtol`, and a star that arrives already split cannot " *
            "be closed by any rotation. Build the lattice from its exact hexagonal form " *
            "(`bravais_lattice(:triangular; a = ...)`, or round the primitive vectors to the " *
            "precision the lattice constants are actually known to) -- the distortion here is " *
            "far below anything that changes the physics, and it is only the classification " *
            "that cannot survive it."))
        orbits
    end
    2 * length(reps) == length(star) || throw(ErrorException(
        "Shell $shell of the reciprocal lattice did not split into ± pairs; this is an internal " *
        "inconsistency, not a usage error."
    ))
    _sort_star!(reps, lattice)

    amp = V * cis(psi)
    return [delta_g(amp; dG=Vector(g), name="$(name)[$(join(Tuple(g), ','))]") for g in reps]
end

function harmonic_potential(cell::UnitCell{D}, V::Real, psi::Real; kwargs...) where {D}
    _require_g_sites(cell, "harmonic_potential")
    return harmonic_potential(cell.lattice, V, psi; kwargs...)
end

"""
    _c3_closed_reps(star, C3) -> Union{Nothing,Vector{SVector{2,Int}}}

One representative per ± pair of `star`, chosen CLOSED UNDER C₃ ({g, C₃g, C₃²g} orbits), or
`nothing` when the star does not decompose into such orbits (the caller then falls back to
canonical-sign representatives). A ± pair never sits inside one C₃ orbit -- ±g in an orbit
would need a 120° or 240° rotation to equal 180° -- so the orbit selection always yields
exactly one representative per pair.
"""
function _c3_closed_reps(star::Vector{SVector{2,Int}}, C3::SMatrix{2,2,Int})
    length(star) % 3 == 0 || return nothing
    starset = Set(star)
    used = Set{SVector{2,Int}}()
    out = SVector{2,Int}[]
    for g in star
        g in used && continue
        orbit = SVector{2,Int}[g, C3 * g, C3 * (C3 * g)]
        all(o -> o in starset, orbit) || return nothing     # star not C₃-closed
        append!(out, orbit)
        for o in orbit
            push!(used, o); push!(used, -o)
        end
    end
    return out
end

"""
    _integer_c3(lattice) -> Union{Nothing,SMatrix{2,2,Int}}

The action of a counter-clockwise 120° rotation on integer Miller indices, when the lattice has
a three-fold axis: `G_cart = bᵀG`, so `G -> b⁻ᵀ R(120°) bᵀ G`. Returns `nothing` when the
lattice has NO C₃ at all, and refuses by name when it has one that cannot be written on this
basis. Shared by the harmonic-potential orbit selection and the solved tunnelling tripod, so
"C₃" means the same rotation everywhere in this file.

Where "has a C₃" is decided -- and why not here:
------------------------------------------------
The returned matrix must be an EXACT three-fold automorphism of the integer lattice (`C₃³ = I`,
`det = +1`); that part is integer arithmetic and is checked here. But whether the *metric* is
close enough to hexagonal to carry a C₃ at all is a crystallography question, and
[`identify_bravais`](@ref) is the layer that owns it (`:hp` is the only 2D class with a
three-fold axis). Asking it, instead of re-deriving a private tolerance, is what keeps the two
answers from drifting apart -- they did, by three decades: this function used to reject any
lattice whose 120° rotation missed integrality by more than `INT_ROUND_ATOL` (1e-11), so a
hexagonal moiré lattice read from a file with nine or ten significant digits was classified
`hp` and yet had "no C₃". `moire_tunnelling` then refused on it while `harmonic_potential`
silently fell back to canonical-sign representatives -- the wrong well lattice at every ψ ≠ 0.
"""
function _integer_c3(lattice::BravaisLattice{2})
    c, s = -0.5, sqrt(3) / 2
    R = SMatrix{2,2,Float64}(c, s, -s, c)
    bT = SMatrix{2,2,Float64}(transpose(lattice.b))
    M = inv(bT) * R * bT
    Mr = round.(M)
    ### A basis so skew that the rotated indices leave Int64 carries no REPRESENTABLE C₃ -- the
    ### conversion below would be an `InexactError` rather than an answer. But this must not
    ### short-circuit to `nothing` before the metric is consulted: `nothing` means "no C₃ at
    ### all", and the caller's silent fallback is only correct under that reading. A hexagonal
    ### lattice written in an extreme unimodular basis HAS a three-fold axis and merely cannot
    ### express it in Int64, so it belongs in the refusal below, not in the fallback.
    representable = all(m -> isfinite(m) && abs(m) < 1e15, Mr)
    C3 = representable ? SMatrix{2,2,Int}(Int.(Mr)) : SMatrix{2,2,Int}(1, 0, 0, 1)
    exact_c3 = representable &&
               C3 * C3 * C3 == SMatrix{2,2,Int}(1, 0, 0, 1) && det(C3) == 1
    ### fast path: integral to float noise on a proper basis -- no classification needed
    exact_c3 && isapprox(M, Mr; atol=INT_ROUND_ATOL) && return C3
    _has_c3_metric(lattice) || return nothing
    exact_c3 || throw(ArgumentError(
        "This lattice is hexagonal (it has a three-fold axis), but its 120° rotation does not " *
        "act on the Miller indices of THIS basis as an exact integer matrix: `b⁻ᵀ R(120°) bᵀ` " *
        "rounds to $(Tuple(C3)), which is not a three-fold integer automorphism. That happens " *
        "when the primitive vectors are a heavily skewed unimodular mixture, or carry too few " *
        "significant digits for their own integer arithmetic. Rebuild the lattice on its " *
        "reduced basis (`bravais_lattice(:triangular; a=...)`, or the `M` returned by " *
        "`identify_bravais`), or pass the tunnelling momenta explicitly with `q=`."))
    return C3
end

"""
    _has_c3_metric(lattice) -> Bool

Does this lattice's METRIC carry a three-fold axis? Delegated to `identify_bravais` -- `:hp`
(the 2D hexagonal net) is the only 2D Bravais class with one -- so that "how close to
symmetric counts as symmetric" is answered in exactly one place in the package. A partially
periodic lattice has no 2D Bravais class at all, and no C₃ with it.
"""
_has_c3_metric(lattice::BravaisLattice{2}) =
    nperiodic(lattice) == 2 && identify_bravais(lattice).name === :hp




#############################################################
###### Interlayer tunnelling harmonics                  #####
#############################################################

"""
    moire_tunnelling(cell, w0, w1; q=nothing, layer=:layer, sub=:sub, phases=nothing,
                     kernel=nothing, kernel_parity=:none, name="moire_tunnelling")
        -> Vector{AbstractHamiltonianTerm}

The three interlayer tunnelling harmonics of a Bistritzer-MacDonald-type moiré model. With
`ℓ₁, ℓ₂` the first and second declared level of the `layer` axis, the emitted couplings are

    Σ_{n=1}^{3} Σ_G |G + ΔG_n, ℓ₂⟩ T_n ⟨G, ℓ₁|  +  h.c.,
    T_n = w0 σ₀ + w1 (σ_x cos φ_n + σ_y sin φ_n)        on the `sub` pseudospin,

equivalently the tunnelling field `T(r) = Σ_n T_n e^{i q_n ⋅ r}` acting from layer `ℓ₁` to
layer `ℓ₂` in the envelope basis (states `|G, ℓ⟩` at momenta `bᵀ(k + G − K_ℓ)`), with the
Hermitian-conjugate block supplied by the compiler. The default `phases` are the standard
`φ_n = 2π(n-1)/3` (Tarnopolsky-Kruchkov-Vishwanath convention, `q_1, q_2, q_3`
counter-clockwise); pass your own three angles when off that convention rather than falling
back to raw terms.

The registration, solved rather than hand-derived:
--------------------------------------------------
`q` is the tuple of three **cartesian** tunnelling momenta. The coupling
`|G_i, K_to⟩⟨G_j, K_from|` transfers the physical momentum `ΔG_n − K_to + K_from` (fractional),
so the integer plane-wave offset each harmonic must carry is

    ΔG_n = frac(q_n) + K_to − K_from

with `K_from/K_to` the momentum origins of the `layer` axis's two declared levels (declare them
with `plane_wave_cell(...; origins=...)`). This is the one bookkeeping step that must be exact
-- getting it wrong by a reciprocal-lattice mismatch silently breaks the C₃ structure and gaps
the moiré Dirac points -- and it is the framework's arithmetic, not the model file's: a
non-integral `ΔG_n` raises, naming the offending transfer.

The tripod, solved from the origins (`q` omitted):
--------------------------------------------------
With no `q` at all, the tunnelling momenta themselves are solved from the cell. On a lattice
with an exact C₃ (its 120° rotation acts integrally on Miller indices) the tripod is the C₃
orbit of the direct transfer between the two declared origins,

    q_n = C₃ⁿ⁻¹ ⋅ bᵀ(K_{ℓ₁} − K_{ℓ₂}),      n = 1, 2, 3      (counter-clockwise),

so harmonic 1 is the direct transfer (`ΔG_1 = 0`) and the integer transfers are
`ΔG_n = C₃ⁿ⁻¹ d − d` with `d = K_{ℓ₁} − K_{ℓ₂}` fractional. Those come out integer exactly
when both origins sit at C₃-registered points (`C₃K ≡ K` modulo reciprocal vectors -- the ±K
corners of a triangular Brillouin zone, for instance); anything else raises, naming the
offending harmonic. This is the twisted-TMD spelling: declare the two valleys once, on the
basis, and the builder needs no momentum bookkeeping at all. An explicit `q=` always wins over
the solve; a lattice without an exact C₃, or identical origins on the two levels, refuses with
the `q=` escape named.

Level order: the LAYER ladder runs from the axis's first declared level to its second (an
inconsistent declaration cannot mis-register silently -- the ΔG integrality check catches it).
The `sub` pseudospin is resolved by LEVEL VALUES where they speak: on a `{+1, -1}` axis
(either declaration order) up = `+1`, agreeing with `pauli_x/y/z`, `raise`/`lower` and
`dirac_cone`; on a named axis the first declared level is up; `up=`/`down=` override
explicitly.

Momentum-dependent tunnelling:
------------------------------
`kernel = f` multiplies EVERY tunnelling element by the real envelope `f(p)` -- the terms are
then emitted through [`k_envelope`](@ref), with `p` the cartesian momentum at the entry's ΔG
midpoint (see the `k_envelope` docstring for the exact convention). The registration arithmetic
above is unchanged; declare the envelope's parity with `kernel_parity` (`:even` for any `f`
that depends on `|p|` only) if the model is to survive `nambu_doubling`.

    T(p) = w1 * exp(-α(|p| - kθ)):
    moire_tunnelling(cell, w0, w1; q=(q1, q2, q3),
                     kernel=p -> exp(-α * (norm(p) - kθ)), kernel_parity=:even)

Scalar (TMD homobilayer) form:
------------------------------
    moire_tunnelling(cell, w; q=nothing, layer=:layer, phases=nothing, kernel=nothing, ...)

The two-argument method needs NO sublattice axis and emits the scalar interlayer transfer of
the TMD continuum models (Wu-Lovorn-MacDonald),

    T(r) = w Σ_{n=1}^{3} e^{-i φ_n} e^{i q_n ⋅ r}       from layer ℓ₁ to layer ℓ₂,  + h.c.,

with the default `phases = zeros(3)`: three transfers of equal amplitude `w` on the bare layer
ladder, and the same solved registration and tripod. (The `q_n − q_1` are moiré reciprocal
vectors -- `bᵀ(ΔG_n − ΔG_1)` -- so this is the familiar `w (1 + e^{i b₁·r} + e^{i b₂·r})`
transfer up to the plane-wave gauge `e^{i q_1·r}` that the per-layer origins absorb.)
"""
function moire_tunnelling(cell::UnitCell{2}, w0::Real, w1::Real;
                          q=nothing, layer::Symbol=:layer, sub::Symbol=:sub,
                          up=nothing, down=nothing,
                          phases::Union{Nothing,AbstractVector{<:Real}}=nothing,
                          kernel::Union{Nothing,Function}=nothing,
                          kernel_parity::Symbol=:none,
                          name::AbstractString="moire_tunnelling")
    φs = _tunnelling_phases(phases, n -> 2π * (n - 1) / 3)

    sch = cell.schema
    as = axis_position(sch, sub)
    as == 0 && throw(ArgumentError(
        "`moire_tunnelling` decomposes on the axis `$sub`, which the cell does not declare " *
        "(its axes are $(axis_names(sch))). For the SCALAR tunnelling of a TMD homobilayer " *
        "(no sublattice structure) use the two-argument form `moire_tunnelling(cell, w; ...)`."))
    n_levels(sch, as) == 2 || throw(ArgumentError(
        "`moire_tunnelling` needs a two-level `$sub` axis (T_n is a 2×2 pseudospin block); " *
        "it declares $(levels(sch, as))."))

    dGs, lift = _moire_registration(cell, q, layer)
    ### pseudospin-up resolved by LEVEL VALUES on ±1 axes (up = +1, the pauli_*/raise
    ### convention and the same rule as `dirac_cone`), declaration order on named axes;
    ### σ₊ = |up⟩⟨down| -- see `TwoLevelMatrixQnum`
    sub_plus = TwoLevelMatrixQnum(sub, _SIGMA_PLUS, up, down)
    sub_minus = TwoLevelMatrixQnum(sub, _SIGMA_MINUS, up, down)
    emit(amp, dG, nm) = _moire_transfer(amp, dG, nm, kernel, kernel_parity)

    terms = AbstractHamiltonianTerm[]
    ### a dropped group is a reweight trap: nothing named `_T{n}_AA` is registered, so a
    ### chiral-limit compile cannot sweep corrugation upward -- say so once, loudly
    iszero(w0) && @warn("moire_tunnelling: w0 = 0 (the chiral limit) emits NO AA terms, so " *
                        "no `$(name)_T{n}_AA` names are registered and `reweight` cannot " *
                        "turn corrugation on later. To sweep w0, compile at a nonzero w0 " *
                        "and sweep DOWN (relative reweight reaches exactly 0).", maxlog=1)
    for n in 1:3
        dG = dGs[n]
        iszero(w0) ||
            push!(terms, emit(w0, dG, "$(name)_T$(n)_AA") ⊗ lift)
        push!(terms, emit(w1 * cis(-φs[n]), dG, "$(name)_T$(n)_AB") ⊗ lift ⊗ sub_plus)
        push!(terms, emit(w1 * cis(+φs[n]), dG, "$(name)_T$(n)_BA") ⊗ lift ⊗ sub_minus)
    end
    return terms
end

function moire_tunnelling(cell::UnitCell{2}, w::Real;
                          q=nothing, layer::Symbol=:layer,
                          phases::Union{Nothing,AbstractVector{<:Real}}=nothing,
                          kernel::Union{Nothing,Function}=nothing,
                          kernel_parity::Symbol=:none,
                          name::AbstractString="moire_tunnelling")
    ### the scalar (TMD homobilayer) form: no sublattice decomposition, zero default phases
    φs = _tunnelling_phases(phases, _ -> 0.0)
    dGs, lift = _moire_registration(cell, q, layer)
    return AbstractHamiltonianTerm[
        _moire_transfer(w * cis(-φs[n]), dGs[n], "$(name)_T$(n)", kernel, kernel_parity) ⊗ lift
        for n in 1:3]
end

### the three tunnelling phases, validated once for both forms
function _tunnelling_phases(phases::Union{Nothing,AbstractVector{<:Real}},
                            default::Function)::Vector{Float64}
    φs = isnothing(phases) ? Float64[default(n) for n in 1:3] : collect(Float64, phases)
    length(φs) == 3 || throw(ArgumentError(
        "`phases` must hold three angles (one per C₃ harmonic); got $(length(φs))."))
    return φs
end

### one transfer term: a constant `delta_g` harmonic, or its `k_envelope` form when a
### momentum-dependent kernel rides along (the amplitude becomes the coefficient)
_moire_transfer(amp, dG::Vector{Int}, nm::String, ::Nothing, ::Symbol) =
    delta_g(amp; dG=dG, name=nm)
_moire_transfer(amp, dG::Vector{Int}, nm::String, kernel::Function, parity::Symbol) =
    k_envelope(kernel; coefficient=amp, parity=parity, dG=dG, name=nm)

"""
    _moire_registration(cell, q, layer) -> (dGs, lift)

The solved ΔG registrations of the three tunnelling harmonics and the layer ladder rule -- the
shared bookkeeping of both `moire_tunnelling` forms (see its docstring for the arithmetic).
`q` is either the explicit tuple of three cartesian tunnelling momenta, or `nothing` to solve
the C₃ tripod from the per-level origins.
"""
function _moire_registration(cell::UnitCell{2}, q, layer::Symbol)
    _require_g_sites(cell, "moire_tunnelling")
    sch = cell.schema
    a = axis_position(sch, layer)
    a == 0 && throw(ArgumentError(
        "`moire_tunnelling` ladders the axis `$layer`, which the cell does not declare " *
        "(its axes are $(axis_names(sch)))."))
    n_levels(sch, a) == 2 || throw(ArgumentError(
        "`moire_tunnelling` needs a two-level `$layer` axis (the tunnelling couples exactly " *
        "two layers); it declares $(levels(sch, a))."))
    lv = levels(sch, a)

    kof = _level_origins(cell, a, lv, layer)
    dGs = isnothing(q) ? _tripod_from_origins(cell.lattice, kof, lv, layer) :
                         _tripod_from_q(cell.lattice, q, kof, layer)
    return dGs, raise(layer; from=lv[1], to=lv[2])
end

"""
    _level_origins(cell, a, lv, layer) -> Vector{SVector{2,Float64}}

The momentum origins of the two levels of axis `a`, read off the states -- and checked UNIFORM
per level: with origins declared on the wrong axis, level 1 would mix two different K's and
the registration solved from the first state would mis-register silently.
"""
function _level_origins(cell::UnitCell{2}, a::Int, lv, layer::Symbol)
    sch = cell.schema
    origins = state_origins(cell)
    internal = state_internals(cell)
    kof = Vector{SVector{2,Float64}}(undef, 2)
    for o in 1:2
        members = [s for s in eachindex(internal) if ordinal_at(sch, internal[s], a) == o]
        isempty(members) && throw(ArgumentError(
            "No state carries level $(lv[o]) of axis `$layer`; the cell is inconsistent."))
        kof[o] = origins[members[1]]
        for s in members
            isapprox(origins[s], kof[o]; atol=LATTICE_ATOL) || throw(ArgumentError(
                "The states at level $(lv[o]) of axis `$layer` carry DIFFERENT momentum " *
                "origins ($(Tuple(kof[o])) vs $(Tuple(origins[s]))): the tunnelling " *
                "registration is solved from one origin per layer level, so mixed origins " *
                "would mis-register silently. Declare `origins=` on the `$layer` axis (one " *
                "origin per level), not on another axis."))
        end
    end
    return kof
end

"""
    _tripod_from_q(lattice, q, kof, layer) -> Vector{Vector{Int}}

The integer transfers of an EXPLICIT tripod: `ΔG_n = frac(q_n) + K_to − K_from`, integrality
asserted per harmonic (see the `moire_tunnelling` docstring).
"""
function _tripod_from_q(lattice::BravaisLattice{2}, q, kof, layer::Symbol)
    length(q) == 3 || throw(ArgumentError(
        "`moire_tunnelling` expects the three cartesian tunnelling momenta; got $(length(q))."))
    K_from, K_to = kof[1], kof[2]
    dGs = Vector{Vector{Int}}(undef, 3)
    for n in 1:3
        t = reciprocal_to_fractional(q[n], lattice)
        raw = t + K_to - K_from
        rounded = round.(raw)
        isapprox(raw, rounded; atol=INT_ROUND_ATOL) || throw(ArgumentError(
            "The tunnelling momentum q_$n = $(Tuple(q[n])) is not commensurate with this cell: " *
            "ΔG_$n = frac(q_$n) + K_to − K_from = $(Tuple(raw)) is not an integer vector. " *
            "Check the moiré lattice construction and the `origins=` of the `$layer` axis -- " *
            "their registration must satisfy K_to − K_from ≡ −frac(q_1) (mod 1) and its C₃ " *
            "images."))
        dGs[n] = Vector{Int}(Int.(rounded))
    end
    return dGs
end

"""
    _tripod_from_origins(lattice, kof, lv, layer) -> Vector{Vector{Int}}

The integer transfers of the C₃ tripod SOLVED from the per-level origins:
`ΔG_n = C₃ⁿ⁻¹d − d` with `d = K_{ℓ₁} − K_{ℓ₂}` fractional (so harmonic 1 is the direct,
`ΔG_1 = 0` transfer and the physical momenta are `q_n = C₃ⁿ⁻¹ bᵀd`). Integrality is asserted
per harmonic; it holds exactly when both origins sit at C₃-registered points.
"""
function _tripod_from_origins(lattice::BravaisLattice{2}, kof, lv, layer::Symbol)
    C3 = _integer_c3(lattice)
    isnothing(C3) && throw(ArgumentError(
        "`moire_tunnelling` was asked to SOLVE the tunnelling tripod from the cell's origins " *
        "(no `q=` given), but this lattice has no exact C₃ -- its metric is not that of a " *
        "hexagonal net (`identify_bravais` calls it " *
        "$(nperiodic(lattice) == 2 ? "`$(identify_bravais(lattice).name)`" :
                                     "partially periodic")), so the three C₃-related transfers " *
        "are not defined. Pass the three cartesian tunnelling momenta explicitly with `q=`."))
    draw = kof[1] - kof[2]
    maximum(abs, draw) > LATTICE_ATOL || throw(ArgumentError(
        "`moire_tunnelling` was asked to SOLVE the tunnelling tripod from the cell's origins " *
        "(no `q=` given), but the two levels of axis `$layer` carry the SAME momentum origin " *
        "($(Tuple(kof[1]))): there is no transfer to rotate. Declare distinct per-layer " *
        "origins on the cell (`plane_wave_cell(...; origins=:$layer => [...])`), or pass the " *
        "three cartesian tunnelling momenta with `q=`."))
    ### The physical tripod is the FIRST tunnelling star, so `d` must be the SHORTEST cartesian
    ### representative of K_ℓ₁ − K_ℓ₂ modulo reciprocal vectors: equivalent mod-G spellings of
    ### the same valleys (declaring −K instead of the (2/3, 2/3) corner) would otherwise solve
    ### a HIGHER harmonic silently -- same integrality, converged, wrong model. The integer
    ### offset W between the declared and reduced representatives is folded back into every
    ### transfer, so the emitted couplings still register the origins exactly as declared.
    B = recip_matrix(lattice)
    d = SVector{2,Float64}(draw)
    best = norm(B * d)
    tol = INT_ROUND_ATOL * max(best, 1.0)
    ### STRICT improvement only: when the declared representative is already first-star it is
    ### KEPT (its C3 star is norm-degenerate, and swapping within the star would silently
    ### rotate which harmonic carries which phase); ties among genuinely shorter candidates
    ### break lexicographically for determinism
    for n1 in -2:2, n2 in -2:2
        cand = SVector{2,Float64}(draw + SVector(n1, n2))
        l = norm(B * cand)
        if l < best - tol || (l < norm(B * SVector{2,Float64}(draw)) - tol &&
                              abs(l - best) <= tol && Tuple(cand) < Tuple(d))
            d, best = cand, l
        end
    end
    W = round.(Int, draw - d)
    dGs = Vector{Vector{Int}}(undef, 3)
    dn = d
    for n in 1:3
        raw = dn - d
        rounded = round.(raw)
        isapprox(raw, rounded; atol=INT_ROUND_ATOL) || throw(ArgumentError(
            "The tunnelling tripod solved from the `$layer` origins is not commensurate with " *
            "this cell: harmonic $n carries ΔG_$n = C₃^$(n - 1)·d − d = $(Tuple(raw)) " *
            "(d = K_$(lv[1]) − K_$(lv[2]) = $(Tuple(d)), reduced fractional), which is not an " *
            "integer vector. The per-level origins must sit at C₃-registered points (C₃K ≡ K " *
            "modulo reciprocal vectors -- the corners of a triangular Brillouin zone, say); " *
            "fix the `origins=` declaration, or pass the physical transfers explicitly with " *
            "`q=`."))
        ### fold the declared-vs-reduced offset back: q_n = bᵀ(ΔG_n + d_declared) must equal
        ### the first-star C₃ⁿ⁻¹ bᵀ d_reduced, so ΔG_n = (C₃ⁿ⁻¹d − d) − W
        dGs[n] = Vector{Int}(Int.(rounded) - W)
        dn = C3 * dn
    end
    return dGs
end

# One representative per ± pair: the first non-zero integer component is positive. The same
# convention as `_is_canonical_bond`, applied to reciprocal integer coordinates.
function _g_canonical_sign(n::SVector{D,Int}) where {D}
    for m in 1:D
        n[m] > 0 && return true
        n[m] < 0 && return false
    end
    return false      # the zero vector never belongs to a star
end

# Deterministic, physically readable order: by cartesian angle in 2D (the Wu-MacDonald g_1, g_3,
# g_5 come out counter-clockwise), lexicographic otherwise.
function _sort_star!(reps::Vector{SVector{D,Int}}, lattice::BravaisLattice{D}) where {D}
    if D == 2
        sort!(reps; by = n -> begin
            g = n[1] * primitive_recip_vector(lattice, 1) + n[2] * primitive_recip_vector(lattice, 2)
            atan(g[2], g[1])
        end)
    else
        sort!(reps; by = Tuple)
    end
    return reps
end
