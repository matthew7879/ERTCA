# Paper-specific ertca routines.

function _recip_scale(lattice::BravaisLattice{D}) where {D}
    nperiodic(lattice) >= 1 || throw(ArgumentError(
        "This lattice has no periodic direction (a Γ-only finite cluster): it has no " *
        "reciprocal vectors, so reciprocal-space shells and plane-wave bases are not defined " *
        "on it."))
    bmin = minimum(norm(primitive_recip_vector(lattice, m)) for m in 1:nperiodic(lattice))
    return bmin, LATTICE_ATOL * bmin
end

g_shells(lattice::BravaisLattice; shells::Int) = g_shells(lattice, shells)

function _g_shells_within(lattice::BravaisLattice{D}, r::Float64) where {D}
    bmin, datol = _recip_scale(lattice)
    P = nperiodic(lattice)
    amax = maximum(norm(primitive_vector(lattice, m)) for m in 1:P)
    N = ceil(Int, r * amax / (2π)) + 1

    entries = Tuple{Float64,SVector{D,Int}}[]
    for n in Iterators.product(ntuple(m -> m <= P ? (-N:N) : (0:0), D)...)
        nv = SVector{D,Int}(n)
        iszero(nv) && continue
        g = sum(nv[m] * primitive_recip_vector(lattice, m) for m in 1:D)
        d = norm(g)
        d <= r + datol && push!(entries, (d, nv))
    end
    sort!(entries; by = first)

    radii = Float64[]
    shells = Vector{Vector{SVector{D,Int}}}()
    for (d, nv) in entries
        if isempty(radii) || d - radii[end] > max(datol, SHELL_RTOL * d)
            push!(radii, d)
            push!(shells, SVector{D,Int}[])
        end
        push!(shells[end], nv)
    end
    return radii, shells
end

function _sort_shells!(shells::Vector{Vector{SVector{D,Int}}}) where {D}
    for s in shells
        sort!(s; by = Tuple)
    end
    return shells
end

function g_shells(lattice::BravaisLattice{D}, nshells::Int) where {D}
    nshells >= 1 || throw(DomainError(nshells, "Number of reciprocal shells must be at least 1."))

    bmin, datol = _recip_scale(lattice)

    # the nshells-th distinct radius is <= nshells * (shortest reciprocal vector) <= nshells * bmin
    r = nshells * bmin * (1 + SHELL_RTOL) + datol
    radii, shells = _g_shells_within(lattice, r)

    length(shells) >= nshells || throw(ErrorException(
        "g_shells resolved only $(length(shells)) shells inside its provably-sufficient box; " *
        "this is an internal inconsistency, not a usage error."
    ))
    return radii[1:nshells], _sort_shells!(shells[1:nshells])
end

g_star(lattice::BravaisLattice{D}, shell::Int) where {D} = g_shells(lattice, shell)[2][shell]

function plane_wave_cell(lattice::BravaisLattice{D};
                         shells::Union{Nothing,Int}=nothing,
                         gcut::Union{Nothing,Real}=nothing,
                         gvectors::Union{Nothing,AbstractVector}=nothing,
                         schema::Union{Nothing,InternalSchema}=nothing,
                         site_name::Symbol=:origin,
                         origin::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         origins::Union{Nothing,Pair}=nothing) where {D}
    count(!isnothing, (shells, gcut, gvectors)) == 1 || throw(ArgumentError(
        "Give exactly one of `shells` (a shell count), `gcut` (a cartesian cutoff), or " *
        "`gvectors` (an explicit list of integer G vectors)."
    ))

    Gs = SVector{D,Int}[]
    if !isnothing(gvectors)
        for g in gvectors
            length(g) == D || throw(DimensionMismatch(
                "Every entry of `gvectors` must have length $D; got one of length $(length(g))."))
            gi = try
                canonicalize_int(g)
            catch
                throw(ArgumentError(
                    "`gvectors` entry $(Tuple(g)) is not an integer vector. `gvectors` takes " *
                    "INTEGER reciprocal labels (Miller indices); passing cartesian momenta " *
                    "`B*g` or fractional momenta is the usual mistake -- convert with " *
                    "`round.(Int, reciprocal_to_fractional(g_cart, lattice))`."))
            end
            all(m -> gi[m] == 0, (nperiodic(lattice)+1):D) || throw(ArgumentError(
                "`gvectors` entry $(Tuple(gi)) is non-zero along an aperiodic direction of " *
                "this partially periodic lattice; there is no reciprocal lattice there."))
            push!(Gs, gi)
        end
        allunique(Gs) || throw(ArgumentError("`gvectors` contains duplicate entries."))
    elseif !isnothing(shells)
        shells >= 0 || throw(DomainError(shells,
            "The shell count must be non-negative (0 keeps only G = 0)."))
        push!(Gs, zero(SVector{D,Int}))
        if shells > 0
            _, sh = g_shells(lattice, shells)
            foreach(s -> append!(Gs, s), sh)
        end
    else
        gcut >= 0 || throw(DomainError(gcut, "The cutoff must be non-negative."))
        push!(Gs, zero(SVector{D,Int}))
        bmin, datol = _recip_scale(lattice)
        radii, sh = _g_shells_within(lattice, (gcut + 2 * datol) * (1 + 4 * SHELL_RTOL) + datol)
        for s in eachindex(radii)
            radii[s] <= gcut + datol || continue
            append!(Gs, sort!(sh[s]; by = Tuple))
        end
    end

    sch = isnothing(schema) ? TRIVIAL_SCHEMA : schema
    k0_of = _resolve_origins(lattice, sch, origin, origins)
    pos = zero(SVector{D,Float64})
    basis = BasisState{D}[]
    sizehint!(basis, length(Gs) * n_internal(sch))
    for g in Gs, internal in 1:n_internal(sch)
        push!(basis, BasisState{D}(pos, g, k0_of[internal], internal, sch))
    end
    return UnitCell{D}(lattice, sch, [site_name], [pos], basis; site_kind=[:plane_wave])
end

function _resolve_origins(lattice::BravaisLattice{D}, sch::InternalSchema,
                          origin::Union{Nothing,AbstractVector{<:Real}},
                          origins::Union{Nothing,Pair}) where {D}
    zero_k = zero(SVector{D,Float64})
    isnothing(origin) && isnothing(origins) && return fill(zero_k, n_internal(sch))
    (isnothing(origin) || isnothing(origins)) || throw(ArgumentError(
        "Give `origin` (one expansion point) or `origins` (per-level), not both."))

    if !isnothing(origin)
        length(origin) == D || throw(DimensionMismatch(
            "`origin` has length $(length(origin)) but the lattice is $D-dimensional."))
        kf = _conform_origin(reciprocal_to_fractional(origin, lattice), lattice)
        return fill(SVector{D,Float64}(kf), n_internal(sch))
    end

    ax = Symbol(origins.first)
    a = axis_position(sch, ax)
    a == 0 && throw(ArgumentError(
        "`origins` names axis `$ax`, but the schema declares $(axis_names(sch))."))
    table = Dict{Int,SVector{D,Float64}}()
    for p in origins.second
        p isa Pair || throw(ArgumentError(
            "`origins` must be `axis => [level => K_cartesian, ...]`; got a non-pair entry."))
        o = level_ordinal(sch, a, _as_level(p.first))
        o == 0 && throw(ArgumentError(
            "`origins` names level $(repr(p.first)) on axis `$ax`, which declares " *
            "$(levels(sch, a))."))
        length(p.second) == D || throw(DimensionMismatch(
            "`origins` entry for level $(repr(p.first)) has length $(length(p.second)); " *
            "expected a cartesian momentum of length $D."))
        haskey(table, o) && throw(ArgumentError(
            "`origins` names level $(repr(p.first)) twice; each level takes one origin."))
        table[o] = _conform_origin(
            SVector{D,Float64}(reciprocal_to_fractional(p.second, lattice)), lattice)
    end
    length(table) == n_levels(sch, a) || throw(ArgumentError(
        "`origins` covers $(length(table)) of the $(n_levels(sch, a)) levels of axis `$ax`; " *
        "every level needs an origin (use `origin=` for a single shared one)."))

    out = Vector{SVector{D,Float64}}(undef, n_internal(sch))
    for internal in 1:n_internal(sch)
        out[internal] = table[ordinal_at(sch, internal, a)]
    end
    return out
end

function _conform_origin(kf::SVector{D,Float64}, lattice::BravaisLattice{D}) where {D}
    P = nperiodic(lattice)
    P == D && return kf
    for m in (P+1):D
        abs(kf[m]) <= LATTICE_ATOL || throw(ArgumentError(
            "The momentum origin resolves to fractional $(Tuple(kf)), with weight along " *
            "aperiodic direction $m; a momentum origin must lie in the periodic subspace."))
    end
    return SVector{D,Float64}(ntuple(m -> m <= P ? kf[m] : 0.0, D))
end

const _SIGMA_PLUS = ComplexF64[0 1; 0 0]        # σ₊ = |up⟩⟨down|

const _SIGMA_MINUS = ComplexF64[0 0; 1 0]       # σ₋ = |down⟩⟨up|

function _require_g_sites(cell::UnitCell, builder::AbstractString)::Nothing
    for s in eachindex(cell.site_kind)
        cell.site_kind[s] === :plane_wave && continue
        (cell.site_kind[s] === :localized && abs(det(cell.site_period[s])) != 1) && continue
        throw(ArgumentError(
            "`$builder` emits ΔG (plane-wave) harmonics, but site `$(cell.site_names[s])` of " *
            "the cell is :localized -- a real-space orbital carries no plane-wave label to " *
            "shift. Build the model on a `plane_wave_cell` (or a folded cell); for " *
            "tight-binding couplings use `hopping`/`translation_hopping` instead."))
    end
    return nothing
end

function harmonic_potential(lattice::BravaisLattice{D}, V::Real, psi::Real;
                            shell::Int=1, name::AbstractString="moire_potential") where {D}
    star = g_star(lattice, shell)
    C3 = D == 2 ? _integer_c3(lattice) : nothing
    reps = if isnothing(C3)
        filter(_g_canonical_sign, star)
    else
        orbits = _c3_closed_reps(star, C3)
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

function _integer_c3(lattice::BravaisLattice{2})
    c, s = -0.5, sqrt(3) / 2
    R = SMatrix{2,2,Float64}(c, s, -s, c)
    bT = SMatrix{2,2,Float64}(transpose(lattice.b))
    M = inv(bT) * R * bT
    Mr = round.(M)
    representable = all(m -> isfinite(m) && abs(m) < 1e15, Mr)
    C3 = representable ? SMatrix{2,2,Int}(Int.(Mr)) : SMatrix{2,2,Int}(1, 0, 0, 1)
    exact_c3 = representable &&
               C3 * C3 * C3 == SMatrix{2,2,Int}(1, 0, 0, 1) && det(C3) == 1
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

_has_c3_metric(lattice::BravaisLattice{2}) =
    nperiodic(lattice) == 2 && identify_bravais(lattice).name === :hp

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
    sub_plus = TwoLevelMatrixQnum(sub, _SIGMA_PLUS, up, down)
    sub_minus = TwoLevelMatrixQnum(sub, _SIGMA_MINUS, up, down)
    emit(amp, dG, nm) = _moire_transfer(amp, dG, nm, kernel, kernel_parity)

    terms = AbstractHamiltonianTerm[]
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
    φs = _tunnelling_phases(phases, _ -> 0.0)
    dGs, lift = _moire_registration(cell, q, layer)
    return AbstractHamiltonianTerm[
        _moire_transfer(w * cis(-φs[n]), dGs[n], "$(name)_T$(n)", kernel, kernel_parity) ⊗ lift
        for n in 1:3]
end

function _tunnelling_phases(phases::Union{Nothing,AbstractVector{<:Real}},
                            default::Function)::Vector{Float64}
    φs = isnothing(phases) ? Float64[default(n) for n in 1:3] : collect(Float64, phases)
    length(φs) == 3 || throw(ArgumentError(
        "`phases` must hold three angles (one per C₃ harmonic); got $(length(φs))."))
    return φs
end

_moire_transfer(amp, dG::Vector{Int}, nm::String, ::Nothing, ::Symbol) =
    delta_g(amp; dG=dG, name=nm)

_moire_transfer(amp, dG::Vector{Int}, nm::String, kernel::Function, parity::Symbol) =
    k_envelope(kernel; coefficient=amp, parity=parity, dG=dG, name=nm)

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
    B = recip_matrix(lattice)
    d = SVector{2,Float64}(draw)
    best = norm(B * d)
    tol = INT_ROUND_ATOL * max(best, 1.0)
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
        dGs[n] = Vector{Int}(Int.(rounded) - W)
        dn = C3 * dn
    end
    return dGs
end

function _g_canonical_sign(n::SVector{D,Int}) where {D}
    for m in 1:D
        n[m] > 0 && return true
        n[m] < 0 && return false
    end
    return false      # the zero vector never belongs to a star
end

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

struct RegistryScan{D,K<:KPeriodicity}
    lattice::BravaisLattice{2}
    models::Vector{HamiltonianModel{D,K}}
    d::Vector{SVector{2,Float64}}
    shifted::BitVector
    labels::Vector{Symbol}
    layer_names::NTuple{2,Symbol}
    permutations::Vector{Vector{Int}}
    drift::Vector{Float64}
    name::String
end

n_stackings(scan::RegistryScan)::Int = length(scan.models)

n_orbitals(scan::RegistryScan)::Int = length(scan.labels)

function Base.show(io::IO, scan::RegistryScan{D}) where {D}
    nsh = count(scan.shifted)
    print(io, "RegistryScan{", D, "}(\"", scan.name, "\": ", n_stackings(scan),
          " stackings, ", n_orbitals(scan), " orbitals -- ",
          n_orbitals(scan) - nsh, " on :", scan.layer_names[1], ", ", nsh, " on :",
          scan.layer_names[2], " (shifted))")
end

Base.show(io::IO, ::MIME"text/plain", scan::RegistryScan) = show(io, scan)

function registry_scan(entries::AbstractVector{<:Pair};
                       layers::AbstractVector,
                       shifted_layer,
                       lattice::Union{Nothing,BravaisLattice}=nothing,
                       labels::Union{Nothing,AbstractVector}=nothing,
                       match::Union{Symbol,AbstractVector}=:centres,
                       atol::Float64=1e-4,
                       name::AbstractString="registry_scan")::RegistryScan
    isempty(entries) && throw(ArgumentError(
        "`registry_scan` needs at least one stacking entry `d => source`."))
    atol > 0 || throw(DomainError(atol, "`registry_scan`: `atol` must be positive."))
    S = length(entries)
    ds = Vector{SVector{2,Float64}}(undef, S)
    loaded = Vector{Any}(undef, S)
    for (s, entry) in pairs(collect(entries))
        dfrac, source = entry
        (dfrac isa AbstractVector && length(dfrac) == 2 && all(isfinite, dfrac)) ||
            throw(ArgumentError(
                "`registry_scan`: stacking $s declares the shift `$dfrac`; a registry shift " *
                "is an IN-PLANE fractional vector with 2 finite components."))
        ds[s] = SVector{2,Float64}(dfrac)
        loaded[s] = _registry_load(source, lattice, s)
    end
    for s in 1:S, t in (s+1):S
        norm(ds[s] - ds[t] - round.(ds[s] - ds[t])) > atol || throw(ArgumentError(
            "`registry_scan`: stackings $s and $t declare the same shift modulo lattice " *
            "translations ($(Tuple(ds[s])) vs $(Tuple(ds[t]))); duplicate stackings make " *
            "the harmonic system singular -- drop one."))
    end

    Dref = _model_dim(loaded[1])
    all(m -> _model_dim(m) == Dref, loaded) || throw(ArgumentError(
        "`registry_scan`: the stackings mix spatial dimensions " *
        "($(sort(unique(_model_dim.(loaded))))); the whole family must share one."))
    Dref == 1 && throw(ArgumentError(
        "`registry_scan` describes a bilayer with in-plane registry shifts; a 1D model " *
        "cannot carry one."))
    models = [m for m in loaded]::Vector{<:HamiltonianModel{Dref}}
    models = [change_gauge(m, :atomic) for m in models]
    nbeta = n_states(models[1].cell)
    for (s, m) in pairs(models)
        same_basis(m.cell.lattice, models[1].cell.lattice) || throw(ArgumentError(
            "`registry_scan`: stacking $s lives on a different Bravais lattice than " *
            "stacking 1. All members of a registry scan share the SAME untwisted bilayer " *
            "lattice; only the basis positions move with `d`."))
        n_states(m.cell) == nbeta || throw(ArgumentError(
            "`registry_scan`: stacking $s holds $(n_states(m.cell)) Wannier orbitals but " *
            "stacking 1 holds $nbeta; the family must share one orbital set."))
        is_folded(m.cell) && throw(ArgumentError(
            "`registry_scan`: stacking $s is a folded model; registry scans ingest plain " *
            "Wannier90 imports (unfolded, localized sites)."))
        all(k -> k === :localized, site_kinds(m.cell)) || throw(ArgumentError(
            "`registry_scan`: stacking $s carries non-localized (plane-wave) sites; " *
            "registry scans ingest real-space Wannier Hamiltonians."))
        has_kdependence(m) && throw(ArgumentError(
            "`registry_scan`: stacking $s carries k-dependent kernels; a Wannier90 import " *
            "is a constant-kernel hopping list, and the ERTCA identification " *
            "`bloch_hamiltonian(model, k̄) = W(k̄, d)` assumes exactly that."))
        maximum(q -> norm(q), m.couplings.q; init=0.0) <= LATTICE_ATOL || throw(ArgumentError(
            "`registry_scan`: stacking $s carries nonzero per-coupling momentum offsets; " *
            "an ordinary Wannier90 import has `q = 0` everywhere."))
        if Dref == 3
            a = m.cell.lattice.a
            atol_cart = distance_atol(m.cell.lattice)
            (abs(a[1, 3]) <= atol_cart && abs(a[2, 3]) <= atol_cart &&
             abs(a[3, 1]) <= atol_cart && abs(a[3, 2]) <= atol_cart) ||
                throw(ArgumentError(
                    "`registry_scan`: stacking $s has a 3D lattice whose first two " *
                    "primitive vectors are not in-plane with `a₃ ∥ ẑ`; re-express the slab " *
                    "so rows 1:2 span the layers and row 3 is the stacking axis."))
            pos3 = state_positions(m.cell)
            c = m.couplings
            bad = findfirst(e -> abs(Float64(c.dR[e][3]) + pos3[c.j[e]][3] -
                                     pos3[c.i[e]][3]) > 0.5, eachindex(c.i))
            isnothing(bad) || throw(ArgumentError(
                "`registry_scan`: stacking $s couples orbitals $(c.i[bad]) and $(c.j[bad]) " *
                "across an out-of-plane separation δ₃ = " *
                "$(round(Float64(c.dR[bad][3]) + pos3[c.j[bad]][3] - pos3[c.i[bad]][3]; digits=4)) " *
                "(fractional, > half the cell height): the slab hops into its own periodic " *
                "image along the stacking axis. A bilayer slab Wannierized on a 2D k-mesh in " *
                "a cell with enough vacuum has no such hopping -- add vacuum, or re-run " *
                "Wannier90 with a single k-point along that axis."))
        end
    end
    lat2 = leading_block(models[1].cell.lattice, 2)
    length(layers) == nbeta || throw(DimensionMismatch(
        "`registry_scan`: `layers` holds $(length(layers)) labels for $nbeta Wannier " *
        "orbitals; declare one label per orbital, in the file order of the first entry."))
    lvals = unique(layers)
    length(lvals) == 2 || throw(ArgumentError(
        "`registry_scan`: `layers` takes $(length(lvals)) distinct values " *
        "($(lvals)); a bilayer registry scan needs exactly two."))
    any(isequal(shifted_layer), lvals) || throw(ArgumentError(
        "`registry_scan`: `shifted_layer = $shifted_layer` is not one of the `layers` " *
        "labels $(lvals)."))
    shifted = BitVector([isequal(l, shifted_layer) for l in layers])
    other = lvals[findfirst(!isequal(shifted_layer), lvals)]
    layer_names = (Symbol(other), Symbol(shifted_layer))
    lbls = if isnothing(labels)
        [Symbol("w", b) for b in 1:nbeta]
    else
        length(labels) == nbeta || throw(DimensionMismatch(
            "`registry_scan`: `labels` holds $(length(labels)) names for $nbeta orbitals."))
        collect(Symbol, labels)
    end
    allunique(lbls) || throw(ArgumentError(
        "`registry_scan`: the orbital `labels` must be distinct (they become the levels of " *
        "the realized model's internal axis); got $lbls."))
    perms, drift = _registry_permutations(models, ds, shifted, match, atol)

    return RegistryScan{Dref,_model_kpar(models[1])}(lat2, models, ds, shifted, lbls, layer_names, perms, drift,
                              String(name))
end

_registry_load(source::HamiltonianModel, ::Union{Nothing,BravaisLattice}, ::Int) = source

function _registry_load(source::NamedTuple, lattice::Union{Nothing,BravaisLattice}, s::Int)
    haskey(source, :hr) || throw(ArgumentError(
        "`registry_scan`: stacking $s's file entry needs an `hr = \"…_hr.dat\"` field."))
    haskey(source, :centres) || throw(ArgumentError(
        "`registry_scan`: stacking $s's file entry carries no `centres=`. The atomic gauge " *
        "is pinned to the TRUE Wannier centres -- and the shifted layer's centres move with " *
        "`d` -- so `centres` (the `…_centres.xyz` path, or a vector of cartesian positions) " *
        "is mandatory for file entries."))
    isnothing(lattice) && throw(ArgumentError(
        "`registry_scan`: stacking $s is a file entry, which is read on an explicit " *
        "Bravais lattice; pass `lattice=`."))
    return read_wannier90_hr(lattice, source.hr; centres=source.centres,
                             name="registry stacking $s")
end

_registry_load(source, ::Union{Nothing,BravaisLattice}, s::Int) = throw(ArgumentError(
    "`registry_scan`: stacking $s's source is a $(typeof(source)); expected a " *
    "`HamiltonianModel` or a NamedTuple `(hr = …, centres = …)`."))

_model_dim(::HamiltonianModel{D}) where {D} = D

@inline function _wrapped_dist(a::SVector{D,Float64}, b::SVector{D,Float64},
                               P::Int)::Float64 where {D}
    v = a - b
    w = SVector{D,Float64}(ntuple(m -> m <= P ? v[m] - round(v[m]) : v[m], Val(D)))
    return norm(w)
end

@inline _pad_shift(shift::SVector{2,Float64}, ::Val{D}) where {D} =
    _pad_components(shift, Val(D))

function _registry_permutations(models::AbstractVector{<:HamiltonianModel{D}},
                                ds::Vector{SVector{2,Float64}}, shifted::BitVector,
                                match::Union{Symbol,AbstractVector},
                                atol::Float64) where {D}
    S = length(models)
    nbeta = length(shifted)
    if match isa AbstractVector
        length(match) == S || throw(DimensionMismatch(
            "`registry_scan`: `match` holds $(length(match)) permutations for $S stackings."))
        perms = [collect(Int, p) for p in match]
        for (s, p) in pairs(perms)
            isperm(p) && length(p) == nbeta || throw(ArgumentError(
                "`registry_scan`: `match[$s]` is not a permutation of 1:$nbeta."))
        end
        return perms, zeros(S)
    end
    match === :none && return [collect(1:nbeta) for _ in 1:S], zeros(S)
    match === :centres || throw(ArgumentError(
        "`registry_scan`: unknown `match = :$match`; supported are :centres (match Wannier " *
        "centres across stackings, the default), :none (trust file order), or a vector of " *
        "explicit permutations."))

    ref = state_positions(models[1].cell)
    P = models[1].cell.lattice.nperiodic
    perms = Vector{Vector{Int}}(undef, S)
    drift = zeros(S)
    perms[1] = collect(1:nbeta)
    for s in 2:S
        pos = state_positions(models[s].cell)
        shift = _pad_shift(ds[s] - ds[1], Val(D))
        pred = [shifted[b] ? ref[b] + shift : ref[b] for b in 1:nbeta]
        used = falses(nbeta)
        perm = zeros(Int, nbeta)
        maxdev = 0.0
        for n in 1:nbeta
            cands = [b for b in 1:nbeta if !used[b] && _wrapped_dist(pos[n], pred[b], P) <= atol]
            isempty(cands) && throw(ArgumentError(
                "`registry_scan`: stacking $s's Wannier orbital $n (at fractional " *
                "$(Tuple(round.(pos[n]; digits=6)))) matches no reference orbital within " *
                "atol = $atol. Either the centres drifted between Wannierizations (raise " *
                "`atol`, and inspect the drift), the declared `d` is wrong, or the file " *
                "belongs to a different family."))
            if length(cands) > 1
                genuine = allequal(shifted[b] for b in cands) &&
                          all(b -> _wrapped_dist(ref[b], ref[cands[1]], P) <= atol, cands) &&
                          all(b -> _wrapped_dist(pred[b], pred[cands[1]], P) <= atol, cands)
                genuine || throw(ArgumentError(
                    "`registry_scan`: stacking $s's Wannier orbital $n sits within " *
                    "atol = $atol of several reference orbitals ($(cands)) that are not a " *
                    "genuinely co-centred same-layer set (an accidental cross-layer " *
                    "coincidence at this stacking, or distinct positions within " *
                    "tolerance); the match is ambiguous -- tighten `atol`, drop or move " *
                    "this stacking, or pass explicit permutations via `match=`."))
            end
            b = cands[1]
            used[b] = true
            perm[n] = b
            maxdev = max(maxdev, _wrapped_dist(pos[n], pred[b], P))
        end
        if maxdev > min(atol / 2, 1e-3)
            @warn "registry_scan: stacking $s's Wannier centres drift up to $maxdev " *
                  "(fractional) from the positions the declared shift predicts. The ERTCA " *
                  "reference assumes rigidly shifted centres; a real drift this size bends " *
                  "the interlayer phases -- check the Wannierization convergence at this " *
                  "d, or the declared shift itself. The per-stacking drifts stay " *
                  "inspectable as `scan.drift`."
        end
        perms[s] = perm
        drift[s] = maxdev
    end
    return perms, drift
end

struct HarmonicBlock
    delta::Vector{SVector{2,Float64}}
    amp::Vector{ComplexF64}
end

Base.isempty(b::HarmonicBlock) = isempty(b.amp)

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

@inline function _g_canonical(g::SVector{2,Int})::Bool
    g[1] > 0 && return true
    g[1] < 0 && return false
    return g[2] >= 0
end

function _harmonic_value(h::RegistryHarmonics, iG::Int, i::Int, j::Int,
                         k::SVector{2,Float64})::ComplexF64
    b = h.blocks[iG, i, j]
    s = zero(ComplexF64)
    @inbounds for m in eachindex(b.amp)
        s += b.amp[m] * cispi(2 * dot(k, b.delta[m]))
    end
    return s
end

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

@inline recip_matrix_times(lattice::BravaisLattice{2}, n::SVector{2,Int}) =
    transpose(lattice.b) * SVector{2,Float64}(n)

@inline function _min_wrapped_norm(v::SVector{2,Float64})::Float64
    w = v - round.(v)
    return norm(w)
end

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
    neg = [findfirst(==(-g), F)::Int for g in F]
    herm_res = maximum(abs(C[m, s] - conj(C[neg[m], s])) for m in eachindex(F), s in 1:S)
    if enforce_hermiticity
        C = [(C[m, s] + conj(C[neg[m], s])) / 2 for m in eachindex(F), s in 1:S]
    end
    canon = [m for m in eachindex(F) if _g_canonical(F[m])]
    gvecs = F[canon]
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

@inline _pz(x::Float64) = x == 0.0 ? 0.0 : x

function _aliased_pair(E::Matrix{ComplexF64}, F::Vector{SVector{2,Int}})
    S = size(E, 1)
    for a in axes(E, 2), b in (a+1):size(E, 2)
        abs(abs(dot(view(E, :, a), view(E, :, b))) - S) <= 1e-8 * S && return (F[a], F[b])
    end
    return nothing
end

@inline _ertca_L(theta_top::Float64) = _rotation2(theta_top) - _rotation2(-theta_top)

@inline function _ertca_rot(shifted_i::Bool, shifted_j::Bool, theta_top::Float64)
    shifted_i && shifted_j && return _rotation2(-theta_top)      # both on the top layer
    !shifted_i && !shifted_j && return _rotation2(+theta_top)    # both on the bottom layer
    return one(SMatrix{2,2,Float64})                        # interlayer
end

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
    foldspec = if folding === :auto && isnothing(gvectors)
        Mtop = supercell_matrix(tb, top == 2 ? 2 : 1)
        Mbot = supercell_matrix(tb, top == 2 ? 1 : 2)
        _ertca_foldspec(h, Mbot, Mtop, kappa_cut, fold_width)
    else
        nothing
    end
    Msh = SMatrix{2,2,Int}(supercell_matrix(tb, top == 2 ? 2 : 1))   # the shifted layer
    Mfx = SMatrix{2,2,Int}(supercell_matrix(tb, top == 2 ? 1 : 2))   # the other one
    Nmat = Mfx - Msh
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
    probe = plane_wave_cell(moire; shells=shells, gcut=gcut, gvectors=gvectors)
    gs = unique(state_g_vectors(probe))
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
    cell = _ertca_cell(moire, schema, Gsets)

    binvT = inv(transpose(h.lattice.b))              # cartesian k̄ → monolayer fractional
    bmT = transpose(moire.b)
    Lmat = _ertca_L(theta_top)
    shell_of = _shell_ids(h)
    transfers = Set{SVector{2,Int}}()
    for i in 1:nb, j in 1:nb, ga in Gsets[i], gb in Gsets[j]
        dG = ga - gb
        push!(transfers, _g_canonical(dG) ? dG : -dG)
    end
    binvbarT = inv(transpose(h.lattice.b))
    LinvT = inv(transpose(Lmat))
    adjP, detP = isnothing(parent) ? (nothing, 0) : (_int_adjugate(parent), _int_det(parent))

    terms = AbstractHamiltonianTerm[]
    for dGc in sort!(collect(transfers); by=g -> (g[1], g[2]))
        gbar = if isnothing(adjP)
            gbar_fl = binvbarT * (LinvT * (bmT * SVector{2,Float64}(-dGc)))
            gbar_i = round.(Int, gbar_fl)
            maximum(abs, gbar_fl - gbar_i) <= INT_ROUND_ATOL || continue
            SVector{2,Int}(gbar_i)
        else
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
    (isnothing(strict) || loose < strict - 1e-9) && return nothing
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
    transfers = Set{SVector{2,Int}}()
    for i in 1:nb, j in 1:nb
        (!h.shifted[i] && h.shifted[j]) || continue
        for ga in Gsets[i], gb in Gsets[j]
            push!(transfers, ga - gb)
        end
    end
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
        gp_diag = detN == 0 ? nothing : _int_solve(adjN, detN, -dG)
        diag_retained = !isnothing(gp_diag) && !isnothing(_harmonic_match(h, gp_diag))
        for m1 in -nbox:nbox, m2 in -nbox:nbox
            nbv = SVector(m1, m2)
            ntv = _int_solve(adjT, detT, fs.Mbot * nbv + dG)
            isnothing(ntv) && continue
            diag_retained && nbv == gp_diag && ntv == gp_diag && continue
            off = bmT * SVector{2,Float64}(fs.Mbot * nbv)    # κ_meet = κ_bra + off
            kmeet_c = off + dGcart / 2                       # the family's κ_meet centroid
            norm(kmeet_c) - kreach > fs.kcut + width && continue     # never in-window
            gp = SVector{2,Int}(round.(Int, binvT * kmeet_c))
            hit = alpha_row(gp)
            isnothing(hit) && continue
            iG, sgn, alpha = hit
            Q = SMatrix{2,2,Float64}(binvT)
            p0 = SVector{2,Float64}(binvT * (bmT * (SVector{2,Float64}(dG) / 2) + off) -
                                    SVector{2,Float64}(gp))
            wshift = SVector{2,Float64}(bbarT * SVector{2,Float64}(gp))
            for i in 1:nb, j in 1:nb
                (!h.shifted[i] && h.shifted[j]) || continue
                block = sgn > 0 ? h.blocks[iG, i, j] : _conj_block(h.blocks[iG, j, i])
                (isempty(block) || sum(abs, block.amp) <= tol) && continue
                phase = cispi(2 * (dot(SVector{2,Float64}(gp - ntv), h.tau[j]) +
                                   dot(SVector{2,Float64}(nbv - gp), h.tau[i]))) * conj(alpha)
                nfam += 1
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
    nemit = 0
    for i in 1:nb, j in 1:nb
        h.shifted[i] == h.shifted[j] || continue
        M = layerM[j]
        adjM, detM = _int_adjugate(M), _int_det(M)
        rot = _ertca_rot(h.shifted[i], h.shifted[j], theta_top)     # L_ℓ⁻¹
        Q = binvT * rot
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

function _harmonic_match(h::RegistryHarmonics,
                         gbar::SVector{2,Int})::Union{Nothing,Tuple{Int,Int}}
    for (iG, g) in pairs(h.gvecs)
        gbar == g && return (iG, 1)
        gbar == -g && return (iG, -1)
    end
    return nothing
end

_conj_block(b::HarmonicBlock) =
    HarmonicBlock([SVector(_pz(-d[1]), _pz(-d[2])) for d in b.delta], conj.(b.amp))

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

@inline _k0_raw(term::KPolynomialTerm) = term.k0_frac

@inline _kcoefficient(term::KPolynomialTerm) = term.coefficient

@inline _kselector(term::KPolynomialTerm) = term.selector

@inline _krule(term::KPolynomialTerm) = term.rule

otimes(term::KPolynomialTerm, rule::AbstractQnumRule) =
    KPolynomialTerm(term.c, term.alpha, term.k0_frac, term.selector,
                    compose(term.rule, rule), term.coefficient, term.name)

function _make_kernel(term::KPolynomialTerm, ::Val{D}) where {D}
    size(term.alpha, 2) == D || throw(DimensionMismatch(
        "Term `$(term.name)`: the exponent table covers $(size(term.alpha, 2)) momentum " *
        "components but the cell is $D-dimensional."))
    nm = length(term.c)
    return KPolynomialKernel(SVector{nm,Float64}(term.c),
                             SMatrix{nm,D,Int}(term.alpha))
end

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
