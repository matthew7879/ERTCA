##################################################################
###### export only the following functions from this module ######
##################################################################

export RegistryScan, registry_scan
export n_stackings, n_orbitals




#############################################################
###### The registry scan container                      #####
#############################################################

### `K` rides along so `models` keeps a CONCRETE element type. Every stacking in a registry
### scan is an imported `_hr.dat` model on one shared lattice, so they necessarily agree on the
### k-periodicity class; spelling `Vector{HamiltonianModel{D}}` after `HamiltonianModel` gained
### its class parameter would make that an abstract-eltype vector, boxing every entry on the
### path `registry_harmonics` walks once per shell.
"""
    RegistryScan{D}

A validated family of commensurate-bilayer Hamiltonians at rigid registry shifts `d` -- the
raw material of the ERTCA pipeline. One member per computed stacking: the same untwisted
bilayer, with the SHIFTED layer rigidly translated by the in-plane fractional vector `d[s]`
relative to the other layer.

Fields:
-------
- `lattice::BravaisLattice{2}`      the in-plane monolayer lattice, unrotated frame
- `models::Vector{HamiltonianModel{D,K}}`  one imported model per stacking, all `:atomic` gauge
- `d::Vector{SVector{2,Float64}}`   the declared fractional in-plane shifts, one per stacking
- `shifted::BitVector`              per REFERENCE orbital: does it sit on the shifted layer?
- `labels::Vector{Symbol}`          the reference orbital labels (levels of the realized
                                    model's internal axis)
- `layer_names::NTuple{2,Symbol}`   `(unshifted, shifted)` -- bookkeeping and display
- `permutations::Vector{Vector{Int}}`  per stacking: state index in `models[s]` → reference
                                    orbital index (identity when the files agree on order)
- `drift::Vector{Float64}`          per stacking: the largest wrapped distance between a
                                    matched centre and the position the declared shift
                                    predicts (fractional) -- the rigid-shift health of the
                                    family, kept inspectable after ingest
- `name::String`

The invariants -- one shared lattice, equal orbital counts, atomic gauge, constant kernels,
orbital identity resolved across stackings -- are established by [`registry_scan`](@ref);
construct through it.
"""
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

"""
    n_stackings(scan::RegistryScan) -> Int

The number of computed stackings `S` -- the number of registry harmonics the scan can
resolve (see [`registry_harmonics`](@ref): first shell needs 7 on a triangular lattice,
second shell 13).
"""
n_stackings(scan::RegistryScan)::Int = length(scan.models)

"""
    n_orbitals(scan::RegistryScan) -> Int

The number of Wannier orbitals per bilayer cell (both layers together) -- the dimension of
each harmonic block `W_{ββ'}`.
"""
n_orbitals(scan::RegistryScan)::Int = length(scan.labels)

function Base.show(io::IO, scan::RegistryScan{D}) where {D}
    nsh = count(scan.shifted)
    print(io, "RegistryScan{", D, "}(\"", scan.name, "\": ", n_stackings(scan),
          " stackings, ", n_orbitals(scan), " orbitals -- ",
          n_orbitals(scan) - nsh, " on :", scan.layer_names[1], ", ", nsh, " on :",
          scan.layer_names[2], " (shifted))")
end
Base.show(io::IO, ::MIME"text/plain", scan::RegistryScan) = show(io, scan)




#############################################################
###### Ingest + cross-stacking validation               #####
#############################################################

"""
    registry_scan(entries; layers, shifted_layer, lattice=nothing, labels=nothing,
                  match=:centres, atol=1e-4, name="registry_scan") -> RegistryScan

Ingest a family of commensurate-bilayer Wannier90 calculations at rigid registry shifts into
a validated [`RegistryScan`](@ref) -- stage one of the ERTCA pipeline.

`entries` is a vector of pairs `d => source`, one per stacking:
- `d` -- the in-plane shift of the SHIFTED layer, in fractional coordinates of the monolayer
  lattice (2 components);
- `source` -- either an already-imported `HamiltonianModel` (any gauge; flipped to `:atomic`,
  an O(1) operation), or a NamedTuple `(hr = "seedname_hr.dat", centres = "…_centres.xyz")`
  loaded through [`read_wannier90_hr`](@ref). `centres` is mandatory for file entries: the
  ERTCA identification `bloch_hamiltonian(model, k̄) = W(k̄, d)` holds in the atomic gauge
  pinned to the TRUE Wannier centres, and the shifted layer's centres move with `d`.
  CO-CENTRED Wannier manifolds (several functions at one centre) need an internal schema
  the file path does not carry: import those yourself with
  `read_wannier90_hr(lattice, hr; centres=…, schema=…)` and pass the MODELS as entries.

Keywords:
---------
- `layers` (REQUIRED): one label per Wannier orbital, in the file order of the FIRST entry,
  taking exactly two distinct values -- layer identity is declared per ORBITAL, never
  inferred from positions.
- `shifted_layer` (REQUIRED): which of the two labels names the layer displaced by `d`.
- `lattice`: the `BravaisLattice` the file entries are read on (required when any entry is a
  file pair; must match the files' dimension). Model entries carry their own.
- `labels`: reference orbital labels (default `:w1, :w2, …`); these become the levels of the
  realized continuum model's internal axis.
- `match`: how orbital identity is resolved across independently-Wannierized stackings:
  `:centres` (default) matches each stacking's centres against the first entry's, displaced
  by `d[s] − d[1]` on the shifted layer -- refusing unmatched or ambiguous orbitals by name
  and auto-permuting reordered files (co-centred orbitals are matched in file order, which
  the files must share); `:none` trusts the file order; a `Vector{Vector{Int}}` applies your
  own permutations (`perms[s][n]` = reference orbital of state `n` in stacking `s`).
- `atol`: the fractional matching tolerance for `:centres`. A genuine centre drift between
  Wannierizations (convergence differing per `d`, or relaxation that depends on the registry)
  surfaces here: matched drifts beyond `min(atol/2, 1e-3)` are reported with a `@warn` --
  capped, deliberately, so that raising `atol` to admit a drifting family does not also raise
  the threshold past which the drift is worth mentioning -- and unmatched orbitals refuse and
  name the stacking. Note `atol` ALSO sets the smallest distinguishable registry shift (two
  stackings closer than it are refused as duplicates), so a very loose value on a dense
  d-grid can collide with the grid spacing itself.

Validation (the method's load-bearing assumption is ONE smooth Wannier gauge across the
whole family; everything checkable is checked here, by name): a shared Bravais lattice
(`same_basis`), equal orbital counts, `:localized` sites only, no folded cells, constant
kernels only (a W90 import carries no k-dependence), zero momentum offsets, and -- for 3D
slab calculations -- in-plane `a₁, a₂` with `a₃ ∥ ẑ` and no out-of-plane hopping images
(`ΔR₃ = 0`; use a 2D k-mesh in the DFT). What is NOT checkable from single stackings (a
random gauge jump between two Wannierizations) surfaces downstream, in
`reconstruction_residual` and the harmonic decay.
"""
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

    ### ---- load every stacking --------------------------------------------------------------
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

    ### ---- gauge and structural validation, stacking by stacking ----------------------------
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
            ### `LATTICE_ATOL` is a DIMENSIONLESS (fractional) tolerance; these are cartesian
            ### lengths, so scale it by the lattice's own length scale -- otherwise the test is
            ### absurdly strict in Å and only passes on exactly-zero input
            atol_cart = distance_atol(m.cell.lattice)
            (abs(a[1, 3]) <= atol_cart && abs(a[2, 3]) <= atol_cart &&
             abs(a[3, 1]) <= atol_cart && abs(a[3, 2]) <= atol_cart) ||
                throw(ArgumentError(
                    "`registry_scan`: stacking $s has a 3D lattice whose first two " *
                    "primitive vectors are not in-plane with `a₃ ∥ ẑ`; re-express the slab " *
                    "so rows 1:2 span the layers and row 3 is the stacking axis."))
            ### Out-of-plane hopping is a property of the SEPARATION δ₃ = ΔR₃ + τ_{j3} − τ_{i3},
            ### not of ΔR₃ alone. Testing ΔR₃ is not wrap-invariant: a Wannier centre sitting
            ### just below the home cell (z = −0.4 Å in a 20 Å slab is entirely ordinary) is
            ### wrapped on import and every cross-layer coupling is compensated with
            ### ΔR₃ = ±1 -- correctly, since δ stays invariant -- whereupon a ΔR₃ test refuses
            ### data whose _hr.dat has ΔR₃ ≡ 0 throughout, and advises a re-run that would
            ### change nothing. Test the separation instead: a hop that does not cross half the
            ### vacuum is intra-slab however the centres happen to be wrapped.
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

    ### ---- the in-plane monolayer lattice ----------------------------------------------------
    ### the vacuum axis is periodic on paper and DECOUPLED in fact -- `leading_block` is the
    ### owner of that projection, and re-checks the orthogonality the per-stacking guard above
    ### already demanded with an ERTCA-specific message
    lat2 = leading_block(models[1].cell.lattice, 2)

    ### ---- the layer map ---------------------------------------------------------------------
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

    ### ---- orbital labels --------------------------------------------------------------------
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

    ### ---- orbital identity across stackings -------------------------------------------------
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

### the wrapped fractional distance between two positions (periodic components only)
@inline function _wrapped_dist(a::SVector{D,Float64}, b::SVector{D,Float64},
                               P::Int)::Float64 where {D}
    v = a - b
    w = SVector{D,Float64}(ntuple(m -> m <= P ? v[m] - round(v[m]) : v[m], Val(D)))
    return norm(w)
end

### the in-plane shift, zero-padded to the DFT model's own dimension
@inline _pad_shift(shift::SVector{2,Float64}, ::Val{D}) where {D} =
    _pad_components(shift, Val(D))

### `AbstractVector{<:...}` rather than `Vector{HamiltonianModel{D}}`: `Vector` is invariant,
### so the concrete-eltype vector every caller actually holds would not match the latter.
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
                ### file order may resolve only a GENUINELY co-centred set: same layer AND
                ### coincident in the reference cell. A shifted centre that merely lands on
                ### an unshifted one at THIS stacking is an accidental cross-layer collision
                ### -- resolving it by file order would silently swap the layers.
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
        ### the threshold must NOT scale away with a raised atol (the refusal message's own
        ### advice): drifts beyond 1e-3 fractional warn regardless of the matching window
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
