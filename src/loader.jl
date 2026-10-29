# =====================================================================================
#  Paper loader for the ERTCA registry scans (archives in Code/Inputs/)
#
#  A 12×12 grid of Wannier90 calculations of an UNTWISTED bilayer, one per rigid registry
#  shift d = (i/12, j/12) of the TOP layer (fractional, monolayer lattice). Each directory
#  holds `<seed>.wout` + `<seed>_hr.dat`. Two materials ship:
#
#    NbSe₂  Inputs/NbSe2_grid_12x12.h5  a₀ = 3.45 Å, 20 Å slab, isolated 4-band manifold
#    WSe₂   Inputs/WSe2_grid_12x12.h5   a₀ = 3.295 Å, 39.87 Å slab, DISENTANGLED from 110 bands
#
#  Each raw tree is 46-87 MB of Wannier90 text, so what the repository carries is a ~1.4 MB
#  lossless archive per material; `ensure_deck!` expands it into byte-identical `_hr.dat`
#  files the first time an example touches that material (see `deck.jl`). Expansion is
#  LAZY and per material — including this file expands nothing. Everything below then reads
#  genuine Wannier90 output through `read_wannier90_hr`, with no idea any of that happened.
#
#  Structure of the manifold (read off the .wout, verified below):
#    * 4 Wannier functions per cell = 2 layers × 1 obstructed orbital × 2 spin;
#    * the projections are s-like at an OBSTRUCTED centre — the manuscript's Sec. V A
#      construction — NOT the metal site. NbSe₂ puts them at the fractional site (2/3, 1/3);
#      WSe₂ puts them on the layer axis at (0, 0);
#    * WFs 1,2 are CO-CENTRED (bottom layer) and 3,4 are CO-CENTRED (top layer) — the two
#      spin components of one orbital. Co-centred manifolds cannot each be their own
#      `:localized` site, so every stacking is imported with an explicit `schema=`
#      and the MODELS (not the file paths) are handed to `registry_scan`.
#    * the calculations are RELAXED: for NbSe₂ the interlayer Wannier separation breathes
#      from 6.34 Å (d ≈ ⅓(a₁+a₂), the AB-like registry) to 7.04 Å (d = 0, the AA registry),
#      an 11% variation, and the in-plane centres drift ≲ 4.2e-3 (fractional) from the
#      rigid-shift prediction. Both are physical; the ingest tolerance must accommodate
#      them (see `match_atol` on `ERTCAMaterial`).
#    * NbSe₂ is an isolated manifold (`num_bands = num_wann = 4`, no disentanglement);
#      WSe₂ is NOT — it is disentangled from 110 input bands with an outer window
#      −0.8 … 1.2 eV and a frozen window 0.6 … 1.2 eV. Nothing in the ingest depends on
#      that difference (`_hr.dat` looks the same either way), but its spreads are larger.
#
#  This file only DEFINES things — `include` it from an example script.
# =====================================================================================

using SingleParticleModels
using LinearAlgebra
using StaticArrays
using Logging

include(joinpath(@__DIR__, "deck.jl"))

# Read the paper's input archives; expand into a paper-local cache.
const ERTCA_DATA_DIR = joinpath(@__DIR__, "..", "Inputs")
const ERTCA_DECK_CACHE = joinpath(@__DIR__, "..", ".cache")

"""
    quiet(f) -> (result, n_warnings)

Run `f()` with warnings collected instead of printed, and report how many there were.
A relaxed registry family legitimately trips `registry_scan`'s rigid-shift drift warning on
most of its 144 stackings (the interlayer distance breathes with the registry); printing 130
copies of it buries everything else, while silencing it outright would hide a real signal.
Callers print the COUNT and the max drift instead -- see `scan.drift`.
"""
struct CountingLogger <: Logging.AbstractLogger
    count::Base.RefValue{Int}
end
Logging.min_enabled_level(::CountingLogger) = Logging.Warn
Logging.shouldlog(::CountingLogger, args...) = true
Logging.catch_exceptions(::CountingLogger) = false
function Logging.handle_message(l::CountingLogger, level, message, _mod, _grp, _id,
                                _file, _line; kwargs...)
    level >= Logging.Warn && (l.count[] += 1)
    return nothing
end

function quiet(f)
    lg = CountingLogger(Ref(0))
    res = with_logger(f, lg)
    return res, lg.count[]
end
"""
    ERTCAMaterial

Everything that distinguishes one registry family from another: where its files are, what
they are called, and the few physical numbers the ingest and the Fermi level need.

Fields:
-------
- `seedname`: the Wannier90 seed, so the files are `<seedname>.wout` / `<seedname>_hr.dat`
- `ngrid`: the registry grid is `ngrid × ngrid`
- `archive` / `deckdir`: the committed `.h5` and the git-ignored tree it expands into
- `site_names`: the two layer sites, bottom first
- `match_atol`: orbital-matching tolerance for `registry_scan`; see the note per material
- `filling`: electrons per Wannier state at the Fermi level, or `nothing` when the manifold
  is not one this repository has pinned an electron count for
- `zshift`: rigid z-relabelling of every centre; `0.0` unless a slab needs the escape hatch
"""
struct ERTCAMaterial
    seedname::String
    ngrid::Int
    archive::String
    deckdir::String
    site_names::Vector{Symbol}
    match_atol::Float64
    filling::Union{Rational{Int},Nothing}
    zshift::Float64
    name::String
end

### --- NbSe₂ ------------------------------------------------------------------------------
### `match_atol`: the centre drift of the relaxed family (4.2e-3 in-plane, 1.9e-2 in z as a
### fraction of the 20 Å slab cell) sits far above the `registry_scan` default 1e-4; 0.05 still
### separates the two layers cleanly (they are 0.35 apart in fractional z) and the two in-plane
### sublattices, so orbital identity stays unambiguous.
###
### `zshift`: the bottom layer's Wannier centre sits at z ≈ −0.37 Å, just BELOW the home cell, so
### `read_wannier90_hr` wraps it to z_frac ≈ 0.98 and compensates every cross-layer coupling
### with ΔR₃ = ±1 -- exactly and correctly, since δ = ΔR + τ_j − τ_i is invariant.
### `registry_scan` used to inspect the COMPENSATED ΔR₃ and refuse this family as out-of-plane
### hopping, although the _hr.dat carries ΔR₃ ≡ 0 on all 409 R-vectors; the workaround was a
### rigid z-offset of every centre (a pure relabelling: it leaves τ_j − τ_i, hence every Bloch
### phase, untouched -- verified bit-identical). The guard now tests the SEPARATION δ₃ instead
### of ΔR₃, which is wrap-invariant, so the centres go in as the .wout reports them.
###
### `filling`: Nb⁴⁺ is d¹, so each Nb donates one electron to the manifold. The bilayer cell
### holds 2 Nb and the Wannier manifold holds 4 spinor states -> HALF FILLING. Every Fermi level
### in the NbSe₂ examples is fixed by that electron count, never fitted.
const NBSE2 = ERTCAMaterial(
    "NbSe2", 12,
    joinpath(ERTCA_DATA_DIR, "NbSe2_grid_12x12_tight.h5"),
    joinpath(ERTCA_DECK_CACHE, "NbSe2_grid_12x12_tight"),
    [:Nb_b, :Nb_t], 0.05, 1 // 2, 0.0, "NbSe2")

### --- WSe₂ -------------------------------------------------------------------------------
### `match_atol`: re-derived, NOT inherited from NbSe₂ -- the justifying geometry is different.
### WSe₂'s layers sit 7.11 Å apart in a 39.87 Å cell, a fractional separation of 0.178 (half
### NbSe₂'s 0.35), so the SAME 0.05 leaves a smaller margin: it is 28% of the layer separation
### here against 14% there. Still unambiguous, and still far above the drift, but it is the
### number to loosen first if a WSe₂ stacking is ever mismatched.
###
### `zshift`: not needed. Both WSe₂ layers' centres (z ≈ 16.4 Å and 23.5 Å) sit inside the
### 39.87 Å cell, so the wrap NbSe₂ runs into never arises.
###
### `filling`: deliberately `nothing`. W is d⁰, so the NbSe₂ electron count says nothing here,
### and this repository has not yet pinned which DFT manifold the 4 WSe₂ Wannier functions
### reproduce (the frozen window is 0.6 … 1.2 eV in the wannierisation's own reference). The
### DFT benchmark compares absolute energies and needs no Fermi level; anything that does need
### one must pass it explicitly rather than inherit a number nobody checked.
const WSE2 = ERTCAMaterial(
    "WSe2", 12,
    joinpath(ERTCA_DATA_DIR, "WSe2_grid_12x12.h5"),
    joinpath(ERTCA_DECK_CACHE, "WSe2_grid_12x12"),
    [:W_b, :W_t], 0.05, nothing, 0.0, "WSe2")

### Expansion is lazy and memoized: including this file must not unpack 130 MB of Wannier90
### text for a material the script never touches.
const _DECK_ROOTS = Dict{String,String}()

"""
    deck_root(mat) -> String

The directory of Wannier90 stackings for `mat`, expanding its committed archive on first use.
"""
function deck_root(mat::ERTCAMaterial)
    return get!(_DECK_ROOTS, mat.seedname) do
        ensure_deck!(mat.seedname, mat.archive, mat.deckdir; ngrid=mat.ngrid)
    end
end

"""
    ertca_lattice(printed::BravaisLattice{3}) -> BravaisLattice{3}

The slab lattice, rebuilt in EXACT triangular form from the `.wout`'s printed vectors.

Wannier90 prints lattice vectors to six decimals, so the hexagonal degeneracy
`|b₁| = |b₂| = |b₁−b₂|` is broken at the ~1e-7 level. `SHELL_RTOL = 1e-8` then reads the
six first-neighbour reciprocal vectors as THREE distinct shells, which fragments both
`gshells=` (it would retain a ±pair at a time instead of a full star, breaking the C₃
structure of the retained set) and the `ertca_shell<s>` term names used for ablation.
The underlying DFT lattice is exactly triangular, so we restore it: `a₀` and `c` are taken
from the file, the 120° angle is imposed exactly.
"""
function ertca_lattice(printed::BravaisLattice{3})
    a0 = norm(primitive_vector(printed, 1))
    c = primitive_vector(printed, 3)[3]
    ### the in-plane net with the 120° angle imposed exactly, then the slab height as a
    ### PERIODIC padding axis -- which is what `embed_into_higher_dimension(...; padding=)` is
    ### for, and byte-for-byte the matrix this used to write out by hand
    return embed_into_higher_dimension(bravais_lattice(:triangular; a=a0), 3;
                                       periodic=true, padding=c)
end

"""
    snap_cocentred(centres; atol=1e-5) -> Vector

Collapse Wannier centres that agree to `atol` onto a single representative.

Wannier90 prints centres to six decimals, and the two spin partners of one orbital are the
SAME centre -- but in 2 of the 144 WSe₂ stackings the printed pair straddles a rounding
boundary and differs by exactly 1e-6 Å in one coordinate (`shift_i01_j07` in y,
`shift_i04_j07` in z). `read_wannier90_hr` groups centres at its file tolerance, also 1e-6,
so those two stackings present THREE distinct centres instead of two and the import refuses a
`site_names` of length 2.

The fix belongs here rather than in the import's `atol`, which additionally bounds the
Hermiticity check on the hoppings: loosening that to 1e-5 would admit 10 μeV of asymmetry in
_hr.dat entries printed to 1 μeV. Snapping is safe by a wide margin -- the real separations
in these decks are 7 Å between layers and ~1 Å between in-plane sublattices, so a 1e-5 Å
tolerance cannot merge two centres that are physically distinct.
"""
function snap_cocentred(centres::AbstractVector; atol::Real=1e-5)
    reps = eltype(centres)[]
    out = similar(collect(centres))
    for (i, c) in pairs(collect(centres))
        k = findfirst(r -> all(abs.(c .- r) .<= atol), reps)
        if isnothing(k)
            push!(reps, c)
            out[i] = c
        else
            out[i] = reps[k]
        end
    end
    return out
end

"""
    ertca_stacking(mat, i, j; root=deck_root(mat)) -> (d, model, spreads)

Import the single stacking `shift_i<ii>_j<jj>` of `mat` as a `HamiltonianModel{3}` in the
atomic gauge pinned to its TRUE Wannier centres, together with its declared registry shift
`d = (i/ngrid, j/ngrid)`.

The 3D slab lattice is kept deliberately: after dropping z, all four centres coincide
in-plane at d = 0 and the cross-layer orbital identity would be ambiguous. `registry_scan`
restricts to the in-plane lattice itself (it checks a₃ ∥ ẑ and ΔR₃ ≡ 0, both true here).
"""
function ertca_stacking(mat::ERTCAMaterial, i::Int, j::Int;
                        root::AbstractString=deck_root(mat))
    dir = joinpath(root, "shift_i" * lpad(i, 2, '0') * "_j" * lpad(j, 2, '0'))
    isdir(dir) || error("missing stacking directory $dir")
    w = read_wannier90_wout(joinpath(dir, "$(mat.seedname).wout"))
    centres, spreads = w.centres, w.spreads
    length(centres) == 4 || error("$dir declares $(length(centres)) Wannier functions, " *
                                  "expected 4 (2 layers × 1 orbital × 2 spin)")
    ### A `.wout` rendered by an OLDER `deck.jl` carries no `Unit Cell Volume` line, so the
    ### package reader finds no lattice block and returns `nothing`. That is a stale unpacked
    ### deck, not a corrupt archive: the fix is to delete the directory and let `ensure_deck!`
    ### expand it again from the committed `.h5`.
    w.lattice === nothing && error(
        "$dir: the `.wout` has no `Lattice Vectors (Ang) ... Unit Cell Volume` block, so it " *
        "was rendered by a deck.jl predating that line. Delete $(dirname(dir)) and re-run; " *
        "`ensure_deck!` will expand the archive again.")
    lat = ertca_lattice(w.lattice)
    centres = snap_cocentred(centres)
    ### `zshift` is 0 for both shipped materials -- kept as a knob because a rigid
    ### z-relabelling is exactly free (it leaves τ_j − τ_i, hence H(k), bit-identical) and is
    ### the escape hatch if a future slab lands its centres somewhere the import dislikes
    zc = iszero(mat.zshift) ? centres :
         [c + SVector(0.0, 0.0, mat.zshift) for c in centres]
    ### spin is the internal axis: the two co-centred functions at each site, in file order
    model = read_wannier90_hr(lat, joinpath(dir, "$(mat.seedname)_hr.dat");
                              centres=zc,
                              schema=InternalSchema(:spin => [:up, :dn]),
                              site_names=mat.site_names,
                              name="$(mat.name) d=($i/$(mat.ngrid), $j/$(mat.ngrid))")
    return (d=[i / mat.ngrid, j / mat.ngrid], model=model, spreads=spreads)
end

"""
    ertca_scan(mat; stride=1, skip=(), root=deck_root(mat), match=:centres) -> RegistryScan

Ingest the registry family. `stride` subsamples the 12×12 grid (`stride=1` takes all 144);
`skip` is a collection of `(i, j)` pairs to HOLD OUT, which is how
`reconstruction_residual` is given honest, unseen stackings.

Layer identity is declared per ORBITAL, never inferred: `[:b, :b, :t, :t]` in file order,
with `:t` (WFs 3,4) the layer that moves with `d`.
"""
function ertca_scan(mat::ERTCAMaterial; stride::Int=1, skip=(),
                    root::AbstractString=deck_root(mat), match::Symbol=:centres,
                    name::AbstractString="$(mat.name) $(mat.ngrid)x$(mat.ngrid) registry scan")
    skipset = Set(Tuple.(skip))
    entries = Pair[]
    for i in 0:stride:(mat.ngrid-1), j in 0:stride:(mat.ngrid-1)
        (i, j) in skipset && continue
        s = ertca_stacking(mat, i, j; root=root)
        push!(entries, s.d => s.model)
    end
    return registry_scan(entries; layers=[:b, :b, :t, :t], shifted_layer=:t,
                         labels=[:b_up, :b_dn, :t_up, :t_dn],
                         match=match, atol=mat.match_atol, name=name)
end

"""
    ertca_holdout(mat, pairs; root=deck_root(mat)) -> RegistryScan

A `RegistryScan` holding only the named `(i, j)` stackings — the held-out set for
`reconstruction_residual`.
"""
function ertca_holdout(mat::ERTCAMaterial, pairs; root::AbstractString=deck_root(mat))
    entries = Pair[]
    for (i, j) in pairs
        s = ertca_stacking(mat, i, j; root=root)
        push!(entries, s.d => s.model)
    end
    return registry_scan(entries; layers=[:b, :b, :t, :t], shifted_layer=:t,
                         labels=[:b_up, :b_dn, :t_up, :t_dn],
                         match=:centres, atol=mat.match_atol,
                         name="$(mat.name) held-out")
end
