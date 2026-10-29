##################################################################
###### export only the following functions from this module ######
##################################################################

export KPeriodicity, ExactlyPeriodic, TruncatedWindow, NoZone, kperiodicity

### ------------------------------------------------------------------------------------
### How H(k) relates to the Brillouin-zone torus -- a TYPE, not a flag
### ------------------------------------------------------------------------------------
###
### This was a `Symbol` on `HamiltonianModel` until 2026-09-16, read back at seventeen sites
### across eight files, and what those sites did with it was overwhelmingly to REFUSE. That is
### the signature of a type wearing a flag's clothing: the caller never passes it, every
### consumer re-reads it, and each consumer decides for itself which of the three cases it can
### serve. A new topological quantity that forgot a case got no help from anything.
###
### As singleton types the three cases dispatch, so a quantity that forgets one is a
### `MethodError` at the call rather than a plausible-looking number. The test that decided
### this, and that separates a legitimate symbol flag from a type in disguise: a symbol is fine
### when every branch it selects returns the same KIND of answer about the same object and the
### caller could flip it without invalidating anything they hold (`matrix = :dense | :sparse`
### passes); it is a type in disguise when the branches answer about DIFFERENT objects, or when
### code the caller never wrote must re-read it to know what it is holding. This failed both
### halves. The CHANGELOG's v0.4.0 section is the tracked account.
###
### This file is VOCABULARY and depends on nothing -- not even on what a cell is -- which
### is why it loads before `bravaislattice.jl`. `UnitCell{D,K}` carries the class in its
### TYPE, so `kperiodicity(cell)` is a type read rather than a derivation; that method,
### and the cell-level forward to the derivation below, live beside the struct.

"""
    KPeriodicity

How `H(k)` relates to the Brillouin-zone torus. A property of the CELL -- the site kinds and
their `G` sets decide it -- which a compiled model carries along. One of three singletons:
[`ExactlyPeriodic`](@ref), [`TruncatedWindow`](@ref), [`NoZone`](@ref).

Ask for it with [`kperiodicity`](@ref), on a cell or on a model.
"""
abstract type KPeriodicity end

"""
    ExactlyPeriodic()

Every site is `:localized` (folded or not): the spectrum is exactly periodic in `k -> k + G`.

On a folded cell `H` itself is periodic only up to a constant coset-relabelling unitary
([`bloch_periodicity`](@ref)); the spectrum and the Berry curvature are exactly periodic.
"""
struct ExactlyPeriodic <: KPeriodicity end

"""
    TruncatedWindow()

Plane-wave sites with a truncated `G` set: periodic up to truncation accuracy only.

`H(k + G)` is a DIFFERENT truncation window, so no unitary relates it to `H(k)` exactly -- only
the spectrum is periodic, to the accuracy of the window. Zone seams are therefore recomputed
rather than glued, and the quantities that need an exact boundary unitary refuse.
"""
struct TruncatedWindow <: KPeriodicity end

"""
    NoZone()

A plane-wave site with `G`-set `{0}` -- a k·p model. The lattice is only a coordinate frame and
`H(k)` is not periodic at all.

Analysis functions take an explicit `kregion` here, and `chern_number` refuses in favour of
[`berry_flux`](@ref): a bare Dirac cone integrates to ±½, which is the honest answer and not an
integer. [`discretize`](@ref) regularizes such a model onto a real lattice when integer topology
is what is wanted.
"""
struct NoZone <: KPeriodicity end

Base.show(io::IO, ::ExactlyPeriodic) = print(io, "ExactlyPeriodic()")
Base.show(io::IO, ::TruncatedWindow) = print(io, "TruncatedWindow()")
Base.show(io::IO, ::NoZone) = print(io, "NoZone()")

"""
    _derive_kperiodicity(site_kind, site_states, basis) -> KPeriodicity

Derive the class from the CONFORMED cell fields, before a `UnitCell` object exists -- which is
what `UnitCell`'s derive-`K` constructor needs. The cell method `_derive_kperiodicity(cell)`
lives beside the struct in `unitcell.jl` and forwards here, so there is one implementation.

The join rule is `NoZone` dominates, then `TruncatedWindow`, then `ExactlyPeriodic`.
"""
function _derive_kperiodicity(site_kind::Vector{Symbol}, site_states::Vector{Vector{Int}},
                              basis::AbstractVector)::KPeriodicity
    out = ExactlyPeriodic()
    for s in eachindex(site_kind)
        site_kind[s] === :plane_wave || continue
        gset = unique(basis[i].G for i in site_states[s])
        if length(gset) <= 1
            return NoZone()                    # NoZone dominates
        else
            out = TruncatedWindow()
        end
    end
    return out
end

### ------------------------------------------------------------------------------------
### Cross-checking a class that was SUPPLIED against the one the contents imply
### ------------------------------------------------------------------------------------
###
### A class read back from a file, or handed to `UnitCell{D,K}` by a transform, is trusted --
### that is the point of supplying it. But trust with no check is how a wrong value re-enables
### every refusal the class exists to keep shut: a file asserting `ExactlyPeriodic` for a k·p
### cell would make `chern_number` answer a number instead of refusing, which is exactly the
### bug the stored field was added to prevent, merely relocated.
###
### The check is cheap because the LEGITIMATE disagreement is exactly one-directional.
### Enumerating every path that sets a class:
###
###   fold of a NoZone cell           stores NoZone           derivation says TruncatedWindow
###   fold of a TruncatedWindow cell  stores TruncatedWindow  derivation agrees
###   fold of an ExactlyPeriodic cell stores ExactlyPeriodic  derivation agrees (localized)
###   supercell / open_boundary /
###     nambu_doubling / submodel     preserve, site kinds untouched -> derivation agrees
###   stack                           joins, and the join is what the derivation computes
###
### So `NoZone` stored where the derivation says `TruncatedWindow` is the folded-plane-wave
### case and is correct; every other mismatch is a bad value and is refused by name.

"""
    _kper_agrees(stored, derived) -> Bool

Whether a SUPPLIED k-periodicity class is consistent with the one the cell's contents imply.
Equality always is. The single legitimate inequality is a stored `NoZone` against a derived
`TruncatedWindow`: folding replicates a plane-wave site's one `G` label into `|det M|` coset
representatives, which the derivation reads as a truncation window, and the physics -- whether
`H(k)` has a Brillouin-zone torus at all -- is the parent's answer.
"""
_kper_agrees(stored::KPeriodicity, derived::KPeriodicity) = stored === derived
_kper_agrees(::NoZone, ::TruncatedWindow) = true

### `_check_supplied_kperiodicity(cell, source)` is the `UnitCell`-typed half of this and
### lives in `unitcell.jl`, beside the other two methods that need to know what a cell is.

### ------------------------------------------------------------------------------------
### The two shared refusals
### ------------------------------------------------------------------------------------
###
### Before these existed, six entry points carried their own hand-written "this model is not
### k-periodic" message and three carried their own "this needs an exact zone". The DIAGNOSIS
### was the same in each; only what the caller was attempting and what it should do instead
### differed. Those are the two arguments here, so every refusal still names the caller, the
### reason and the fix -- the package's contract for an error -- with one copy of the reason.

"""
    _require_zone(k, what, fix)

Refuse a `NoZone` cell by name. `what` names the operation ("`diagonalize_over_bz` tiles the
Brillouin zone"), `fix` is the sentence telling the caller what to do instead.
"""
_require_zone(::KPeriodicity, ::AbstractString, ::AbstractString) = nothing
_require_zone(::NoZone, what::AbstractString, fix::AbstractString) = throw(ArgumentError(
    "$what, and this model has no Brillouin-zone torus: its k-periodicity is `NoZone()` -- a " *
    "k·p frame, whose lattice is a coordinate frame rather than a zone, so a fractional mesh " *
    "over it samples nothing physical. $fix"))

"""
    _require_exact_zone(k, what, truncated_fix, none_fix)

Refuse anything but `ExactlyPeriodic` by name, with a different fix for each of the two
refused cases -- a truncated window and a k·p frame fail for genuinely different reasons and a
shared sentence would be wrong for one of them.
"""
_require_exact_zone(::ExactlyPeriodic, ::AbstractString, ::AbstractString, ::AbstractString) = nothing
_require_exact_zone(::TruncatedWindow, what::AbstractString, truncated_fix::AbstractString,
                    ::AbstractString) = throw(ArgumentError(
    "$what needs an exactly k-periodic model, and this one is a truncated plane-wave window " *
    "(`TruncatedWindow()`): `H(k+G)` is a DIFFERENT truncation window and no unitary relates " *
    "the two exactly -- only the spectrum is periodic, to truncation accuracy. $truncated_fix"))
_require_exact_zone(::NoZone, what::AbstractString, ::AbstractString,
                    none_fix::AbstractString) = throw(ArgumentError(
    "$what needs an exactly k-periodic model, and this one has no Brillouin-zone torus at all " *
    "(`NoZone()` -- a k·p frame). $none_fix"))
