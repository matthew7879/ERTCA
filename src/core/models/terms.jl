##################################################################
###### export only the following functions from this module ######
##################################################################

export AbstractHamiltonianTerm, BondTerm, KDependentTerm
export AbstractBondSelector, OnSiteSelector, TranslationSelector, ShellSelector, RadiusSelector,
       DeltaGSelector, ExplicitBondSelector
export AbstractBondWeight, UnitWeight, Chirality, ChiralityPhase, ExponentialDecay
export bond_weight, validate_weight, is_symmetric_weight
export HoppingTerm, KLinearTerm, KQuadraticTerm, KEnvelopeTerm
export CompileContext
export emit_term!, max_shell, max_radius, needs_neighbors, is_serializable, term_name
export onsite, hopping, translation_hopping, delta_g, explicit_bonds
export kinetic, dirac, k_linear, k_quadratic, k_envelope
export kvalue, kernel_parity, kernel_tag
export AbstractEnvelope, GaussianEnvelope, TabulatedEnvelope, BlochSumEnvelope




#############################################################
###### Term hierarchy                                   #####
#############################################################

"""
    AbstractHamiltonianTerm

A contribution to a Hamiltonian, expressed as a rule for generating couplings rather than as a
matrix.

Two families:
-------------
- `BondTerm` -- the amplitude is independent of `k`, so the term lowers **entirely** into the
  compiled `CouplingList` and costs nothing per k-point beyond the phase already being applied.
- `KDependentTerm` -- the amplitude is a function of `k` (a continuum kinetic energy
  `ħ²|k+G|²/2m`, a Dirac `ħv k⋅σ`). These cannot become bonds, but their *sparsity pattern* is just
  as compilable: which `(i,j)` they touch, with what prefactor and what momentum offset, is fixed
  once. The per-k cost then reduces to one dot product per non-zero, not a re-run of the selection
  logic.

Every term is built from three composable pieces -- a spatial **selector**, a quantum-number
**rule**, and a per-bond geometric **weight** -- which is what lets `⊗` attach physics uniformly.

Dimension:
----------
Terms are deliberately **dimension-free**: `hopping(-1.0; shell=1)` mentions no `D`. The dimension
is a property of the cell, is stated exactly once (on the cell), and every dimension-carrying
datum a term stores (`dR`, `dG`, `k0`, `c`, `M`) is validated against the cell and materialized
into statically-sized arrays when `compile` lowers the term. Term construction and emission are
cold paths, so nothing is lost by deferring; what is gained is that the same term object works
against any cell it fits.
"""
abstract type AbstractHamiltonianTerm end

abstract type BondTerm       <: AbstractHamiltonianTerm end
abstract type KDependentTerm <: AbstractHamiltonianTerm end




#############################################################
###### Spatial selectors                                #####
#############################################################

"""
    AbstractBondSelector

Chooses which `(i, j, ΔR)` triples a term acts on, before quantum numbers are consulted.
"""
abstract type AbstractBondSelector end

"""
    OnSiteSelector(sites)

All state pairs living on the same site, with `ΔR = 0`. `sites` empty means every site.

Notes:
------
This is deliberately *not* `i == j`. A term like `onsite(B) ⊗ pauli_x(:spin)` must couple A↑ to A↓,
which are two different states on one site. Restricting to `i == j` would silently drop every
internal off-diagonal on-site term -- in-plane Zeeman fields, sublattice mixing, Rashba-like
on-site pieces.
"""
struct OnSiteSelector <: AbstractBondSelector
    sites::Vector{Symbol}
end
OnSiteSelector() = OnSiteSelector(Symbol[])

"""
    TranslationSelector(dR, sites_from, sites_to)

Every state pair connected by the fixed Bravais translation `dR`.

Notes:
------
At `dR = 0` the translation orients nothing, and `(i, j, 0)` / `(j, i, 0)` are each other's
Hermitian partner -- so there each unordered pair is emitted **once**, in an orientation admitted
by the site filters (source `sites_from` → target `sites_to`) and by the rule, and
`finalize_bonds` supplies the conjugate. Same-site pairs behave exactly like `onsite`; cross-site
pairs are the intra-cell couplings -- the two layers of a bilayer at the same in-plane position,
say.
"""
struct TranslationSelector <: AbstractBondSelector
    dR::Vector{Int}
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}
end

"""
    ShellSelector(shell, sites_from, sites_to)

Every state pair whose sites lie in the given distance shell. `shell = 1` is nearest-neighbour.
"""
struct ShellSelector <: AbstractBondSelector
    shell::Int
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}

    function ShellSelector(shell::Int, from::Vector{Symbol}, to::Vector{Symbol})
        shell >= 1 || throw(DomainError(shell, "Shell index must be at least 1 (1 = nearest neighbour)."))
        return new(shell, from, to)
    end
end

"""
    RadiusSelector(rcut, sites_from, sites_to)

Every state pair whose sites lie within the cartesian cutoff distance `rcut` -- **all** shells
at once, without naming any of them.

Notes:
------
This is the selector for relaxed, corrugated or imported geometry, where the distance spectrum
is quasi-continuous and a shell index is not a stable name for anything: real-space twisted
bilayers, strained supercells, structures read from files. It consumes the neighbour table's
per-entry distances directly and never reads a shell id, so `compile` may build the table
**ungrouped** (`neighbor_table(...; group_shells=false)`) and the near-degenerate-shell
warnings cannot fire. Like `ShellSelector` it enumerates unordered geometric pairs
(`has_intrinsic_direction` is `false`), so a complex constant amplitude gets the same
construction-time warning and callback amplitudes obey the same conjugate-symmetry contract.
"""
struct RadiusSelector <: AbstractBondSelector
    rcut::Float64
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}

    function RadiusSelector(rcut::Real, from::Vector{Symbol}, to::Vector{Symbol})
        (isfinite(rcut) && rcut > 0) || throw(DomainError(rcut,
            "Cutoff radius `rcut` must be positive and finite (a cartesian distance, in the " *
            "units of the lattice vectors)."))
        return new(Float64(rcut), from, to)
    end
end

"""
    DeltaGSelector(dG, sites_from, sites_to)

Plane-wave coupling `|G⟩ → |G + ΔG⟩`, found through the exact-integer `by_G` table.

Notes:
------
For a bond term, a ΔG coupling is a degenerate bond: in a plane-wave basis every state sits at one
site, and `⟨G'|T|G⟩` is k-independent, so this selector emits ordinary bonds with `ΔR = 0` and unit
phase, and continuum models share the compiled bond list with tight-binding ones.

For a k-dependent term it is the *non-local* coupling of continuum models -- a momentum-dependent
tunnelling `w(k)|G+ΔG⟩⟨G|` -- and the entry's momentum offset uses the midpoint convention; see [`emit_term!`](@ref).
"""
struct DeltaGSelector <: AbstractBondSelector
    dG::Vector{Int}
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}
end

"""
    ExplicitBondSelector(i, j, dR)

A hand-written or imported list of `(i, j, ΔR)` triples. The escape hatch, and the natural entry
point for a Wannier90 `_hr.dat` reader.
"""
struct ExplicitBondSelector <: AbstractBondSelector
    i::Vector{Int}
    j::Vector{Int}
    dR::Vector{Vector{Int}}

    function ExplicitBondSelector(i::Vector{Int}, j::Vector{Int}, dR::Vector{Vector{Int}})
        (length(i) == length(j) == length(dR)) || throw(DimensionMismatch(
            "ExplicitBondSelector arrays must have equal length; got $(length(i)), $(length(j)), $(length(dR))."
        ))
        if !isempty(dR)
            d1 = length(first(dR))
            all(r -> length(r) == d1, dR) || throw(DimensionMismatch(
                "ExplicitBondSelector translations must all have the same length."
            ))
        end
        return new(i, j, dR)
    end
end

"""
    max_shell(term_or_selector) -> Int

The deepest distance shell a term needs from the neighbour table, `0` when it needs none --
consumed by `compile`'s neighbour-table decision (only `ShellSelector` returns non-zero; its
metric analog is [`max_radius`](@ref)).
"""
max_shell(::AbstractBondSelector) = 0
max_shell(s::ShellSelector) = s.shell

"""
    max_radius(term_or_selector) -> Float64

The cartesian neighbour-search radius a term needs, `0.0` when it needs none -- the metric
analog of [`max_shell`](@ref), consumed by `compile`'s neighbour-table decision. Only
`RadiusSelector` (the `hopping(t; rcut=...)` spelling) returns a non-zero value.
"""
max_radius(::AbstractBondSelector)::Float64 = 0.0
max_radius(s::RadiusSelector)::Float64 = s.rcut

"""
    has_intrinsic_direction(selector) -> Bool

Whether the selector defines its own source→target orientation.

`ShellSelector` and `OnSiteSelector` do not: they enumerate unordered geometric pairs, so which
endpoint counts as "source" is fixed by basis ordering, not by physics. A complex amplitude on such
a selector is almost always a modelling error -- the phase should come from a direction-
antisymmetric weight instead.

`DeltaGSelector`, `TranslationSelector` and `ExplicitBondSelector` do: `|G⟩ → |G+ΔG⟩`, a named
translation, and a hand-written triple are all oriented by construction. A complex amplitude there
is normal and correct -- the moiré tunnelling amplitudes `w₁e^{±2πi/3}` are exactly this.
"""
has_intrinsic_direction(::AbstractBondSelector) = true
has_intrinsic_direction(::ShellSelector) = false
has_intrinsic_direction(::RadiusSelector) = false
has_intrinsic_direction(::OnSiteSelector) = false




#############################################################
###### Dimension materialization                        #####
#############################################################

### `what` may be a string OR a thunk returning one. `_emit!(::ExplicitBondSelector)` calls
### this once PER BOND, and building the `"dR[$e]"` label eagerly cost 1.02 ms of a 6.96 ms
### compile and 45% of its allocations on a 4060-bond term -- for a message that is almost
### never raised. The cost is linear in bond count, so a Wannier90-scale import (10^5-10^6
### bonds) pays 25-250 ms and millions of allocations for nothing. Same text, deferred.
"""
    _as_svec_int(v, D, what, name) -> SVector{D,Int}

Materialize a stored integer vector against the cell's dimension, with an error that names the
term. This is where a dimension-free term meets a concrete cell.
"""
_what_string(what::AbstractString) = what
_what_string(what) = what()

function _as_svec_int(v::Vector{Int}, D::Int, what, name::AbstractString)
    length(v) == D || throw(DimensionMismatch(
        "Term `$name`: `$(_what_string(what))` has length $(length(v)) but the cell is " *
        "$D-dimensional."
    ))
    return SVector{D,Int}(v)
end

"""
    _materialize_k0(k0, D, name) -> SVector{D,Float64}

An empty stored `k0` means the origin, whatever the dimension.
"""
function _materialize_k0(k0::Vector{Float64}, D::Int, name::AbstractString)
    isempty(k0) && return zero(SVector{D,Float64})
    length(k0) == D || throw(DimensionMismatch(
        "Term `$name`: `k0` has length $(length(k0)) but the cell is $D-dimensional."
    ))
    return SVector{D,Float64}(k0)
end

# A stored linear coefficient is either a full vector or `direction => value` (what `dirac` needs
# before the dimension is known).
function _materialize_c(c::Vector{Float64}, D::Int, name::AbstractString)
    length(c) == D || throw(DimensionMismatch(
        "Term `$name`: `c` has length $(length(c)) but the cell is $D-dimensional."
    ))
    return SVector{D,Float64}(c)
end
function _materialize_c(c::Pair{Int,Float64}, D::Int, name::AbstractString)
    1 <= c.first <= D || throw(DimensionMismatch(
        "Term `$name`: direction $(c.first) does not exist in a $D-dimensional cell."
    ))
    return SVector{D,Float64}(ntuple(m -> m == c.first ? c.second : 0.0, D))
end

# A stored quadratic form is either a full matrix or a scalar λ meaning λ·I -- the isotropic mass,
# expressible before the dimension is known.
function _materialize_M(M::Matrix{Float64}, D::Int, name::AbstractString)
    size(M) == (D, D) || throw(DimensionMismatch(
        "Term `$name`: `M` is $(size(M)) but the cell is $D-dimensional."
    ))
    return SMatrix{D,D,Float64}(M)
end
_materialize_M(lam::Float64, D::Int, ::AbstractString) = lam * one(SMatrix{D,D,Float64})

# Site filters accept a single name or a list; empty list means "any".
_site_list(s::Symbol) = Symbol[s]
_site_list(v::AbstractVector{Symbol}) = Vector{Symbol}(v)




#############################################################
###### Per-bond geometric weights                       #####
#############################################################

"""
    AbstractBondWeight

A geometric factor evaluated **once per bond, at compile time** -- never per k-point.

The contract, for a weight defined outside the package:
-------------------------------------------------------
- [`bond_weight`](@ref) (required): the factor of one site-level bond;
- [`is_symmetric_weight`](@ref) (optional, default `true`): whether reversing the bond leaves
  the factor unchanged. A direction-carrying weight (a Peierls phase, a chirality) adds a
  method returning `false` and must satisfy the reversal contract spelled out there;
- [`validate_weight`](@ref) (optional): a compile-time refusal for cells on which the weight
  cannot mean anything.

Shipped subtypes: [`UnitWeight`](@ref) (the default everywhere), [`ExponentialDecay`](@ref),
[`Chirality`](@ref), [`ChiralityPhase`](@ref), and [`PeierlsPhase`](@ref) -- built by
[`uniform_flux`](@ref) for a uniform orbital field, paired with [`magnetic_cell`](@ref).
"""
abstract type AbstractBondWeight end

"""
    UnitWeight()

No geometric modulation; factor `1`.
"""
struct UnitWeight <: AbstractBondWeight end

"""
    Chirality()

The Haldane/Kane-Mele chirality `ν_ij = ±1` of a next-nearest-neighbour bond: the sign of the cross
product of the two nearest-neighbour hops taken through the shared intermediate site.

Notes:
------
Defined only in `D = 2`, and only when the two endpoints have **exactly one** common nearest
neighbour. On a square lattice a second-shell bond has two common neighbours and the chirality is
ambiguous -- that case raises rather than picking one silently.

`ν` is antisymmetric under bond reversal, which is what makes `im*λ ⊗ Chirality()` Hermitian for
free: the partner amplitude `conj(iλν) = iλ(-ν)` is exactly the physical value of the reversed bond.
"""
struct Chirality <: AbstractBondWeight end

"""
    ChiralityPhase(phi)

`exp(i φ ν_ij)` with `ν_ij` the chirality above -- the Haldane phase factor.

Sign anchor, so the resulting topology is predictable without an experiment: on the honeycomb
lattice, `hopping(t2; shell=2, weight=ChiralityPhase(φ))` gives the lower Haldane band the
Chern number `C = -sign(sin φ)` in the package's cartesian (right-handed) flux orientation
(the one `berry_curvature`/`chern_number` report).
"""
struct ChiralityPhase <: AbstractBondWeight
    phi::Float64
end

"""
    ExponentialDecay(lambda, d0)

`exp(-(d - d0)/λ)` with `d` the cartesian bond length. The common Slater-Koster-like envelope.
"""
struct ExponentialDecay <: AbstractBondWeight
    lambda::Float64
    d0::Float64

    function ExponentialDecay(lambda::Real, d0::Real=0.0)
        lambda > 0 || throw(DomainError(lambda, "Decay length must be positive."))
        return new(Float64(lambda), Float64(d0))
    end
end

"""
    is_symmetric_weight(w::AbstractBondWeight) -> Bool

Whether the weight's factor is unchanged under bond reversal -- the direction declaration of
the weight contract, beside [`bond_weight`](@ref) and [`validate_weight`](@ref).

- `true` (the generic default): the factor of the reversed bond `(sj, si, -ΔR)` equals the
  factor of `(si, sj, ΔR)`. The emitters evaluate `bond_weight` once per geometric bond and
  REUSE that value for the reversed orientation.
- `false`: reversing the bond changes the factor, and the emitters call
  `bond_weight(w, ctx, sj, si, -ΔR, dist)` wherever the reversed element is needed: a ladder
  rule that accepts only the reversed orientation emits the reversed-bond element with the
  reversed weight, and a directional rule's per-bond Hermiticity check judges the full product
  `t·M(-d)·w(-d)` against `conj(t·M(d)·w(d))`. The construction-time warning about a complex
  constant amplitude on an unoriented selector is also suppressed -- an antisymmetric weight
  is exactly what gives such a phase a well-defined direction.

What `false` must mean, exactly:
--------------------------------
The compiled list stores each bond ONCE and `finalize_couplings` supplies `conj(entry)` as the
Hermitian partner, so the term's full per-bond product must satisfy

    t · w(reversed bond) == conj(t · w(bond))

for that supplied partner to BE the physical reversed-bond element. Two shapes are supported:
a weight **antisymmetric in phase**, `w(-d) = conj(w(d))`, under a real amplitude -- what
makes Hermiticity free for [`PeierlsPhase`](@ref) and [`ChiralityPhase`](@ref) -- and the real
sign flip `w(-d) = -w(d)` under a purely imaginary amplitude ([`Chirality`](@ref) with
`im*λ`). A weight/amplitude product that is neither symmetric nor conjugate-antisymmetric is
unsupported: only one orientation is ever evaluated and stored, so the supplied partner would
silently disagree with the reversed bond's physical value -- and the result stays perfectly
Hermitian, so nothing downstream can catch it.
"""
is_symmetric_weight(::AbstractBondWeight) = true
is_symmetric_weight(::Chirality) = false
is_symmetric_weight(::ChiralityPhase) = false




#############################################################
###### Compile context                                  #####
#############################################################

"""
    CompileContext{D}

Everything a term needs while emitting, assembled once per compilation.

`aT` and `bT` are cached as `SMatrix` locals rather than re-derived from the lattice inside the
emit loops: the transposes are used once per candidate translation, and going through
`fractional_to_real` there would add a function call in the tightest part of the search.
"""
struct CompileContext{D,K<:KPeriodicity}
    cell::UnitCell{D,K}
    tables::CellTables{D}
    neighbors::Union{Nothing,NeighborTable{D}}
    aT::SMatrix{D,D,Float64}
    bT::SMatrix{D,D,Float64}
    ### On a FOLDED cell (localized sites with a shared period M_s ≠ I) the neighbour table is
    ### built on the FINE (parent) lattice, and its integer fine translations convert to model
    ### coordinates through the exact rational M⁻ᵀ stored here; `nothing` on an ordinary cell.
    fold_MinvT::Union{Nothing,SMatrix{D,D,Rational{Int}}}
    ### the companion per-site fine-lattice wrap shifts (see `_fold_fine_wrapshifts`), computed
    ### ONCE per compilation instead of once per shell/radius term
    fold_wrapshifts::Union{Nothing,Vector{SVector{D,Int}}}
end

### the class as a TYPE, for `EmitSink`, so its `ctx` field stays concrete
_ctx_kpar(::CompileContext{D,K}) where {D,K} = K

function CompileContext(cell::UnitCell{D,K};
                        tables::CellTables{D}=cell_tables(cell),
                        neighbors::Union{Nothing,NeighborTable{D}}=nothing) where {D,K}
    MinvT = _fold_MinvT(cell)
    return CompileContext{D,K}(cell, tables, neighbors,
                             SMatrix{D,D,Float64}(transpose(cell.lattice.a)),
                             SMatrix{D,D,Float64}(transpose(cell.lattice.b)),
                             MinvT,
                             isnothing(MinvT) ? nothing : _fold_fine_wrapshifts(cell))
end

"""
    _fold_MinvT(cell) -> Union{Nothing,SMatrix{D,D,Rational{Int}}}

The exact rational `M⁻ᵀ` of a folded cell's (cell-uniform) period matrix, or `nothing` when the
cell is not folded. `M⁻¹ = adj(M)/det(M)` keeps the arithmetic exact.
"""
function _fold_MinvT(cell::UnitCell{D}) where {D}
    is_folded(cell) || return nothing
    s = findfirst(k -> cell.site_kind[k] === :localized &&
                       abs(det(cell.site_period[k])) != 1, eachindex(cell.site_kind))
    return _int_invT(SMatrix{D,D,Int}(cell.site_period[s]))
end

"""
    _require_neighbors(ctx, what) -> NeighborTable

Fetch the neighbour table, raising an informative error when the term needed one but compilation
was not told to build it.
"""
function _require_neighbors(ctx::CompileContext{D}, what::AbstractString)::NeighborTable{D} where {D}
    isnothing(ctx.neighbors) && throw(ArgumentError(
        "$what requires a neighbour table, but none was built. This is an internal inconsistency: " *
        "`compile` should have scanned the terms with `max_shell` before assembling the context."
    ))
    return ctx.neighbors
end




#############################################################
###### Weight evaluation                                #####
#############################################################

"""
    bond_weight(w::AbstractBondWeight, ctx::CompileContext, si, sj, dR, dist) -> ComplexF64

Evaluate a geometric weight for one site-level bond -- **once per bond, at compile time**.
This is the public override point of the weight family: a new `AbstractBondWeight` subtype
defines a `bond_weight` method (and, when reversing the bond changes its value, an
[`is_symmetric_weight`](@ref) method returning `false` -- see there for the exact reversal
contract the factor must then satisfy, and for how the emitters consume it).

Args:
-----
- `ctx`: the compile context (cell, tables, cached `aT`/`bT`)
- `si`, `sj`: TARGET and SOURCE site ids -- the bond hops from site `sj` in the cell at `dR`
  to site `si` in the home cell, the package's global bond convention
- `dR`: the integer Bravais translation of the source cell (exact rational on a folded cell)
- `dist`: the cartesian bond length

The optional companion hook is [`validate_weight`](@ref), called once per term at compile.
"""
@inline bond_weight(::UnitWeight, ::CompileContext, ::Int, ::Int, ::SVector, ::Float64) = one(ComplexF64)

bond_weight(w::ExponentialDecay, ::CompileContext, ::Int, ::Int, ::SVector, dist::Float64) =
    ComplexF64(exp(-(dist - w.d0) / w.lambda))

function bond_weight(::Chirality, ctx::CompileContext{D}, si::Int, sj::Int,
                     dR::SVector{D,Int}, ::Float64)::ComplexF64 where {D}
    return ComplexF64(_chirality(ctx, si, sj, dR))
end

function bond_weight(w::ChiralityPhase, ctx::CompileContext{D}, si::Int, sj::Int,
                     dR::SVector{D,Int}, ::Float64)::ComplexF64 where {D}
    return cis(w.phi * _chirality(ctx, si, sj, dR))
end

### a rational translation means a folded cell, where the chirality's common-neighbour search
### (which reads the fine-lattice table against model positions) is not defined yet
bond_weight(::Union{Chirality,ChiralityPhase}, ::CompileContext{D}, ::Int, ::Int,
            ::SVector{D,Rational{Int}}, ::Float64) where {D} = throw(ArgumentError(
    "`Chirality`/`ChiralityPhase` weights are not supported on a folded cell; compile the " *
    "parent model and `fold` it instead."))

### the internal spelling the emitters call; a one-line forwarder, so the public `bond_weight`
### is the single override point for every weight type
@inline _bond_weight(w::AbstractBondWeight, ctx::CompileContext, si::Int, sj::Int,
                     dR::SVector, dist::Float64) = bond_weight(w, ctx, si, sj, dR, dist)

"""
    validate_weight(w::AbstractBondWeight, ctx::CompileContext, name::String) -> Nothing

Refuse, at compile time, a weight that cannot mean anything on this cell -- called **once per
term** by `emit_term!`, beside the selector validation. The default is a no-op; a weight type
adds a method when it has a cell-level contract to enforce (the shipped example is
`PeierlsPhase`, whose gauge must be translation-commensurate on the compiled cell). Refusals
should name the term (`name`), the reason, and the fix, in the style of the neighbouring
selector refusals.
"""
validate_weight(::AbstractBondWeight, ::CompileContext, ::String)::Nothing = nothing

"""
    _chirality(ctx, si, sj, dR) -> Float64

Sign of the cross product of the two nearest-neighbour hops joining site `si` (home cell) to site
`sj` in cell `dR` through their unique common nearest neighbour.

Raises when the dimension is not 2, or when the number of common nearest neighbours is not exactly
one -- both cases mean the caller asked for a chirality that is not defined.
"""
function _chirality(ctx::CompileContext{D}, si::Int, sj::Int, dR::SVector{D,Int})::Float64 where {D}
    D == 2 || throw(ArgumentError(
        "`Chirality` is defined only in D = 2 (it is the sign of a 2D cross product); got D = $D."
    ))
    nt = _require_neighbors(ctx, "`Chirality`")
    isempty(nt.radii) && throw(ArgumentError("`Chirality` needs a nearest-neighbour shell, but the neighbour table has none."))

    t = ctx.tables
    r_i = ctx.aT * t.site_positions[si]
    r_j = ctx.aT * (SVector{D,Float64}(dR) + t.site_positions[sj])

    ### The common nearest neighbours are read off the neighbour TABLE, which already holds
    ### every shell-1 half-bond: the neighbours of `si` intersected with those of `sj` shifted by
    ### `dR`. This replaces a per-bond rescan of every site × candidate translation -- an
    ### O(n_sites · box) search whose cost multiplied by nbonds × internal pairs on large cells --
    ### with an O(z²) set intersection at coordination z.
    nnj = Set{Tuple{Int,SVector{D,Int}}}()
    for e in shell_entries(nt, 1)
        nt.site_i[e] == sj && push!(nnj, (nt.site_j[e], dR + nt.dR[e]))
        nt.site_j[e] == sj && push!(nnj, (nt.site_i[e], dR - nt.dR[e]))
    end
    commons = SVector{D,Float64}[]
    for e in shell_entries(nt, 1)
        nt.site_i[e] == si && (nt.site_j[e], nt.dR[e]) in nnj &&
            push!(commons, ctx.aT * (SVector{D,Float64}(nt.dR[e]) + t.site_positions[nt.site_j[e]]))
        nt.site_j[e] == si && (nt.site_i[e], -nt.dR[e]) in nnj &&
            push!(commons, ctx.aT * (SVector{D,Float64}(-nt.dR[e]) + t.site_positions[nt.site_i[e]]))
    end

    length(commons) == 1 || throw(ArgumentError(
        "`Chirality` requires exactly one common nearest neighbour between the two endpoints of a " *
        "bond, found $(length(commons)) for sites ($si, $sj) at ΔR = $(Tuple(dR)). " *
        "On a square lattice, second-shell bonds have two common neighbours and the chirality is " *
        "ambiguous; select those bonds with an explicit `TranslationSelector` instead."
    ))

    d1 = commons[1] - r_i
    d2 = r_j - commons[1]
    cross2 = d1[1] * d2[2] - d1[2] * d2[1]
    abs(cross2) > LATTICE_ATOL || throw(ArgumentError(
        "`Chirality` is degenerate: the two hops for sites ($si, $sj) at ΔR = $(Tuple(dR)) are collinear."
    ))
    return sign(cross2)
end




#############################################################
###### Bond terms                                       #####
#############################################################

"""
    HoppingTerm

A Hamiltonian term with a k-independent amplitude: on-site energies, tight-binding hoppings,
continuum ΔG couplings, imported bond lists. All of them.

Fields:
-------
- `selector`: which `(i, j, ΔR)` triples
- `rule`: the quantum-number selection rule and factor
- `weight`: the per-bond geometric factor
- `t`: the overall amplitude -- a number, or a callback `(bs_i, bs_j, ΔR) -> Number`
- `name`: label used in diagnostics

Notes:
------
A callback amplitude is a **compile-time builder**: it runs once while the term lowers into the
bond list and leaves nothing behind, so evaluation and HDF5 round-tripping are unaffected. What it
does cost is symbolic re-derivation from the provenance record, and `is_serializable` reports that
so the loss is announced rather than discovered.
"""
struct HoppingTerm <: BondTerm
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    weight::AbstractBondWeight
    t::Union{ComplexF64,Function}
    name::String

    function HoppingTerm(selector::AbstractBondSelector, rule::AbstractQnumRule,
                         weight::AbstractBondWeight, t::Union{ComplexF64,Function},
                         name::String)
        if t isa ComplexF64 && !iszero(imag(t)) &&
           is_symmetric_weight(weight) && !has_intrinsic_direction(selector)
            @warn "HoppingTerm `$name` has a complex amplitude ($t) on an unoriented selector " *
                  "($(typeof(selector))) with a direction-symmetric weight $(typeof(weight)). " *
                  "The phase would be applied in whichever direction the basis ordering makes " *
                  "canonical, which is not physics. Use a direction-antisymmetric weight " *
                  "(e.g. `Chirality()`) for a directed phase."
        end
        return new(selector, rule, weight, t, name)
    end
end

max_shell(term::HoppingTerm) = max_shell(term.selector)
max_radius(term::HoppingTerm)::Float64 = max_radius(term.selector)
max_radius(term::KDependentTerm)::Float64 = max_radius(_kselector(term))
max_radius(::AbstractHamiltonianTerm)::Float64 = 0.0
needs_neighbors(term::AbstractHamiltonianTerm) = max_shell(term) > 0 || max_radius(term) > 0
"""
    is_serializable(term) -> Bool

Whether the TERM itself could be re-derived symbolically from a file -- not whether a model
carrying it can be saved.

A callback amplitude, a [`directional_rule`](@ref), or a raw-closure [`k_envelope`](@ref) all
lower to exact numbers at compile, so the compiled MODEL round-trips through [`h5_write`](@ref)
whenever every k-kernel carries a registered tag (see [`kernel_tag`](@ref) and
[`register_kernel!`](@ref) -- a `k_envelope` with a registered [`AbstractEnvelope`](@ref)
payload reports `true` here, a raw closure reports `false`). What a `false` means in practice:
the term's symbolic form is gone from the provenance record, so a reloaded model cannot be
`reweight`-ed by that term's recorded amplitude alone and must be recompiled from source to
change ITS parameters.
"""
is_serializable(term::HoppingTerm) =
    !(term.t isa Function) && isempty(_split_directional(term.rule)[2])
term_name(term::AbstractHamiltonianTerm) = term.name

"""
    _has_zero_amplitude(term) -> Bool

Whether a term carries a literal zero amplitude, and so is expected to emit nothing.

Used to suppress the "emitted no bonds" diagnostic: sweeping a coupling through zero is routine,
whereas a term that matched nothing structurally is a modelling bug.
"""
_has_zero_amplitude(term::HoppingTerm) = (term.t isa ComplexF64) && iszero(term.t)
_has_zero_amplitude(term::KDependentTerm) = iszero(_kcoefficient(term))
_has_zero_amplitude(::AbstractHamiltonianTerm) = false

"""
    _has_zero_rule(term, schema) -> Bool

Whether the term's quantum-number rule compiles to an identically zero matrix -- the OTHER
legitimate way a term emits nothing (e.g. `matrix_rule(:layer, diagm(u))` with `u .== 0` at a
parameter-sweep endpoint). Distinguished from a structural mismatch by the same logic as
`_has_zero_amplitude`: a zero the user wrote is routine, a rule that matched no pairs is a bug.
"""
function _has_zero_rule(term::AbstractHamiltonianTerm, schema)
    static, directional = _split_directional(_term_rule(term))
    ### a directional rule resolves per bond, so whether it vanishes cannot be known
    ### statically -- the empty-term warning must never be suppressed for it
    isempty(directional) || return false
    return iszero(qnum_matrix(static, schema))
end

"""
    _amplitude(t, bs_i, bs_j, dR) -> ComplexF64

Resolve a term's amplitude, calling the builder when one was supplied.
"""
@inline _amplitude(t::ComplexF64, ::BasisState, ::BasisState, ::SVector) = t
@inline _amplitude(f::Function, bs_i::BasisState, bs_j::BasisState, dR::SVector) =
    ComplexF64(f(bs_i, bs_j, dR))

"""
    _checked_amplitude(t, bs_i, bs_j, dR, name) -> ComplexF64

Resolve a term's amplitude on an element whose Hermitian partner will be **supplied** rather
than enumerated, verifying a callback in both orientations.

On such elements only one direction is ever evaluated and `finalize_bonds` conjugates it, so a
callback that is not conjugate-symmetric would be Hermitized silently -- precisely what the
package refuses to do for rules (`reverse ≠ conj(forward)` raises). Constants skip the check:
a complex constant on an unoriented selector already gets the construction-time warning, and on
an oriented one it is the intended one-directional matrix element. Callbacks on **oriented**
selectors (`DeltaGSelector`, `TranslationSelector` at ΔR ≠ 0, `ExplicitBondSelector`) are
likewise trusted one-directionally and do not come through here.
"""
@inline _checked_amplitude(t::ComplexF64, ::BasisState, ::BasisState, ::SVector, ::String) = t
function _checked_amplitude(f::Function, bs_i::BasisState, bs_j::BasisState,
                            dR::SVector, name::String)::ComplexF64
    fwd = ComplexF64(f(bs_i, bs_j, dR))
    rev = ComplexF64(f(bs_j, bs_i, -dR))
    ### a non-finite value falls through to the finalize funnel, whose DomainError names the
    ### bond and the NaN -- this check is about symmetry, not finiteness
    (isfinite(fwd) && isfinite(rev)) || return fwd
    isapprox(rev, conj(fwd); atol=AMPLITUDE_ATOL) || throw(ArgumentError(
        "Term `$name`: its callback amplitude is not conjugate-symmetric on a pair whose " *
        "Hermitian partner is supplied automatically: forward $fwd, reverse $rev ≠ " *
        "conj(forward). Only one orientation of this element is stored, so a non-symmetric " *
        "callback would be silently Hermitized. Make the callback satisfy " *
        "t(j, i, -ΔR) = conj(t(i, j, ΔR)), or select the bonds with an oriented selector " *
        "(`translation_hopping`, `delta_g`, `explicit_bonds`), where one-directional values " *
        "are the contract."
    ))
    return fwd
end

"""
    _diagonal_checked_amplitude(t, bs, dR, name) -> ComplexF64

Resolve a term's amplitude on a self-partnered (i == j, ΔR = 0) element, verifying that a
callback is DETERMINISTIC.

The conjugate-symmetry check of `_checked_amplitude` degenerates to a tautology on a
self-partnered element (forward and reverse are the same call), so a callback drawing fresh
randomness per call used to compile silently -- into a realization no recompile can reproduce
and any transform re-emitting the term silently reshuffles. Disorder belongs in a pre-drawn
table the callback reads (the worked recipe is
`examples/hamiltonianconstruction/04_disorder_and_defects.jl`); two draws that disagree are
refused by name.
"""
@inline _diagonal_checked_amplitude(t::ComplexF64, ::BasisState, ::SVector, ::String) = t
function _diagonal_checked_amplitude(f::Function, bs::BasisState, dR::SVector,
                                     name::String)::ComplexF64
    a = ComplexF64(f(bs, bs, dR))
    b = ComplexF64(f(bs, bs, dR))
    ### finiteness stays the finalize funnel's job -- this check is about determinism
    (isfinite(a) && isfinite(b)) || return a
    isapprox(a, b; atol=AMPLITUDE_ATOL) || throw(ArgumentError(
        "Term `$name`: its callback amplitude is not deterministic -- two evaluations on the " *
        "same on-site element returned $a and then $b. A per-call random draw compiles into " *
        "an irreproducible disorder realization; draw the realization into a table first and " *
        "let the callback read it (the worked recipe is " *
        "examples/hamiltonianconstruction/04_disorder_and_defects.jl)."
    ))
    return a
end

"""
    _site_ok(allowed, site) -> Bool

Whether a site name passes a filter list; an empty list means "any".

Notes:
------
Selection by *site* is what replaced selection by orbital index. The two were conflated before:
`orbitals=[1]` on graphene meant "sublattice A", which is a position, while on a multi-orbital atom
it would have meant an envelope character. Those are now separate handles -- `sites=[:A]` here, and
`⊗ project(:orbital, :dxy)` on the rule side.
"""
@inline _site_ok(allowed::Vector{Symbol}, site::Symbol)::Bool = isempty(allowed) || (site in allowed)

"""
    _validate_selector(sel, ctx, name) -> Nothing

Refuse, at compile time, a selector that names something the cell does not have.

Notes:
------
This closes the same silent-failure path the rule layer already closes for axes and levels
(`_rule_axis`, `_check_rule_levels`): a mistyped site name (`sites=[:AA]`) used to match nothing
and make the term emit **no bonds at all**, downgraded to the generic empty-term `@warn`. A name
is either valid or an error -- the policy must not depend on which layer the typo landed in.
`ExplicitBondSelector` gets the analogous check on its state indices, so an out-of-range triple
raises here with the term's name instead of surfacing as a bare `BoundsError` mid-emission.
"""
function _validate_selector(sel::AbstractBondSelector, ctx::CompileContext, name::String)::Nothing
    for (field, sites) in _selector_site_lists(sel)
        for s in sites
            s in ctx.tables.site_names || throw(ArgumentError(
                "Term `$name` selects $field = :$s, but the cell declares no such site (its " *
                "sites are $(ctx.tables.site_names)). No pair would match, so the term would " *
                "silently emit nothing."
            ))
        end
    end
    sel isa TranslationSelector && _check_dR_periodic(sel.dR, ctx, name)
    sel isa DeltaGSelector && _check_dR_periodic(sel.dG, ctx, name)
    return nothing
end

function _validate_selector(sel::ExplicitBondSelector, ctx::CompileContext, name::String)::Nothing
    n = n_states(ctx.cell)
    for e in eachindex(sel.i)
        (1 <= sel.i[e] <= n && 1 <= sel.j[e] <= n) || throw(ArgumentError(
            "Term `$name` names the explicit bond ($(sel.i[e]), $(sel.j[e]), " *
            "$(Tuple(sel.dR[e]))), but the cell has only $n states."
        ))
        _check_dR_periodic(sel.dR[e], ctx, name)
    end
    return nothing
end

"""
    _check_dR_periodic(v, ctx, name) -> Nothing

Refuse a user-supplied integer translation (`dR=`) or reciprocal transfer (`dG=`) with a
non-zero component along an aperiodic direction of a partially periodic lattice.
"""
function _check_dR_periodic(v::AbstractVector{<:Integer}, ctx::CompileContext,
                            name::String)::Nothing
    P = nperiodic(ctx.cell.lattice)
    D = ndim(ctx.cell.lattice)
    P == D && return nothing
    for m in (P+1):min(D, length(v))
        v[m] == 0 || throw(ArgumentError(
            "Term `$name` carries the integer vector $(Tuple(v)), non-zero along direction " *
            "$m -- but the lattice is periodic only along its first $P direction" *
            "$(P == 1 ? "" : "s"), so there is no translation (or reciprocal vector) there."))
    end
    return nothing
end

_selector_site_lists(sel::OnSiteSelector) = (("sites", sel.sites),)
_selector_site_lists(sel::Union{TranslationSelector,ShellSelector,RadiusSelector,DeltaGSelector}) =
    (("sites_from", sel.sites_from), ("sites_to", sel.sites_to))

"""
    _candidate_sites(sel, tables) -> Vector{Int}

The site ids a selector can possibly involve, given its filters -- the domain the kind checks
run over.
"""
_candidate_sites(sel::OnSiteSelector, t::CellTables) =
    [s for s in 1:n_sites(t) if _site_ok(sel.sites, t.site_names[s])]
_candidate_sites(sel::Union{TranslationSelector,ShellSelector,RadiusSelector,DeltaGSelector},
                 t::CellTables) =
    [s for s in 1:n_sites(t)
     if _site_ok(sel.sites_from, t.site_names[s]) || _site_ok(sel.sites_to, t.site_names[s])]
_candidate_sites(sel::ExplicitBondSelector, t::CellTables) =
    unique(vcat([t.state_site[i] for i in sel.i], [t.state_site[j] for j in sel.j]))

"""
    _validate_kinds(term, sel, ctx) -> Nothing

Refuse, at compile time, a term whose kernel or selector is meaningless for the KIND of basis
function it addresses. The two rules (see the design notes and `site_kinds`):

- a **k-kernel on a `:localized` site** is an error -- "`p` on a comb": a Wannier-like orbital
  has no free-particle dispersion, its matrix elements are *given* as bonds. Without this
  check, `compile(graphene, [hopping(-1.0), kinetic()])` would add `|bᵀk|²/2m` to every
  orbital's diagonal without a word -- on a tight-binding cell every state carries `G = 0`, so
  the on-site k-pattern enumerator matches every orbital and nothing else the compiler checks
  can see the mistake. Refusing on the orbital KIND makes it a named error instead of a
  silently wrong model.
- a **ΔG selector on a `:localized` site** is an error: `ΔG` reshapes a plane-wave label, which
  a comb does not have. (On a *folded* cell -- `fold`, where localized sites legitimately carry
  coset labels -- same-site ΔG harmonics are the moiré-tight-binding coupling and are accepted.)

Real-space selectors on `:plane_wave` sites are deliberately allowed: `hopping` there is the
lattice-translation operator `Σ t·T_ΔR`, which is exactly consistent.
"""
function _validate_kinds(term::AbstractHamiltonianTerm, sel::AbstractBondSelector,
                         ctx::CompileContext)::Nothing
    t = ctx.tables
    if term isa KDependentTerm
        for s in _candidate_sites(sel, t)
            t.site_kind[s] === :plane_wave || throw(ArgumentError(
                "k-dependent term `$(term_name(term))` addresses site `$(t.site_names[s])`, " *
                "which is :localized -- a momentum kernel on a localized (Wannier-like) orbital " *
                "is meaningless (`p` on a comb): its matrix elements are given as bonds, not " *
                "computed from a dispersion. Use `hopping`/`onsite`/`explicit_bonds` there, or " *
                "restrict the term with `sites=` to the :plane_wave sites."))
        end
    elseif sel isa DeltaGSelector
        for s in _candidate_sites(sel, t)
            t.site_kind[s] === :plane_wave ||
                (t.site_kind[s] === :localized && abs(det(t.site_period[s])) != 1) ||
                throw(ArgumentError(
                    "Term `$(term_name(term))` uses a ΔG selector on site " *
                    "`$(t.site_names[s])`, which is :localized -- a real-space orbital carries " *
                    "no plane-wave label to shift. ΔG couplings apply to :plane_wave sites, or " *
                    "to the coset labels of a folded cell (`fold`). For tight-binding couplings " *
                    "use `onsite`/`hopping`/`translation_hopping`."))
        end
    end
    return nothing
end




### `K` rides along so `ctx` keeps a CONCRETE type. `CompileContext` gained its k-periodicity
### parameter on 2026-09-17, which turned a plain `CompileContext{D}` field here into a
### `UnionAll` -- boxed, and re-dispatched on every emitter call. Spelling it out is what the
### step-6 field audit was supposed to catch and did not.
"""
    EmitSink{D}

Everything ONE term needs while emitting into the unified coupling buffer: the compiled rule
matrix, the geometric weight, the amplitude, the kernel id, and the per-term momentum-origin
sugar. One sink per term; the emitters below are shared by **every kernel** -- this is the
single enumerator + single fan-out that replaces the former bond/k-term emitter families.
"""
struct EmitSink{D,K<:KPeriodicity}
    buf::CouplingBuffer{D}
    ctx::CompileContext{D,K}
    ### `Mrule` is the compiled STATIC part of the rule; for a term with no directional
    ### factors (the common case) it is the whole rule and the emitters read it directly.
    Mrule::Matrix{ComplexF64}
    ### the directional factors of the rule, resolved per site pair by `_pair_rule_matrix`
    ### (composed with `static_rule` BEFORE the unnamed-axis completion); empty = static term.
    static_rule::AbstractQnumRule
    ### Deliberately an ABSTRACT eltype, and not the container regression that parameterizing
    ### a type usually causes: one composed rule can carry directional factors with DIFFERENT
    ### level orders (a `TwoLevelOrder` on spin ⊗ a `NamedOrder` on orbital), so this vector is
    ### genuinely heterogeneous. Narrowing it would be wrong, not faster.
    directional::Vector{DirectionalQnum}
    weight::AbstractBondWeight
    amp::Union{ComplexF64,Function}
    kid::Int
    k0::SVector{D,Float64}
    name::String
end

@inline _is_directional(sink::EmitSink) = !isempty(sink.directional)

# The five-knob decomposition, read off any term: geometry, rule, weight, amplitude, kernel.
@inline _term_selector(t::HoppingTerm) = t.selector
@inline _term_selector(t::KDependentTerm) = _kselector(t)
@inline _term_rule(t::HoppingTerm) = t.rule
@inline _term_rule(t::KDependentTerm) = _krule(t)
@inline _term_weight(t::HoppingTerm) = t.weight
@inline _term_weight(::KDependentTerm) = UnitWeight()
@inline _term_amp(t::HoppingTerm) = t.t
@inline _term_amp(t::KDependentTerm) = _kcoefficient(t)
@inline _term_k0raw(::HoppingTerm) = Float64[]
@inline _term_k0raw(t::KDependentTerm) = _k0_raw(t)
_term_kernel(::HoppingTerm, ::Val{D}) where {D} = ConstantKernel()
_term_kernel(t::KDependentTerm, ::Val{D}) where {D} = _make_kernel(t, Val(D))

"""
    _check_kernel_periodic(kern, lattice, name) -> Nothing

On a partially periodic lattice, refuse a momentum kernel that responds along an aperiodic
direction: every legal momentum has zero weight there, so the response is IDENTICALLY dead --
a `k_linear([0, 1])` on a 1-periodic frame would compile into a silently-zero term, the exact
opposite of the package's loud-refusal policy. Probed numerically along each padding axis
(covers every kernel type, callbacks included); a kernel that ignores the aperiodic momenta
passes untouched.
"""
function _check_kernel_periodic(kern, lattice::BravaisLattice{D},
                                name::AbstractString)::Nothing where {D}
    kern isa ConstantKernel && return nothing
    P = nperiodic(lattice)
    P == D && return nothing
    ### the aperiodic momentum axes ARE the padding rows (orthonormal complement)
    for m in (P+1):D
        pad = primitive_vector(lattice, m)
        for t in (0.7, -1.3)
            dead = _kvalue(kern, SVector{D,Float64}(t .* pad)) - _kvalue(kern, zero(SVector{D,Float64}))
            abs(dead) <= AMPLITUDE_ATOL || throw(ArgumentError(
                "Term `$name` carries a momentum kernel that responds along aperiodic " *
                "direction $m of this partially periodic lattice (probe response $dead). " *
                "Momentum is identically zero there, so that part of the kernel would be a " *
                "silently dead term -- restrict the kernel to the periodic directions."))
        end
    end
    return nothing
end

"""
    _split_directional(rule) -> (static::AbstractQnumRule, directional::Vector{DirectionalQnum})

Partition a (possibly composed) rule into its static part and its bond-direction-dependent
factors. The static part compiles to one matrix exactly as before; the directional factors are
re-composed with it per site pair by `_pair_rule_matrix` -- composition happens BEFORE the
unnamed-axis completion, which is what keeps the Hadamard semantics of `compose` correct.
"""
function _split_directional(rule::AbstractQnumRule)
    rule isa DirectionalQnum && return (AnyQnum(), DirectionalQnum[rule])
    rule isa ProductQnumRule || return (rule, DirectionalQnum[])
    directional = DirectionalQnum[]
    static = AbstractQnumRule[]
    for r in rule.rules
        s, d = _split_directional(r)
        s isa AnyQnum || push!(static, s)
        append!(directional, d)
    end
    stat = isempty(static) ? AnyQnum() :
           length(static) == 1 ? static[1] : ProductQnumRule(static)
    return (stat, directional)
end

"""
    _validate_directional_selector(sel, dr, name) -> Nothing

Refuse, at compile time, a directional rule on a selector that gives its pairs no bond
direction: an on-site pair has none, and a ΔG coupling reshapes a plane-wave label at
`ΔR = 0` (a plane wave has no centre, so there is no displacement to point along).
"""
function _validate_directional_selector(sel::AbstractBondSelector, dr::DirectionalQnum,
                                        name::String)::Nothing
    sel isa OnSiteSelector && throw(ArgumentError(
        "Term `$name` carries a directional rule on axis `$(dr.key)`, but its on-site " *
        "selector pairs states on one site: a bond-direction-dependent matrix has no " *
        "direction on an on-site pair. Put the directional physics on a bond term " *
        "(`hopping`, `translation_hopping` at ΔR ≠ 0, `explicit_bonds`)."))
    sel isa DeltaGSelector && throw(ArgumentError(
        "Term `$name` carries a directional rule on axis `$(dr.key)`, but its ΔG selector " *
        "couples plane-wave labels at ΔR = 0 -- there is no bond direction to evaluate " *
        "`f(d)` at. Directional rules apply to real-space bond selectors."))
    return nothing
end

"""
    _probe_directional(dr, ::Val{D}, schema, name) -> Nothing

Spot-check a directional rule's callback before emission: `f` must return a square matrix of
the size its level mode implies, with finite entries, at a handful of non-collinear probe
directions. A constructor cannot validate a closure, and a wrong-size matrix (or a NaN from a
`0/0` in a hand-written direction cosine) should be a named error at compile, not a
`BoundsError` mid-emission or a non-finite band.
"""
function _probe_directional(dr::DirectionalQnum, ::Val{D}, s::InternalSchema,
                            name::AbstractString)::Nothing where {D}
    for (s1, s2) in ((1.0, 0.0), (-0.611, 0.788), (0.5, -0.207))
        d = SVector{D,Float64}(ntuple(m -> s1 * cospi(0.29m + 0.13) + s2 * sinpi(0.41m - 0.17), D))
        _directional_matrix(dr, d / norm(d), s, name)
    end
    return nothing
end

"""
    _directional_matrix(dr, d, schema, name) -> Matrix{ComplexF64}

Evaluate a directional rule's callback at one bond direction, refusing a wrong-size or
non-finite matrix with the term named. The expected size comes from the rule's level mode:
the axis's level count (`DeclaredOrder`), 2 (`TwoLevelOrder`), or the length of its label
list (`NamedOrder`).
"""
function _directional_matrix(dr::DirectionalQnum, d::SVector{D,Float64}, s::InternalSchema,
                             name::AbstractString)::Matrix{ComplexF64} where {D}
    Mraw = dr.f(d)
    Mraw isa AbstractMatrix || throw(ArgumentError(
        "Term `$name`: the directional rule on axis `$(dr.key)` returned a " *
        "$(typeof(Mraw)) at d = $(Tuple(d)); `f(d_cart)` must return an AbstractMatrix."))
    n = _order_rank(dr.order, dr, s)
    size(Mraw) == (n, n) || throw(DimensionMismatch(
        "Term `$name`: the directional rule on axis `$(dr.key)` returned a " *
        "$(size(Mraw, 1))×$(size(Mraw, 2)) matrix at d = $(Tuple(d)), but " *
        _order_rank_reason(dr.order, dr, s, n)))
    M = Matrix{ComplexF64}(Mraw)
    all(isfinite, M) || throw(DomainError(M,
        "Term `$name`: the directional rule on axis `$(dr.key)` returned a non-finite " *
        "entry at d = $(Tuple(d)). A hand-written direction cosine often divides by " *
        "`norm(d)` -- normalize once, inside `f`."))
    return M
end

### How many rows and columns the rule's matrix must have, and -- when it does not -- which
### half of the sentence names the requirement. One method per level order rather than a
### re-read of a tag; a fourth `LevelOrder` would be a MethodError here, by name, instead of
### silently taking the `:declared` branch of a chain of ternaries.
_order_rank(::DeclaredOrder, dr::DirectionalQnum, s::InternalSchema) =
    n_levels(s, _rule_axis(s, dr.key, dr))
_order_rank(::TwoLevelOrder, ::DirectionalQnum, ::InternalSchema) = 2
_order_rank(o::NamedOrder, ::DirectionalQnum, ::InternalSchema) = length(o.levels)

_order_rank_reason(::LevelOrder, dr::DirectionalQnum, s::InternalSchema, n::Int) =
    "the axis declares $n levels ($(levels(s, _rule_axis(s, dr.key, dr))))."
_order_rank_reason(o::NamedOrder, ::DirectionalQnum, ::InternalSchema, n::Int) =
    "its level labels $(o.levels) require $n×$n."

"""
    _directional_inner(dr, M, schema) -> AbstractQnumRule

Wrap one evaluated directional matrix as the static rule its level mode implies: a plain
declared-order `MatrixQnum`, a value-keyed `TwoLevelMatrixQnum`, or (for a `NamedOrder`) the
matrix permuted from the rule's label order into the axis's declared order.
"""
_directional_inner(dr::DirectionalQnum{DeclaredOrder}, M::Matrix{ComplexF64},
                   ::InternalSchema)::AbstractQnumRule = MatrixQnum(dr.key, M)

_directional_inner(dr::DirectionalQnum{TwoLevelOrder}, M::Matrix{ComplexF64},
                   ::InternalSchema)::AbstractQnumRule =
    TwoLevelMatrixQnum(dr.key, M, dr.order.up, dr.order.down)

### validated by `validate_rule`: every declared level appears in the rule's label order
function _directional_inner(dr::DirectionalQnum{NamedOrder}, M::Matrix{ComplexF64},
                            s::InternalSchema)::AbstractQnumRule
    a = _rule_axis(s, dr.key, dr)
    lv = s.levels[a]
    pos = [findfirst(l -> isequal(l, v), dr.order.levels)::Int for v in lv]
    return MatrixQnum(dr.key, ComplexF64[M[pos[r], pos[c]] for r in eachindex(lv),
                                                               c in eachindex(lv)])
end

"""
    _pair_rule_matrix(sink, si, sj, dR, dist) -> Matrix{ComplexF64}

The compiled rule matrix of ONE site pair. For a purely static rule this is `sink.Mrule` --
zero cost, today's path bit-identical. For a term with directional factors it evaluates each
`f` at the pair's cartesian bond direction, composes the results with the static part, and
completes the unnamed axes -- a compile-time cold path, run once per site pair.

Direction convention: `d` points FROM the source (site `sj`, cell `dR`) TO the target
(site `si`, home cell), `d = aᵀ(τ_i − τ_j − ΔR)` -- so reversing the bond flips `d` exactly,
and `f(d̂)` is the matrix of the stored element `⟨i, home | H | j, dR⟩`.
"""
function _pair_rule_matrix(sink::EmitSink{D}, si::Int, sj::Int,
                           dR::Union{SVector{D,Int},SVector{D,Rational{Int}}},
                           dist::Float64)::Matrix{ComplexF64} where {D}
    _is_directional(sink) || return sink.Mrule
    t = sink.ctx.tables
    d = sink.ctx.aT * (t.site_positions[si] - t.site_positions[sj] - SVector{D,Float64}(dR))
    nd = norm(d)
    nd > distance_atol(sink.ctx.cell.lattice) || throw(ArgumentError(
        "Term `$(sink.name)` carries a directional rule, but sites ($si, $sj) at " *
        "ΔR = $(Tuple(dR)) form a zero-length pair: a bond-direction-dependent matrix has " *
        "no direction on an on-site pair. Restrict the selector (site filters, or a " *
        "ΔR ≠ 0 translation) so only genuine bonds remain."))
    s = sink.ctx.cell.schema
    rule = sink.static_rule
    for dr in sink.directional
        rule = compose(rule, _directional_inner(dr, _directional_matrix(dr, d, s, sink.name), s))
    end
    return qnum_matrix(rule, s)
end

"""
    _throw_directional_asymmetry(name, i, j, dR, fwd, rev) -> Union{}

The directional conjugate-symmetry refusal, shared by the stored-once and undirected-bond
seams: on a pair whose Hermitian partner is supplied by `finalize_couplings`, the full product
must satisfy `amp·M(-d)[b, a] == conj(amp·M(d)[a, b])`, else the element would be silently
Hermitized -- exactly the policy of the static rule refusal and `_checked_amplitude`.
"""
_throw_directional_asymmetry(name::String, i::Int, j::Int, dR, fwd::ComplexF64,
                             rev::ComplexF64) = throw(ArgumentError(
    "Term `$name`: its directional rule is not conjugate-symmetric on the bond pair " *
    "(states $i, $j, ΔR = $(Tuple(dR))): forward element amp·M(d)[a,b] = $fwd, reverse " *
    "element amp·M(-d)[b,a] = $rev ≠ conj(forward). Only one orientation of this element " *
    "is stored and the partner is supplied automatically, so a non-symmetric directional " *
    "rule would be silently Hermitized. Make the full product satisfy " *
    "amp·f(-d)[b,a] = conj(amp·f(d)[a,b]) (for a real amplitude: f(-d) = f(d)†, e.g. an " *
    "odd anti-Hermitian f), or select the bonds with an oriented selector " *
    "(`translation_hopping`, `explicit_bonds`), where one-directional values are the contract."))

"""
    emit_term!(buf::CouplingBuffer, term, ctx) -> Nothing

Lower ONE term into the unified coupling buffer. Called once per term, during compilation.

The pipeline (see the design notes): validate the selector against the cell (names AND kinds),
compile the rule to its `n_internal × n_internal` matrix once, materialize the kernel and
register it in the buffer's kernel table, then fan the selector's geometric pairs out over the
basis states -- one enumerator per geometry, shared by every kernel. Each entry is pushed with
its momentum offset

    q = (G_i + G_j)/2 - (K_i + K_j)/2 - k0_term

(midpoint convention, symmetric under `i ↔ j`, which is what keeps Hermitization one rule for
every kernel).
"""
function emit_term!(buf::CouplingBuffer{D}, term::AbstractHamiltonianTerm,
                    ctx::CompileContext{D})::Nothing where {D}
    sel = _term_selector(term)
    # Validation runs even for zero-amplitude terms: a typo does not stop being a typo because
    # the coupling is currently swept through zero.
    _validate_selector(sel, ctx, term_name(term))
    _validate_kinds(term, sel, ctx)
    validate_weight(_term_weight(term), ctx, term_name(term))
    ### the static part compiles to ONE matrix, exactly as before; the directional factors are
    ### carried on the sink and resolved per site pair (`_pair_rule_matrix`)
    static, directional = _split_directional(_term_rule(term))
    for dr in directional
        validate_rule(dr, ctx.cell.schema)
        _validate_directional_selector(sel, dr, term_name(term))
        _probe_directional(dr, Val(D), ctx.cell.schema, term_name(term))
    end
    # A literally-zero amplitude emits nothing at all: keeping a pattern alive would charge the
    # full per-k cost to a term swept through zero, which is routine in a phase diagram.
    _has_zero_amplitude(term) && return nothing

    Mrule = qnum_matrix(static, ctx.cell.schema)
    kern = _term_kernel(term, Val(D))
    _check_kernel_periodic(kern, ctx.cell.lattice, term_name(term))
    kid = kernel_id!(buf, kern)
    k0 = _materialize_k0(_term_k0raw(term), D, term_name(term))
    if !iszero(k0) && any(!iszero, ctx.tables.k0)
        @warn "Term `$(term_name(term))` carries a per-term `k0` while the cell declares " *
              "per-state momentum origins; the two COMPOSE " *
              "(q = (G_i+G_j)/2 − (K_i+K_j)/2 − k0_term). Prefer origins on the cell -- the " *
              "per-term keyword is sugar for cells without them." maxlog = 1
    end
    sink = EmitSink{D,_ctx_kpar(ctx)}(buf, ctx, Mrule, static, directional, _term_weight(term),
                       _term_amp(term), kid, k0, term_name(term))
    _emit!(sink, sel)
    return nothing
end

"""
    _push_entry!(sink, i, j, dR, amp) -> Nothing

Push one coupling with its endpoint-derived momentum offset. Zero amplitudes are dropped here,
in one place.
"""
@inline function _push_entry!(sink::EmitSink{D}, i::Int, j::Int,
                              dR::Union{SVector{D,Int},SVector{D,Rational{Int}}},
                              amp::ComplexF64)::Nothing where {D}
    iszero(amp) && return nothing
    t = sink.ctx.tables
    q = _endpoint_q(t.G[i], t.G[j], t.k0[i], t.k0[j]) - sink.k0
    push_coupling!(sink.buf, i, j, dR, q, amp, sink.kid)
    return nothing
end


# ----- on-site: every state pair sharing a site, ΔR = 0 -----
function _emit!(sink::EmitSink{D}, sel::OnSiteSelector)::Nothing where {D}
    t = sink.ctx.tables
    zeroR = zero(SVector{D,Int})
    for s in 1:n_sites(t)
        _site_ok(sel.sites, t.site_names[s]) || continue
        w = _bond_weight(sink.weight, sink.ctx, s, s, zeroR, 0.0)
        _emit_unordered_pairs!(sink, s, s, w, w, true)
    end
    return nothing
end

"""
    _emit_unordered_pairs!(sink, si, sj, w, wrev, allow_rev, seen) -> Nothing

Emit ΔR = 0 couplings between two state groups, counting each unordered pair **once**.

Why this is separate from the ordered fan-out:
----------------------------------------------
At `ΔR = 0`, `(i,j,0)` and `(j,i,0)` are each other's Hermitian partner. Emitting both and then
letting `finalize_couplings` add partners would double every such matrix element. At `ΔR ≠ 0`
the situation is different -- the partner of `(i,j,ΔR)` is `(j,i,-ΔR)`, so `(i,j,ΔR)` and
`(j,i,ΔR)` are genuinely independent elements and both must be emitted.

The pair enumeration goes through the exact-integer `by_G` index rather than the full
`rows × cols` product: a real-space coupling is diagonal in the plane-wave label, and on a
moiré cell the full product is ~10⁶ tuples to keep the ~10³ on the G-diagonal. (`DeltaGSelector`
is the one selector that deliberately changes `G`, and it does not use this path.)

Args:
-----
- `si`, `sj`: the TARGET and SOURCE sites; the state pairs are drawn from their bases.
- `w`, `wrev`: the geometric weight in the enumerated orientation (targets on `si`, sources on
  `sj`) and in the reversed one. They differ only for direction-antisymmetric weights; callers
  with a symmetric weight pass the same value twice.
- `allow_rev`: whether the reversed orientation is admissible (see `_emit_hermitized_pair!`).
- `seen`: the dedup set. Passed in by the ΔR = 0 `TranslationSelector` path, whose site-pair loop
  can reach one unordered state pair from both orientations, so the dedup must span the calls.
- `Mfwd`/`Mrev`: the pair's rule matrices in the two orientations; `sink.Mrule` (the defaults)
  for a static rule, per-pair matrices for a directional one (see `_emit_hermitized_pair!`).
"""
function _emit_unordered_pairs!(sink::EmitSink{D},
                                si::Int, sj::Int,
                                w::ComplexF64, wrev::ComplexF64, allow_rev::Bool,
                                seen::Set{Tuple{Int,Int}}=Set{Tuple{Int,Int}}();
                                Mfwd::Matrix{ComplexF64}=sink.Mrule,
                                Mrev::Matrix{ComplexF64}=sink.Mrule)::Nothing where {D}
    t = sink.ctx.tables
    for i in t.site_states[si]
        ### the partners are `site_states[sj] ∩ by_G[G[i]]`, read straight off the (site, G)
        ### table instead of walking one factor and testing membership of the other -- either
        ### factor can be the entire basis (see `states_on_site_with_G`)
        for j in states_on_site_with_G(t, sj, t.G[i])
            key = i <= j ? (i, j) : (j, i)
            key in seen && continue
            push!(seen, key)
            _emit_hermitized_pair!(sink, i, j, w, wrev, allow_rev; Mfwd=Mfwd, Mrev=Mrev)
        end
    end
    return nothing
end

"""
    _emit_hermitized_pair!(sink, i, j, w, wrev, allow_rev) -> Nothing

Emit one **stored-once** pair: a ΔR = 0 coupling whose reverse `(j, i)` is its Hermitian
partner, so exactly one directed entry may enter the buffer, and `finalize_couplings` supplies
the conjugate. Shared by every kernel -- the k-dependent scalar being real (and `q` being
`i ↔ j` symmetric) is precisely what makes the one rule valid for all of them.

Choosing the direction:
-----------------------
A rule need only be non-zero in one direction. A ladder operator such as `raise(:layer)` couples
`layer=-1 → layer=+1` and nothing else, so whichever ordering the rule accepts is the one emitted
-- never whichever ordering the enumeration happened to present first, which is basis ordering,
not physics. When the rule specifies BOTH directions, the reverse entry is never read, and
silently discarding it is only sound if it equals conj(forward); a rule where it does not
(composable, so a user can build one) is refused rather than silently Hermitized. The same
policy applies to callback amplitudes, through `_checked_amplitude`.

`allow_rev`:
------------
Whether the reversed orientation `(j ← i)` is admissible at all. Directed site filters make it
false: `sites_from=[:A], sites_to=[:B]` admits only elements whose source sits on `:A`, so a rule
that accepts only the reverse orientation selects an element the filter excludes -- the pair is
then simply not coupled. With `allow_rev = false` the reverse is neither emitted nor held against
the rule in the consistency check, because the discarded direction was never part of the term.

`Mfwd`/`Mrev`:
--------------
The pair's rule matrices in the enumerated orientation and in the reversed one (see
`_pair_rule_matrix`). For a static rule both are `sink.Mrule` -- the defaults, today's path.
They differ only for a directional rule on a cross-site ΔR = 0 pair, where the reversed
orientation evaluates `f` at `-d`; the reverse factor `g` is then the genuine reversed-bond
element, and the check below becomes the full directional contract
`amp·M(-d)[b,a] == conj(amp·M(d)[a,b])` (amplitude included, since a directional matrix and a
phase-carrying amplitude can only be judged as a product).
"""
function _emit_hermitized_pair!(sink::EmitSink{D}, i::Int, j::Int,
                                w::ComplexF64, wrev::ComplexF64, allow_rev::Bool;
                                Mfwd::Matrix{ComplexF64}=sink.Mrule,
                                Mrev::Matrix{ComplexF64}=sink.Mrule)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    zeroR = zero(SVector{D,Int})
    f = @inbounds Mfwd[t.internal[i], t.internal[j]]
    g = @inbounds Mrev[t.internal[j], t.internal[i]]
    if _is_directional(sink)
        (iszero(f) && iszero(g)) && return nothing
        Afwd = iszero(f) ? zero(ComplexF64) :
               _amplitude(sink.amp, basis[i], basis[j], zeroR) * f * w
        Arev = iszero(g) ? zero(ComplexF64) :
               _amplitude(sink.amp, basis[j], basis[i], zeroR) * g * wrev
        if allow_rev && i != j && !isapprox(Arev, conj(Afwd); atol=AMPLITUDE_ATOL)
            _throw_directional_asymmetry(sink.name, i, j, zeroR, Afwd, Arev)
        end
        ### the check above forces g = 0 whenever f = 0, so only the forward element exists
        _push_entry!(sink, i, j, zeroR, Afwd)
        return nothing
    end
    if !iszero(f)
        if allow_rev && i != j && !iszero(g) && !isapprox(g, conj(f); atol=AMPLITUDE_ATOL)
            throw(ArgumentError(
                "Term `$(sink.name)`: its quantum-number rule is non-Hermitian on the unordered " *
                "pair (states $i, $j): forward factor $f, reverse factor $g ≠ conj(forward). " *
                "Such a pair is stored once and Hermitized, so only conjugate-symmetric " *
                "or one-sided (ladder) rules are meaningful there."
            ))
        end
        ### the partner of this element is supplied by `finalize_couplings`, so a callback must
        ### be conjugate-symmetric here -- checked, mirroring the rule check just above; on a
        ### SELF-partnered element (i == j) that check is a tautology, so determinism is what
        ### gets verified instead
        amp0 = if allow_rev && i != j
            _checked_amplitude(sink.amp, basis[i], basis[j], zeroR, sink.name)
        elseif i == j
            _diagonal_checked_amplitude(sink.amp, basis[i], zeroR, sink.name)
        else
            _amplitude(sink.amp, basis[i], basis[j], zeroR)
        end
        amp = amp0 * f * w
        _push_entry!(sink, i, j, zeroR, amp)
        return nothing
    end
    # the rule may only accept the reverse ordering (a ladder operator)
    (allow_rev && !iszero(g)) || return nothing
    amp = (i != j ? _checked_amplitude(sink.amp, basis[j], basis[i], zeroR, sink.name) :
                    _amplitude(sink.amp, basis[j], basis[i], zeroR)) * g * wrev
    _push_entry!(sink, j, i, zeroR, amp)
    return nothing
end


# ----- distance shell -----
function _emit!(sink::EmitSink{D}, sel::ShellSelector)::Nothing where {D}
    nt = _require_neighbors(sink.ctx, "`ShellSelector` (shell $(sel.shell))")
    sel.shell <= n_shells(nt) || throw(ArgumentError(
        "Term `$(sink.name)` asks for shell $(sel.shell) but the neighbour table resolved only " *
        "$(n_shells(nt)) shells out to radius $(nt.rcut)."
    ))
    MinvT = sink.ctx.fold_MinvT
    ### the fine table's positions are wrapped into the fine cell, so each site carries an
    ### integer wrap shift that must ride along in the conversion (cached on the context)
    rho = sink.ctx.fold_wrapshifts
    for e in shell_entries(nt, sel.shell)
        si, sj, dRf, d = nt.site_i[e], nt.site_j[e], nt.dR[e], nt.dist[e]
        if isnothing(MinvT)
            _emit_site_pair!(sink, si, sj, dRf, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        else
            ### FOLDED cell: the table was built on the fine (parent) lattice, so its integer
            ### translations convert to model coordinates through the exact rational M⁻ᵀ --
            ### this is what lets a folded comb find its true neighbours (a bond at fractional
            ### ΔR of the new cell) instead of the silent dimer of the old :bz_fold trap.
            dRm = SVector{D,Rational{Int}}(
                MinvT * SVector{D,Rational{Int}}(dRf + rho[sj] - rho[si]))
            _emit_site_pair!(sink, si, sj, dRm, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        end
    end
    return nothing
end


# ----- cutoff radius: every table entry with dist ≤ rcut, no shell ids read -----
function _emit!(sink::EmitSink{D}, sel::RadiusSelector)::Nothing where {D}
    nt = _require_neighbors(sink.ctx, "`RadiusSelector` (rcut $(sel.rcut))")
    datol = distance_atol(sink.ctx.cell.lattice)
    sel.rcut <= nt.rcut + datol || throw(ArgumentError(
        "Term `$(sink.name)` asks for bonds out to rcut = $(sel.rcut), but the neighbour " *
        "table was searched only to radius $(nt.rcut). This is an internal inconsistency: " *
        "`compile` should have scanned the terms with `max_radius` before building the table."
    ))
    MinvT = sink.ctx.fold_MinvT
    ### the fine table's positions are wrapped into the fine cell, so each site carries an
    ### integer wrap shift that must ride along in the conversion (cached on the context)
    rho = sink.ctx.fold_wrapshifts
    for e in eachindex(nt.dist)
        d = nt.dist[e]
        d <= sel.rcut + datol || continue
        si, sj, dRf = nt.site_i[e], nt.site_j[e], nt.dR[e]
        if isnothing(MinvT)
            _emit_site_pair!(sink, si, sj, dRf, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        else
            ### FOLDED cell: same exact rational M⁻ᵀ conversion as the shell emitter above
            dRm = SVector{D,Rational{Int}}(
                MinvT * SVector{D,Rational{Int}}(dRf + rho[sj] - rho[si]))
            _emit_site_pair!(sink, si, sj, dRm, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        end
    end
    return nothing
end


# ----- explicit Bravais translation -----
function _emit!(sink::EmitSink{D}, sel::TranslationSelector)::Nothing where {D}
    t = sink.ctx.tables
    dR = _as_svec_int(sel.dR, D, "dR", sink.name)

    ### At ΔR = 0 the translation orients nothing: (i,j,0) and (j,i,0) are each other's Hermitian
    ### partner, so the ordered site product below would emit every cross-site element in both
    ### orientations and `finalize_couplings` would coalesce them to t + conj(t) -- doubled
    ### magnitude, phase silently stripped. So ΔR = 0 takes the stored-once path: each unordered
    ### state pair is emitted once, in an orientation admitted by the site filters and the rule,
    ### with one dedup set spanning both site-pair orientations.
    if iszero(dR)
        zeroR = zero(SVector{D,Int})
        seen = Set{Tuple{Int,Int}}()
        for si in 1:n_sites(t), sj in 1:n_sites(t)
            _site_ok(sel.sites_to, t.site_names[si]) || continue      # si holds the targets (bra)
            _site_ok(sel.sites_from, t.site_names[sj]) || continue    # sj holds the sources (ket)
            t.site_kind[si] === t.site_kind[sj] || continue           # no cross-kind couplings
            d = norm(sink.ctx.aT * (t.site_positions[sj] - t.site_positions[si]))
            w = _bond_weight(sink.weight, sink.ctx, si, sj, zeroR, d)
            wrev = is_symmetric_weight(sink.weight) ? w :
                   _bond_weight(sink.weight, sink.ctx, sj, si, zeroR, d)
            ### the reversed orientation (source on si, target on sj) is admissible only if the
            ### filters also pass that way round
            allow_rev = _site_ok(sel.sites_to, t.site_names[sj]) &&
                        _site_ok(sel.sites_from, t.site_names[si])
            ### directional rules resolve per pair (same-site pairs refuse: zero direction);
            ### for a static rule both matrices ARE `sink.Mrule` and nothing changes
            Mf = _pair_rule_matrix(sink, si, sj, zeroR, d)
            Mr = _is_directional(sink) ? _pair_rule_matrix(sink, sj, si, zeroR, d) : Mf
            _emit_unordered_pairs!(sink, si, sj, w, wrev, allow_rev, seen; Mfwd=Mf, Mrev=Mr)
        end
        return nothing
    end

    for si in 1:n_sites(t), sj in 1:n_sites(t)
        d = norm(sink.ctx.aT * (SVector{D,Float64}(dR) + t.site_positions[sj] - t.site_positions[si]))
        _emit_site_pair!(sink, si, sj, dR, d, sel.sites_from, sel.sites_to,
                         has_intrinsic_direction(sel))
    end
    return nothing
end


"""
    _emit_site_pair!(sink, si, sj, dR, dist, from, to, directed) -> Nothing

Fan a site-level coupling out over the basis states living on those two sites, applying the
site filters and the quantum-number rule.
"""
function _emit_site_pair!(sink::EmitSink{D}, si::Int, sj::Int,
                          dR::Union{SVector{D,Int},SVector{D,Rational{Int}}}, dist::Float64,
                          from::Vector{Symbol}, to::Vector{Symbol},
                          _directed::Bool)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    ### `to` filters the target (row, bra) site; `from` filters the source (column, ket) site.
    ###
    ### For a selector with no intrinsic direction -- a distance shell, or an on-site term -- the
    ### orientation stored in the neighbour table is *arbitrary*: `_is_canonical_pair` picks the
    ### representative lexicographically on `dR`, not by physics. Applying a directed filter to
    ### that orientation keeps whichever bonds happen to have been stored the matching way round.
    ### So an undirected selector matches the pair in either orientation.
    if _directed
        _site_ok(to, t.site_names[si]) || return nothing
        _site_ok(from, t.site_names[sj]) || return nothing
    else
        (_site_ok(to, t.site_names[si]) && _site_ok(from, t.site_names[sj])) ||
            (_site_ok(from, t.site_names[si]) && _site_ok(to, t.site_names[sj])) ||
            return nothing
    end

    ### a coupling between a comb and a smooth wave needs orbital form factors the package does
    ### not model: cross-KIND pairs are skipped (give them with `explicit_bonds` if you must)
    t.site_kind[si] === t.site_kind[sj] || return nothing

    w = _bond_weight(sink.weight, sink.ctx, si, sj, dR, dist)

    # At ΔR = 0 on one site, (i,j) and (j,i) are Hermitian partners, so each unordered pair may be
    # emitted only once. At ΔR ≠ 0 they are independent matrix elements and both are needed.
    if si == sj && iszero(dR)
        ### unreachable from the geometric selectors (the neighbour table stores no self-pairs),
        ### but a directional rule must still refuse the zero direction rather than fall through
        _is_directional(sink) && _pair_rule_matrix(sink, si, sj, dR, dist)
        _emit_unordered_pairs!(sink, si, sj, w, w, true)
        return nothing
    end

    wrev = _directed || is_symmetric_weight(sink.weight) ? w :
           _bond_weight(sink.weight, sink.ctx, sj, si, -dR, dist)

    ### the pair's rule matrices: `sink.Mrule` for a static rule (both, zero cost); for a
    ### directional rule, `f` evaluated at the bond direction and -- on an undirected selector,
    ### where both orientations are one Hermitian element -- at the reversed one
    Mfwd = _pair_rule_matrix(sink, si, sj, dR, dist)
    dirn = _is_directional(sink)
    Mrev = (dirn && !_directed) ? _pair_rule_matrix(sink, sj, si, -dR, dist) : Mfwd

    for i in t.site_states[si]
        ### real-space couplings are G-diagonal AND site-resolved, so the partners are exactly
        ### `site_states[sj] ∩ by_G[G[i]]`: one keyed lookup, not a walk over the G bucket (the
        ### whole basis on a localized cell) testing membership of `site_states[sj]`
        for j in states_on_site_with_G(t, sj, t.G[i])
            f = @inbounds Mfwd[t.internal[i], t.internal[j]]
            if _directed
                iszero(f) && continue
                amp = _amplitude(sink.amp, basis[i], basis[j], dR) * f * w
                _push_entry!(sink, i, j, dR, amp)
                continue
            end
            ### An UNDIRECTED selector applies the rule in BOTH orientations of the geometric
            ### bond -- never only in the orientation the neighbour table happens to store,
            ### which is lexicographic bookkeeping, not physics. The elements (i,j,ΔR) and
            ### (j,i,-ΔR) are one Hermitian degree of freedom: when the rule speaks in both
            ### orientations they must agree (g = conj(f), refused otherwise, exactly like the
            ### stored-once path); when it speaks only in reverse (a ladder), the
            ### reverse-orientation element is emitted -- basis-declaration order must never
            ### change the physics.
            g = @inbounds Mrev[t.internal[j], t.internal[i]]
            if dirn
                ### the directional contract judges the FULL product, amplitude and weight
                ### included: amp·M(-d)[b,a] must equal conj(amp·M(d)[a,b]), else the stored
                ### element and its supplied partner would disagree with the physics of the
                ### reversed bond -- silent Hermitization, refused
                (iszero(f) && iszero(g)) && continue
                Afwd = iszero(f) ? zero(ComplexF64) :
                       _amplitude(sink.amp, basis[i], basis[j], dR) * f * w
                Arev = iszero(g) ? zero(ComplexF64) :
                       _amplitude(sink.amp, basis[j], basis[i], -dR) * g * wrev
                isapprox(Arev, conj(Afwd); atol=AMPLITUDE_ATOL) ||
                    _throw_directional_asymmetry(sink.name, i, j, dR, Afwd, Arev)
                _push_entry!(sink, i, j, dR, Afwd)
                continue
            end
            if !iszero(f)
                if !iszero(g) && !isapprox(g, conj(f); atol=AMPLITUDE_ATOL)
                    throw(ArgumentError(
                        "Term `$(sink.name)`: its quantum-number rule is non-Hermitian on the " *
                        "bond pair (states $i, $j, ΔR = $(Tuple(dR))): forward factor $f, " *
                        "reverse factor $g ≠ conj(forward). The two orientations of an " *
                        "undirected bond are one Hermitian element, so only " *
                        "conjugate-symmetric or one-sided (ladder) rules are meaningful there."
                    ))
                end
                amp = _checked_amplitude(sink.amp, basis[i], basis[j], dR, sink.name) * f * w
                _push_entry!(sink, i, j, dR, amp)
            elseif !iszero(g)
                # the rule accepts only the reversed orientation: emit the reversed-bond element
                amp = _checked_amplitude(sink.amp, basis[j], basis[i], -dR, sink.name) * g * wrev
                _push_entry!(sink, j, i, -dR, amp)
            end
        end
    end
    return nothing
end


# ----- plane-wave ΔG coupling -----
### this emitter reads `sink.Mrule` directly: a directional rule on a ΔG selector is refused
### at `emit_term!` (`_validate_directional_selector` -- a plane-wave coupling has no bond
### direction), so the static matrix is always the whole rule here
function _emit!(sink::EmitSink{D}, sel::DeltaGSelector)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    zeroR = zero(SVector{D,Int})
    dG = _as_svec_int(sel.dG, D, "dG", sink.name)
    w = _bond_weight(sink.weight, sink.ctx, 1, 1, zeroR, 0.0)

    # ΔG = 0 makes the coupling its own reverse, so each unordered pair may be emitted only once.
    # For ΔG ≠ 0 the reverse never arises: it would need G_j = G_i + ΔG as well as G_i = G_j + ΔG.
    unordered = iszero(dG)
    seen = unordered ? Set{Tuple{Int,Int}}() : nothing

    ### FOLDED localized sites carry coset labels: a harmonic at ΔG couples |G_j⟩ to the state
    ### whose label is CONGRUENT to G_j + ΔG mod the period lattice, and a wrap-around picks up
    ### the exact matrix element e^{2πi(ΔG + G_j − G_i)·τ} of the harmonic between same-τ
    ### fine-comb states (the phase is computed HERE, by the compiler -- never user-supplied).
    ### Cross-site pairs between combs are identically zero and are skipped.
    folded = !isnothing(sink.ctx.fold_MinvT)
    coset_states = folded ? _folded_coset_index(t) : nothing

    for j in 1:n_states(t)                                  # j is the source (ket)
        sj = t.state_site[j]
        _site_ok(sel.sites_from, t.site_names[sj]) || continue

        if folded && t.site_kind[sj] === :localized
            _site_ok(sel.sites_to, t.site_names[sj]) || continue     # same-site only
            M = SMatrix{D,D,Int}(t.site_period[sj])
            lbl = _coset_label(_int_adjugate(M), _int_det_abs(M), t.G[j] + dG)
            for i in get(coset_states, (sj, lbl), EMPTY_INDICES)
                gwrap = dG + t.G[j] - t.G[i]                # ∈ M ℤᴰ by construction
                phase = cispi(2 * dot(SVector{D,Float64}(gwrap), t.site_positions[sj]))
                if unordered && iszero(gwrap)
                    key = i <= j ? (i, j) : (j, i)
                    key in seen && continue
                    push!(seen, key)
                    _emit_hermitized_pair!(sink, i, j, w, w, true)
                    continue
                end
                f = @inbounds sink.Mrule[t.internal[i], t.internal[j]]
                iszero(f) && continue
                amp = _amplitude(sink.amp, basis[i], basis[j], zeroR) * f * w * phase
                if i == j
                    ### a harmonic with ΔG ∈ M_s ℤᴰ (a FINE-lattice reciprocal vector) couples a
                    ### coset state to itself: the element and its Hermitian conjugate land on
                    ### the SAME diagonal slot, so both halves are emitted here -- a lone
                    ### one-directional push would be silently halved (real phase) or spuriously
                    ### refused (complex wrap phase) by the self-conjugate reality check
                    _push_entry!(sink, i, i, zeroR, amp + conj(amp))
                    continue
                end
                _push_entry!(sink, i, j, zeroR, amp)
            end
            continue
        end

        for i in states_with_G(t, t.G[j] + dG)              # i is the target (bra), exact integer key
            _site_ok(sel.sites_to, t.site_names[t.state_site[i]]) || continue
            ### a comb has no plane-wave label to shift against a smooth wave: cross-KIND
            ### couplings need orbital form factors the package does not model
            t.site_kind[t.state_site[i]] === t.site_kind[sj] || continue
            if unordered
                ### stored-once: emitted in whichever direction the site filters and the rule
                ### admit (a ladder rule must work regardless of basis ordering), the Hermitian
                ### partner supplied by `finalize_couplings` -- the same contract as the on-site
                ### path, and now the ONE code path for every kernel (the k-side used to drop
                ### `i < j` pairs outright under directed filters; that drift class is gone).
                key = i <= j ? (i, j) : (j, i)
                key in seen && continue
                push!(seen, key)
                allow_rev = _site_ok(sel.sites_to, t.site_names[t.state_site[j]]) &&
                            _site_ok(sel.sites_from, t.site_names[t.state_site[i]])
                _emit_hermitized_pair!(sink, i, j, w, w, allow_rev)
                continue
            end
            f = @inbounds sink.Mrule[t.internal[i], t.internal[j]]
            iszero(f) && continue
            ### an oriented selector: the amplitude is a one-directional matrix element
            ### (`w₁ e^{iφ}`), its conjugate supplied automatically -- no symmetry check
            amp = _amplitude(sink.amp, basis[i], basis[j], zeroR) * f * w
            _push_entry!(sink, i, j, zeroR, amp)
        end
    end
    return nothing
end

"""
    _folded_coset_index(t) -> Dict{Tuple{Int,SVector{D,Int}},Vector{Int}}

States of each folded localized site, keyed by `(site, coset label of G)` -- the exact-integer
lookup behind congruent-`G` matching.
"""
function _folded_coset_index(t::CellTables{D}) where {D}
    out = Dict{Tuple{Int,SVector{D,Int}},Vector{Int}}()
    for s in 1:n_sites(t)
        t.site_kind[s] === :localized || continue
        M = SMatrix{D,D,Int}(t.site_period[s])
        nd = _int_det_abs(M)
        nd == 1 && continue
        adjM = _int_adjugate(M)
        for i in t.site_states[s]
            push!(get!(Vector{Int}, out, (s, _coset_label(adjM, nd, t.G[i]))), i)
        end
    end
    return out
end


# ----- explicit bond list -----
function _emit!(sink::EmitSink{D}, sel::ExplicitBondSelector)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    n = n_states(t)
    for e in eachindex(sel.i)
        i, j = sel.i[e], sel.j[e]
        dR = _as_svec_int(sel.dR[e], D, () -> "dR[$e]", sink.name)
        (1 <= i <= n) || throw(BoundsError(1:n, i))
        (1 <= j <= n) || throw(BoundsError(1:n, j))
        ### The unified explicit-selector policy on a mismatched plane-wave label is RAISE, for
        ### every kernel: the user named a specific pair, so a mismatch is a modelling error
        ### rather than a pair to skip (the bond path used to ignore G here while the k path
        ### raised -- the drift class the unified fan-out exists to remove).
        t.G[i] == t.G[j] || throw(ArgumentError(
            "Term `$(sink.name)` was explicitly asked to couple states $i and $j, which carry " *
            "different plane-wave labels ($(Tuple(t.G[i])) and $(Tuple(t.G[j]))). For a " *
            "deliberate ΔG ≠ 0 coupling use `delta_g`/`DeltaGSelector` (`dG = ...`), which " *
            "fixes the momentum-offset convention."
        ))
        d = norm(sink.ctx.aT * (SVector{D,Float64}(dR) + t.tau[j] - t.tau[i]))
        ### an oriented, hand-written pair: the directional matrix (if any) is evaluated at
        ### this orientation and trusted one-directionally, like a complex amplitude here
        Mp = _pair_rule_matrix(sink, t.state_site[i], t.state_site[j], dR, d)
        f = @inbounds Mp[t.internal[i], t.internal[j]]
        iszero(f) && continue
        w = _bond_weight(sink.weight, sink.ctx, t.state_site[i], t.state_site[j], dR, d)
        amp = _amplitude(sink.amp, basis[i], basis[j], dR) * f * w
        _push_entry!(sink, i, j, dR, amp)
    end
    return nothing
end




#############################################################
###### k-dependent terms                                #####
#############################################################

"""
    KLinearTerm

A term whose amplitude is linear in momentum: `coefficient * (c ⋅ p)` with
`p = b^T (k_frac + q)` the cartesian momentum at the entry's offset `q` (see `emit_term!`).

The archetype is the Dirac Hamiltonian `ħv(kₓσₓ + k_yσ_y)`, written as two `KLinearTerm`s carrying
`pauli_x` and `pauli_y` rules.

Notes:
------
`c` is constrained **real** while `coefficient` may be complex. Hermitizing a k-dependent entry
requires `conj(H_ij(k)) = H_ji(k)` at the *same* momentum, which holds exactly when the
k-dependent scalar is real -- so all the `i`s belong in the quantum-number rule, which is also how
`ħv k⋅σ` is physically written. This makes Hermitization literally the same code for bonds and for
k-terms.

`c` is stored either as a full vector or as `direction => value` -- the latter is what `dirac`
needs before the dimension is known. Materialized against the cell at compile time.
"""
struct KLinearTerm <: KDependentTerm
    c::Union{Vector{Float64},Pair{Int,Float64}}
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KLinearTerm(c::Union{Vector{Float64},Pair{Int,Float64}}, k0::Vector{Float64},
                         selector::AbstractBondSelector, rule::AbstractQnumRule,
                         coefficient::ComplexF64, name::String)
        ### mirror of KQuadraticTerm's validating constructor: reject the inputs that would
        ### otherwise surface as a bare error (or NaN bands) far from the term that caused them
        if c isa Pair{Int,Float64}
            first(c) >= 1 || throw(DomainError(first(c),
                "`KLinearTerm` `$name`: the direction index must be at least 1."))
            isfinite(last(c)) || throw(DomainError(last(c),
                "`KLinearTerm` `$name`: the coefficient along direction $(first(c)) is not finite."))
        else
            all(isfinite, c) || throw(DomainError(c,
                "`KLinearTerm` `$name`: the coefficient vector contains a non-finite entry."))
        end
        all(isfinite, k0) || throw(DomainError(k0,
            "`KLinearTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KLinearTerm` `$name`: the prefactor is not finite."))
        return new(c, k0, selector, rule, coefficient, name)
    end
end

"""
    KQuadraticTerm

A term whose amplitude is quadratic in momentum: `coefficient * (pᵀ M p)` with
`p = b^T (k_frac + q)`.

The archetype is the continuum kinetic energy `ħ²|k+G|²/2m`, for which `M = (ħ²/2m) I`.

`M` must be real and symmetric, for the Hermitization reason given on `KLinearTerm`. It is stored
either as a full matrix or as a scalar `λ` meaning `λ·I` -- the isotropic form, expressible before
the dimension is known.
"""
struct KQuadraticTerm <: KDependentTerm
    M::Union{Matrix{Float64},Float64}
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KQuadraticTerm(M::Union{Matrix{Float64},Float64}, k0::Vector{Float64},
                            selector::AbstractBondSelector, rule::AbstractQnumRule,
                            coefficient::ComplexF64, name::String)
        if M isa Matrix{Float64}
            issymmetric(M) || throw(ArgumentError(
                "`KQuadraticTerm` `$name` requires a symmetric M; got $M. An antisymmetric part " *
                "would make the k-dependent scalar non-real and break Hermitization."
            ))
            all(isfinite, M) || throw(DomainError(M,
                "`KQuadraticTerm` `$name`: M contains a non-finite entry."))
        else
            isfinite(M) || throw(DomainError(M, "`KQuadraticTerm` `$name`: M is not finite."))
        end
        all(isfinite, k0) || throw(DomainError(k0,
            "`KQuadraticTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KQuadraticTerm` `$name`: the prefactor is not finite."))
        return new(M, k0, selector, rule, coefficient, name)
    end
end

"""
    KEnvelopeTerm

A term whose amplitude is an arbitrary **real envelope** of momentum:
`coefficient * f(p)` with `p = b^T (k_frac + q)` and `f(p) -> Real` a user callback.

The archetype is a momentum-dependent tunnelling form factor,
`T(p) = w1 * exp(-α(|p| - kθ))`, at a `ΔG ≠ 0` transfer.

The contract is the one every k-term obeys -- the k-dependent scalar is **real**, every phase
lives in `coefficient` or in the quantum-number rule -- which is what lets the one Hermitization
rule handle it with no new code. Realness cannot be enforced by a constructor for a closure, so
`compile` spot-checks `f` at a few momenta and raises, naming the `coefficient`/rule escape.

`parity` (`:even`, `:odd`, or `:none`) declares the behaviour of `f` under `p -> -p`; it is what
`nambu_doubling` needs, and `:none` makes doubling refuse loudly.

A raw closure is not serialisable: `is_serializable` reports `false`, and `h5_write` refuses
before touching the file -- unlike a callback *amplitude*, which lowers to numbers at compile
and round-trips exactly, a callback kernel would reload as a different Hamiltonian. A
registered [`AbstractEnvelope`](@ref) payload IS serialisable (`is_serializable` reports the
payload's [`kernel_tag`](@ref) truthfully) and round-trips exactly.
"""
struct KEnvelopeTerm <: KDependentTerm
    f::Function
    parity::Symbol
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KEnvelopeTerm(f::Function, parity::Symbol, k0::Vector{Float64},
                           selector::AbstractBondSelector, rule::AbstractQnumRule,
                           coefficient::ComplexF64, name::String)
        parity in (:even, :odd, :none) || throw(ArgumentError(
            "`KEnvelopeTerm` `$name`: `parity` must be :even, :odd or :none; got :$parity."))
        all(isfinite, k0) || throw(DomainError(k0,
            "`KEnvelopeTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KEnvelopeTerm` `$name`: the prefactor is not finite."))
        return new(f, parity, k0, selector, rule, coefficient, name)
    end
end

max_shell(term::KDependentTerm) = max_shell(_kselector(term))
is_serializable(::KDependentTerm) = true
### a k-envelope term is serializable exactly when its payload carries an on-disk tag: a
### registered AbstractEnvelope round-trips, a raw closure does not
is_serializable(t::KEnvelopeTerm) = !isnothing(kernel_tag(t.f))

@inline _k0_raw(term::KLinearTerm) = term.k0_frac
@inline _k0_raw(term::KQuadraticTerm) = term.k0_frac
@inline _k0_raw(term::KEnvelopeTerm) = term.k0_frac
@inline _kcoefficient(term::KLinearTerm) = term.coefficient
@inline _kcoefficient(term::KQuadraticTerm) = term.coefficient
@inline _kcoefficient(term::KEnvelopeTerm) = term.coefficient
@inline _kselector(term::KLinearTerm) = term.selector
@inline _kselector(term::KQuadraticTerm) = term.selector
@inline _kselector(term::KEnvelopeTerm) = term.selector
@inline _krule(term::KLinearTerm) = term.rule
@inline _krule(term::KQuadraticTerm) = term.rule
@inline _krule(term::KEnvelopeTerm) = term.rule


"""
    KLinearKernel{D} / KQuadraticKernel{D,L}

The **materialized, statically-sized** evaluation kernel of a k-dependent term: the only part of
the term the per-k loop touches. Concrete by construction (`SVector{D}` and `SMatrix{D,D,·,L}`
carry every type parameter), so the per-kernel-group accumulation loops specialize -- the
function-barrier property the abstractly-typed `term` field used to only approximate.
"""
struct KLinearKernel{D}
    c::SVector{D,Float64}
end

struct KQuadraticKernel{D,L}
    M::SMatrix{D,D,Float64,L}   # the implicit constructor infers {D,L} from the field
end

"""
    CallbackKernel{F}

A user-supplied real envelope of momentum, wrapped so the inner accumulation loop specializes.

The `{F}` wrapper is mandatory: a bare closure argument would hit Julia's `Function`-argument
nospecialize heuristic (the kernel is only passed through to `kvalue`, never called at the
dispatch site) and box the inner loop. `parity` is the declared behaviour under `p -> -p`
(`+1` even, `-1` odd, `0` undeclared -- doubling refuses on `0`).
"""
struct CallbackKernel{F}
    f::F
    parity::Int
end

"""
    KPolynomialKernel{D,NM,L}

The materialized, statically-sized kernel of a momentum polynomial: `NM` monomials over `D`
momentum components (see `KPolynomialTerm` in `hamiltonians/kdotp/kdotp.jl`).
"""
struct KPolynomialKernel{D,NM,L}
    c::SVector{NM,Float64}
    alpha::SMatrix{NM,D,Int,L}
end




#############################################################
###### The public kernel contract                       #####
#############################################################

"""
    kvalue(kern, p::SVector{D,Float64}) -> Real

The k-dependent scalar `f_κ(p)` of a kernel, evaluated at the cartesian momentum `p` --
REQUIRED for every kernel type. The per-k assembly loops call it once per coupling entry, as

    H[i, j] += w * kvalue(kern, bᵀ(k + q)) * exp(2πi (k + q)·δ)

The return value MUST be real: this is the one invariant Hermitization rests on
(`w_ji(-ΔR, q) = conj(w_ij(ΔR, q))` holds for every kernel *because* the kernel scalar is
real; see the coupling convention in `couplinglist.jl`). Put every phase in the coupling
amplitude or in a quantum-number rule, never in the kernel.

The other two methods of the kernel contract are [`kernel_parity`](@ref) (REQUIRED by
`nambu_doubling`) and [`kernel_tag`](@ref) (OPTIONAL; enables HDF5 round-trips together with
[`register_kernel!`](@ref)).
"""
function kvalue end

@inline kvalue(::ConstantKernel, ::SVector) = 1.0
@inline kvalue(kern::KLinearKernel{D}, p::SVector{D,Float64}) where {D} = dot(kern.c, p)
@inline kvalue(kern::KQuadraticKernel{D}, p::SVector{D,Float64}) where {D} = dot(p, kern.M * p)
@inline kvalue(kern::CallbackKernel, p::SVector) = kern.f(p)

@inline function kvalue(kern::KPolynomialKernel{D,NM}, p::SVector{D,Float64}) where {D,NM}
    acc = 0.0
    @inbounds for m in 1:NM
        t = kern.c[m]
        for d in 1:D
            e = kern.alpha[m, d]
            e == 0 && continue
            t *= p[d]^e
        end
        acc += t
    end
    return acc
end

### the internal spelling the assembly loops and compile-time probes call; a one-line
### forwarder, so the public `kvalue` is the single override point for every kernel type
@inline _kvalue(kern, p) = kvalue(kern, p)

"""
    kernel_parity(kern) -> Int

The parity of a k-kernel under `p -> -p`: `+1` even, `-1` odd. The hole block of the BdG
doubling needs it because it evaluates every kernel at reversed momentum -- REQUIRED for a
model to survive [`nambu_doubling`](@ref). A kernel with no method here refuses by name rather
than guessing (a wrong parity is a silently wrong hole block); a mixed-parity kernel must be
split into its even and odd parts first.
"""
kernel_parity(k) = throw(ArgumentError(
    "`nambu_doubling` needs the parity of the k-kernel `$(typeof(k))` under p -> -p; " *
    "define `kernel_parity(::$(typeof(k)))` (+1 even, -1 odd)."))

kernel_parity(::ConstantKernel) = 1
kernel_parity(::KLinearKernel) = -1
kernel_parity(::KQuadraticKernel) = 1
function kernel_parity(k::CallbackKernel)
    k.parity == 0 && throw(ArgumentError(
        "`nambu_doubling` needs the parity of a k-envelope under p -> -p, and this one was " *
        "built with `parity=:none`. Rebuild the term with `k_envelope(f; parity=:even)` or " *
        "`:odd` (split a mixed-parity envelope into its even and odd parts first)."))
    return k.parity
end
### the degree-parity body lives beside the term type in `hamiltonians/kdotp/kdotp.jl`,
### under the private spelling; the public name forwards to it
kernel_parity(kern::KPolynomialKernel) = _kernel_parity(kern)

"""
    kernel_tag(kern) -> Union{String,Nothing}

The on-disk tag a kernel serializes under, or `nothing` (the default) for a kernel that
cannot round-trip -- OPTIONAL. `h5_write` refuses a model whose kernel table holds an
untagged kernel with entries, before the target file is touched. To make a kernel type
shareable, define `kernel_tag(::T) = "mytag"` and register the tag's writer/reader with
[`register_kernel!`](@ref) (an [`AbstractEnvelope`](@ref) with [`register_envelope!`](@ref)).

A `CallbackKernel` delegates to its payload: a registered envelope carries the envelope's
tag; a raw closure keeps `nothing` and the loud refusal.
"""
kernel_tag(::Any) = nothing
kernel_tag(::ConstantKernel) = "constant"
kernel_tag(::KLinearKernel) = "linear"
kernel_tag(::KQuadraticKernel) = "quadratic"
kernel_tag(::KPolynomialKernel) = "polynomial"
kernel_tag(k::CallbackKernel) = k.f isa AbstractEnvelope ? kernel_tag(k.f) : nothing


#############################################################
###### Serializable envelopes                           #####
#############################################################

"""
    AbstractEnvelope <: Function

A **serializable, named** k-envelope: a callable value object `env(p::SVector) -> Real` that
flows anywhere a plain envelope function does (`k_envelope`, `moire_tunnelling`'s `kernel=`)
while, unlike a raw closure, round-tripping through HDF5.

A subtype must:
- be an **immutable** callable struct returning a REAL number (the kernel contract);
- define [`kernel_parity`](@ref) (`+1` even, `-1` odd under `p -> -p`, `0` when the envelope
  has no definite parity) -- `k_envelope` defaults its `parity` from a ±1 value, so such an
  envelope never hits the `:none` doubling refusal by accident (a `0` keeps `:none`, and
  `nambu_doubling` then refuses loudly rather than guess);
- define [`kernel_tag`](@ref) and register the tag with [`register_envelope!`](@ref)
  (typically at module load / in `__init__`) to enable the HDF5 round-trip;
- define `Base.:(==)` (structural equality) when it holds arrays, so equal envelopes share
  one kernel group (`kernel_id!` compares payloads structurally).

Shipped: [`GaussianEnvelope`](@ref) and [`TabulatedEnvelope`](@ref).
"""
abstract type AbstractEnvelope <: Function end

"""
    kvalue(env::AbstractEnvelope, p) -> Real

Envelopes are callable; `kvalue` on one is simply `env(p)`, so the kernel contract can be
probed uniformly (an envelope enters the compiled kernel table wrapped in a `CallbackKernel`).
"""
@inline kvalue(env::AbstractEnvelope, p::SVector) = env(p)

"""
    GaussianEnvelope(sigma; p0=0.0) <: AbstractEnvelope

The radial Gaussian envelope `exp(-(|p| - p0)² / 2σ²)` -- a smooth momentum-shell form
factor peaked at radius `p0` (momenta in the package's cartesian units, `p = bᵀ(k + q)`).
Even under `p -> -p` (it depends on `|p|` only), tag `"gaussian_radial"`: models built from
it round-trip through HDF5 exactly.

    moire_tunnelling(cell, w; kernel=GaussianEnvelope(0.4; p0=norm(Kc)))
"""
struct GaussianEnvelope <: AbstractEnvelope
    sigma::Float64
    p0::Float64

    function GaussianEnvelope(sigma::Real; p0::Real=0.0)
        (isfinite(sigma) && sigma > 0) || throw(DomainError(sigma,
            "`GaussianEnvelope` needs a positive, finite width `sigma`."))
        isfinite(p0) || throw(DomainError(p0, "`GaussianEnvelope`: `p0` is not finite."))
        return new(Float64(sigma), Float64(p0))
    end
end

(env::GaussianEnvelope)(p::AbstractVector) = exp(-(norm(p) - env.p0)^2 / (2 * env.sigma^2))

kernel_parity(::GaussianEnvelope) = 1
kernel_tag(::GaussianEnvelope) = "gaussian_radial"

"""
    TabulatedEnvelope(r, v; extrapolation=:zero) <: AbstractEnvelope

A radial envelope tabulated on the grid `r` (values of `|p|`, strictly increasing) with
values `v`, evaluated by linear interpolation between the nodes. Outside the grid,
`extrapolation` decides: `:zero` (the default) makes the envelope vanish, `:flat` holds the
nearest endpoint value; any other kind refuses by name. Even under `p -> -p` (a function of
`|p|` only), tag `"tabulated_radial"`.

This is the exchange format for measured or computed form factors: the stored numbers ARE
the model, so the HDF5 round-trip is exact -- unlike a raw closure, which `h5_write` refuses.

    T = TabulatedEnvelope([0.0, 0.5, 1.0, 2.0], [1.0, 0.8, 0.4, 0.1])
    moire_tunnelling(cell, w; kernel=T)                # parity inferred, file-shareable
"""
struct TabulatedEnvelope <: AbstractEnvelope
    r::Vector{Float64}
    v::Vector{Float64}
    extrapolation::Symbol

    function TabulatedEnvelope(r::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                               extrapolation::Symbol=:zero)
        length(r) == length(v) || throw(DimensionMismatch(
            "`TabulatedEnvelope`: the radial grid `r` has $(length(r)) nodes but `v` holds " *
            "$(length(v)) values; each node needs exactly one value."))
        length(r) >= 2 || throw(ArgumentError(
            "`TabulatedEnvelope` interpolates linearly and needs at least two grid nodes; " *
            "got $(length(r))."))
        all(isfinite, r) && all(isfinite, v) || throw(DomainError((r, v),
            "`TabulatedEnvelope`: the grid and its values must be finite."))
        all(m -> r[m] < r[m+1], 1:(length(r) - 1)) || throw(ArgumentError(
            "`TabulatedEnvelope`: the radial grid `r` must be STRICTLY increasing (it holds " *
            "values of |p|); sort the table and merge duplicate nodes first."))
        extrapolation in (:zero, :flat) || throw(ArgumentError(
            "`TabulatedEnvelope`: unknown extrapolation kind `:$extrapolation`; supported " *
            "kinds are :zero (the envelope vanishes outside the grid) and :flat (the " *
            "nearest endpoint value is held)."))
        return new(collect(Float64, r), collect(Float64, v), extrapolation)
    end
end

function (env::TabulatedEnvelope)(p::AbstractVector)
    x = norm(p)
    r = env.r
    if x < r[1] || x > r[end]
        env.extrapolation === :flat || return 0.0
        return x < r[1] ? env.v[1] : env.v[end]
    end
    m = searchsortedlast(r, x)
    m == length(r) && return env.v[end]
    t = (x - r[m]) / (r[m+1] - r[m])
    return (1 - t) * env.v[m] + t * env.v[m+1]
end

kernel_parity(::TabulatedEnvelope) = 1
kernel_tag(::TabulatedEnvelope) = "tabulated_radial"

### structural equality, so two terms built from equal tables share ONE kernel group
### (`kernel_id!` compares envelope payloads by type and ==, not by identity)
Base.:(==)(a::TabulatedEnvelope, b::TabulatedEnvelope) =
    a.r == b.r && a.v == b.v && a.extrapolation === b.extrapolation
Base.hash(env::TabulatedEnvelope, h::UInt) =
    hash((TabulatedEnvelope, env.r, env.v, env.extrapolation), h)

### constructor echoes (the default Function display is a method table, useless here)
Base.show(io::IO, env::GaussianEnvelope) =
    print(io, "GaussianEnvelope(", env.sigma, iszero(env.p0) ? "" : "; p0=$(env.p0)", ")")
Base.show(io::IO, ::MIME"text/plain", env::GaussianEnvelope) = show(io, env)
Base.show(io::IO, env::TabulatedEnvelope) =
    print(io, "TabulatedEnvelope(", env.r, ", ", env.v,
          env.extrapolation === :zero ? "" : "; extrapolation=$(repr(env.extrapolation))", ")")
Base.show(io::IO, ::MIME"text/plain", env::TabulatedEnvelope) = show(io, env)

"""
    BlochSumEnvelope(Q, p0, delta, amp, component;
                     wmap=nothing, wshift=(0, 0), wradius=Inf, wwidth=0) <: AbstractEnvelope

The real (`component = :re`) or imaginary (`:im`) part of a finite Bloch sum, evaluated at an
affinely mapped momentum:

    env(p) = win(|wmap·u + wshift|) · re/im( Σₘ amp[m] · e^{2πi u·delta[m]} ),   u = Q·p + p0.

`Q` maps the package's cartesian kernel momentum `p = bᵀ(k + q)` into the fractional
coordinates the separations `delta` are expressed in (any fixed rotation -- e.g. a layer
frame -- composes into `Q`), and `p0` is a constant offset (an expansion point). The stored
numbers ARE the envelope, so the HDF5 round-trip is exact (tag `"bloch_sum"`, or
`"bloch_sum_windowed"` when a window is declared).

The optional WINDOW keeps the sum only inside a radius of a declared point: `win ≡ 1` for
`r ≤ wradius − wwidth`, `≡ 0` for `r ≥ wradius + wwidth`, with a smooth cosine roll-off in
between (`wwidth = 0` is a hard cutoff). `wmap`/`wshift` map the evaluation coordinate `u`
to whatever frame the radius lives in (e.g. back to cartesian momentum, recentered on a
resonance). Sampled data is trustworthy only within its aliasing radius -- the window is how
an envelope carries that limit with itself; the default `wradius = Inf` is windowless.

This is the exact carrier for k-dependent hopping data: a Wannier Hamiltonian block, or any
finite Fourier series in k, evaluated with no interpolation and no periodization -- the sum
is exact at every `p`, including far outside the first Brillouin zone. It is 2D
(`length(p) == 2`); no definite parity (`kernel_parity == 0`), so terms built from it keep
`parity = :none` and `nambu_doubling` refuses them loudly rather than guess.

A complex-valued Bloch sum `W(p)` enters a Hermitian model as TWO terms sharing the data:
`k_envelope(re_env; coefficient=1) + k_envelope(im_env; coefficient=im)` -- the package
Hermitizes each stored coupling by conjugating the coefficient at the same midpoint momentum,
which reproduces `conj(W(p))` exactly.
"""
struct BlochSumEnvelope <: AbstractEnvelope
    Q::SMatrix{2,2,Float64,4}
    p0::SVector{2,Float64}
    delta::Vector{SVector{2,Float64}}
    amp::Vector{ComplexF64}
    component::Symbol
    wmap::SMatrix{2,2,Float64,4}
    wshift::SVector{2,Float64}
    wradius::Float64
    wwidth::Float64
    ### ---- derived separability tables (see `_bloch_tables`); NOT part of the envelope's
    ### identity: `==`/`hash`/HDF5 read them back off `delta` and `amp`, and the constructor
    ### rebuilds them, so they never need storing or comparing.
    dx::Vector{Float64}          # the distinct δ¹ values
    dy::Vector{Float64}          # the distinct δ² values
    xptr::Vector{Int32}          # CSR over `dx`: group a spans xptr[a]:xptr[a+1]-1
    iy::Vector{Int32}            # δ² index of each term (`delta`/`amp` are stored x-grouped)
    fast::Bool                   # do the tables pay for themselves?

    function BlochSumEnvelope(Q::AbstractMatrix{<:Real}, p0::AbstractVector{<:Real},
                              delta::AbstractVector{<:AbstractVector{<:Real}},
                              amp::AbstractVector{<:Number}, component::Symbol;
                              wmap::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                              wshift::AbstractVector{<:Real}=SVector(0.0, 0.0),
                              wradius::Real=Inf, wwidth::Real=0.0)
        size(Q) == (2, 2) || throw(DimensionMismatch(
            "`BlochSumEnvelope`: the momentum map `Q` must be 2×2; got $(size(Q)). The " *
            "envelope is two-dimensional -- momenta of 2D plane-wave models."))
        length(p0) == 2 || throw(DimensionMismatch(
            "`BlochSumEnvelope`: the offset `p0` must have 2 components; got $(length(p0))."))
        length(delta) == length(amp) || throw(DimensionMismatch(
            "`BlochSumEnvelope`: $(length(delta)) separations but $(length(amp)) amplitudes; " *
            "each δ needs exactly one amplitude."))
        isempty(delta) && throw(ArgumentError(
            "`BlochSumEnvelope` needs at least one (δ, amplitude) entry; an identically zero " *
            "envelope is a term you simply do not add."))
        all(d -> length(d) == 2, delta) || throw(DimensionMismatch(
            "`BlochSumEnvelope`: every separation in `delta` must have 2 components."))
        all(isfinite, Q) && all(isfinite, p0) &&
            all(d -> all(isfinite, d), delta) && all(isfinite, amp) || throw(DomainError(
            "BlochSumEnvelope", "`BlochSumEnvelope`: Q, p0, delta and amp must be finite."))
        component in (:re, :im) || throw(ArgumentError(
            "`BlochSumEnvelope`: unknown component `:$component`; supported components are " *
            ":re and :im (a complex Bloch sum enters a Hermitian model as two real terms)."))
        wm = isnothing(wmap) ? one(SMatrix{2,2,Float64}) : SMatrix{2,2,Float64}(wmap)
        length(wshift) == 2 || throw(DimensionMismatch(
            "`BlochSumEnvelope`: the window shift must have 2 components; got $(length(wshift))."))
        (wradius > 0 && !isnan(wradius)) || throw(DomainError(wradius,
            "`BlochSumEnvelope`: the window radius must be positive (Inf = no window)."))
        (isfinite(wwidth) && wwidth >= 0) || throw(DomainError(wwidth,
            "`BlochSumEnvelope`: the window roll-off width must be finite and non-negative."))
        ### a roll-off wider than the radius reaches inside r = 0: the window would then never
        ### equal 1 anywhere and would silently attenuate the whole envelope, which is not what
        ### "trust this data out to `wradius`" is supposed to mean
        (wwidth < wradius || !isfinite(wradius)) || throw(ArgumentError(
            "`BlochSumEnvelope`: the roll-off width $wwidth reaches the origin at radius " *
            "$wradius (need wwidth < wradius). The window would be below 1 everywhere, " *
            "attenuating the envelope at every momentum rather than bounding where it is " *
            "trusted; narrow `wwidth` or widen `wradius`."))
        all(isfinite, wm) && all(isfinite, wshift) || throw(DomainError(
            "BlochSumEnvelope", "`BlochSumEnvelope`: wmap and wshift must be finite."))
        dl, av, dx, dy, xptr, iy, fast =
            _bloch_tables([SVector{2,Float64}(d) for d in delta], collect(ComplexF64, amp))
        return new(SMatrix{2,2,Float64}(Q), SVector{2,Float64}(p0),
                   dl, av, component, wm, SVector{2,Float64}(wshift), Float64(wradius),
                   Float64(wwidth), dx, dy, xptr, iy, fast)
    end
end

"""
    _bloch_tables(delta, amp) -> (delta', amp', dx, dy, xptr, iy, fast)

Precompute the SEPARABILITY tables that make a Bloch sum cheap to evaluate, returning the
entries reordered to match them.

`e^{2πi u·δ} = e^{2πi u₁δ¹} · e^{2πi u₂δ²}` factorizes, and the two COMPONENT sets are far
smaller than the term list: separations built from lattice sums reuse the same handful of
first and second components over and over. An ERTCA interlayer harmonic solved from `S`
registry stackings holds `S ×` (number of R-vectors) terms -- tens of thousands -- but only
`S ×` (R-range) distinct components, an order of magnitude fewer. So the `N` transcendental
calls of the direct sum collapse to `|dx| + |dy|`, and one complex multiply-add per term is
all that remains.

Entries are GROUPED by their δ¹ index (a stable counting sort into CSR form), which factors
the evaluation one step further:

    Σₘ ampₘ e^{2πi u·δₘ} = Σ_a e^{2πi u₁ dx[a]} · ( Σ_{m ∈ group a} ampₘ e^{2πi u₂ dy[iy[m]]} )

-- ONE complex multiply per term instead of two. Reordering `delta`/`amp` in place, rather
than keeping a second copy, costs 4 bytes per term instead of 20; the sort is deterministic
and stable, so envelopes built from equal inputs stay `==`, and the HDF5 round trip (which
writes the stored order and re-sorts an already-sorted list on read) stays exact.

`fast` is false when the factorization would not pay -- separations sharing few components --
and then no tables are stored and the direct sum is used. The threshold is a COST comparison,
not a memory one: a `cispi` costs several times a complex multiply-add, so the tables still
win well past `|dx| + |dy| = n/2`.
"""
function _bloch_tables(delta::Vector{SVector{2,Float64}}, amp::Vector{ComplexF64})
    n = length(delta)
    xid = Dict{Float64,Int32}()
    yid = Dict{Float64,Int32}()
    dx = Float64[]
    dy = Float64[]
    ix = Vector{Int32}(undef, n)
    iy0 = Vector{Int32}(undef, n)
    for m in 1:n
        d = delta[m]
        ix[m] = get!(xid, d[1]) do
            push!(dx, d[1]); Int32(length(dx))
        end
        iy0[m] = get!(yid, d[2]) do
            push!(dy, d[2]); Int32(length(dy))
        end
    end
    if length(dx) + length(dy) > (7 * n) ÷ 10
        return delta, amp, Float64[], Float64[], Int32[], Int32[], false
    end
    ndx = length(dx)
    xptr = zeros(Int32, ndx + 1)
    xptr[1] = 1
    for m in 1:n
        xptr[ix[m]+1] += Int32(1)
    end
    for a in 1:ndx
        xptr[a+1] += xptr[a]
    end
    pos = xptr[1:ndx]
    iy = Vector{Int32}(undef, n)
    dl = Vector{SVector{2,Float64}}(undef, n)
    av = Vector{ComplexF64}(undef, n)
    for m in 1:n
        a = ix[m]
        q = pos[a]
        pos[a] = q + Int32(1)
        iy[q] = iy0[m]
        dl[q] = delta[m]
        av[q] = amp[m]
    end
    return dl, av, dx, dy, xptr, iy, true
end

### the direct sum -- the definition, and the fallback when the tables do not pay
@inline function _bloch_sum_direct(env::BlochSumEnvelope, u::SVector{2,Float64})::ComplexF64
    s = zero(ComplexF64)
    @inbounds for m in eachindex(env.amp)
        s += env.amp[m] * cispi(2 * dot(u, env.delta[m]))
    end
    return s
end

### the factorized sum: `tx`/`ty` are scratch of length ≥ |dx| / |dy| (see `_bloch_tables`)
@inline function _bloch_sum_fast(env::BlochSumEnvelope, u::SVector{2,Float64},
                                 tx::Vector{ComplexF64},
                                 ty::Vector{ComplexF64})::ComplexF64
    u1, u2 = u[1], u[2]
    @inbounds for a in eachindex(env.dx)
        tx[a] = cispi(2 * u1 * env.dx[a])
    end
    @inbounds for b in eachindex(env.dy)
        ty[b] = cispi(2 * u2 * env.dy[b])
    end
    s = zero(ComplexF64)
    @inbounds for a in eachindex(env.dx)
        t = zero(ComplexF64)
        for m in env.xptr[a]:(env.xptr[a+1]-Int32(1))
            t += env.amp[m] * ty[env.iy[m]]
        end
        s += tx[a] * t
    end
    return s
end

@inline _bloch_scratch(env::BlochSumEnvelope) =
    (Vector{ComplexF64}(undef, length(env.dx)), Vector{ComplexF64}(undef, length(env.dy)))

_has_window(env::BlochSumEnvelope)::Bool = isfinite(env.wradius)

### the smooth radial window: 1 inside, 0 outside, cosine roll-off across
### [wradius − wwidth, wradius + wwidth] (wwidth = 0: hard cutoff at wradius)
@inline function _bloch_window(env::BlochSumEnvelope, u::SVector{2,Float64})::Float64
    isfinite(env.wradius) || return 1.0
    r = norm(env.wmap * u + env.wshift)
    r <= env.wradius - env.wwidth && return 1.0
    r >= env.wradius + env.wwidth && return 0.0
    env.wwidth == 0 && return r <= env.wradius ? 1.0 : 0.0
    return 0.5 * (1 + cospi((r - env.wradius + env.wwidth) / (2 * env.wwidth)))
end

function (env::BlochSumEnvelope)(p::AbstractVector)
    length(p) == 2 || throw(DimensionMismatch(
        "`BlochSumEnvelope` is two-dimensional but was evaluated at a momentum with " *
        "$(length(p)) components."))
    u = env.Q * SVector{2,Float64}(p[1], p[2]) + env.p0
    win = _bloch_window(env, u)
    win == 0.0 && return 0.0
    ### the scalar entry point allocates its own scratch; the assembly loop hoists it out
    ### across a whole kernel group instead (see the `CallbackKernel{BlochSumEnvelope}`
    ### methods of `_accumulate!` / `_accumulate_sparse!`)
    s = if env.fast
        tx, ty = _bloch_scratch(env)
        _bloch_sum_fast(env, u, tx, ty)
    else
        _bloch_sum_direct(env, u)
    end
    return win * (env.component === :re ? real(s) : imag(s))
end

### the value at a momentum, with caller-supplied scratch (the hot path)
@inline function _bloch_value(env::BlochSumEnvelope, p::SVector{2,Float64},
                              tx::Vector{ComplexF64}, ty::Vector{ComplexF64})::Float64
    u = env.Q * p + env.p0
    win = _bloch_window(env, u)
    win == 0.0 && return 0.0
    s = env.fast ? _bloch_sum_fast(env, u, tx, ty) : _bloch_sum_direct(env, u)
    return win * (env.component === :re ? real(s) : imag(s))
end

kernel_parity(::BlochSumEnvelope) = 0
kernel_tag(env::BlochSumEnvelope) = _has_window(env) ? "bloch_sum_windowed" : "bloch_sum"

### structural equality over ALL fields: two envelopes holding the same table but different
### momentum maps (the two layers' frames) must occupy DISTINCT kernel groups -- merging them
### would silently evaluate one layer's data in the other layer's frame
Base.:(==)(a::BlochSumEnvelope, b::BlochSumEnvelope) =
    a.Q == b.Q && a.p0 == b.p0 && a.delta == b.delta && a.amp == b.amp &&
    a.component === b.component && a.wmap == b.wmap && a.wshift == b.wshift &&
    isequal(a.wradius, b.wradius) && a.wwidth == b.wwidth
Base.hash(env::BlochSumEnvelope, h::UInt) =
    hash((BlochSumEnvelope, env.Q, env.p0, env.delta, env.amp, env.component,
          env.wmap, env.wshift, env.wradius, env.wwidth), h)

### a full constructor echo would print hundreds of amplitudes; summarize instead
### (display-only -- nothing parses this back)
Base.show(io::IO, env::BlochSumEnvelope) =
    print(io, "BlochSumEnvelope(", length(env.amp), " δ-terms, component=",
          repr(env.component),
          _has_window(env) ? ", window r=$(env.wradius)±$(env.wwidth)" : "", ")")
Base.show(io::IO, ::MIME"text/plain", env::BlochSumEnvelope) = show(io, env)




_make_kernel(term::KLinearTerm, ::Val{D}) where {D} =
    KLinearKernel{D}(_materialize_c(term.c, D, term.name))
_make_kernel(term::KQuadraticTerm, ::Val{D}) where {D} =
    KQuadraticKernel(_materialize_M(term.M, D, term.name))

function _make_kernel(term::KEnvelopeTerm, ::Val{D}) where {D}
    p = term.parity === :even ? 1 : term.parity === :odd ? -1 : 0
    _check_envelope_real(term.f, Val(D), p, term.name)
    return CallbackKernel(term.f, p)
end

"""
    _check_envelope_real(f, ::Val{D}, parity, name) -> Nothing

Spot-check that a callback envelope returns a real number, AND that a declared parity is one
the callback actually has. A constructor cannot validate a closure, and every user who copies
a complex `T(p)` from a paper would otherwise discover the contract as a wrong spectrum
instead of a named error; a mis-declared parity would surface even later, as a silently wrong
hole block under `nambu_doubling`. Deterministic probe momenta -- exactness is not needed,
only coverage away from symmetry points.
"""
function _check_envelope_real(f::Function, ::Val{D}, parity::Int,
                              name::AbstractString)::Nothing where {D}
    ### probes must span NON-COLLINEAR directions: parity is a property over all of momentum
    ### space, and three probes on one line through the origin miss any envelope engineered to
    ### vanish on that line (a genuinely odd f that passed as :even would silently break the
    ### hole block under `nambu_doubling`)
    probes = SVector{D,Float64}[]
    for scale in (0.317, -1.234, 2.718)
        push!(probes, SVector{D,Float64}(ntuple(m -> scale * (0.4321 + 0.317 * m), D)))
    end
    for (s1, s2) in ((0.911, -0.407), (-0.286, 1.618), (1.207, 0.093))
        push!(probes, SVector{D,Float64}(ntuple(
            m -> s1 * cospi(0.37m + 0.11) + s2 * sinpi(0.53m - 0.29), D)))
    end
    for p in probes
        v = f(p)
        v isa Real || throw(ArgumentError(
            "k-envelope term `$name`: the callback returned $(typeof(v)) at p = $(Tuple(p)); " *
            "the k-dependent scalar must be REAL -- put every phase in `coefficient` or in the " *
            "quantum-number rule. This is what keeps H(k) Hermitian by construction."))
        isfinite(v) || throw(DomainError(v,
            "k-envelope term `$name`: the callback returned a non-finite value at p = $(Tuple(p))."))
        if parity != 0
            vm = f(-p)
            isapprox(Float64(vm), parity * Float64(v);
                     atol=AMPLITUDE_ATOL * max(1.0, abs(v))) || throw(ArgumentError(
                "k-envelope term `$name` declares parity :$(parity == 1 ? "even" : "odd"), " *
                "but f(-p) = $vm while " *
                "$(parity == 1 ? "f(p)" : "-f(p)") = $(parity * v) at p = $(Tuple(p)). A " *
                "mis-declared parity gives a silently wrong hole block under " *
                "`nambu_doubling` -- declare the parity the callback actually has (split a " *
                "mixed-parity envelope into its even and odd parts)."))
        end
    end
    return nothing
end






#############################################################
###### Attaching quantum-number rules with ⊗            #####
#############################################################

"""
    term ⊗ rule

Attach a quantum-number rule to a term, composing with any rule it already carries.

    hopping(-t; shell=1) ⊗ conserve(:spin)
    onsite(B) ⊗ pauli_z(:spin)
    delta_g(w1; dG=[1,0]) ⊗ pauli_x(:sublattice) ⊗ raise(:layer)
"""
otimes(term::HoppingTerm, rule::AbstractQnumRule) =
    HoppingTerm(term.selector, compose(term.rule, rule), term.weight, term.t, term.name)

otimes(term::KLinearTerm, rule::AbstractQnumRule) =
    KLinearTerm(term.c, term.k0_frac, term.selector, compose(term.rule, rule),
                term.coefficient, term.name)

otimes(term::KQuadraticTerm, rule::AbstractQnumRule) =
    KQuadraticTerm(term.M, term.k0_frac, term.selector, compose(term.rule, rule),
                   term.coefficient, term.name)

otimes(term::KEnvelopeTerm, rule::AbstractQnumRule) =
    KEnvelopeTerm(term.f, term.parity, term.k0_frac, term.selector, compose(term.rule, rule),
                  term.coefficient, term.name)




#############################################################
###### User-facing term constructors                    #####
#############################################################

_to_amplitude(t::Number) = ComplexF64(t)
_to_amplitude(f::Function) = f

"""
    onsite(t; sites=Symbol[], weight=UnitWeight(), name="onsite") -> HoppingTerm

An on-site term: couples state pairs sharing a site, with `ΔR = 0`. `sites` may be a single name
or a list; empty means every site.

Combine with a rule to get the familiar objects:

    onsite(eps)                        # a uniform potential
    onsite(B) ⊗ pauli_z(:spin)         # Zeeman splitting
    onsite(M) ⊗ pauli_z(:sublattice)   # a sublattice mass
"""
function onsite(t; sites=Symbol[], weight::AbstractBondWeight=UnitWeight(),
                name::AbstractString="onsite")
    return HoppingTerm(OnSiteSelector(_site_list(sites)), AnyQnum(), weight,
                       _to_amplitude(t), String(name))
end

"""
    hopping(t; shell=1, sites_from=Symbol[], sites_to=Symbol[], weight=UnitWeight(), name) -> HoppingTerm
    hopping(t; rcut, sites_from=Symbol[], sites_to=Symbol[], weight=UnitWeight(), name) -> HoppingTerm

A tight-binding hopping, selected either by a distance **shell** (`shell = 1` is
nearest-neighbour, the default) or by a cartesian **cutoff radius** (`rcut=`, in the units of
the lattice vectors: every pair with `dist ≤ rcut`, all shells at once). The two are mutually
exclusive -- naming both raises.

`rcut` is the spelling for relaxed, corrugated or imported geometry, where the distance
spectrum is quasi-continuous and shells are not stable names: the neighbour search is then run
**ungrouped**, so the near-degenerate-shell warnings cannot fire, and the standard
tight-binding parametrization becomes one term:

    hopping((bi, bj, dR) -> t_sk(bi, bj, dR); rcut=8.0)          # relaxed TBG, one line
    hopping(t0; rcut=2.5, weight=ExponentialDecay(0.8))           # decay envelope over all pairs

`weight=` attaches a per-bond geometric factor, evaluated once at compile time (the contract
is [`AbstractBondWeight`](@ref)); shipped: [`UnitWeight`](@ref) (the default),
[`Chirality`](@ref), [`ChiralityPhase`](@ref), [`ExponentialDecay`](@ref) and
[`PeierlsPhase`](@ref). For a uniform orbital magnetic field -- a Hofstadter problem -- the
first-class route is the flux family: `weight=`[`uniform_flux`](@ref)`(cell, p//q; dir=...)`
compiled on the matching [`magnetic_cell`](@ref), never a hand-written phase.

`t` may be a number or a **function** `t(bs_i::BasisState, bs_j::BasisState, dR) -> Number`,
evaluated per bond with the two endpoint states and the integer translation -- the general
escape for distance- or position-dependent amplitudes no shipped weight covers (an exponential
interlayer profile, a custom gauge). On an undirected selector like this one a callback must
satisfy the conjugate-symmetry contract `t(bs_j, bs_i, -dR) == conj(t(bs_i, bs_j, dR))`; the
compiler verifies it per bond and raises otherwise (`onsite` and
[`translation_hopping`](@ref)/[`explicit_bonds`](@ref) accept the same callback form).
"""
function hopping(t; shell::Union{Nothing,Int}=nothing, rcut::Union{Nothing,Real}=nothing,
                 sites_from=Symbol[], sites_to=Symbol[],
                 weight::AbstractBondWeight=UnitWeight(), name::AbstractString="hopping")
    isnothing(shell) || isnothing(rcut) || throw(ArgumentError(
        "hopping `$name`: `shell=` and `rcut=` are mutually exclusive -- a hopping is selected " *
        "either by a distance shell or by a cutoff radius, never both. Drop one."))
    sel = isnothing(rcut) ?
          ShellSelector(something(shell, 1), _site_list(sites_from), _site_list(sites_to)) :
          RadiusSelector(rcut, _site_list(sites_from), _site_list(sites_to))
    return HoppingTerm(sel, AnyQnum(), weight, _to_amplitude(t), String(name))
end

"""
    translation_hopping(t; dR, sites_from=Symbol[], sites_to=Symbol[], weight, name) -> HoppingTerm

A hopping along one explicit Bravais translation, for cases where a distance shell is ambiguous.

`dR = 0` couples the state pairs of the home cell: same-site pairs behave exactly like `onsite`,
and cross-site pairs give the intra-cell coupling -- interlayer tunnelling at the same in-plane
position, say. Each such pair is counted **once** (`t` is the matrix element; the Hermitian
conjugate is supplied automatically), in the orientation `sites_from` → `sites_to` when filters
are given.

Direction convention, spelled out because a complex Peierls phase hangs on it: the stored bond
is `t = ⟨i, home | H | j, cell dR⟩` with the SOURCE state `j` (a `sites_from` site) in the cell
at `dR` and the TARGET state `i` (a `sites_to` site) in the home cell -- `t` multiplies
`c†_{i,0} c_{j,dR}`. Amplitude callbacks (see [`hopping`](@ref)) receive exactly that oriented
triple `(bs_i, bs_j, dR)`.
"""
function translation_hopping(t; dR::AbstractVector{<:Integer},
                             sites_from=Symbol[], sites_to=Symbol[],
                             weight::AbstractBondWeight=UnitWeight(),
                             name::AbstractString="translation")
    return HoppingTerm(TranslationSelector(Vector{Int}(dR), _site_list(sites_from), _site_list(sites_to)),
                       AnyQnum(), weight, _to_amplitude(t), String(name))
end

"""
    delta_g(t; dG, sites_from=Symbol[], sites_to=Symbol[], name) -> HoppingTerm

A plane-wave coupling `|G⟩ → |G + ΔG⟩`, the building block of moiré continuum models.

At `dG = 0` each pair is counted once, in whichever direction an attached ladder rule accepts --
`delta_g(w; dG=[0,0]) ⊗ raise(:layer)` is the ΔG = 0 harmonic of an interlayer tunnelling.
"""
function delta_g(t; dG::AbstractVector{<:Integer},
                 sites_from=Symbol[], sites_to=Symbol[],
                 name::AbstractString="delta_g")
    return HoppingTerm(DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to)),
                       AnyQnum(), UnitWeight(), _to_amplitude(t), String(name))
end

"""
    explicit_bonds(t; i, j, dR, name) -> HoppingTerm

A hand-written or imported bond list -- the escape hatch when no geometric selector fits.
(For a uniform orbital magnetic field, do NOT hand-write the phases: the first-class route is
`hopping(t; weight=`[`uniform_flux`](@ref)`(cell, p//q))` on the matching
[`magnetic_cell`](@ref).) Four facts fix the contract:

1. `i` and `j` are **basis-state indices** (positions in `cell.basis`), not site indices; the
   two coincide only for a trivial schema.
2. Direction: entry `(i, j, dR)` is the hop FROM state `j` in the cell at `dR` TO state `i` in
   the home cell, `t = ⟨i, home | H | j, dR⟩` multiplying `c†_{i,0} c_{j,dR}` -- the package's
   global bond convention. For a complex `t` (a Peierls phase) this orientation IS the sign.
3. Each bond appears **once**; the Hermitian conjugate is supplied automatically. Listing both
   orientations -- `(i, j, dR)` and `(j, i, -dR)`, the convention of many imported formats --
   would silently double the coupling, so the constructor refuses duplicates and reversed
   pairs. Halve an imported both-directions list first (`wannier90` does this for you).
4. `t` may be a single number for the whole list, or a callback `t(bs_i, bs_j, dR) -> Number`
   evaluated per bond (see [`hopping`](@ref) for the callback contract).
"""
function explicit_bonds(t; i::Vector{Int}, j::Vector{Int},
                        dR::Vector{<:AbstractVector{<:Integer}},
                        name::AbstractString="explicit")
    ### equal lengths are checked FIRST: the dedup scan below indexes all three arrays by the
    ### indices of `i`, so a mismatched hand-written list would otherwise die in a bare
    ### BoundsError that names neither the function nor the contract
    (length(i) == length(j) == length(dR)) || throw(DimensionMismatch(
        "explicit_bonds `$name`: `i`, `j` and `dR` must have equal lengths; got " *
        "$(length(i)), $(length(j)) and $(length(dR)). Each bond is one (i, j, dR) triple."))
    dRs = [Vector{Int}(r) for r in dR]
    ### the stored-once contract makes a repeated or reversed entry a silent factor-2 bug (the
    ### result stays perfectly Hermitian, so nothing downstream can catch it) -- refuse here
    seen = Dict{Tuple{Int,Int,Vector{Int}},Int}()
    for a in eachindex(i)
        key = (i[a], j[a], dRs[a])
        rkey = (j[a], i[a], -dRs[a])
        haskey(seen, key) && throw(ArgumentError(
            "explicit_bonds `$name`: entries $(seen[key]) and $a are the same bond " *
            "(i=$(i[a]), j=$(j[a]), dR=$(Tuple(dRs[a]))). Each bond is stored once; a repeat " *
            "would double the coupling. Merge the amplitudes into one entry."))
        haskey(seen, rkey) && throw(ArgumentError(
            "explicit_bonds `$name`: entries $(seen[rkey]) and $a are the two orientations of " *
            "one bond (the second is the Hermitian conjugate of the first). Each bond is " *
            "stored ONCE and the conjugate is supplied automatically -- listing both would " *
            "silently double the coupling. Drop one orientation (imported both-direction " *
            "lists must be halved first)."))
        seen[key] = a
    end
    return HoppingTerm(ExplicitBondSelector(i, j, dRs), AnyQnum(), UnitWeight(),
                       _to_amplitude(t), String(name))
end

"""
    kinetic(; mass=1.0, hbar=1.0, M=nothing, k0=Float64[], sites=Symbol[], name) -> KQuadraticTerm

The free-particle kinetic energy `ħ²|k + G - k0|² / 2m`, diagonal in the basis.

Pass an explicit symmetric matrix `M` (the coefficient of `pᵀMp`, e.g. an anisotropic inverse-mass
tensor already including `ħ²/2`) to override the isotropic `mass`/`hbar` form.
"""
function kinetic(; mass::Real=1.0, hbar::Real=1.0,
                 M::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                 k0::AbstractVector{<:Real}=Float64[],
                 sites=Symbol[], name::AbstractString="kinetic")
    Mstored = if isnothing(M)
        mass > 0 || throw(DomainError(mass,
            "`kinetic` describes an electron-like band and requires a positive mass. For a " *
            "HOLE band (a TMD valence band, say) write the quadratic term directly: " *
            "`k_quadratic(-hbar^2 / (2 * m_star))`."))
        Float64(hbar^2 / (2 * mass))
    else
        Matrix{Float64}(M)
    end
    return KQuadraticTerm(Mstored, Vector{Float64}(k0), OnSiteSelector(_site_list(sites)),
                          AnyQnum(), one(ComplexF64), String(name))
end

"""
    dirac(v; direction, k0=Float64[], hbar=1.0, sites=Symbol[], name) -> KLinearTerm

One cartesian component of a Dirac term: `ħ v p_direction`, to be tensored with a Pauli rule.

    dirac(v; direction=1) ⊗ pauli_x(:sublattice)
    dirac(v; direction=2) ⊗ pauli_y(:sublattice)
"""
function dirac(v::Real; direction::Int, k0::AbstractVector{<:Real}=Float64[],
               hbar::Real=1.0, sites=Symbol[], name::AbstractString="dirac")
    direction >= 1 || throw(DomainError(direction, "Direction must be a positive axis index."))
    return KLinearTerm(direction => Float64(hbar * v), Vector{Float64}(k0),
                       OnSiteSelector(_site_list(sites)), AnyQnum(), one(ComplexF64), String(name))
end

"""
    k_linear(c; coefficient=1.0, dG=nothing, k0=Float64[], sites=Symbol[],
             sites_from=Symbol[], sites_to=Symbol[], name="k_linear") -> KLinearTerm

The general momentum-linear term `coefficient * (c ⋅ p)` with `c` a real vector.

With `dG = nothing` the term is G-diagonal (an `OnSiteSelector`; `sites` filters it). With an
integer vector `dG` it becomes a **non-local** plane-wave coupling `|G⟩ → |G + ΔG⟩` whose
amplitude is linear in `k` -- a gradient correction to a moiré tunnelling, say -- evaluated at the
midpoint momentum (see [`emit_term!`](@ref)); `sites_from`/`sites_to` filter that form.

The complex `coefficient` carries any phase; `c` itself must be real (see `KLinearTerm`).
"""
function k_linear(c::AbstractVector{<:Real}; coefficient::Number=1.0,
                  dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                  k0::AbstractVector{<:Real}=Float64[],
                  sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                  name::AbstractString="k_linear")
    sel = isnothing(dG) ? OnSiteSelector(_site_list(sites)) :
          DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to))
    return KLinearTerm(Vector{Float64}(c), Vector{Float64}(k0), sel, AnyQnum(),
                       ComplexF64(coefficient), String(name))
end

"""
    k_quadratic(M; coefficient=1.0, dG=nothing, k0=Float64[], sites=Symbol[],
                sites_from=Symbol[], sites_to=Symbol[], name="k_quadratic") -> KQuadraticTerm

The general momentum-quadratic term `coefficient * (pᵀ M p)`. `M` may be a real symmetric matrix
or a scalar `λ` meaning `λ·I`. The `dG` keyword works exactly as in [`k_linear`](@ref): `nothing`
for a G-diagonal term, an integer vector for a non-local `|G⟩ → |G + ΔG⟩` coupling evaluated at
the midpoint momentum.
"""
function k_quadratic(M::Union{Real,AbstractMatrix{<:Real}}; coefficient::Number=1.0,
                     dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                     k0::AbstractVector{<:Real}=Float64[],
                     sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                     name::AbstractString="k_quadratic")
    sel = isnothing(dG) ? OnSiteSelector(_site_list(sites)) :
          DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to))
    Mstored = M isa Real ? Float64(M) : Matrix{Float64}(M)
    return KQuadraticTerm(Mstored, Vector{Float64}(k0), sel, AnyQnum(),
                          ComplexF64(coefficient), String(name))
end

"""
    k_envelope(f; coefficient=1.0, parity=:none, dG=nothing, k0=Float64[], sites=Symbol[],
               sites_from=Symbol[], sites_to=Symbol[], name="k_envelope") -> KEnvelopeTerm

A term whose k-dependence is an arbitrary **real envelope**: `coefficient * f(p)` with
`p = bᵀ(k + q)` the cartesian momentum at the entry's offset.

`f(p::SVector) -> Real` is evaluated at the state's own momentum for a G-diagonal term
(`dG = nothing`) and at the **midpoint** momentum for a `dG ≠ 0` coupling -- the two endpoint
momenta are then `p ± bᵀ(G_i − G_j)/2` (plus any origin difference), a constant shift per
origin pair. The scalar must be real: every phase belongs in `coefficient` or in the attached
quantum-number rule (a complex `f` raises at compile with that exact message). The name says the
contract -- this is an *envelope*, not an arbitrary complex function of k.

`parity` declares `f`'s behaviour under `p -> -p` (`:even`, `:odd`); it is what `nambu_doubling`
needs, and the default `:none` makes doubling refuse loudly rather than guess. When `f` is an
[`AbstractEnvelope`](@ref), a left-alone `parity=:none` defaults to the envelope's own
[`kernel_parity`](@ref) -- a serializable envelope never hits the doubling refusal by accident
(an envelope with no definite parity, `kernel_parity == 0`, keeps `:none`).

The twisted-bilayer use, a momentum-dependent interlayer tunnelling:

    T(p) = w1 * exp(-α * (norm(p) - kθ))
    k_envelope(T; dG=dG_n, coefficient=cis(-φ_n), parity=:even) ⊗ raise(:layer) ⊗ raise(:sub)

!!! note "Conventions agree on ±1 axes"
    On an axis whose two levels are `{+1, -1}` (either declaration order) the builders
    (`dirac_cone`, `moire_tunnelling`) resolve pseudospin-up = `+1`, the same convention as
    `raise`/`lower`'s defaults (`from = -1, to = +1`) -- so this hand-written spelling and the
    builder spelling always match there. On a NAMED two-level axis (`:band => [:c, :v]`) the
    builders take the first declared level as up, and `raise`/`lower` need explicit
    `from`/`to`; pass the builders' `up=`/`down=` when matching an external convention.

A raw closure cannot round-trip through HDF5, so `h5_write` refuses a model carrying one
(before touching the file); recompile from source, or use a registered
[`AbstractEnvelope`](@ref) ([`GaussianEnvelope`](@ref), [`TabulatedEnvelope`](@ref), or your
own via [`register_envelope!`](@ref)) -- those serialize exactly.
"""
function k_envelope(f::Function; coefficient::Number=1.0, parity::Symbol=:none,
                    dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                    k0::AbstractVector{<:Real}=Float64[],
                    sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                    name::AbstractString="k_envelope")
    ### a serializable envelope KNOWS its parity; defaulting from it keeps `moire_tunnelling`
    ### and hand-written spellings from hitting the `:none` doubling refusal by accident.
    ### An explicit :even/:odd always wins (and is still spot-checked at compile). An envelope
    ### with NO definite parity (kernel_parity == 0, e.g. `BlochSumEnvelope`) keeps :none.
    if parity === :none && f isa AbstractEnvelope
        pf = kernel_parity(f)
        parity = pf == 1 ? :even : pf == -1 ? :odd : :none
    end
    sel = isnothing(dG) ? OnSiteSelector(_site_list(sites)) :
          DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to))
    return KEnvelopeTerm(f, parity, Vector{Float64}(k0), sel, AnyQnum(),
                         ComplexF64(coefficient), String(name))
end




#############################################################
###### Builder-shaped display                           #####
#############################################################

### Terms and rules echo as the CONSTRUCTOR CALL that builds them -- the constructors are
### invertible, so the echo is the spelling the user typed (`hopping(-1.0 + 0.0im; shell=1)
### ⊗ conserve(:spin)`), and it parses back through `Meta.parse`. Display-only: nothing here
### is read by the compiler or the assembly.

### a callback amplitude/envelope has no literal spelling; `var"#12#34"` at least parses
function _show_callable(f::Function)::String
    f isa AbstractEnvelope && return repr(f)
    s = string(nameof(f))
    return Base.isidentifier(s) ? s : "var\"$s\""
end

_show_amp(t::ComplexF64)::String = repr(t)
_show_amp(f::Function)::String = _show_callable(f)

function _push_fromto!(kw::Vector{String}, from::Vector{Symbol}, to::Vector{Symbol})
    isempty(from) || push!(kw, "sites_from=$(repr(from))")
    isempty(to) || push!(kw, "sites_to=$(repr(to))")
    return kw
end

function _print_call(io::IO, head::String, arg::String, kw::Vector{String})
    print(io, head, "(", arg)
    isempty(kw) || print(io, "; ", join(kw, ", "))
    print(io, ")")
    return nothing
end

_show_rule_suffix(::IO, ::AnyQnum) = nothing
_show_rule_suffix(io::IO, r::AbstractQnumRule) = print(io, " ⊗ ", r)

function Base.show(io::IO, term::HoppingTerm)
    sel = term.selector
    kw = String[]
    if sel isa OnSiteSelector
        head, default = "onsite", "onsite"
        isempty(sel.sites) || push!(kw, "sites=$(repr(sel.sites))")
    elseif sel isa ShellSelector
        head, default = "hopping", "hopping"
        push!(kw, "shell=$(sel.shell)")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa RadiusSelector
        head, default = "hopping", "hopping"
        push!(kw, "rcut=$(sel.rcut)")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa TranslationSelector
        head, default = "translation_hopping", "translation"
        push!(kw, "dR=$(repr(sel.dR))")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa DeltaGSelector
        head, default = "delta_g", "delta_g"
        push!(kw, "dG=$(repr(sel.dG))")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa ExplicitBondSelector
        head, default = "explicit_bonds", "explicit"
        push!(kw, "i=$(repr(sel.i))", "j=$(repr(sel.j))", "dR=$(repr(sel.dR))")
    else
        ### a selector this file does not know (there are none today): fall back honestly
        head, default = "HoppingTerm", term.name
        push!(kw, "selector=$(repr(sel))")
    end
    term.weight isa UnitWeight || push!(kw, "weight=$(repr(term.weight))")
    term.name == default || push!(kw, "name=$(repr(term.name))")
    _print_call(io, head, _show_amp(term.t), kw)
    _show_rule_suffix(io, term.rule)
    return nothing
end

### the keywords shared by the k-term builders (coefficient, geometry, k0, name)
function _kterm_kwargs(term::KDependentTerm, default_name::String;
                       coefficient::Bool=true)::Vector{String}
    kw = String[]
    coefficient && _kcoefficient(term) != one(ComplexF64) &&
        push!(kw, "coefficient=$(repr(_kcoefficient(term)))")
    sel = _kselector(term)
    if sel isa DeltaGSelector
        push!(kw, "dG=$(repr(sel.dG))")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa OnSiteSelector
        isempty(sel.sites) || push!(kw, "sites=$(repr(sel.sites))")
    end
    isempty(_k0_raw(term)) || push!(kw, "k0=$(repr(_k0_raw(term)))")
    term_name(term) == default_name || push!(kw, "name=$(repr(term_name(term)))")
    return kw
end

function Base.show(io::IO, term::KLinearTerm)
    if term.c isa Pair{Int,Float64}
        ### the direction => value form only `dirac` builds (coefficient is 1 by construction)
        kw = String["direction=$(term.c.first)"]
        append!(kw, _kterm_kwargs(term, "dirac"; coefficient=false))
        _print_call(io, "dirac", repr(term.c.second), kw)
    else
        _print_call(io, "k_linear", repr(term.c), _kterm_kwargs(term, "k_linear"))
    end
    _show_rule_suffix(io, term.rule)
    return nothing
end

function Base.show(io::IO, term::KQuadraticTerm)
    _print_call(io, "k_quadratic", repr(term.M), _kterm_kwargs(term, "k_quadratic"))
    _show_rule_suffix(io, term.rule)
    return nothing
end

function Base.show(io::IO, term::KEnvelopeTerm)
    kw = String[]
    _kcoefficient(term) != one(ComplexF64) &&
        push!(kw, "coefficient=$(repr(_kcoefficient(term)))")
    term.parity === :none || push!(kw, "parity=$(repr(term.parity))")
    append!(kw, _kterm_kwargs(term, "k_envelope"; coefficient=false))
    _print_call(io, "k_envelope", _show_callable(term.f), kw)
    _show_rule_suffix(io, term.rule)
    return nothing
end

### terms are shown through the 2-arg echo in every context, never as raw struct dumps
Base.show(io::IO, ::MIME"text/plain",
          term::Union{HoppingTerm,KLinearTerm,KQuadraticTerm,KEnvelopeTerm}) = show(io, term)


# ----- rule echoes ------------------------------------------------------------------------

### levels are Symbols or Ints; `repr` spells both back (`:up`, `1`)
_show_qlevel(x)::String = repr(x)

Base.show(io::IO, ::AnyQnum) = print(io, "AnyQnum()")

Base.show(io::IO, r::ConserveQnum) =
    print(io, "conserve(", join((_show_qlevel(k) for k in r.keys), ", "), ")")

### `pauli_*`/`raise`/`lower` default to the {+1, -1} levels; print the kwargs only off that
function _print_two_level(io::IO, head::String, key::Symbol, up, down)
    if isequal(up, 1) && isequal(down, -1)
        print(io, head, "(", repr(key), ")")
    else
        print(io, head, "(", repr(key), "; up=", _show_qlevel(up),
              ", down=", _show_qlevel(down), ")")
    end
    return nothing
end

function Base.show(io::IO, r::DiagonalQnum)
    vals = sort!(collect(r.values); by = kv -> _show_qlevel(kv[1]))
    if length(vals) == 1 && vals[1][2] == one(ComplexF64)
        print(io, "project(", repr(r.key), ", ", _show_qlevel(vals[1][1]), ")")
    elseif length(vals) == 2 && Set(last.(vals)) == Set([one(ComplexF64), -one(ComplexF64)])
        up = vals[findfirst(kv -> kv[2] == one(ComplexF64), vals)][1]
        down = vals[findfirst(kv -> kv[2] == -one(ComplexF64), vals)][1]
        _print_two_level(io, "pauli_z", r.key, up, down)
    elseif length(vals) == 2 && all(kv -> kv[2] == one(ComplexF64), vals)
        ### pauli_0: both factors 1 -- up/down are interchangeable, prefer the value-keyed order
        up, down = vals[1][1], vals[2][1]
        (isequal(down, 1) || isequal(up, -1)) && ((up, down) = (down, up))
        _print_two_level(io, "pauli_0", r.key, up, down)
    else
        print(io, "DiagonalQnum(", repr(r.key), ", Dict(",
              join(("$(_show_qlevel(l)) => $(repr(f))" for (l, f) in vals), ", "), "))")
    end
    return nothing
end

function Base.show(io::IO, r::OffDiagonalQnum)
    prs = sort!(collect(r.pairs); by = kv -> _show_qlevel(kv[1][1]))
    if length(prs) == 1 && prs[1][2] == one(ComplexF64)
        (to, from) = prs[1][1]
        if isequal(to, 1) && isequal(from, -1)
            print(io, "raise(", repr(r.key), ")")
        elseif isequal(to, -1) && isequal(from, 1)
            print(io, "lower(", repr(r.key), ")")
        else
            print(io, "raise(", repr(r.key), "; from=", _show_qlevel(from),
                  ", to=", _show_qlevel(to), ")")
        end
        return nothing
    end
    if length(prs) == 2 && prs[1][1] == reverse(prs[2][1])
        (k1, v1), (k2, v2) = prs
        if v1 == one(ComplexF64) && v2 == one(ComplexF64)
            ### pauli_x: symmetric in up/down -- prefer the value-keyed labelling
            up, down = k1
            (isequal(down, 1) || isequal(up, -1)) && ((up, down) = (down, up))
            _print_two_level(io, "pauli_x", r.key, up, down)
            return nothing
        end
        if Set([v1, v2]) == Set([ComplexF64(im), ComplexF64(-im)])
            up, down = v1 == -im ? k1 : k2            # the (up, down) element of σ_y is -i
            _print_two_level(io, "pauli_y", r.key, up, down)
            return nothing
        end
    end
    print(io, "OffDiagonalQnum(", repr(r.key), ", Dict(",
          join(("($(_show_qlevel(a)), $(_show_qlevel(b))) => $(repr(f))"
                for ((a, b), f) in prs), ", "), "))")
    return nothing
end

Base.show(io::IO, r::MatrixQnum) =
    print(io, "matrix_rule(", repr(r.key), ", ", repr(r.M), ")")

### the callback has no literal spelling (same policy as callback amplitudes). The default
### order echoes the builder that produces it; the other two echo the full constructor
### honestly, since they are builder plumbing and each `LevelOrder` shows its own data.
Base.show(io::IO, r::DirectionalQnum{DeclaredOrder}) =
    print(io, "directional_rule(", _show_callable(r.f), ", ", repr(r.key), ")")

Base.show(io::IO, r::DirectionalQnum) =
    print(io, "DirectionalQnum(", repr(r.key), ", ", _show_callable(r.f), ", ",
          repr(r.order), ")")

Base.show(io::IO, ::DeclaredOrder) = print(io, "DeclaredOrder()")
Base.show(io::IO, o::TwoLevelOrder) =
    print(io, "TwoLevelOrder(", repr(o.up), ", ", repr(o.down), ")")
Base.show(io::IO, o::NamedOrder) = print(io, "NamedOrder(", repr(o.levels), ")")

Base.show(io::IO, r::TwoLevelMatrixQnum) =
    print(io, "TwoLevelMatrixQnum(", repr(r.key), ", ", repr(r.M), ", ",
          repr(r.up), ", ", repr(r.down), ")")

Base.show(io::IO, r::ProductQnumRule) = join(io, r.rules, " ⊗ ")
