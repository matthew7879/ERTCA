##################################################################
###### export only the following functions from this module ######
##################################################################

export HamiltonianModel, compile, reweight, term_names
export bloch_hamiltonian, bloch_hamiltonian!, bands, bands!
export change_gauge, gauge_matrix, bloch_periodicity, conserved_qnums, qnum_blocks
export has_kdependence, kperiodicity
export gauge, term_amplitudes




#############################################################
###### The model                                        #####
#############################################################

"""
    HamiltonianModel{D}

A single-particle Hamiltonian: a geometric `UnitCell` plus ONE compiled, Hermitian list of
couplings -- bonds and k-dependent entries alike, a bond being the constant-kernel special case.

The Hamiltonian is never stored as a matrix. It is stored as the flat sparse list
`(i, j, ΔR, q, w, kernel)` -- the representation Wannier90, tbmodels, pythtb and kwant all
converge on independently, extended by the momentum offset and the kernel column -- and `H(k)`
is built from it on demand, one branchless pass per kernel group:

    H_ij(k) += w · f_κ( bᵀ(k + q) ) · exp( 2πi (k + q)·δ )

Fields:
-------
- `cell::UnitCell{D}`: the geometry
- `couplings::CouplingList{D}`: the compiled, Hermitian, coalesced couplings. **Gauge-free.**
- `delta`: ATOMIC-convention phase displacements, parallel to `couplings` (the only stored
  convention; see `_phase_displacements`)
- `gauge::Symbol`: `:atomic` (default) or `:periodic`

The k-periodicity class is NOT a field: it is the cell's type parameter `K`, so
[`kperiodicity`](@ref) reads it off the type and model and cell cannot disagree. It was a
cached `Symbol` here until 2026-09-17, which made the invariant sometimes derived and sometimes
carried -- `fold` had to rebuild through the private inner constructor to correct it
which left the invariant able to disagree with itself.
- `name::String`

The model deliberately keeps NO symbolic term list: the compiled couplings are the single truth
(a provenance list was empty after every transform and every read anyway, and nothing could
re-derive terms from it). What the couplings DO carry is a per-entry term tag and the term
registry (name + scalar compile-time amplitude), which survives every transform -- enough for
[`reweight`](@ref) to rescale a named term's amplitude in O(couplings) without recompiling.
Recompile from source to change anything structural.

On the gauge:
-------------
The gauge lives on the model rather than on a per-call keyword, because it is a property of the
*representation*: it changes the eigenvectors. A keyword would invite silently mixing conventions
between a band-structure call and a Berry-phase call on the same object. The STORED assembly is
always the atomic convention; the flag only selects whether the assembly exit conjugates by the
diagonal translator `W(k)` ([`gauge_matrix`](@ref)) -- so `change_gauge` is an O(1) flag flip,
and the periodic view can never drift from the atomic truth. Physical-curvature consumers
(`berry_curvature` & co.) read the atomic assembly regardless of the flag.
"""
struct HamiltonianModel{D,K<:KPeriodicity}
    cell::UnitCell{D,K}
    couplings::CouplingList{D}
    delta::Vector{SVector{D,Float64}}
    gauge::Symbol
    name::String

    function HamiltonianModel{D,K}(cell::UnitCell{D,K}, couplings::CouplingList{D},
                                   delta::Vector{SVector{D,Float64}}, gauge::Symbol,
                                   name::String) where {D,K<:KPeriodicity}
        gauge in GAUGE_MODES || throw(ArgumentError(
            "Unknown gauge `$gauge`; expected one of $GAUGE_MODES."
        ))
        couplings.n == n_states(cell) || throw(DimensionMismatch(
            "CouplingList dimension $(couplings.n) does not match n_states(cell) = $(n_states(cell))."
        ))
        length(delta) == n_couplings(couplings) || throw(DimensionMismatch(
            "Phase-displacement table has length $(length(delta)) but the coupling list has " *
            "$(n_couplings(couplings))."
        ))
        return new{D,K}(cell, couplings, delta, gauge, name)
    end
end

"""
    HamiltonianModel(cell::UnitCell{D}, buf::CouplingBuffer{D};
                     gauge=DEFAULT_GAUGE, name="", atol=AMPLITUDE_ATOL, prune=false)
        -> HamiltonianModel{D}

Finalize a buffer of emitted couplings into a model -- the ONE public gate from raw couplings
to a `HamiltonianModel`, and the seam every transform (shipped or user-written) goes through.

It runs [`finalize_couplings`](@ref) (canonicalize, coalesce, check on-site reality,
materialize every Hermitian partner), then derives the phase-displacement table and the
k-periodicity from the cell. `delta` and `kperiodic` are **derived here, never accepted**:
there is deliberately no way to supply them from outside, so the phase-table and periodicity
invariants cannot be violated by anything built through this door.

Args:
-----
- `cell`: the unit cell the couplings live on; state indices in `buf` address its basis
- `buf`: the coupling buffer -- push ONE direction of every Hermitian pair only (see below)
- `gauge`: `:atomic` (default) or `:periodic`
- `name`: the model's name
- `atol`: tolerance for the reality check on self-conjugate entries
- `prune`: drop couplings whose coalesced amplitude falls below `atol`

Writing a transform:
--------------------
Every shipped model transform (`supercell`, `fold`, `open_boundary`, `nambu_doubling`,
`discretize`) is the same five-step recipe over the compiled list -- entirely public, and the
shape any new transform (strain, vacancy projection, symmetrization) should take:

    function my_transform(model::HamiltonianModel{D}, ...) where {D}
        new_cell = ...                                    # build the transformed UnitCell
        c = model.couplings
        buf, kids = adopted_buffer(c)                     # 1.+2. registry adopted, kernels
                                                          #       re-registered, capacity hinted
        for cpl in each_canonical(c)                      # 3. canonical halves ONLY
            i2, j2, dR2, q2, w2 = ...                     # 4. map the entry onto the new cell
            push_coupling!(buf, i2, j2, dR2, q2, w2, kids[cpl.kernel], cpl.term)
        end
        return HamiltonianModel(new_cell, buf; gauge=model.gauge, name=model.name)  # 5. gate
    end

([`adopted_buffer`](@ref) is the one-call spelling of the former three-line preamble --
`CouplingBuffer` + `adopt_provenance!` + a `kernel_id!` re-registration loop -- which every
shipped transform now shares; the manual spelling remains valid.)

The one hazard worth restating (it is also on [`each_canonical`](@ref)): the source list
stores BOTH directions of every Hermitian pair. Map ONLY the canonical halves -- the finalize
inside this constructor re-materializes every partner, and mapping both directions coalesces
each coupling with its own image and silently doubles every amplitude.

Notes:
------
[`compile`](@ref) is this constructor with the term-emission pipeline in front of it. A
transform that also adds NEW terms registers them with [`begin_term!`](@ref) *after* adopting
the source's provenance; their couplings land in the same buffer before the finalize. To wrap
an already-finished, both-directions list (a file reader, an importer), use the
`HamiltonianModel(cell, cl::CouplingList)` method instead, which checks Hermiticity rather
than re-finalizing.
"""
function HamiltonianModel(cell::UnitCell{D}, buf::CouplingBuffer{D};
                          gauge::Symbol=DEFAULT_GAUGE, name::AbstractString="",
                          atol::Float64=AMPLITUDE_ATOL,
                          prune::Bool=false)::HamiltonianModel{D} where {D}
    couplings = finalize_couplings(buf, n_states(cell); atol=atol, prune=prune)
    delta = _phase_displacements(cell, couplings)
    return HamiltonianModel{D,_kpar(cell)}(cell, couplings, delta, gauge,
                               String(name))
end

"""
    HamiltonianModel(cell::UnitCell{D}, cl::CouplingList{D};
                     gauge=DEFAULT_GAUGE, name="", atol=AMPLITUDE_ATOL) -> HamiltonianModel{D}

Wrap a FINISHED coupling list into a model -- the door for lists that did not come out of this
process's own `finalize_couplings`: a file reader, an importer, a hand-assembled list.

`finalize_couplings` guarantees Hermiticity by construction for lists IT built; a finished
list is input. `bands` skips its per-k Hermiticity check for constant-only models on that very
guarantee, and `Hermitian(H)` reads one triangle, so a broken list would be silently
symmetrized into a wrong spectrum. The list is therefore checked once, here, with
[`is_hermitian`](@ref): a list in which some `(i, j, ΔR, q)` entry lacks or contradicts its
`(j, i, -ΔR, q)` partner is refused by name. Both directions of every pair must be present --
to build a model from one-direction couplings, push them into a [`CouplingBuffer`](@ref) and
use the buffer method, whose finalize materializes the partners.
"""
function HamiltonianModel(cell::UnitCell{D}, cl::CouplingList{D};
                          gauge::Symbol=DEFAULT_GAUGE, name::AbstractString="",
                          atol::Float64=AMPLITUDE_ATOL)::HamiltonianModel{D} where {D}
    is_hermitian(cl; atol=atol) || throw(ArgumentError(
        "The coupling list is not Hermitian: some (i, j, ΔR, q) entry lacks or contradicts " *
        "its (j, i, -ΔR, q) partner. A finished `CouplingList` must carry both directions of " *
        "every Hermitian pair exactly -- a corrupt or hand-edited file usually lands here. " *
        "Fix the list, or push one direction per pair into a `CouplingBuffer` and build " *
        "`HamiltonianModel(cell, buffer)`, whose finalize materializes the partners."))
    delta = _phase_displacements(cell, cl)
    return HamiltonianModel{D,_kpar(cell)}(cell, cl, delta, gauge,
                               String(name))
end

Base.length(m::HamiltonianModel) = n_states(m.cell)
ndim(::HamiltonianModel{D}) where {D} = D

"""
    has_kdependence(model) -> Bool

Whether `H(k)` carries any k-dependence beyond the Bloch phases -- i.e. whether any non-constant
kernel holds entries. Bond-only models skip the per-k Hermiticity check on this guarantee.
"""
has_kdependence(m::HamiltonianModel)::Bool = has_kdependence(m.couplings)

"""
    kperiodicity(model) -> KPeriodicity

How `H(k)` relates to the Brillouin-zone torus: [`ExactlyPeriodic`](@ref),
[`TruncatedWindow`](@ref) or [`NoZone`](@ref), each documented on its own type.

Derived from the CELL at compile -- `kperiodicity(cell)` answers the same question with no
Hamiltonian in hand -- and cached here. The join rule for mixed cells is `NoZone` dominates,
then `TruncatedWindow`, then `ExactlyPeriodic`.
"""
kperiodicity(::HamiltonianModel{D,K}) where {D,K} = K()

### the class as a TYPE, for containers and structs that must keep a concrete element type
_model_kpar(::HamiltonianModel{D,K}) where {D,K} = K





#############################################################
###### Compilation                                      #####
#############################################################

"""
    _phase_displacements(cell, couplings) -> Vector{SVector{D,Float64}}

Precompute the fractional displacement whose dot product with `k + q` gives each coupling's
ATOMIC-convention phase: `δ = ΔR + τ_j - τ_i` with the effective (kind-aware) positions.

The atomic convention is the ONLY stored representation: the amplitudes `w` are the physical
matrix elements and `delta` matches them, which is the invariant every transform, Hermiticity
check and amplitude callback relies on. The `:periodic` view is produced at the single assembly
exit by the diagonal gauge translator `W(k)` (see [`gauge_matrix`](@ref) and
[`bloch_hamiltonian!`](@ref)) -- derived, never stored, so it cannot drift out of sync with the
atomic truth (the folded-cell wrap harmonics did exactly that when both gauges materialized
their own displacement tables).

Everything is in fractional coordinates: with rows of `a` as primitive vectors,
`k_cart ⋅ R_cart = 2π (k_frac ⋅ R_frac)` identically, so no lattice metric ever enters the
phase. `ΔR` is rational (fractional after a `fold`) and converts exactly here.
"""
function _phase_displacements(cell::UnitCell{D},
                              c::CouplingList{D})::Vector{SVector{D,Float64}} where {D}
    tau = _effective_positions(cell)
    delta = Vector{SVector{D,Float64}}(undef, n_couplings(c))
    for e in eachindex(c.i)
        delta[e] = SVector{D,Float64}(c.dR[e]) + tau[c.j[e]] - tau[c.i[e]]
    end
    return delta
end

"""
    _effective_positions(cell) -> Vector{SVector{D,Float64}}

The per-state positions that may enter a Bloch PHASE: the declared position for a `:localized`
state, and **zero** for a `:plane_wave` state -- a plane wave has no centre, its declared
position is gauge-only, and the compiler must never let it enter a phase. Without this, a
`ΔG` loop over three or more plane-wave sites at distinct declared positions would make the
two gauges physically different Hamiltonians (the per-entry factor `e^{2πi(k+q)·(τ_j-τ_i)}`
with midpoint `q` is not a similarity transform around such a loop).
"""
function _effective_positions(cell::UnitCell{D})::Vector{SVector{D,Float64}} where {D}
    z = zero(SVector{D,Float64})
    return [cell.site_kind[cell.state_site[i]] === :plane_wave ? z : b.r_frac
            for (i, b) in pairs(cell.basis)]
end

"""
    _all_localized_at_origin(cell) -> Bool

`true` when the cell carries more than one localized state and every one of them sits at the
origin. In that configuration the atomic assembly coincides entrywise with the zero-position
(periodic) convention, so no position-dependent observable can tell whether the stored
positions are the physical ones: genuine for a single multi-orbital site whose states really
do share one centre, and the classic missing-`centres=` artefact on a Wannier90 import. The
position-dependent observables (`berry_curvature`, `spectral_weight`) warn on this predicate --
the observable-layer relocation of the importer's atomic-gauge refusal. Plane-wave states are
ignored: their positions are gauge-only and never enter a phase (`_effective_positions`).
"""
function _all_localized_at_origin(cell::UnitCell{D})::Bool where {D}
    nloc = 0
    for (i, b) in pairs(cell.basis)
        cell.site_kind[cell.state_site[i]] === :plane_wave && continue
        nloc += 1
        _same_site(b.r_frac, zero(SVector{D,Float64}), cell.atol) || return false
    end
    return nloc > 1
end

"""
    compile(cell, terms; gauge=DEFAULT_GAUGE, name="", rcut=nothing, atol, prune) -> HamiltonianModel{D}

Lower a collection of terms into a `HamiltonianModel`.

Args:
-----
- `cell`: the unit cell
- `terms`: the Hamiltonian terms
- `gauge`: `:atomic` (default) or `:periodic`
- `rcut`: explicit neighbour cutoff; by default the table is grown to reach the deepest shell
  any term requests, and covers at least the largest `max_radius` any `rcut=` hopping asks for
- `atol`: tolerance for the Hermiticity/reality checks
- `prune`: drop couplings whose coalesced amplitude falls below `atol`

Returns:
--------
- `HamiltonianModel{D}`

Notes:
------
Every term -- constant or k-dependent -- lowers through ONE emission pipeline (`emit_term!`)
into one coupling list; there are no parallel bond/k-term paths left to drift apart.

The neighbour table is built **only** if some term requests a shell, so continuum models never
pay for a distance search they do not use.

A term that emits no couplings triggers a warning naming it. That case is almost always a
quantum-number mismatch -- `pauli_z(:spin)` applied to a spinless cell yields factor 0 everywhere
and the term silently vanishes -- and naming it turns a subtle physics bug into an immediate fix.
"""
function compile(cell::UnitCell{D}, terms::Union{AbstractVector,Tuple};
                 gauge::Symbol=DEFAULT_GAUGE,
                 name::AbstractString="",
                 rcut::Union{Nothing,Real}=nothing,
                 atol::Float64=AMPLITUDE_ATOL,
                 prune::Bool=false)::HamiltonianModel{D} where {D}
    gauge in GAUGE_MODES || throw(ArgumentError("Unknown gauge `$gauge`; expected one of $GAUGE_MODES."))
    termvec = _flatten_terms(terms)

    tables = cell_tables(cell)

    # Build the neighbour table only when some term actually needs one. On a FOLDED cell the
    # search runs on the fine (parent) lattice, where the physical neighbours live.
    # `max_radius` is the metric analog of `max_shell` (RadiusSelector / `hopping(t; rcut=)`):
    # when only radius terms consume the table it is built UNGROUPED, so the shell-degeneracy
    # warnings cannot fire on the quasi-continuous distance spectra radius terms exist for.
    deepest = isempty(termvec) ? 0 : maximum(max_shell, termvec)
    rmax = isempty(termvec) ? 0.0 : maximum(max_radius, termvec)
    neighbors = if deepest == 0 && rmax == 0.0
        nothing
    elseif is_folded(cell)
        _fold_fine_neighbor_table(cell, deepest;
                                  rcut=rmax > 0.0 ? max(rmax, something(rcut, 0.0)) : rcut)
    elseif rmax > 0.0
        r = max(rmax, something(rcut, 0.0))
        if deepest > 0
            ### Shell terms are present too. Build ONE table at the radius the radius terms
            ### need and ask whether it already resolves the deepest requested shell -- that is
            ### `neighbor_table_by_shell`'s own completeness test (`> deepest` radii, so the
            ### deepest one lies strictly inside the searched radius and is complete). It
            ### almost always does, because a `rcut=` term reaches further than the shells a
            ### model names. Only when it does not do we pay for the shell probe and rebuild:
            ### the probe used to run unconditionally and its table was then thrown away.
            nt = neighbor_table(cell, r; tables=tables, warn_degenerate=false)
            if length(nt.radii) > deepest
                _warn_near_degenerate(nt.radii, distance_atol(cell.lattice); rtol=SHELL_RTOL)
                nt
            else
                rs = neighbor_table_by_shell(cell, deepest; tables=tables,
                                             warn_degenerate=false).rcut
                neighbor_table(cell, max(r, rs); tables=tables)
            end
        else
            neighbor_table(cell, r; tables=tables, group_shells=false)
        end
    elseif isnothing(rcut)
        neighbor_table_by_shell(cell, deepest; tables=tables)
    else
        neighbor_table(cell, rcut; tables=tables)
    end

    ctx = CompileContext(cell; tables=tables, neighbors=neighbors)

    buf = CouplingBuffer{D}()
    for term in termvec
        before = length(buf)
        ### register the term's provenance: every coupling it emits is tagged with its id,
        ### and the registry records its scalar amplitude for `reweight`
        begin_term!(buf, term_name(term), _scalar_amplitude(term))
        emit_term!(buf, term, ctx)
        # A term that matched nothing *structurally* is a modelling bug worth naming. A term
        # whose amplitude -- or whose rule MATRIX -- is literally zero is not: sweeping a
        # parameter through 0 is routine, and warning there would make every phase diagram
        # noisy (and misread a U = 0 endpoint as a spelling error).
        if length(buf) == before && !_has_zero_amplitude(term) &&
           !_has_zero_rule(term, cell.schema)
            @warn "Term `$(term_name(term))` emitted no couplings. This usually means a " *
                  "quantum-number rule rejected every pair -- for example a rule on `:spin` " *
                  "applied to a cell that carries no `:spin` quantum number."
        end
    end

    ### the same public gate every transform uses: finalize, then derive delta and kperiodic
    return HamiltonianModel(cell, buf; gauge=gauge, name=name, atol=atol, prune=prune)
end

compile(cell::UnitCell{D}, terms::AbstractHamiltonianTerm...; kwargs...) where {D} =
    compile(cell, collect(terms); kwargs...)

"""
    compile(model, terms; atol=AMPLITUDE_ATOL, prune=false) -> HamiltonianModel{D}

Extend a COMPILED model with new terms: every coupling and the whole term registry of `model`
are adopted, `terms` are compiled on `model.cell` into the same buffer, and one finalize builds
the merged model. Gauge and name are inherited. The result equals what
`compile(model.cell, all_terms_ever)` would build -- except that it also works when the
parent's terms are gone: a Wannier90 import, a colleague's HDF5 file, the far end of a
transform chain. "Import a DFT-derived model, add a field, look at the response" is:

    m  = read_wannier90_hr(cell, "silicon_hr.dat")     # couplings only: no term source
    ms = add_axis(m, :spin, [+1, -1])                  # give the import a spin axis
    mz = compile(ms, [onsite(0.05; name="zeeman") ⊗ pauli_z(:spin)])
    up = reweight(mz, "zeeman" => 0.10)                # sweep the field, no recompile

Provenance:
-----------
The parent's per-entry term tags are carried through with their orientation SIGN preserved
(negative = the stored amplitude is the conjugate of the emitted one; see `CouplingList`), and
the new terms' registry ids are spliced after the adopted block -- the `nambu_doubling` pairing
splice, generalized. [`reweight`](@ref) therefore keeps addressing every parent term AND every
new term, complex factors included.

On a parent with NO provenance (imported, or read from a legacy file) the adopted couplings
carry tag 0 and stay un-reweightable, while the NEW terms are reweightable. That asymmetry is
the point: the import is frozen data; the perturbations added on top are the parameters being
swept.

A new term whose name collides with an adopted one is CO-ADDRESSED by `reweight` -- several
same-named terms rescale together, the documented registry semantics -- which is almost never
what a collision means here, so it triggers an `@warn` naming the term. (`nambu_doubling` and
`direct_sum` instead RENAME the colliding side -- `"bdg_chemical_potential"`, `"pairing_onsite"`
-- because there the two registries come from two models being joined, and neither spelling is
the one the caller just typed. Here the new term's name IS what the caller just typed, so this
method warns rather than renaming under them.) Give the new term a distinct `name=`.

Notes:
------
`terms` go through the ordinary emission pipeline on `model.cell` -- neighbour table grown to
the deepest shell any new term requests, zero-emission warning included -- exactly as in
`compile(cell, terms)`. Anything structural (a new site, a new axis) is a cell transform, not
a term: use [`add_axis`](@ref) or recompile from source.
"""
function compile(model::HamiltonianModel{D}, terms::Union{AbstractVector,Tuple};
                 atol::Float64=AMPLITUDE_ATOL,
                 prune::Bool=false)::HamiltonianModel{D} where {D}
    termvec = _flatten_terms(terms)
    c = model.couplings

    ### ---- adopt the parent: registry first, then the canonical halves verbatim ------------
    ### (each_canonical + finalize-in-the-gate is the standard recipe; pushing both stored
    ### directions would coalesce each coupling with its own image and double every amplitude)
    buf, kids = adopted_buffer(c)
    for cpl in each_canonical(c)
        push_coupling!(buf, cpl.i, cpl.j, cpl.dR, cpl.q, cpl.w, kids[cpl.kernel], cpl.term)
    end

    ### ---- the new terms: ordinary compile on the parent's cell, then the registry splice ---
    if !isempty(termvec)
        for term in termvec
            tn = term_name(term)
            tn in c.term_names && @warn(
                "compile: the new term `$tn` shares its name with a term adopted from the " *
                "parent model, so `reweight(m, \"$tn\" => ...)` will co-address BOTH (every " *
                "term with a matching name rescales together). Give the new term a distinct " *
                "`name=` to keep them separately sweepable.")
        end
        em = compile(model.cell, termvec; gauge=model.gauge, atol=atol)
        ec = em.couplings
        ekids = [kernel_id!(buf, kern) for kern in ec.kernels]
        tbase = length(buf.term_names)
        append!(buf.term_names, ec.term_names)
        append!(buf.term_amps, ec.term_amps)
        for cpl in each_canonical(ec)
            ### the splice shifts the term-tag MAGNITUDE by tbase; the tag's SIGN is the
            ### entry's orientation and must survive -- `-t + tbase` would silently point a
            ### flipped-orientation new term, positively, at an adopted parent term
            tid = cpl.term == 0 ? 0 : (cpl.term > 0 ? cpl.term + tbase : cpl.term - tbase)
            push_coupling!(buf, cpl.i, cpl.j, cpl.dR, cpl.q, cpl.w, ekids[cpl.kernel], tid)
        end
    end

    return HamiltonianModel(model.cell, buf; gauge=model.gauge, name=model.name,
                            atol=atol, prune=prune)
end

compile(model::HamiltonianModel{D}, terms::AbstractHamiltonianTerm...; kwargs...) where {D} =
    compile(model, collect(terms); kwargs...)

"""
    add_axis(model, key, levels) -> HamiltonianModel{D}

Add an internal axis to a COMPILED model: the cell gains the axis (the `UnitCell` method of
[`add_axis`](@ref), level-outer per `STATE_ORDER`), and every coupling is replicated
diagonally over the new levels -- with `n = n_states(model.cell)` and `L = length(levels)`,
each canonical `(i, j, ΔR, q, w, κ)` is pushed at `(i + (l-1)n, j + (l-1)n)` for every
`l in 1:L`. The Hamiltonian is unchanged physics on a bigger basis: `H_new(k) = 1_L ⊗ H(k)`
in the level-outer ordering, so the spectrum is the parent's, `L`-fold degenerate, until
axis-addressing terms are added (typically via `compile(model, extras)`):

    ms = add_axis(m, :spin, [+1, -1])                         # spectrum unchanged, doubled
    mz = compile(ms, [onsite(0.05; name="zeeman") ⊗ pauli_z(:spin)])   # now split

`q` rides along UNCHANGED: the replicated states carry the parent's `G` and `k0` verbatim, so
the endpoint-derived momentum offset of every entry is the same on every level. (Contrast
`nambu_doubling`, whose hole block flips `-q` -- that flip encodes `-h(-k)ᵀ`, a Hamiltonian
transform, not the basis replication happening here.) Term tags are carried through with their
orientation sign, so [`reweight`](@ref) addresses each term across ALL levels at once, complex
factors included. Gauge and name are inherited.

Refused, by name: an axis the cell already declares -- replicating over it again would nest
label spaces (the `nambu_doubling` refusal, shared reasoning). Pick another name if `key`
means something else in this model.
"""
function add_axis(model::HamiltonianModel{D}, key::Symbol,
                  lvls::AbstractVector)::HamiltonianModel{D} where {D}
    has_axis(model.cell.schema, key) && throw(ArgumentError(
        "The cell already declares an internal axis `$key` (levels " *
        "$(levels(model.cell.schema, key))) -- adding it twice would nest label spaces. Pick " *
        "another `key` name if `$key` means something else in this model."))

    n = n_states(model.cell)
    new_cell = add_axis(model.cell, key, lvls)      # level-outer: level l of state i = i+(l-1)n
    L = n_levels(new_cell.schema, n_axes(new_cell.schema))

    c = model.couplings
    buf, kids = adopted_buffer(c; per_canonical=L)
    for cpl in each_canonical(c)
        for l in 1:L
            off = (l - 1) * n
            ### same dR, q, w, kernel and SIGNED term tag on every level: pure diagonal
            ### replication (no -q, no conj -- those belong to nambu's hole block, not here)
            push_coupling!(buf, cpl.i + off, cpl.j + off, cpl.dR, cpl.q, cpl.w,
                           kids[cpl.kernel], cpl.term)
        end
    end
    return HamiltonianModel(new_cell, buf; gauge=model.gauge, name=model.name)
end

"""
    term_names(model) -> Vector{String}

The distinct term names registered in the model's provenance -- exactly the names
[`reweight`](@ref) accepts. Builders may register several suffixed names per `name=` they were
given (`moire_tunnelling`: `<name>_T<n>_{AA,AB,BA}`; `dirac_cone`: `<name>_x`/`<name>_y`).
Empty for models without provenance (imported, or read from a legacy file).
"""
term_names(model::HamiltonianModel)::Vector{String} = unique(model.couplings.term_names)

"""
    gauge(model) -> Symbol

The model's Bloch gauge: `:atomic` (physical positions in the phases, the stored convention)
or `:periodic` (the view produced by the diagonal translator [`gauge_matrix`](@ref)). Flip it
with [`change_gauge`](@ref), an O(1) flag change.
"""
gauge(model::HamiltonianModel)::Symbol = model.gauge

"""
    n_couplings(model::HamiltonianModel) -> Int

Number of stored couplings of the compiled model, counting both directions of every Hermitian
pair and every kernel -- `n_couplings(model.couplings)`.
"""
n_couplings(model::HamiltonianModel)::Int = n_couplings(model.couplings)

"""
    term_amplitudes(model) -> Vector{Pair{String,ComplexF64}}

The provenance registry as `name => scalar compile-time amplitude` pairs, in registry order
with duplicate names dropped (the FIRST registered amplitude is kept -- give terms distinct
names to see each). `NaN` marks a term with no scalar amplitude (a callback). Empty for
models without provenance (imported, or read from a legacy file). These are exactly the
names [`reweight`](@ref) accepts, with the amplitudes its absolute mode divides by.
"""
function term_amplitudes(model::HamiltonianModel)::Vector{Pair{String,ComplexF64}}
    cl = model.couplings
    out = Pair{String,ComplexF64}[]
    seen = Set{String}()
    for (t, nm) in pairs(cl.term_names)
        nm in seen && continue
        push!(seen, nm)
        push!(out, nm => cl.term_amps[t])
    end
    return out
end

"""
    _scalar_amplitude(term) -> ComplexF64

The term's scalar amplitude, as recorded in the compiled provenance registry for
[`reweight`](@ref) -- every emitted coupling is LINEAR in this number. `NaN` when the term
has no scalar amplitude (a callback), which makes absolute reweighting refuse it by name.
"""
_scalar_amplitude(term::BondTerm)::ComplexF64 =
    term.t isa Function ? ComplexF64(NaN, NaN) : ComplexF64(term.t)
_scalar_amplitude(term::KDependentTerm)::ComplexF64 = ComplexF64(term.coefficient)
_scalar_amplitude(::AbstractHamiltonianTerm)::ComplexF64 = ComplexF64(NaN, NaN)

"""
    reweight(model, pairs::Pair...; relative=false) -> HamiltonianModel

A new model in which the named terms carry new amplitudes -- the O(couplings) alternative to
recompiling when ONLY amplitudes change (a phase-diagram sweep, a self-consistent loop).
Everything shared is reused: the cell, the phase-displacement table, the neighbour geometry
baked into the couplings -- and the sparsity pattern: a [`sparse_bloch_workspace`](@ref)
built for the source model accepts every reweighted (and `change_gauge`d) model, so a sweep
pays for the CSC pattern once.

    m2 = reweight(m, "nnn" => 0.3im)               # the t₂ term now has amplitude 0.3i
    m3 = reweight(m, "mass" => 2.0; relative=true) # every `mass` coupling scaled ×2

Args:
-----
- `pairs`: `name => value`. With `relative=false` (default) `value` is the term's NEW scalar
  amplitude: its couplings are scaled by `value / amp₀` with `amp₀` the amplitude recorded at
  compile time. With `relative=true`, `value` is a bare scale factor -- this works for every
  term with a NONZERO amplitude, including callback amplitudes, because emitted couplings are
  linear in the amplitude.
- Every term whose registered NAME matches is rescaled; several terms may share a name.
  Discover the registered names with [`term_names`](@ref) (builders may suffix the `name=`
  they were given -- `moire_tunnelling` registers `<name>_T<n>_{AA,AB,BA}` per harmonic and
  sublattice channel, `dirac_cone` registers `<name>_x`/`<name>_y`).
- Repeated names in one call COMPOSE: `reweight(m, "x" => 2.0, "x" => 3.0; relative=true)`
  scales by 6.
- Reweight is MULTIPLICATIVE, so zero is absorbing: a term at amplitude 0 (compiled there, or
  driven there) cannot be rescaled back -- absolute mode refuses, relative mode warns and
  no-ops. Sweeps through 0 should anchor every point on the original nonzero compile, not on
  the previous sweep point.

Complex factors are exact: every stored entry's term tag carries an orientation sign
(negative = the stored amplitude is the conjugate of the emitted one -- a canonicalization
flip, a Hermitian partner, a BdG hole entry), and negative-tagged entries scale by `conj(f)`.
A Peierls-phase sweep on `explicit_bonds` written in either direction, or on a Nambu-doubled
model, therefore reweights identically to a recompile.

Refused, by name, rather than silently mis-scaled:
- an unknown term name (the error lists the registered names);
- absolute mode on a term with no recorded scalar amplitude (a callback -- use
  `relative=true`), with a zero recorded amplitude (the couplings are all zero; there is
  nothing to scale -- recompile), or on several same-named terms with DIFFERENT recorded
  amplitudes (say which one you mean by giving terms distinct names);
- a non-real factor on a term with on-site (self-conjugate) couplings, which must stay real;
- a model with no provenance at all (imported from Wannier90 or a legacy HDF5 file).
"""
function reweight(model::HamiltonianModel{D}, pairs::Pair...;
                  relative::Bool=false)::HamiltonianModel{D} where {D}
    isempty(pairs) && return model
    cl = model.couplings
    isempty(cl.term_names) && throw(ArgumentError(
        "This model carries no term provenance (imported or legacy-file models do not), so " *
        "`reweight` cannot identify any term. Recompile from terms to rescale amplitudes."))

    factor = ones(ComplexF64, length(cl.term_names))     # per-term scale, identity default
    amps = copy(cl.term_amps)
    for (name, value) in pairs
        ids = findall(==(String(name)), cl.term_names)
        isempty(ids) && throw(ArgumentError(
            "No term named `$name` in this model. Registered terms: " *
            "$(join(unique(cl.term_names), ", "))."))
        if relative
            ### a zero-amplitude term (compiled at 0, or reweighted to 0) has no couplings to
            ### scale: 0·f = 0 for every f, so this would be a SILENT no-op mid-sweep
            any(t -> iszero(amps[t]), ids) && @warn(
                "reweight: term `$name` carries amplitude 0 (compiled at 0, or reweighted " *
                "to 0 earlier), so scaling it by a factor is a no-op -- reweight is " *
                "multiplicative. Anchor sweeps on a nonzero-amplitude compile.")
            for t in ids
                factor[t] *= ComplexF64(value)
                amps[t] *= ComplexF64(value)             # keep the registry consistent
            end
        else
            amp0 = cl.term_amps[ids[1]]
            all(t -> cl.term_amps[t] == amp0, ids) || throw(ArgumentError(
                "The $(length(ids)) terms named `$name` were compiled with different " *
                "amplitudes, so `$name => value` is ambiguous. Give the terms distinct " *
                "names, or use `relative=true` to scale them all by one factor."))
            isfinite(amp0) || throw(ArgumentError(
                "Term `$name` has no recorded scalar amplitude (a callback amplitude). " *
                "Use `relative=true` to scale its couplings by a factor."))
            iszero(amp0) && throw(ArgumentError(
                "Term `$name` carries amplitude 0 -- compiled at 0, or reweighted to 0 " *
                "earlier -- so its couplings are all zero and no multiplicative rescaling " *
                "can recover them. Recompile with the new amplitude (and anchor sweeps " *
                "through 0 on the original nonzero-amplitude model)."))
            iszero(ComplexF64(value)) && @warn(
                "reweight: term `$name` is being set to amplitude 0. That is exact, but the " *
                "term becomes irrecoverable by further reweights (reweight is " *
                "multiplicative); anchor the rest of the sweep on the original model.")
            for t in ids
                factor[t] *= ComplexF64(value) / amp0
                amps[t] = ComplexF64(value)
            end
        end
    end

    w = copy(cl.w)
    for e in eachindex(w)
        t = cl.term[e]
        t == 0 && continue
        f = factor[abs(t)]
        f == one(ComplexF64) && continue
        ### the SIGN of the term tag records the entry's orientation relative to how the term
        ### emitted it (negative = the stored amplitude is the conjugate of the emitted one --
        ### a canonicalization flip, a Hermitian partner, or a BdG hole entry). Scaling those
        ### by conj(f) is what makes a complex amplitude change EXACT for bonds written in
        ### either direction. A self-conjugate (on-site) entry must stay real regardless.
        if _is_self_conjugate(cl.i[e], cl.j[e], cl.dR[e])
            abs(imag(f)) <= AMPLITUDE_ATOL || throw(ArgumentError(
                "Term `$(cl.term_names[abs(t)])` has on-site (self-conjugate) couplings, " *
                "which must stay real; rescaling them by the non-real factor $f would make " *
                "H(k) non-Hermitian."))
            w[e] *= real(f)
        else
            w[e] *= t > 0 ? f : conj(f)
        end
    end
    ### same keys, same sort, same groups -- only the amplitudes (and the registry) change,
    ### so the sparsity-pattern fingerprint is carried forward: a `sparse_bloch_workspace`
    ### built for the source model accepts every reweighted model of the sweep
    couplings = CouplingList{D}(cl.n, cl.i, cl.j, cl.dR, cl.q, w, cl.kernel, cl.kernels,
                                cl.groups, cl.term, copy(cl.term_names), amps, cl.pattern_id)
    return HamiltonianModel{D,_kpar(model.cell)}(model.cell, couplings, model.delta,
                                                 model.gauge, model.name)
end

"""
    _flatten_terms(terms) -> Vector{AbstractHamiltonianTerm}

Flatten arbitrarily nested collections of terms into one flat vector.

Convenience builders (a moiré harmonic star, both components of a Dirac term) naturally return
*groups* of terms; flattening here means the user passes them alongside single terms without
splatting gymnastics: `compile(cell, [kinetic(mass=m), harmonic_potential(cell, V, ψ)])`.
"""
function _flatten_terms(terms)::Vector{AbstractHamiltonianTerm}
    out = AbstractHamiltonianTerm[]
    _flatten_terms!(out, terms)
    return out
end

function _flatten_terms!(out::Vector{AbstractHamiltonianTerm}, x)
    if x isa AbstractHamiltonianTerm
        push!(out, x)
    elseif x isa Union{AbstractVector,Tuple}
        for y in x
            _flatten_terms!(out, y)
        end
    else
        throw(ArgumentError(
            "`compile` expected Hamiltonian terms (or nested collections of them); got $(typeof(x))."
        ))
    end
    return out
end

"""
    change_gauge(model, gauge) -> HamiltonianModel

Return the same Hamiltonian in the other Bloch gauge.

O(1): the stored representation is ALWAYS the atomic convention (physical amplitudes, atomic
`delta`), and the gauge flag only selects whether the assembly exit applies the diagonal
translator `W(k)` (see [`gauge_matrix`](@ref)). Nothing is recomputed. The eigenvalues are
unchanged; the eigenvectors differ by `W(k)`.
"""
function change_gauge(model::HamiltonianModel{D}, gauge::Symbol)::HamiltonianModel{D} where {D}
    gauge in GAUGE_MODES || throw(ArgumentError("Unknown gauge `$gauge`; expected one of $GAUGE_MODES."))
    gauge === model.gauge && return model
    return HamiltonianModel{D,_kpar(model.cell)}(model.cell, model.couplings, model.delta,
                                                 gauge, model.name)
end

"""
    gauge_matrix(cell, k_frac) -> Diagonal

The diagonal gauge translator `W(k) = diag(exp(2πi (k + G_s)⋅τ_s))` -- effective (kind-aware)
positions, plane-wave G-labels included -- relating the two Bloch conventions:

    H_periodic(k) = W(k) * H_atomic(k) * W(k)'          ψ_periodic = W(k) * ψ_atomic

Exposed explicitly because this relation is the trap everyone hits when comparing eigenvectors
or computing Berry phases across conventions. The identity is exact on EVERY cell -- folded
cells included -- by construction: the periodic view IS this conjugation of the stored atomic
assembly (see `_phase_displacements`), never an independently materialized table. On a `G ≡ 0`
tight-binding cell it reduces to the familiar `diag(e^{2πi k·τ})`.
"""
function gauge_matrix(cell::UnitCell{D}, k_frac::AbstractVector{<:Real}) where {D}
    length(k_frac) == D || throw(DimensionMismatch("k_frac has length $(length(k_frac)), expected $D."))
    k = SVector{D,Float64}(k_frac)
    _check_k_periodic(cell.lattice, k)
    return Diagonal(_gauge_phases(cell, k))
end

"""
    _check_k_periodic(lattice, k) -> Nothing

Refuse a momentum with weight along an aperiodic direction: on a partially periodic lattice
(`nperiodic < D`) the trailing components of `k` are not quantum numbers, and letting one
through would multiply real transverse displacements into the Bloch phases silently.
"""
@inline function _check_k_periodic(lat::BravaisLattice{D}, k::SVector{D,Float64})::Nothing where {D}
    P = nperiodic(lat)
    P == D && return nothing
    for m in (P+1):D
        abs(k[m]) <= LATTICE_ATOL || throw(ArgumentError(
            "k = $(Tuple(k)) has a non-zero component along direction $m, but the lattice is " *
            "periodic only along its first $P direction$(P == 1 ? "" : "s") -- momentum is " *
            "not defined there. Set the trailing component(s) to 0."))
    end
    return nothing
end

"""
    _gauge_phases(cell, k) -> Vector{ComplexF64}

The diagonal of the gauge translator `W(k)`: `e^{2πi (k + G_s)·τ_eff,s}` per state. A
plane-wave state has no centre (`τ_eff = 0`), so its factor is 1 -- both conventions coincide
for it by construction. Shared by `gauge_matrix` and the dense/sparse assembly exits.
"""
function _gauge_phases(cell::UnitCell{D}, k::SVector{D,Float64})::Vector{ComplexF64} where {D}
    ### runs once per k-point on every `:periodic` assembly exit, so the effective-position
    ### table is not materialized (a plane-wave state's factor is 1 -- τ_eff = 0 -- directly)
    return [cell.site_kind[cell.state_site[s]] === :plane_wave ? one(ComplexF64) :
            cispi(2 * dot(k + SVector{D,Float64}(cell.basis[s].G), cell.basis[s].r_frac))
            for s in eachindex(cell.basis)]
end

"""
    _integer_shift_or_refuse(Gp, caller) -> Vector{Int}

The exact integer reciprocal shift a float `Gp` denotes, or a named refusal if it denotes none.

Notes:
------
The ONE owner of this refusal, in the spirit of [`_require_periodic_shift`](@ref): a float `G`
has no sewing matrix, because only an exact reciprocal-lattice shift relates `H(k+Gp)` to
`H(k)`. Every entry point into the wrap machinery -- [`bloch_periodicity`](@ref) and each
[`wrap_unitary`](@ref) method, model, cell and `BlochSpectrum` alike -- reaches this through
its own `<:Real` method, which is strictly less specific than its `<:Integer` one and so fires
exactly when the eltype is not integer.

Why a shared owner rather than one check per entry point: the `BlochSpectrum` method used to
convert with `Int.(G)` before handing a model-less grid (the `read_qe_spectrum` import shape) to
the cell method. That cast made the SAME argument behave two ways depending on whether a
compiled model happened to be attached -- `[0.5, 0.0]` raised a bare `InexactError: Int64(0.5)`
instead of a named refusal, while the model path refused it properly.

Why it ROUNDS rather than refusing every float: an integer-valued float is an exact integer
carry that happens to be spelled in floating point, and this package already reads it that way
at a user-facing boundary -- `wilson_loop` closes its path with
`Gf = round.(Int, dk); maximum(abs, dk - Gf) <= INT_ROUND_ATOL` and hands the integer on
(analysis/topology/wilson.jl). Refusing `[1.0, 0.0]` here while accepting it there would be the
same helper answering one question two ways, which is the asymmetry this repository keeps
finding bugs in. It matters for exactly the callers the original defect was about: a
`read_qe_spectrum` import works in floats throughout, because QE's k-points are floats.
"""
@inline function _integer_shift_or_refuse(Gp::AbstractVector{<:Real},
                                          caller::AbstractString)::Vector{Int}
    Gi = round.(Int, Gp)
    maximum(abs, Gp .- Gi) <= INT_ROUND_ATOL || throw(ArgumentError(
        "`$caller` relates H(k+G) to H(k) for INTEGER reciprocal vectors G only " *
        "(got $(Tuple(Gp))): a fractional shift is just a different momentum, with no exact " *
        "unitary relating the two Hamiltonians. Pass the exact integer carry (`kshift` and " *
        "`momentum_conservation` return one), or evaluate at the unwrapped momentum. An " *
        "integer-valued float such as `[1.0, 0.0]` IS accepted -- it names an exact carry, " *
        "just spelled in floating point, as a QE import naturally does."))
    return Gi
end

"""
    bloch_periodicity(model, Gp) -> Matrix{ComplexF64}

The Brillouin-zone periodicity unitary `W(Gp)` of an exactly k-periodic model in the ATOMIC
convention:

    H(k + Gp) = W' * H(k) * W          for every k,

a k-INDEPENDENT permutation × phase matrix -- the object Wilson-loop and hybrid-Wannier
consumers need to glue eigenvectors across the zone boundary.

How it is built: shifting `k` by the reciprocal vector `Gp` relabels every basis function,
`|τ, G_s; k+Gp⟩ = |τ, G_s+Gp; k⟩` exactly. On an ordinary tight-binding cell (`G ≡ 0`,
period `I`) the relabelled state IS the original (every integer label is congruent to 0), and
`W = diag(e^{2πi Gp·τ_s})`. On a FOLDED cell the relabelled state is the stored representative
of the coset of `G_s + Gp`, reached with the same wrap-around phase
`e^{2πi (G_s + Gp − G_target)·τ_s}` the ΔG compiler uses -- a genuine coset permutation.

Restrictions, stated rather than fudged: `TruncatedWindow()` is refused (the finite
G-window makes `H(k+Gp)` a DIFFERENT truncation -- only the spectrum is periodic, to truncation
accuracy), `:none` is refused (no torus), and the model must be in the `:atomic` gauge (the
periodic view's boundary unitary is k-dependent; `change_gauge(model, :atomic)` is O(1)).
"""
bloch_periodicity(model::HamiltonianModel{D}, Gp::AbstractVector{<:Real}) where {D} =
    bloch_periodicity(model, _integer_shift_or_refuse(Gp, "bloch_periodicity"))

function bloch_periodicity(model::HamiltonianModel{D},
                           Gp::AbstractVector{<:Integer})::Matrix{ComplexF64} where {D}
    length(Gp) == D || throw(DimensionMismatch("Gp has length $(length(Gp)), expected $D."))
    _require_periodic_shift(model.cell.lattice, Gp, "bloch_periodicity")
    _require_exact_zone(kperiodicity(model), "`bloch_periodicity`",
        "Only the spectrum can be compared across the zone boundary here, not the states.",
        "Use `discretize` to regularize the frame onto a real lattice first.")
    model.gauge === :atomic || throw(ArgumentError(
        "`bloch_periodicity` is defined for the atomic assembly (the periodic view's boundary " *
        "unitary is k-dependent). `change_gauge(model, :atomic)` is an O(1) flag flip."))
    return _wrap_unitary_cell(model.cell, Gp)
end

"""
    _wrap_unitary_cell(cell, Gp) -> Matrix{ComplexF64}

The zone-boundary relabelling unitary `W` of [`bloch_periodicity`](@ref), built from the CELL
alone -- the body of that function, factored out because it never needed the Hamiltonian.

`W` is a permutation times a phase: shifting `k` by `Gp` relabels every basis function,
`|τ, G_s; k+Gp⟩ = |τ, G_s+Gp; k⟩`, and on a folded cell that is a genuine coset permutation
onto the stored representative. The Hamiltonian enters only through the guards its callers
apply (exact k-periodicity, the atomic assembly, no k-dependent kernels), which is what lets an
imported eigenvector grid with no compiled model use the same machinery.
"""
function _wrap_unitary_cell(cell::UnitCell{D},
                            Gp::AbstractVector{<:Integer})::Matrix{ComplexF64} where {D}
    n = n_states(cell)
    Gv = SVector{D,Int}(Gp)
    tau = _effective_positions(cell)

    ### state lookup by (site, coset label of G, internal): on an unfolded site (period I)
    ### every label is congruent to 0, so the entry is the state itself
    labelof(s, G) = begin
        site = cell.state_site[s]
        M = SMatrix{D,D,Int}(cell.site_period[site])
        _coset_label(_int_adjugate(M), _int_det_abs(M), G)
    end
    index = Dict{Tuple{Int,SVector{D,Int},Int},Int}(
        (cell.state_site[s], labelof(s, cell.basis[s].G), cell.basis[s].internal) => s
        for s in 1:n)

    W = zeros(ComplexF64, n, n)
    for s in 1:n
        b = cell.basis[s]
        key = (cell.state_site[s], labelof(s, b.G + Gv), b.internal)
        t = get(index, key, 0)
        t == 0 && throw(ErrorException(
            "No stored state carries the coset of G + Gp for state $s; an exactly periodic " *
            "cell must be closed under coset relabelling. This is an internal inconsistency."))
        ### the wrap-around phase of the relabelling, same convention as the ΔG compiler
        W[t, s] = cispi(2 * dot(SVector{D,Float64}(b.G + Gv - cell.basis[t].G), tau[s]))
    end
    return W
end




#############################################################
###### Bloch Hamiltonian                                #####
#############################################################

"""
    bloch_hamiltonian!(H, model, k_frac) -> Matrix{ComplexF64}

Fill `H` in place with the Bloch Hamiltonian at fractional momentum `k_frac`.

One branchless pass per kernel group over the atomic-convention assembly (`model.delta` is
always atomic), the phase computed entirely in fractional coordinates -- the constant group is
literally the historical bond loop (its `q·δ` vanishes on every unfolded model). A `:periodic`
model applies the diagonal gauge translator at the exit (one per-entry phase pass; see
[`gauge_matrix`](@ref)).
"""
function bloch_hamiltonian!(H::AbstractMatrix{ComplexF64}, model::HamiltonianModel{D},
                            k_frac::AbstractVector{<:Real}) where {D}
    n = n_states(model.cell)
    size(H) == (n, n) || throw(DimensionMismatch("H is $(size(H)), expected ($n, $n)."))
    length(k_frac) == D || throw(DimensionMismatch("k_frac has length $(length(k_frac)), expected $D."))
    k = SVector{D,Float64}(k_frac)
    _check_k_periodic(model.cell.lattice, k)

    fill!(H, zero(ComplexF64))
    c = model.couplings
    bT = SMatrix{D,D,Float64}(transpose(model.cell.lattice.b))
    for κ in eachindex(c.kernels)
        isempty(c.groups[κ]) && continue
        ### dispatch once per kernel group per k-point; the inner loops below are concrete
        _accumulate!(H, c.kernels[κ], c, c.groups[κ], k, bT, model.delta)
    end
    ### the single gauge exit: the stored assembly is the atomic convention; the periodic view
    ### is its conjugation by the diagonal translator, H_per = W H_at W† -- derived here, never
    ### stored, so it is correct BY CONSTRUCTION on every cell (folded wrap harmonics included)
    if model.gauge === :periodic
        Ws = _gauge_phases(model.cell, k)
        @inbounds for j in axes(H, 2)
            wj = conj(Ws[j])                 # hoisted: `Ws` and `H` could alias for LLVM
            for i in axes(H, 1)
                H[i, j] *= Ws[i] * wj
            end
        end
    end
    return H
end

@inline function _accumulate!(H::AbstractMatrix{ComplexF64}, ::ConstantKernel,
                              c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                              ::SMatrix{D,D,Float64}, δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        H[c.i[e], c.j[e]] += c.w[e] * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

### A Bloch-sum envelope carries tens of thousands of terms and is evaluated once per coupling,
### so it dominates assembly. Its separability scratch (`_bloch_tables`) is allocated ONCE for
### the whole kernel group here rather than once per coupling -- the group shares one envelope,
### only the momentum moves. Locals, so this stays thread-safe under `threaded_chunks`.
@inline function _accumulate!(H::AbstractMatrix{ComplexF64},
                              kern::CallbackKernel{BlochSumEnvelope},
                              c::CouplingList{2}, r::UnitRange{Int}, k::SVector{2,Float64},
                              bT::SMatrix{2,2,Float64}, δ::Vector{SVector{2,Float64}})::Nothing
    env = kern.f
    tx, ty = _bloch_scratch(env)
    @inbounds for e in r
        kq = k + c.q[e]
        H[c.i[e], c.j[e]] += c.w[e] * _bloch_value(env, bT * kq, tx, ty) *
                             bloch_phase(kq, δ[e])
    end
    return nothing
end

@inline function _accumulate!(H::AbstractMatrix{ComplexF64}, kern,
                              c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                              bT::SMatrix{D,D,Float64}, δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        p = bT * (k + c.q[e])
        H[c.i[e], c.j[e]] += c.w[e] * _kvalue(kern, p) * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

"""
    bloch_hamiltonian(model, k_frac) -> Matrix{ComplexF64}

The Bloch Hamiltonian at fractional momentum `k_frac`.
"""
function bloch_hamiltonian(model::HamiltonianModel{D},
                           k_frac::AbstractVector{<:Real})::Matrix{ComplexF64} where {D}
    n = n_states(model.cell)
    ### undef, not zeros: `bloch_hamiltonian!` starts with `fill!`, so zeroing twice is waste
    return bloch_hamiltonian!(Matrix{ComplexF64}(undef, n, n), model, k_frac)
end

"""
    bands(model, k_frac; atol=HERMITICITY_ATOL, cartesian=false) -> Vector{Float64}

Eigenvalues of `H(k)`, ascending. With `cartesian=true`, `k` is a cartesian momentum and is
converted through `reciprocal_to_fractional(k, lattice)` -- the natural spelling on a
`continuum_cell`, whose lattice is only a coordinate frame.

On a PLANE-WAVE (continuum) cell, equivalent representatives of the same physical momentum --
`k` and `k + G` -- give spectra that agree only to TRUNCATION accuracy, because the finite
G-basis window travels with the representative (`TruncatedWindow()` says exactly
this). Two symmetry-related points must therefore be entered as symmetry-related
representatives (e.g. `κ₊ = (1/3, 1/3)` against its inversion partner `(2/3, 2/3)`, not
`(-1/3, 2/3)`) before concluding that a degeneracy is broken: the 0.1 meV-scale mismatch of a
generic representative pair is the truncation window, not the physics.

Notes:
------
Hermiticity is verified *before* wrapping in `Hermitian`. That wrapper reads only one triangle, so
applying it blindly would silently symmetrize away a genuine asymmetry -- exactly the bug the check
is meant to catch. The check is skipped for constant-only models: `finalize_couplings` makes the
list Hermitian by construction, so there the check could only ever confirm what is already
guaranteed, at O(n²) per k-point.
"""
function bands(model::HamiltonianModel{D}, k_frac::AbstractVector{<:Real};
               atol::Float64=HERMITICITY_ATOL, cartesian::Bool=false)::Vector{Float64} where {D}
    k = cartesian ? reciprocal_to_fractional(k_frac, model.cell.lattice) : k_frac
    H = bloch_hamiltonian(model, k)
    has_kdependence(model) && _check_hermitian(H, k_frac, atol)
    ### H is a per-call temporary: the in-place solver skips LAPACK's n×n working copy
    return eigvals!(Hermitian(H))
end

function _check_hermitian(H::AbstractMatrix{ComplexF64}, k_frac, atol::Float64)
    # Allocation-free upper-triangle sweep; `abs.(H - H')` would materialize two n×n temporaries
    # per k-point, which at continuum basis sizes is megabytes of garbage per band-structure run.
    dev = 0.0
    n = size(H, 1)
    @inbounds for j in 1:n
        d = abs(imag(H[j, j]))
        d > dev && (dev = d)
        for i in 1:(j - 1)
            d = abs(H[i, j] - conj(H[j, i]))
            d > dev && (dev = d)
        end
    end
    dev <= atol || throw(DomainError(dev,
        "H(k) is not Hermitian at k_frac = $(collect(k_frac)): max |H - H'| = $dev > atol = $atol. " *
        "This is a bug in a term, not a numerical artefact -- the compiled coupling list is " *
        "Hermitian by construction, so check any k-dependent term's prefactors (a k-envelope " *
        "returning a complex value would land here)."
    ))
    return nothing
end

"""
    bands!(buf, H, model, kpath) -> Matrix{Float64}

Fill `buf` (nbands × nk) with the band structure along `kpath`, reusing `H` for every point.
"""
function bands!(buf::AbstractMatrix{Float64}, H::AbstractMatrix{ComplexF64},
                model::HamiltonianModel{D}, kpath; atol::Float64=HERMITICITY_ATOL) where {D}
    check = has_kdependence(model)   # constant lists are Hermitian by construction; see `bands`
    for (idx, k) in enumerate(kpath)
        bloch_hamiltonian!(H, model, k)
        check && _check_hermitian(H, k, atol)
        ### in-place: H is refilled from scratch at the next point anyway, so LAPACK may
        ### destroy it -- this saves one n×n complex copy per k-point
        buf[:, idx] = eigvals!(Hermitian(H))
    end
    return buf
end

"""
    bands(model, kpath) -> Matrix{Float64}

Band structure along a path of fractional momenta, as an `nbands × nk` matrix.
"""
function bands(model::HamiltonianModel{D}, kpath::AbstractVector{<:AbstractVector{<:Real}};
               atol::Float64=HERMITICITY_ATOL, cartesian::Bool=false)::Matrix{Float64} where {D}
    n = n_states(model.cell)
    ks = cartesian ? [reciprocal_to_fractional(k, model.cell.lattice) for k in kpath] :
                     collect(kpath)
    buf = Matrix{Float64}(undef, n, length(ks))
    ### Threaded, like every other k-loop in the package (`spectrum_along_line`,
    ### `eigenstates_along_line`, `berry_curvature` all go through `threaded_chunks`); this one
    ### was the last serial holdout, and it is the loop a band structure or a BZ sweep actually
    ### spends its time in -- measured 4.4x on a 30x30 grid at 292 states, and the result is
    ### BIT-IDENTICAL to the serial version (verified against `-t 1` and against the old code).
    ### One `H` buffer per chunk, and the chunks write disjoint columns of `buf`.
    threaded_chunks(length(ks)) do chunk
        H = zeros(ComplexF64, n, n)
        bands!(view(buf, :, chunk), H, model, view(ks, chunk); atol=atol)
    end
    return buf
end




#############################################################
###### Derived conservation facts                       #####
#############################################################

"""
    conserved_qnums(model) -> Vector{Symbol}

The quantum numbers that the Hamiltonian conserves, **derived** from the compiled couplings.

Notes:
------
This is a function rather than a stored field on purpose. An asserted `Vector{Symbol}` would go
stale the moment a term is added, and would assert rather than verify. Deriving it from the ONE
coupling list costs O(ncouplings) and cannot be wrong -- and needs exactly one loop, where the
split representation needed two.
"""
function conserved_qnums(model::HamiltonianModel{D})::Vector{Symbol} where {D}
    schema = model.cell.schema
    is_trivial(schema) && return Symbol[]
    internal = state_internals(model.cell)
    c = model.couplings

    conserved = Symbol[]
    for (a, key) in pairs(axis_names(schema))
        ### An axis is conserved when no coupling changes its level. With the internal state
        ### interned, that is integer arithmetic on the mixed-radix ordinal -- no dictionary
        ### lookup, and no default value to get wrong.
        ok = true
        for e in eachindex(c.i)
            if ordinal_at(schema, internal[c.i[e]], a) != ordinal_at(schema, internal[c.j[e]], a)
                ok = false
                break
            end
        end
        ok && push!(conserved, key)
    end
    return conserved
end

"""
    qnum_blocks(model, key) -> Dict{QLevel,Vector{Int}}

Partition the basis indices by the level they take on internal axis `key`.

When `key` is conserved, `H(k)` is block-diagonal with respect to this partition.
"""
function qnum_blocks(model::HamiltonianModel{D}, key::Symbol)::Dict{QLevel,Vector{Int}} where {D}
    schema = model.cell.schema
    a = axis_position(schema, key)
    a == 0 && throw(ArgumentError(
        "The cell declares no internal axis `$key`; its axes are $(axis_names(schema))."
    ))
    blocks = Dict{QLevel,Vector{Int}}()
    for (i, internal) in pairs(state_internals(model.cell))
        push!(get!(Vector{Int}, blocks, level_at(schema, internal, a)), i)
    end
    return blocks
end




#############################################################
###### Display                                          #####
#############################################################

function Base.show(io::IO, ::MIME"text/plain", m::HamiltonianModel{D}) where {D}
    c = m.couplings
    ngrp = count(!isempty, c.groups)
    nc = n_couplings(c)
    print(io, "HamiltonianModel{", D, "}")
    isempty(m.name) || print(io, " ", repr(m.name))
    print(io, ": ", n_states(m.cell), " states on ", n_sites(m.cell), " sites, ",
          nc, " coupling", nc == 1 ? "" : "s",
          " in ", ngrp, " kernel group", ngrp == 1 ? "" : "s",
          ", gauge ", repr(m.gauge), ", k-periodicity ", repr(kperiodicity(m)))
    if isempty(c.term_names)
        print(io, "\n  terms: none recorded (imported or legacy model; `reweight` unavailable)")
        return nothing
    end
    print(io, "\n  terms:")
    nterms = length(c.term_names)
    nshow = nterms > 11 ? 10 : nterms
    for t in 1:nshow
        amp = c.term_amps[t]
        print(io, "\n    ", c.term_names[t], " => ",
              isfinite(amp) ? repr(amp) : "(callback amplitude)")
    end
    nterms > nshow && print(io, "\n    … ", nterms - nshow,
                            " more (see `term_amplitudes`)")
    return nothing
end

Base.show(io::IO, m::HamiltonianModel{D}) where {D} =
    print(io, "HamiltonianModel{", D, "}(", repr(m.name), ", n=", n_states(m.cell),
          ", couplings=", n_couplings(m.couplings), ")")
