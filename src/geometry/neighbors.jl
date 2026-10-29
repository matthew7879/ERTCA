##################################################################
###### export only the following functions from this module ######
##################################################################

export NeighborTable, neighbor_table, neighbor_table_by_shell
export n_shells, shell_radius, shell_entries




#############################################################
###### Enumerating candidate Bravais translations       #####
#############################################################

"""
    _translation_range(bnorms, rcut, dtau) -> NTuple{D,UnitRange{Int}}

Return the box of integer translations `n` that can possibly satisfy
`‖aᵀ(n + dtau)‖ ≤ rcut`.

Notes:
------
With **rows** of `a` as the primitive vectors, a cartesian displacement `d` has fractional
components `n_m = (1/2π) * b[m,:] ⋅ d`, because `(aᵀ)⁻¹ = b/2π`. Cauchy-Schwarz then gives the
per-axis projection bound

    |n_m + dtau_m| ≤ ‖b[m,:]‖ * rcut / 2π

The row norms are passed in rather than recomputed: this runs once per site *pair*, and `lat.b` is
a computed property that rebuilds the whole matrix on each access.

This is a *bound*, not a heuristic: nothing within `rcut` can fall outside the returned box. It is
also tight for orthogonal lattices and degrades gracefully for skewed ones, which is why it is
preferred over an isotropic `σ_min(a)` estimate -- that one requires an SVD and is loose in exactly
the skewed cases where the search is most expensive.
"""
@inline function _translation_range(bnorms::NTuple{D,Float64}, rcut::Float64,
                                    dtau::SVector{D,Float64}, nperiodic::Int=D) where {D}
    return ntuple(D) do m
        ### aperiodic directions have no translations: the only image is ΔR[m] = 0
        m > nperiodic && return 0:0
        c = bnorms[m] * rcut / (2π)
        lo = floor(Int, -dtau[m] - c - LATTICE_ATOL)
        hi = ceil(Int, -dtau[m] + c + LATTICE_ATOL)
        lo:hi
    end
end

"""
    _is_canonical_pair(si, sj, dR) -> Bool

Whether `(si, sj, dR)` is the canonical representative of the unordered pair it forms with
`(sj, si, -dR)`.

The rule is lexicographic on `dR`, falling back to the site order when `dR` vanishes: the first
non-zero component of `dR` must be positive, else `si <= sj`. Exactly one of the two orderings
satisfies it, except for the self-conjugate case `si == sj, dR == 0`.
"""
@inline function _is_canonical_pair(si::Int, sj::Int, dR::SVector{D,Int})::Bool where {D}
    for m in 1:D
        dR[m] > 0 && return true
        dR[m] < 0 && return false
    end
    return si <= sj
end




#############################################################
###### Grouping pair distances into shells              #####
#############################################################

"""
    _group_shells(dists, atol; rtol=SHELL_RTOL) -> (radii, assignment)

Cluster non-negative distances into shell radii, returning both the radii and the shell index of
every input distance.

Returning the assignment keeps **one** notion of "same shell" in this file. Deriving the radii here
and then re-deriving each membership with a separate distance-to-radius test used two different
equivalences, and a distance satisfying neither came back as shell `0`. That then broke the
`offset` scan below so that *every* shell reported itself empty and every distance term compiled to
no bonds -- silently.

Notes:
------
A new shell starts whenever the gap between consecutive sorted distances exceeds
`max(atol, rtol*d)`. Neither tolerance alone suffices: an absolute tolerance fails at large cutoffs
because floating-point error grows with distance, and a relative one fails near zero.

`atol` is a **cartesian** tolerance and is required rather than defaulted, because the right value
depends on the lattice: see `distance_atol`. Defaulting it to `LATTICE_ATOL` -- a *fractional*
tolerance -- made the whole search unit-system dependent, and at a lattice constant near that floor
it silently dropped the nearest-neighbour shell so that `shell=1` selected second neighbours.

Distance 0 between *distinct* sites cannot arise: a site is a position, and `UnitCell` rejects a
registry whose members are closer than `2*atol`. The `d > atol` filter therefore guards only against
a degenerate lattice, and `radii[1]` is always the true nearest-neighbour distance -- so `shell=1`
means the same thing in every model.
"""
function _group_shells(dists::Vector{Float64}, atol::Float64;
                       rtol::Float64=SHELL_RTOL)
    n = length(dists)
    assign = zeros(Int, n)
    radii = Float64[]
    n == 0 && return radii, assign

    ### Membership is measured from the cluster's REPRESENTATIVE (its smallest member), not from
    ### the previous element. Chaining off the predecessor is not transitive: three distances each
    ### within tolerance of the next can span more than the tolerance end to end. `_derive_sites`
    ### in `unitcell.jl` documents itself as avoiding exactly this; the shell grouping used to
    ### reintroduce it.
    for e in sortperm(dists)
        d = dists[e]
        d > atol || continue          # only reachable for a degenerate lattice
        if isempty(radii) || d - radii[end] > max(atol, rtol * d)
            push!(radii, d)
        end
        assign[e] = length(radii)
    end
    return radii, assign
end

"""
    _warn_near_degenerate(radii, atol; rtol)

Warn when two adjacent shells are separated by less than `SHELL_DEGENERACY_FACTOR` times the
clustering tolerance.

Notes:
------
This detects **numerically ambiguous clustering**, not physical near-degeneracy. `_group_shells`
already guarantees a gap above `max(atol, rtol*r)`, so the window this fires in is
`1x` to `SHELL_DEGENERACY_FACTOR x` that tolerance -- a *relative* separation of roughly `1e-8` to
`1e-7`. Two shells that close together were split by a hair, and a slightly different `rtol` would
have merged them, so `shell=k` is not a stable way to name them.

It deliberately does **not** fire on the percent-level near-degeneracies of a distorted lattice:
those are physically distinct shells that `shell=` names unambiguously, and warning about them would
be noise on every low-symmetry cell.
"""
function _warn_near_degenerate(radii::Vector{Float64}, atol::Float64;
                               rtol::Float64=SHELL_RTOL)
    for k in 2:length(radii)
        gap = radii[k] - radii[k-1]
        tol = max(atol, rtol * radii[k])
        if gap < SHELL_DEGENERACY_FACTOR * tol
            @warn "Neighbour shells $(k-1) and $k are nearly degenerate " *
                  "(radii $(radii[k-1]) and $(radii[k]), gap $gap ≈ tolerance $tol). " *
                  "Selecting by `shell=` is ambiguous here; select by explicit `distance=` instead."
        end
    end
    return nothing
end




#############################################################
###### The neighbour table                              #####
#############################################################

"""
    _order_entries(key, ci, cj, cR, keep) -> Vector{Int}

The candidate indices that pass `keep`, ordered by `(key[e], ci[e], cj[e], dR[e])`.

Notes:
------
A free function, NOT a `sort(...; by = e -> ...)` closure written inline in `neighbor_table`:
the closure there captured `shells`/`ci`/`cj`/`cR`, and because `shells` is assigned in two
branches of the `group_shells` test Julia boxes it, so every comparison the sort performs
resolves `shells[e]` dynamically and allocates. Passing the columns as arguments makes them
concretely typed and the comparator static -- same order, no dynamic dispatch.
"""
function _order_entries(key::Vector{T}, ci::Vector{Int}, cj::Vector{Int},
                        cR::Vector{SVector{D,Int}}, keep::F)::Vector{Int} where {T,D,F}
    return sort(findall(keep, key); by = e -> (key[e], ci[e], cj[e], Tuple(cR[e])))
end

"""
    NeighborTable{D}

Canonical half-bonds between the **sites** of a unit cell, grouped into distance shells.

Purpose:
--------
This is the geometry, computed once and shared by every distance-based term. It is expressed over
sites rather than basis states so that its cost tracks the geometry rather than the basis size:
spinful graphene has 4 states but 2 sites, and a moiré cell can have hundreds of states on 1 site.

Only canonical half-bonds are stored -- see `_is_canonical_pair`. The Hermitian partner is
materialised later, by the bond compiler.

Fields:
-------
- `rcut::Float64`                        cutoff radius actually searched
- `radii::Vector{Float64}`               ascending non-zero shell radii; `shell s` ↔ `radii[s]`
- `site_i`, `site_j`, `dR`, `dist`       the half-bonds, sorted by shell
- `shell::Vector{Int}`                   shell index of each entry
- `offset::Vector{Int}`                  `offset[s]:offset[s+1]-1` are the entries of shell `s`
"""
struct NeighborTable{D}
    rcut::Float64
    radii::Vector{Float64}
    site_i::Vector{Int}
    site_j::Vector{Int}
    dR::Vector{SVector{D,Int}}
    dist::Vector{Float64}
    shell::Vector{Int}
    offset::Vector{Int}
end

"""
    neighbor_table(cell::UnitCell{D}, rcut; tables=cell_tables(cell), rtol, atol,
                   group_shells=true) -> NeighborTable{D}

Find every canonical half-bond between sites of `cell` out to the cutoff radius `rcut`
(a *cartesian* distance, in the same units as the lattice vectors).

Args:
-----
- `cell`: the unit cell
- `rcut`: cartesian cutoff radius
- `tables`: precomputed `CellTables` (pass one in to avoid rebuilding it)
- `rtol`, `atol`: tolerances for clustering distances into shells. `atol` is dimensionless and
  is scaled by the lattice's `length_scale` before any distance is compared against it.
- `group_shells`: with the default `true`, distances are clustered into shells exactly as
  before. With `false` the clustering is skipped entirely: the table resolves **no** shells
  (`n_shells == 0`, every `shell` id `0`, entries sorted by distance), so the
  near-degenerate-shell warnings cannot fire and any shell consumer (`shell=`, `Chirality`)
  raises its own named error. This is what a pure radius consumer (`hopping(t; rcut=...)`)
  requests on quasi-continuous distance spectra -- relaxed or imported geometry -- where
  clustering is meaningless and its warnings would be noise.

Returns:
--------
- `NeighborTable{D}` with entries sorted by shell (by distance when `group_shells=false`).
"""
function neighbor_table(cell::UnitCell{D}, rcut::Real;
                        tables::CellTables{D}=cell_tables(cell),
                        rtol::Float64=SHELL_RTOL,
                        atol::Float64=LATTICE_ATOL,
                        warn_degenerate::Bool=true,
                        group_shells::Bool=true)::NeighborTable{D} where {D}
    ### `atol` arrives as a dimensionless tolerance and is scaled to this lattice's length before
    ### touching a distance. Without the scaling the search is unit-system dependent.
    datol = distance_atol(cell.lattice; atol=atol)
    rcut > 0 || throw(DomainError(rcut, "Neighbour cutoff radius must be positive."))
    rc = Float64(rcut)
    lat = cell.lattice
    aT = transpose(lat.a)
    bnorms = ntuple(m -> norm(lat.brows[m]), D)     # hoisted: constant over the site-pair loop
    ns = n_sites(tables)

    ci = Int[]; cj = Int[]; cR = SVector{D,Int}[]; cd = Float64[]

    ### `sj` starts at `si`, not at 1. The translation box of the ordered pair (sj, si) is the
    ### exact mirror of the box of (si, sj) -- `dtau` flips sign and the distance is invariant --
    ### so scanning both halves enumerated every candidate twice and threw one copy away in
    ### `_is_canonical_pair`. Here each unordered pair is enumerated ONCE and the canonical
    ### orientation is chosen from the two, which is the same set of half-bonds at half the
    ### search.
    for si in 1:ns, sj in si:ns
        ### The table is KIND-AWARE: a :plane_wave site's declared position is gauge-only and
        ### carries no geometry, so cross-site distances involving one are meaningless and are
        ### excluded -- on a mixed cell they would silently renumber the physical shells of the
        ### localized subsystem. Same-site pairs survive: their distances are pure lattice
        ### translations (the translation-operator shells of a plane-wave cell).
        if si != sj && (tables.site_kind[si] === :plane_wave ||
                        tables.site_kind[sj] === :plane_wave)
            continue
        end
        dtau = tables.site_positions[sj] - tables.site_positions[si]
        for n in Iterators.product(_translation_range(bnorms, rc, dtau,
                                                      nperiodic(lat))...)
            dR = SVector{D,Int}(n)
            si == sj && iszero(dR) && continue                 # the state itself, not a bond

            d = norm(aT * (SVector{D,Float64}(dR) + dtau))
            d <= rc + datol || continue

            ### No zero-distance branch: a site is a position, and `UnitCell` rejects a registry
            ### whose members are closer than `2*atol`, so two *distinct* sites at distance 0
            ### cannot arise. `si == sj` with `dR = 0` is excluded above.
            if _is_canonical_pair(si, sj, dR)
                push!(ci, si); push!(cj, sj); push!(cR, dR); push!(cd, d)
            elseif si != sj
                ### the reversed orientation is the canonical one. Its distance is the SAME
                ### float: aᵀ(-dR - dtau) = -aᵀ(dR + dtau) exactly, and `norm` is sign-blind.
                push!(ci, sj); push!(cj, si); push!(cR, -dR); push!(cd, d)
            end
            ### `si == sj` with a non-canonical dR needs no branch: its twin at +dR is
            ### enumerated by this same box.
        end
    end

    if group_shells
        radii, shells = _group_shells(cd, datol; rtol=rtol)
        warn_degenerate && _warn_near_degenerate(radii, datol; rtol=rtol)

        ### Every kept entry carries a real shell, so nothing can sort ahead of shell 1 and
        ### strand the `offset` scan below. A zero means a degenerate lattice and is dropped
        ### here rather than silently emptying the whole table.
        order = _order_entries(shells, ci, cj, cR, !iszero)
    else
        ### UNGROUPED: no shells are resolved at all (`radii` empty, every shell id 0), so no
        ### clustering equivalence exists to warn about and no shell id can be read by mistake
        ### -- a shell consumer hits its own named error. Entries are sorted by distance; the
        ### degenerate-lattice guard (`d > datol`) is kept so `dist` stays meaningful.
        radii = Float64[]
        shells = zeros(Int, length(cd))
        order = _order_entries(cd, ci, cj, cR, d -> d > datol)
    end

    site_i = ci[order]; site_j = cj[order]
    ### NOT `dR = cR[order]`: `dR` names the SVector built inside the search loop above, and
    ### rebinding it here to a Vector makes the local a Union for the WHOLE method, so every
    ### `dR = SVector{D,Int}(n)` in the hot inner loop boxes on the heap.
    dRs = cR[order]; dist = cd[order]; shell = shells[order]

    # offset[s]:offset[s+1]-1 spans shell s, exploiting that `shell` is now sorted
    offset = Vector{Int}(undef, length(radii) + 1)
    pos = 1
    for s in 1:length(radii)
        offset[s] = pos
        while pos <= length(shell) && shell[pos] == s
            pos += 1
        end
    end
    offset[end] = pos

    return NeighborTable{D}(rc, radii, site_i, site_j, dRs, dist, shell, offset)
end

"""
    neighbor_table_by_shell(cell::UnitCell{D}, nshells::Integer; kwargs...) -> NeighborTable{D}

Build a neighbour table guaranteed to contain at least `nshells` distance shells.

Notes:
------
A shell index carries no radius on its own -- "the second shell" is only defined once the search has
run. So the cutoff is grown adaptively: seed it from the shortest primitive vector, search, count,
and double until enough shells appear or `SHELL_MAX_GROW` doublings have been spent.
"""
function neighbor_table_by_shell(cell::UnitCell{D}, nshells::Integer;
                                 tables::CellTables{D}=cell_tables(cell),
                                 rtol::Float64=SHELL_RTOL,
                                 atol::Float64=LATTICE_ATOL,
                        warn_degenerate::Bool=true)::NeighborTable{D} where {D}
    ### `atol` arrives as a dimensionless tolerance and is scaled to this lattice's length before
    ### touching a distance. Without the scaling the search is unit-system dependent.
    datol = distance_atol(cell.lattice; atol=atol)
    nshells >= 1 || throw(DomainError(nshells, "Requested shell count must be at least 1."))
    lat = cell.lattice
    if nperiodic(lat) == 0
        ### a Γ-only finite cluster: no primitive vector to seed a cutoff from, and no lattice
        ### images -- every pair distance is realized at ΔR = 0, so ONE search at the cluster
        ### diameter is complete (there is no incomplete outermost shell to grow past)
        aT = transpose(lat.a)
        dmax = 0.0
        for si in 1:n_sites(tables), sj in (si+1):n_sites(tables)
            dmax = max(dmax, norm(aT * (tables.site_positions[sj] - tables.site_positions[si])))
        end
        dmax > datol || throw(ArgumentError(
            "This cell is a finite cluster (0 periodic directions) with " *
            "$(n_sites(tables) == 1 ? "a single site" : "all sites coincident") -- there are " *
            "no pair distances, so no neighbour shell exists."))
        nt = neighbor_table(cell, dmax + datol; tables=tables, rtol=rtol, atol=atol,
                            warn_degenerate=warn_degenerate)
        length(nt.radii) >= nshells || throw(ArgumentError(
            "This cell is a finite cluster (0 periodic directions) and realizes only " *
            "$(length(nt.radii)) distance shell$(length(nt.radii) == 1 ? "" : "s") in total; " *
            "shell $nshells does not exist. A cluster has finitely many pair distances -- " *
            "request a shallower shell, or address bonds with `explicit_bonds`."))
        return nt
    end
    ### seed from the PERIODIC rows only: unit-norm padding rows are coordinate axes, not
    ### translations, and would seed a nonsense cutoff on a partially periodic lattice
    P = nperiodic(lat)
    rmin = minimum(norm(lat.arows[m]) for m in 1:P)
    ### The primitive-vector seed alone is catastrophic on a SUPERCELL: there the shortest
    ### primitive vector is L·a while the nearest-neighbour distance is still ~a, so the very
    ### first search would accept O(n_sites²) pairs (and then sort and shell-group all of
    ### them). For a roughly uniform site distribution the nearest-neighbour distance scales
    ### as (cell volume / n_sites)^(1/P); 2.5× that safely contains the first shells, and the
    ### historical seed remains as a cap so few-site cells behave exactly as before. A too-
    ### small estimate only costs extra doublings, which the loop bound below accounts for.
    est = 2.5 * length_scale(lat) / Float64(n_sites(tables))^(1 / P)
    rcut = min(rmin, max(est, 4 * datol))
    extra = rcut < rmin ? ceil(Int, log2(rmin / rcut)) : 0

    ### The degeneracy warning is suppressed while growing and re-checked once on the answer:
    ### otherwise a single call could emit the same warning once per doubling.
    tried = 0
    last = rcut
    for _ in 0:(SHELL_MAX_GROW + extra)
        last = rcut
        tried += 1
        nt = neighbor_table(cell, rcut; tables=tables, rtol=rtol, atol=atol, warn_degenerate=false)
        # Only shells strictly inside the searched radius are complete: the outermost one may be
        # missing neighbours that sit just beyond the cutoff.
        if length(nt.radii) > nshells
            _warn_near_degenerate(nt.radii, distance_atol(cell.lattice; atol=atol); rtol=rtol)
            return nt
        end
        rcut *= 2
    end
    throw(ArgumentError(
        "Could not reach $nshells neighbour shells after $tried searches, the largest with cutoff " *
        "$last. Either the lattice is extremely anisotropic or `nshells` is too large; " *
        "call `neighbor_table(cell, rcut)` with an explicit radius instead."
    ))
end




#############################################################
###### Accessors                                        #####
#############################################################

"""
    n_shells(nt::NeighborTable) -> Int

Number of distance shells resolved in the table.
"""
@inline n_shells(nt::NeighborTable)::Int = length(nt.radii)

"""
    shell_radius(nt::NeighborTable, s::Integer) -> Float64

Radius of shell `s`; `s = 1` is the nearest-neighbour shell.
"""
function shell_radius(nt::NeighborTable, s::Integer)::Float64
    1 <= s <= length(nt.radii) || throw(BoundsError(nt.radii, s))
    return nt.radii[s]
end

"""
    shell_entries(nt::NeighborTable, s::Integer) -> UnitRange{Int}

Indices into the table's parallel arrays belonging to shell `s`.
"""
function shell_entries(nt::NeighborTable, s::Integer)::UnitRange{Int}
    1 <= s <= length(nt.radii) || throw(BoundsError(nt.radii, s))
    return nt.offset[s]:(nt.offset[s+1] - 1)
end
