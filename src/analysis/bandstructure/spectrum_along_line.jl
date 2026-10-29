##################################################################
###### export only the following functions from this module ######
##################################################################

export kpath_line, spectrum_along_line, eigenstates_along_line, standard_kpath_nodes


#############################################################
###### Building a k-path through named nodes            #####
#############################################################

"""
    kpath_line(lattice, nodes; npoints=200) -> (kpts, dist, ticks, labels)
    kpath_line(lattice, segments::Vector{<:AbstractVector}; npoints=200)

Interpolate a piecewise-linear path through the given k-space `nodes` (fractional coordinates),
distributing about `npoints` samples over the segments **proportionally to their cartesian
length** -- so a plot against `dist` has the conventional geometry, where a long Γ-M stretch
takes more horizontal room than a short M-K one.

Args:
-----
- `lattice`: fixes the cartesian metric of reciprocal space (fractional k alone has none)
- `nodes`: the corners of the path, either plain fractional vectors or `label => k_frac` pairs,
  e.g. `["Γ" => [0,0], "K" => [1/3,1/3], "M" => [1/2,0], "Γ" => [0,0]]`
- `segments` (second method): a vector of such node lists, one per CONTINUOUS run of a
  discontinuous path -- the shape `standard_kpath_nodes(...; segments=:all)` returns. The runs
  are glued along the arc-length axis in the standard band-plot convention: the jump between
  consecutive runs takes ZERO horizontal room and no samples, and the seam carries one double
  tick labeled `"K|U"` (last node of one run, first of the next). Both bounding k-points of a
  jump ARE sampled, at the same `dist` -- a curve drawn through them crosses the seam guide
  vertically, exactly where `plot_bands` draws its node line.
- `npoints`: approximate total sample count (each straight piece gets at least 2)

Returns:
--------
- `kpts::Vector{SVector{D,Float64}}`: the sampled fractional momenta, node points included once
- `dist::Vector{Float64}`: cumulative **cartesian** arc length at each sample -- the x axis
  (nondecreasing; equal values appear exactly at the seams of a segmented path)
- `ticks::Vector{Float64}`: the arc-length positions of the nodes
- `labels::Vector{String}`: the node labels (auto-numbered when plain vectors were given),
  one per tick; seam ticks carry the merged `"K|U"` label

Notes:
------
The high-symmetry nodes of a named lattice come from `high_symmetry_points` /
`high_symmetry_path` in the crystallography layer; this function deliberately takes raw
coordinates so it also serves cells those tables do not know -- a moiré lattice, a supercell.
"""
function kpath_line(lattice::BravaisLattice{D}, nodes::AbstractVector;
                    npoints::Int=200) where {D}
    return _kpath_through_nodes(lattice, nodes; npoints=npoints)
end

function kpath_line(lattice::BravaisLattice{D}, segments::AbstractVector{<:AbstractVector};
                    npoints::Int=200) where {D}
    ### a vector of PLAIN COORDINATE nodes ([[0,0], [1/2,0], ...]) matches this signature too;
    ### only a vector whose elements are themselves node lists is a segmented path
    all(seg -> seg isa AbstractVector{<:Real}, segments) &&
        return _kpath_through_nodes(lattice, segments; npoints=npoints)

    isempty(segments) && throw(ArgumentError("A segmented k-path needs at least one segment."))
    npoints >= 2 || throw(DomainError(npoints, "`npoints` must be at least 2."))
    for (g, seg) in enumerate(segments)
        length(seg) >= 2 || throw(ArgumentError(
            "Path segment $g has $(length(seg)) node(s); every segment needs at least two."))
    end

    labels = [String[_node_label(nd, i) for (i, nd) in enumerate(seg)] for seg in segments]
    corners = [SVector{D,Float64}[_node_coords(nd, D) for nd in seg] for seg in segments]

    B = recip_matrix(lattice)
    seglen = [[norm(B * (c[s+1] - c[s])) for s in 1:length(c)-1] for c in corners]
    total = sum(sum, seglen)
    total > 0 || throw(ArgumentError("The k-path has zero total length; the nodes coincide."))

    kpts = SVector{D,Float64}[]
    dist = Float64[]
    ticks = Float64[0.0]
    ticklabels = String[labels[1][1]]
    acc = 0.0
    for g in eachindex(segments)
        ### the seam: the previous tick position is reused (the jump has zero arc length) and
        ### its label becomes the conventional double label. Sampling of the run itself is the
        ### ONE core `_sample_run!` -- the same code the single-path method uses.
        g > 1 && (ticklabels[end] = ticklabels[end] * "|" * labels[g][1])
        acc = _sample_run!(kpts, dist, ticks, corners[g], seglen[g], total, npoints, acc)
        append!(ticklabels, @view labels[g][2:end])
    end
    return (kpts=kpts, dist=dist, ticks=ticks, labels=ticklabels)
end

"""
    _sample_run!(kpts, dist, ticks, corners, seglen, total, npoints, acc) -> Float64

Sample ONE continuous node run into `(kpts, dist, ticks)`, returning the updated arc-length
accumulator -- the shared sampling core of both `kpath_line` methods. `total` is the WHOLE
path's cartesian length (the `npoints` budget is distributed over every piece of every run);
each piece gets ≥ 2 samples so every node appears exactly, and the shared endpoint of two
consecutive pieces is emitted once (the FIRST node of the run is always emitted -- on a
segmented path it is a different k-point sitting at the seam's arc length).
"""
function _sample_run!(kpts::Vector{SVector{D,Float64}}, dist::Vector{Float64},
                      ticks::Vector{Float64}, corners::Vector{SVector{D,Float64}},
                      seglen::Vector{Float64}, total::Float64, npoints::Int,
                      acc::Float64)::Float64 where {D}
    for s in 1:length(seglen)
        ns = max(2, round(Int, npoints * seglen[s] / total))
        for t in range(0.0, 1.0; length=ns)
            s > 1 && t == 0.0 && continue
            push!(kpts, corners[s] + t * (corners[s+1] - corners[s]))
            push!(dist, acc + t * seglen[s])
        end
        acc += seglen[s]
        push!(ticks, acc)
    end
    return acc
end

"""
    _kpath_through_nodes(lattice, nodes; npoints) -> (kpts, dist, ticks, labels)

The single-continuous-path core of [`kpath_line`](@ref) -- shared by both public methods, since
a `Vector` of plain coordinate vectors dispatches to the segmented signature but IS a node list.
"""
function _kpath_through_nodes(lattice::BravaisLattice{D}, nodes::AbstractVector;
                              npoints::Int=200) where {D}
    length(nodes) >= 2 || throw(ArgumentError("A k-path needs at least two nodes."))
    npoints >= 2 || throw(DomainError(npoints, "`npoints` must be at least 2."))

    labels = String[_node_label(nd, i) for (i, nd) in enumerate(nodes)]
    corners = SVector{D,Float64}[_node_coords(nd, D) for nd in nodes]

    ### segment lengths in CARTESIAN reciprocal space: fractional coordinates carry no metric,
    ### and equal-fraction segments can differ in true length by the lattice anisotropy
    B = recip_matrix(lattice)
    seglen = [norm(B * (corners[s+1] - corners[s])) for s in 1:length(corners)-1]
    total = sum(seglen)
    total > 0 || throw(ArgumentError("The k-path has zero total length; the nodes coincide."))

    kpts = SVector{D,Float64}[]
    dist = Float64[]
    ticks = Float64[0.0]
    _sample_run!(kpts, dist, ticks, corners, seglen, total, npoints, 0.0)
    return (kpts=kpts, dist=dist, ticks=ticks, labels=labels)
end

_node_label(nd::Pair, ::Int) = String(first(nd))
_node_label(::AbstractVector, i::Int) = "k$i"

"""
    standard_kpath_nodes(lattice_or_cell; rtol=BRAVAIS_RTOL, bravais=nothing, segments=:first)
        -> Vector{Pair{String,SVector}}            (segments = :first)
        -> Vector{Vector{Pair{String,SVector}}}    (segments = :all)

The Setyawan-Curtarolo recommended path of a lattice given as an OBJECT, as ready-to-use
`label => k_frac` nodes: `identify_bravais` classifies the lattice, the tabulated path supplies
the labels, and `high_symmetry_points` supplies the coordinates in the lattice's own basis.
This is the default `nodes` of `spectrum_along_line` and `plot_bands`.

Args:
-----
- `bravais`: DECLARE the Bravais type instead of classifying -- the route for 3D lattices, where
  `identify_bravais` raises. The declaration is validated against the lattice metric and refused
  by name if the lattice is not the declared type's standard primitive basis (up to a rigid
  rotation); see [`high_symmetry_points`](@ref). Aliases (`:fcc`, `:hexagonal`) resolve as
  everywhere else. Padded (partially periodic) lattices refuse a declaration -- they classify
  automatically by their periodic block.
- `segments`: `:first` (default) returns the first continuous segment of the recommended path
  as one node list -- the 2D tables are all single-segment, so shipped 2D behavior is unchanged.
  `:all` returns EVERY continuous segment (most 3D recommended paths have several, separated by
  discontinuous jumps) as a vector of node lists, the shape `kpath_line` glues with `"K|U"`
  seam ticks -- pass it straight through as the `nodes` of `spectrum_along_line`/`plot_bands`.
- `rtol`: forwarded to the classification or the declaration check.

Notes:
------
Labels are prettified to the conventional symbols (`GAMMA` -> `Γ`).
"""
function standard_kpath_nodes(lat::BravaisLattice{D}; rtol::Float64=BRAVAIS_RTOL,
                              bravais::Union{Nothing,Symbol}=nothing,
                              segments::Symbol=:first) where {D}
    segments in (:first, :all) || throw(ArgumentError(
        "`segments` must be `:first` (the first continuous segment of the recommended path) " *
        "or `:all` (every segment, ready for `kpath_line`'s seam gluing); got `:$segments`."))
    ### without a declaration, partially periodic lattices classify by their periodic block
    ### (see `periodic_block`): a ribbon gets its Γ-X and a slab its in-plane 2D path, kz
    ### pinned to 0. A declared `bravais` is validated (and padded lattices refused) upstream.
    pts = Dict(high_symmetry_points(lat; rtol=rtol, bravais=bravais))
    segs = high_symmetry_path(lat; rtol=rtol, bravais=bravais)
    nodes = [[_pretty_hs_label(l) => pts[l] for l in seg] for seg in segs]
    return segments === :first ? nodes[1] : nodes
end

standard_kpath_nodes(cell::UnitCell; kwargs...) = standard_kpath_nodes(cell.lattice; kwargs...)

### the conventional symbols for the tabulated ASCII names
function _pretty_hs_label(l::String)
    replacements = ("GAMMA" => "Γ", "DELTA" => "Δ", "SIGMA" => "Σ", "LAMBDA" => "Λ")
    for (from, to) in replacements
        l == from && return to
        startswith(l, from) && return to * l[length(from)+1:end]   # "SIGMA0" -> "Σ0"
    end
    return l
end
_node_coords(nd::Pair, D::Int) = _node_coords(last(nd), D)
function _node_coords(v::AbstractVector{<:Real}, D::Int)
    length(v) == D || throw(DimensionMismatch(
        "k-path node $(collect(v)) has length $(length(v)), expected the lattice dimension $D."))
    return SVector{D,Float64}(v)
end


#############################################################
###### Band structure along the path                    #####
#############################################################

"""
    spectrum_along_line(model, nodes; npoints=200, nev=nothing, which=:smallest,
                        sigma=nothing, kwargs...)
        -> (kpts, dist, ticks, labels, energies)

The band structure of `model` along the piecewise-linear path through `nodes`: `kpath_line`
for the geometry, one [`diagonalize`](@ref) per sample for the physics.

Args:
-----
- `nodes`, `npoints`: as in [`kpath_line`](@ref)
- `nev`, `which`, `sigma`, `kwargs...`: forwarded to `diagonalize`. The defaults compute the
  full spectrum; for a large continuum model ask for what you need instead --
  `nev=10, sigma=0.0` gives the ten bands around zero by shift-invert at a fraction of the
  dense cost.

Returns:
--------
- the `kpath_line` fields, plus `energies::Matrix{Float64}` of size `nbands × nk`
  (ascending within each column), ready to plot as `plot(dist, energies')` with vertical
  lines at `ticks`.

Notes:
------
The k-points are independent, so the loop is parallel over the Julia threads (see
[`threaded_chunks`](@ref); start Julia with `-t auto` to use it). Each chunk allocates ONE Hamiltonian
buffer, refills it per k-point and diagonalizes in place, so the batch performs no per-k-point
matrix allocation; and the Hermiticity sweep follows the `bands` policy -- skipped for pure bond
models, whose compiled list is Hermitian by construction.
"""
function spectrum_along_line(model::HamiltonianModel{D}, nodes::AbstractVector;
                             npoints::Int=200,
                             nev::Union{Nothing,Int}=nothing,
                             which::Symbol=:smallest,
                             sigma::Union{Nothing,Real}=nothing,
                             matrix::Symbol=:dense,
                             kwargs...) where {D}
    path = kpath_line(model.cell.lattice, nodes; npoints=npoints)
    nk = length(path.kpts)
    n = n_states(model.cell)
    nb = isnothing(nev) ? n : nev
    energies = Matrix{Float64}(undef, nb, nk)
    check = has_kdependence(model)      # constant lists are Hermitian by construction; see `bands`
    _check_matrix_mode(matrix, nev)

    threaded_chunks(nk) do chunk
        solve = bloch_solver(model; matrix=matrix)
        for idx in chunk
            r = solve(path.kpts[idx]; nev=nev, which=which, sigma=sigma, vectors=false,
                      check=check, kwargs...)
            energies[:, idx] = r.values
        end
    end
    return (kpts=path.kpts, dist=path.dist, ticks=path.ticks, labels=path.labels,
            energies=energies)
end

"""
    _check_matrix_mode(matrix, nev) -> Nothing

The entry validation behind the `matrix = :dense | :sparse` switch of the batch drivers:
`:sparse` computes a partial spectrum by construction, so `nev` is required there.
"""
function _check_matrix_mode(matrix::Symbol, nev::Union{Nothing,Int})::Nothing
    matrix in (:dense, :sparse) || throw(ArgumentError(
        "`matrix` must be `:dense` or `:sparse`; got `$matrix`."))
    matrix === :sparse && isnothing(nev) && throw(ArgumentError(
        "`matrix = :sparse` computes a partial spectrum through the Krylov paths; give `nev` " *
        "(and optionally `which` or `sigma`). For the full spectrum use `matrix = :dense`."))
    return nothing
end


"""
    spectrum_along_line(model; bravais=nothing, segments=:first, kwargs...)

Band structure along the lattice's own standard path: the nodes default to
[`standard_kpath_nodes`](@ref) of the model's lattice, so a moiré cell, a supercell or any
hand-built lattice gets its Setyawan-Curtarolo path without naming a Bravais type.

`bravais` DECLARES the Bravais type where the classifier cannot derive it -- 3D lattices, where
`identify_bravais` raises: `spectrum_along_line(model; bravais=:cF)` uses the declared type's
tabulated path after validating the declaration against the lattice metric (a wrong declaration
refuses by name; see [`standard_kpath_nodes`](@ref)).

`segments` is forwarded to `standard_kpath_nodes`: 3D recommended paths are mostly
multi-segment, and `segments=:all` glues every continuous run along the arc-length axis with
`"X|M"` seam ticks (`:first`, the default, keeps only the first run).

A model with `NoZone()` (a k·p frame) is refused HERE, with the default path
only: its "lattice" is a coordinate frame, so the tabulated Γ-X-M labels would name high-
symmetry points of a Brillouin zone that does not exist. Pass explicit `nodes` around the
expansion point instead -- the same policy `chern_number` and whole-BZ `berry_curvature`
already enforce.
"""
function spectrum_along_line(model::HamiltonianModel{D};
                             bravais::Union{Nothing,Symbol}=nothing,
                             segments::Symbol=:first, kwargs...) where {D}
    _require_zone(kperiodicity(model),
        "The default k-path is tabulated from the lattice's Bravais type, so Γ-X-M labels " *
        "need a Brillouin zone",
        "Pass explicit `nodes` (fractional momenta around the expansion point), e.g. " *
        "`spectrum_along_line(model, [\"-k\" => [-kf, 0], \"0\" => [0, 0], \"+k\" => [kf, 0]])`.")
    return spectrum_along_line(model, standard_kpath_nodes(model.cell.lattice; bravais=bravais,
                                                           segments=segments);
                               kwargs...)
end

"""
    eigenstates_along_line(model[, nodes]; npoints=200, nev=nothing, which=:smallest,
                           sigma=nothing, matrix=:dense, kwargs...)
        -> (kpts, dist, ticks, labels, energies, states, gauge)

The vector-carrying sibling of [`spectrum_along_line`](@ref): the same path geometry and the
same solver options, additionally returning the eigenvectors at every sampled k -- the entry
point for custom along-path analyses (spin textures, sublattice weights, overlaps).

Returns:
--------
- the `spectrum_along_line` fields, plus `states::Array{ComplexF64,3}` of size
  `n_states × nbands × nk`: `states[:, b, i]` is the eigenvector of `energies[b, i]` at
  `kpts[i]`, components in the cell's state order, in the model's stored gauge -- echoed as
  the `gauge` field, because eigenvectors in the two Bloch conventions differ by the
  k-dependent translator [`gauge_matrix`](@ref) and untagged vectors invite silently mixing
  them.

Notes:
------
Memory is the price of the vectors -- `n_states × nbands × nk` complex entries -- so ask for
the `nev` you need instead of the full spectrum on a large continuum model. The dense path
reuses one buffer per chunk and the vectors are copied out of it here: the returned array owns
its memory. Without `nodes` the lattice's own standard path is used, refusing
`NoZone()` models exactly as `spectrum_along_line` does, with the same `bravais=`
declaration route and the same `segments=` forwarding (3D recommended paths are mostly
multi-segment; `segments=:all` glues them with `"X|M"` seam ticks).
"""
function eigenstates_along_line(model::HamiltonianModel{D}, nodes::AbstractVector;
                                npoints::Int=200,
                                nev::Union{Nothing,Int}=nothing,
                                which::Symbol=:smallest,
                                sigma::Union{Nothing,Real}=nothing,
                                matrix::Symbol=:dense,
                                kwargs...) where {D}
    path = kpath_line(model.cell.lattice, nodes; npoints=npoints)
    nk = length(path.kpts)
    n = n_states(model.cell)
    nb = isnothing(nev) ? n : nev
    energies = Matrix{Float64}(undef, nb, nk)
    states = Array{ComplexF64,3}(undef, n, nb, nk)
    check = has_kdependence(model)      # constant lists are Hermitian by construction; see `bands`
    _check_matrix_mode(matrix, nev)

    threaded_chunks(nk) do chunk
        solve = bloch_solver(model; matrix=matrix)
        for idx in chunk
            r = solve(path.kpts[idx]; nev=nev, which=which, sigma=sigma, vectors=true,
                      check=check, kwargs...)
            energies[:, idx] = r.values
            ### the dense solver's buffer is refilled on the next iteration -- copy out NOW
            states[:, :, idx] = @view r.vectors[:, 1:nb]
        end
    end
    return (kpts=path.kpts, dist=path.dist, ticks=path.ticks, labels=path.labels,
            energies=energies, states=states, gauge=model.gauge)
end

function eigenstates_along_line(model::HamiltonianModel{D};
                                bravais::Union{Nothing,Symbol}=nothing,
                                segments::Symbol=:first, kwargs...) where {D}
    _require_zone(kperiodicity(model),
        "The default k-path is tabulated from the lattice's Bravais type, so Γ-X-M labels " *
        "need a Brillouin zone",
        "Pass explicit `nodes` (fractional momenta around the expansion point).")
    return eigenstates_along_line(model,
                                  standard_kpath_nodes(model.cell.lattice; bravais=bravais,
                                                       segments=segments);
                                  kwargs...)
end
