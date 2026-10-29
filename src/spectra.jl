# Paper-specific spectra routines.

function kpath_line(lattice::BravaisLattice{D}, nodes::AbstractVector;
                    npoints::Int=200) where {D}
    return _kpath_through_nodes(lattice, nodes; npoints=npoints)
end

function kpath_line(lattice::BravaisLattice{D}, segments::AbstractVector{<:AbstractVector};
                    npoints::Int=200) where {D}
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
        g > 1 && (ticklabels[end] = ticklabels[end] * "|" * labels[g][1])
        acc = _sample_run!(kpts, dist, ticks, corners[g], seglen[g], total, npoints, acc)
        append!(ticklabels, @view labels[g][2:end])
    end
    return (kpts=kpts, dist=dist, ticks=ticks, labels=ticklabels)
end

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

function _kpath_through_nodes(lattice::BravaisLattice{D}, nodes::AbstractVector;
                              npoints::Int=200) where {D}
    length(nodes) >= 2 || throw(ArgumentError("A k-path needs at least two nodes."))
    npoints >= 2 || throw(DomainError(npoints, "`npoints` must be at least 2."))

    labels = String[_node_label(nd, i) for (i, nd) in enumerate(nodes)]
    corners = SVector{D,Float64}[_node_coords(nd, D) for nd in nodes]
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

function standard_kpath_nodes(lat::BravaisLattice{D}; rtol::Float64=BRAVAIS_RTOL,
                              bravais::Union{Nothing,Symbol}=nothing,
                              segments::Symbol=:first) where {D}
    segments in (:first, :all) || throw(ArgumentError(
        "`segments` must be `:first` (the first continuous segment of the recommended path) " *
        "or `:all` (every segment, ready for `kpath_line`'s seam gluing); got `:$segments`."))
    pts = Dict(high_symmetry_points(lat; rtol=rtol, bravais=bravais))
    segs = high_symmetry_path(lat; rtol=rtol, bravais=bravais)
    nodes = [[_pretty_hs_label(l) => pts[l] for l in seg] for seg in segs]
    return segments === :first ? nodes[1] : nodes
end

standard_kpath_nodes(cell::UnitCell; kwargs...) = standard_kpath_nodes(cell.lattice; kwargs...)

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

function _check_matrix_mode(matrix::Symbol, nev::Union{Nothing,Int})::Nothing
    matrix in (:dense, :sparse) || throw(ArgumentError(
        "`matrix` must be `:dense` or `:sparse`; got `$matrix`."))
    matrix === :sparse && isnothing(nev) && throw(ArgumentError(
        "`matrix = :sparse` computes a partial spectrum through the Krylov paths; give `nev` " *
        "(and optionally `which` or `sigma`). For the full spectrum use `matrix = :dense`."))
    return nothing
end

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
