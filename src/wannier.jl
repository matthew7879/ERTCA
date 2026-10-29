# Paper-specific wannier routines.

function read_wannier90_hr(cell::UnitCell{D}, filename::AbstractString;
                           gauge::Symbol=:periodic, name::AbstractString="wannier90",
                           atol::Float64=1e-6, prune::Bool=false,
                           centres::Union{Nothing,AbstractString,AbstractVector}=nothing
                           )::HamiltonianModel{D} where {D}
    D <= 3 || throw(ArgumentError("Wannier90 files are at most three-dimensional; the cell has D = $D."))
    gauge in GAUGE_MODES || throw(ArgumentError(
        "Unknown gauge `$gauge`; expected one of $GAUGE_MODES."
    ))
    W = isnothing(centres) ? nothing :
        _wannier_wrap_shifts(cell, _wannier_centres_cartesian(centres, Val(D), atol);
                             atol=atol)
    if gauge === :atomic && isnothing(centres) && n_states(cell) > 1 &&
       all(b -> _same_site(b.r_frac, zero(SVector{D,Float64}), cell.atol), cell.basis)
        throw(ArgumentError(
            "Atomic-gauge Wannier90 import needs the true Wannier centres, but all " *
            "$(n_states(cell)) states of the cell sit at the origin. The :atomic gauge phases " *
            "couplings with e^{2πi k·(ΔR + τ_n - τ_m)}; with every τ = 0 it silently equals the " *
            ":periodic convention while claiming to be atomic, and Berry curvature, spectral " *
            "weight and every position-dependent quantity come out in the unphysical " *
            "zero-position convention. Either (a) build the cell with its states at the Wannier " *
            "centres, or pass `centres=` (a vector of cartesian positions, or the " *
            "seedname_centres.xyz path) to declare and verify them, or (b) keep the default " *
            "`gauge=:periodic`, which accepts the zero-position convention openly (spectra " *
            "are exact; position-dependent observables warn instead of claiming physicality)."
        ))
    end

    lines = readlines(filename)
    length(lines) >= 4 || throw(ArgumentError(
        "`$filename` is not a Wannier90 _hr.dat file: expected a comment line, num_wann, nrpts " *
        "and a degeneracy list."
    ))
    nw = parse(Int, strip(lines[2]))
    nrpts = parse(Int, strip(lines[3]))
    nw == n_states(cell) || throw(DimensionMismatch(
        "`$filename` holds num_wann = $nw Wannier functions but the cell declares " *
        "$(n_states(cell)) states. The cell must carry one state per Wannier function, in the " *
        "file's order."
    ))
    ndeg = Int[]
    row = 4
    while length(ndeg) < nrpts && row <= length(lines)
        append!(ndeg, parse.(Int, split(strip(lines[row]))))
        row += 1
    end
    length(ndeg) == nrpts || throw(ArgumentError(
        "`$filename` lists $(length(ndeg)) Wigner-Seitz degeneracies for nrpts = $nrpts."
    ))
    all(>(0), ndeg) || throw(ArgumentError("`$filename` carries a non-positive WS degeneracy."))
    table = Dict{Tuple{Int,Int,NTuple{D,Int}},ComplexF64}()
    rindex = Dict{NTuple{3,Int},Int}()
    for ln in @view lines[row:end]
        s = strip(ln)
        isempty(s) && continue
        f = split(s)
        length(f) >= 7 || throw(ArgumentError("Malformed _hr.dat line in `$filename`: `$ln`"))
        R3 = (parse(Int, f[1]), parse(Int, f[2]), parse(Int, f[3]))
        m, n = parse(Int, f[4]), parse(Int, f[5])
        (1 <= m <= nw && 1 <= n <= nw) || throw(ArgumentError(
            "`$filename` addresses Wannier index outside 1:$nw: `$ln`"
        ))
        all(R3[d] == 0 for d in (D+1):3) || throw(ArgumentError(
            "`$filename` carries the translation $R3, which is non-zero outside the cell's " *
            "$D dimension(s)."
        ))
        irpt = get!(rindex, R3, length(rindex) + 1)
        irpt <= nrpts || throw(ArgumentError(
            "`$filename` uses more distinct R vectors than its declared nrpts = $nrpts."
        ))
        key = (m, n, ntuple(d -> R3[d], D))
        haskey(table, key) && throw(ArgumentError(
            "`$filename` repeats the matrix element (R=$R3, m=$m, n=$n)."
        ))
        t = ComplexF64(parse(Float64, f[6]), parse(Float64, f[7])) / ndeg[irpt]
        iszero(t) || (table[key] = t)
    end
    buf = CouplingBuffer{D}()
    sizehint!(buf, length(table) ÷ 2 + 1)
    tables = cell_tables(cell)
    qof(i, j) = _endpoint_q(tables.G[i], tables.G[j], tables.k0[i], tables.k0[j])
    for ((m, n, Rt), t) in table
        R = SVector{D,Int}(Rt)
        partner = get(table, (n, m, ntuple(d -> -Rt[d], D)), nothing)
        if !isnothing(partner) && abs(partner - conj(t)) > atol
            throw(ArgumentError(
                "`$filename` is not Hermitian: H[$n,$m](-R) = $partner but conj(H[$m,$n](R)) = " *
                "$(conj(t)) at R = $Rt, deviation $(abs(partner - conj(t))) > atol = $atol."
            ))
        end
        Rc = isnothing(W) ? R : R + W[n] - W[m]
        if is_canonical_coupling(m, n, SVector{D,Rational{Int}}(Rc))
            push_coupling!(buf, m, n, Rc, qof(m, n), t, 1)
        elseif isnothing(partner)
            push_coupling!(buf, n, m, -Rc, qof(n, m), conj(t), 1)
        end
    end
    return HamiltonianModel(cell, buf; gauge=gauge, name=name, atol=atol, prune=prune)
end

function read_wannier90_hr(lattice::BravaisLattice{D}, filename::AbstractString;
                           centres::Union{Nothing,AbstractString,AbstractVector}=nothing,
                           schema::InternalSchema=TRIVIAL_SCHEMA,
                           site_names::Union{Nothing,AbstractVector{Symbol}}=nothing,
                           atol::Float64=1e-6, kwargs...)::HamiltonianModel{D} where {D}
    isnothing(centres) && throw(ArgumentError(
        "The BravaisLattice method builds the cell FROM the Wannier centres; pass `centres=` " *
        "(a vector of cartesian positions, or the seedname_centres.xyz path). To use a cell " *
        "you already built, call `read_wannier90_hr(cell, filename)` instead."
    ))
    cart = _wannier_centres_cartesian(centres, Val(D), atol)
    isempty(cart) && throw(ArgumentError("`centres` declares no Wannier centres."))
    reps = SVector{D,Float64}[]              # cluster representatives, canonicalized
    first_at = Int[]                         # the WF that opened each cluster, for diagnostics
    mult = Int[]                             # running multiplicity per cluster
    state_site = Vector{Int}(undef, length(cart))
    internal = Vector{Int}(undef, length(cart))
    for (n, c) in pairs(cart)
        fc = canonicalize_fracpos(real_to_fractional(c, lattice))
        s = findfirst(r -> _same_site(fc, r, LATTICE_ATOL), reps)
        if isnothing(s)
            push!(reps, fc); push!(first_at, n); push!(mult, 0)
            s = length(reps)
        end
        mult[s] += 1
        mult[s] <= n_internal(schema) || throw(ArgumentError(
            "Wannier centres $(first_at[s]) and $n coincide at $(Tuple(reps[s])) (fractional), " *
            "and the schema holds only $(n_internal(schema)) internal state(s). Co-centred " *
            "Wannier functions cannot each be their own localized site: pass `schema=` " *
            "declaring an internal axis with at least the shared multiplicity (the g-th " *
            "function at a centre, in file order, becomes the schema's g-th internal state), " *
            "or build the `UnitCell` yourself and call the cell method with `centres=`."
        ))
        state_site[n] = s
        internal[n] = mult[s]
    end

    names = if isnothing(site_names)
        [Symbol("w", s) for s in eachindex(reps)]
    else
        length(site_names) == length(reps) || throw(DimensionMismatch(
            "`site_names` holds $(length(site_names)) names for $(length(reps)) distinct " *
            "Wannier centres (one name per distinct centre, in first-appearance order)."
        ))
        collect(Symbol, site_names)
    end

    basis = [BasisState{D}(canonicalize_fracpos(real_to_fractional(cart[n], lattice)),
                           zero(SVector{D,Int}), internal[n], schema) for n in eachindex(cart)]
    cell = UnitCell{D}(lattice, schema, names, reps, basis;
                       site_kind=fill(:localized, length(reps)), state_site=state_site)
    return read_wannier90_hr(cell, filename; centres=cart, atol=atol, kwargs...)
end

function _wannier_centres_cartesian(centres::AbstractString, ::Val{D},
                                    atol::Float64)::Vector{SVector{D,Float64}} where {D}
    lines = readlines(centres)
    length(lines) >= 3 || throw(ArgumentError(
        "`$centres` is not a Wannier90 _centres.xyz file: expected a record count, a comment " *
        "line and the centre records."
    ))
    total = tryparse(Int, strip(lines[1]))
    isnothing(total) && throw(ArgumentError(
        "`$centres` is not an .xyz file: its first line (`$(strip(lines[1]))`) is not the " *
        "record count."
    ))
    raw = SVector{3,Float64}[]
    nrec = 0
    past_centres = false
    for ln in @view lines[3:end]
        s = strip(ln)
        isempty(s) && continue
        f = split(s)
        length(f) >= 4 || throw(ArgumentError("Malformed record in `$centres`: `$ln`"))
        nrec += 1
        if f[1] == "X"
            past_centres && throw(ArgumentError(
                "`$centres` holds an `X` centre record after an atom record; Wannier90 writes " *
                "all centres first, so the leading-`X` rule cannot identify the centres here."
            ))
            push!(raw, SVector{3,Float64}(parse(Float64, f[2]), parse(Float64, f[3]),
                                          parse(Float64, f[4])))
        else
            past_centres = true
        end
    end
    nrec == total || throw(ArgumentError(
        "`$centres` declares $total records but holds $nrec."
    ))
    isempty(raw) && throw(ArgumentError(
        "`$centres` holds no `X`-tagged Wannier-centre records."
    ))
    for (n, c) in pairs(raw)
        all(abs(c[d]) <= atol for d in (D+1):3) || throw(ArgumentError(
            "Wannier centre $n in `$centres` is $(Tuple(c)), non-zero outside the cell's " *
            "$D dimension(s)."
        ))
    end
    return [SVector{D,Float64}(ntuple(d -> c[d], D)) for c in raw]
end

function _wannier_centres_cartesian(centres::AbstractVector, ::Val{D},
                                    ::Float64)::Vector{SVector{D,Float64}} where {D}
    out = Vector{SVector{D,Float64}}(undef, length(centres))
    for (n, c) in pairs(centres)
        c isa AbstractVector{<:Real} || throw(ArgumentError(
            "`centres[$n]` is $(repr(c)); each centre must be a real-valued cartesian position " *
            "of length $D (or pass the seedname_centres.xyz path instead)."
        ))
        length(c) == D || throw(DimensionMismatch(
            "`centres[$n]` has length $(length(c)); centres are cartesian positions of the " *
            "cell's dimension D = $D."
        ))
        out[n] = SVector{D,Float64}(c)
    end
    return out
end

function _wannier_wrap_shifts(cell::UnitCell{D},
                              cart::Vector{SVector{D,Float64}};
                              atol::Float64=1e-6)::Vector{SVector{D,Int}} where {D}
    for s in 1:n_sites(cell)
        cell.site_kind[s] === :plane_wave && throw(ArgumentError(
            "`centres=` declares Wannier centres, but site `$(cell.site_names[s])` is " *
            ":plane_wave -- its position is gauge-only and never enters a phase. Wannier90 " *
            "import needs a cell of :localized states."
        ))
    end
    length(cart) == n_states(cell) || throw(DimensionMismatch(
        "`centres` declares $(length(cart)) Wannier centres but the cell holds " *
        "$(n_states(cell)) states; one centre per state, in the file's Wannier-function order."
    ))
    W = Vector{SVector{D,Int}}(undef, length(cart))
    for n in eachindex(cart)
        cfrac = real_to_fractional(cart[n], cell.lattice)
        d = cfrac - cell.basis[n].r_frac
        Wn = round.(d)
        tol = max(atol, cell.atol)
        all(abs.(d .- Wn) .<= tol) || throw(ArgumentError(
            "Wannier centre $n sits at $(Tuple(cfrac)) in fractional coordinates, but state $n " *
            "of the cell sits at $(Tuple(cell.basis[n].r_frac)): they differ by " *
            "$(Tuple(d .- Wn)) beyond an integer lattice translation (tolerance = " *
            "$(tol), the larger of the reader's `atol` and the cell's). The cell's states " *
            "must sit at the true Wannier centres, in the file's order -- rebuild the cell " *
            "at the centres, or use the `BravaisLattice` method, which builds it for you."
        ))
        W[n] = SVector{D,Int}(round.(Int, d))
    end
    return W
end

function read_wannier90_wout(path::AbstractString)
    isfile(path) || throw(ArgumentError("`read_wannier90_wout`: no such file `$path`."))
    txt = read(path, String)
    lm = match(r"Lattice Vectors \(Ang\)(.*?)Unit Cell Volume"s, txt)
    lat = nothing
    if lm !== nothing
        rows = SVector{3,Float64}[]
        for m in eachmatch(r"a_\d\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)", lm.captures[1])
            push!(rows, SVector{3,Float64}(parse.(Float64, m.captures)))
        end
        length(rows) == 3 && (lat = BravaisLattice(permutedims(hcat(rows...))))
    end
    fm = match(r"Final State(.*?)(?:Spreads|\z)"s, txt)
    fm === nothing && throw(ArgumentError(
        "`read_wannier90_wout`: `$path` has no `Final State` block, so this run did not " *
        "reach a converged answer (it may have stopped on an error -- check the log's tail, " *
        "and the `.werr` file beside it). Returning no centres would make a failed " *
        "Wannierization look like a successful one."))
    centres = SVector{3,Float64}[]
    spreads = Float64[]
    for m in eachmatch(r"WF centre and spread\s+\d+\s+\(\s*([-\d.Ee+]+)\s*,\s*([-\d.Ee+]+)\s*,\s*([-\d.Ee+]+)\s*\)\s+([-\d.Ee+]+)",
                       fm.captures[1])
        push!(centres, SVector{3,Float64}(parse.(Float64, m.captures[1:3])))
        push!(spreads, parse(Float64, m.captures[4]))
    end
    isempty(centres) && throw(ArgumentError(
        "`read_wannier90_wout`: `$path` has a `Final State` block with no " *
        "`WF centre and spread` records."))
    om = match(r"Omega Total\s*=\s*([-\d.Ee+]+)", txt)
    nw = match(r"Number of Wannier Functions\s*:\s*(\d+)", txt)
    nb = match(r"Number of input Bloch states\s*:\s*(\d+)", txt)
    return (lattice=lat, centres=centres, spreads=spreads,
            omega=om === nothing ? sum(spreads) : parse(Float64, om.captures[1]),
            converged=occursin("Wannierisation convergence criteria satisfied", txt),
            num_wann=nw === nothing ? length(centres) : parse(Int, nw.captures[1]),
            num_bands=nb === nothing ? nothing : parse(Int, nb.captures[1]))
end

function _w90_tokens(path::AbstractString, caller::AbstractString; skip_header::Bool)
    isfile(path) || throw(ArgumentError("`$caller`: no such file `$path`."))
    open(path, "r") do io
        skip_header && readline(io)          # the provenance comment every writer stamps
        return split(read(io, String))
    end
end

function read_wannier90_spn(path::AbstractString)
    toks = _w90_tokens(path, "read_wannier90_spn"; skip_header=true)
    length(toks) >= 2 || throw(ArgumentError(
        "`read_wannier90_spn`: `" * path * "` has no `num_bands num_kpts` header line after " *
        "the leading comment. If this is an UNFORMATTED `.spn` (the default `postw90.x` " *
        "writes and reads without `spn_formatted`), it is a Fortran record stream and not " *
        "readable here -- convert it with `w90spn2spn.x -export`."))
    nb, nk = parse.(Int, toks[1:2])
    ntri = div(nb * (nb + 1), 2)
    want = 2 + 6 * ntri * nk
    length(toks) == want || throw(ArgumentError(
        "`read_wannier90_spn`: `" * path * "` declares " * string(nb) * " bands and " *
        string(nk) * " k-points, so its upper triangle needs " * string(want) *
        " fields; it holds " * string(length(toks)) * ". Each of the " * string(ntri * nk) *
        " elements is three `re im` pairs (sigma_x, sigma_y, sigma_z)."))
    S = Array{ComplexF64,4}(undef, nb, nb, 3, nk)
    p = 3
    for i in 1:nk, m in 1:nb, n in 1:m               # upper triangle, column-major
        for a in 1:3
            z = complex(parse(Float64, toks[p]), parse(Float64, toks[p+1]))
            p += 2
            S[n, m, a, i] = z
            S[m, n, a, i] = conj(z)                  # as Wannier90 rebuilds it
        end
    end
    return (spin=S, num_bands=nb, num_kpoints=nk)
end
