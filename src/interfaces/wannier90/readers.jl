export read_wannier90_wout, read_wannier90_spn

function read_wannier90_wout(path::AbstractString)
    isfile(path) || throw(ArgumentError("`read_wannier90_wout`: no such file `$path`."))
    txt = read(path, String)
    ### the lattice
    lm = match(r"Lattice Vectors \(Ang\)(.*?)Unit Cell Volume"s, txt)
    lat = nothing
    if lm !== nothing
        rows = SVector{3,Float64}[]
        for m in eachmatch(r"a_\d\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)", lm.captures[1])
            push!(rows, SVector{3,Float64}(parse.(Float64, m.captures)))
        end
        length(rows) == 3 && (lat = BravaisLattice(permutedims(hcat(rows...))))
    end
    ### the final state: centres and spreads
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

