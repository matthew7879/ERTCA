#!/usr/bin/env julia
# Pack the 144 NbSe2 Wannier Hamiltonians after check_wannier.py succeeds.
# The archive stores all printed hoppings, lattice vectors, centres and spreads.
# Retain original .wout logs separately; their other contents are not archived.
using HDF5, Printf

const NEG_ZERO = typemin(Int32)
_hr_line(r1, r2, r3, m, n, re, im) =
    @sprintf("%5d%5d%5d%5d%5d%12.6f%12.6f", r1, r2, r3, m, n, re, im)

function _parse_hr(path::AbstractString)
    io = open(path)
    header = readline(io)
    nw = parse(Int, strip(readline(io)))
    nr = parse(Int, strip(readline(io)))
    ndegen = Int32[]
    while length(ndegen) < nr
        append!(ndegen, parse.(Int32, split(readline(io))))
    end
    length(ndegen) == nr || error("$path: degeneracy table has $(length(ndegen)) entries, expected $nr")
    rvecs = Matrix{Int32}(undef, 3, nr)
    vals = Vector{Int32}(undef, 2 * nw * nw * nr)
    k = 0
    for ir in 1:nr, n in 1:nw, m in 1:nw
        p = split(readline(io))
        length(p) == 7 || error("$path: malformed hopping line `$(join(p, ' '))`")
        if m == 1 && n == 1
            rvecs[1, ir] = parse(Int32, p[1])
            rvecs[2, ir] = parse(Int32, p[2])
            rvecs[3, ir] = parse(Int32, p[3])
        end
        for s in (p[6], p[7])
            x = parse(Float64, s) * 1_000_000
            q = round(Int32, x)
            abs(x - q) < 1e-3 || error("$path: value `$s` is not a whole number of μeV — " *
                                       "this deck is not the six-decimal Wannier90 dialect " *
                                       "this packer preserves exactly")
            vals[k += 1] = (q == 0 && startswith(lstrip(s), '-')) ? NEG_ZERO : q
        end
    end
    close(io)
    return header, nw, rvecs, ndegen, vals
end

function _parse_wout(path::AbstractString)
    lines = readlines(path)
    a = zeros(3, 3)
    got = false
    for (n, l) in pairs(lines)
        occursin("Lattice Vectors (Ang)", l) || continue
        for m in 1:3
            a[m, :] = parse.(Float64, split(lines[n+m])[2:4])
        end
        got = true
        break
    end
    got || error("no `Lattice Vectors (Ang)` block in $path")
    cen = Vector{Float64}[]
    spr = Float64[]
    infinal = false
    for l in lines
        if occursin("Final State", l)
            infinal = true
            continue
        end
        infinal || continue
        occursin("WF centre and spread", l) || break
        inner = l[findfirst('(', l)+1:findfirst(')', l)-1]
        push!(cen, parse.(Float64, split(inner, ',')))
        push!(spr, parse(Float64, split(l[findfirst(')', l)+1:end])[1]))
    end
    isempty(cen) && error("no `Final State` Wannier centres in $path")
    return a, reduce(hcat, cen), spr
end

function _render_hr(header::AbstractString, nw::Integer, rvecs::AbstractMatrix,
                    ndegen::AbstractVector, vals::AbstractVector)
    nr = length(ndegen)
    io = IOBuffer()
    println(io, header)
    println(io, lpad(nw, 12))
    println(io, lpad(nr, 12))
    for a in 1:15:nr
        println(io, join((lpad(ndegen[b], 5) for b in a:min(a + 14, nr)), ""))
    end
    k = 0
    for ir in 1:nr, n in 1:nw, m in 1:nw
        re = vals[k+1] == NEG_ZERO ? -0.0 : vals[k+1] / 1_000_000
        im = vals[k+2] == NEG_ZERO ? -0.0 : vals[k+2] / 1_000_000
        k += 2
        println(io, _hr_line(rvecs[1, ir], rvecs[2, ir], rvecs[3, ir], m, n, re, im))
    end
    return String(take!(io))
end

function pack_grid(srcdir::AbstractString, archive::AbstractString)
    (ispath(archive) || islink(archive)) && error("Output exists; choose a new archive: $archive")
    ij = [(i, j) for i in 0:11 for j in 0:11]
    headers = Vector{String}(undef, 144)
    centres = Array{Float64}(undef, 3, 4, 144)
    spreads = Matrix{Float64}(undef, 4, 144)
    lat0, rv0, nd0, values = nothing, nothing, nothing, nothing
    for (index, (i, j)) in pairs(ij)
        folder = joinpath(srcdir, @sprintf("shift_i%02d_j%02d", i, j))
        a, cen, spr = _parse_wout(joinpath(folder, "NbSe2.wout"))
        hrpath = joinpath(folder, "NbSe2_hr.dat")
        hdr, nw, rv, nd, vals = _parse_hr(hrpath)
        nw == 4 && size(cen) == (3, 4) && length(spr) == 4 ||
            error("$folder: expected four Wannier functions")
        all(isfinite, a) && all(isfinite, cen) && all(isfinite, spr) ||
            error("$folder: non-finite geometry or spread")
        if index == 1
            lat0, rv0, nd0 = a, rv, nd
            values = Matrix{Int32}(undef, 144, length(vals))
        else
            a == lat0 || error("$folder: inconsistent lattice")
            rv == rv0 && nd == nd0 || error("$folder: inconsistent R-vector grid")
        end
        # Check every byte before creating an archive, including signs of zero.
        _render_hr(hdr, nw, rv, nd, vals) == read(hrpath, String) ||
            error("$hrpath: cannot reproduce the exact input text; no archive written")
        headers[index] = hdr
        centres[:, :, index] = cen
        spreads[:, index] = spr
        values[index, :] = vals
    end
    datasets = (lattice=lat0, rvectors=rv0, ndegen=nd0,
                shift_index=Int32[p[k] for k in 1:2, p in ij],
                centres=centres, spreads=spreads, headers=headers,
                hoppings_ueV=values)
    mkpath(dirname(abspath(archive)))
    temporary, io = mktemp(dirname(abspath(archive)))
    close(io)
    try
        h5open(temporary, "w") do f
            attrs(f)["format"] = "ertca-registry-deck"
            attrs(f)["version"] = 1
            attrs(f)["num_wann"] = 4
            attrs(f)["ngrid"] = 12
            attrs(f)["seedname"] = "NbSe2"
            attrs(f)["units"] = "hoppings: Int32 micro-eV; lattice and centres: Angstrom; spreads: Angstrom^2"
            attrs(f)["layout"] = "hoppings[stacking, value], stacking index fastest"
            for (name, data) in pairs(datasets)
                if name == :hoppings_ueV
                    d = create_dataset(f, String(name), Int32, size(data);
                        chunk=(144, min(1024, size(data, 2))), shuffle=true, deflate=9)
                    write(d, data)
                else
                    f[String(name)] = data
                end
            end
        end
        h5open(temporary, "r") do f
            for (name, data) in pairs(datasets)
                read(f[String(name)]) == data || error("Archive verification failed: $name")
            end
        end
        mv(temporary, archive; force=false)
    finally
        isfile(temporary) && rm(temporary)
    end
    println("Packed and verified 144 stackings: ", archive)
    return archive
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: julia --project=<ERTCA> pack_grid.jl GRID NEW_ARCHIVE.h5")
    pack_grid(ARGS[1], ARGS[2])
end
