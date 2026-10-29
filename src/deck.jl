# =====================================================================================
#  The registry-scan deck codec — one archive per material
#
#  An ERTCA case study needs 144 Wannier90 calculations of an untwisted bilayer, one per
#  rigid registry shift d = (i/12, j/12). As Wannier90 ships them — a `.wout` and a
#  `_hr.dat` per directory — that is 56 MB of ASCII for NbSe₂ and 87 MB for WSe₂, far too
#  much to carry in a git repository. This file is the codec that makes each ~1.4 MB
#  instead, losslessly:
#
#    * `<seed>.wout` is 77 kB (NbSe₂) to 300 kB (WSe₂) of Wannier90 log per stacking, of
#      which the loader reads about twenty-five numbers (the lattice, the Wannier centres,
#      the spreads). Only those are kept — the lattice is byte-identical across all 144
#      stackings, so it is stored once. NOTE the asymmetry: the `_hr.dat` round-trips byte
#      for byte, the `.wout` does NOT. Keep the original logs elsewhere if the
#      disentanglement history matters to you.
#    * `<seed>_hr.dat` prints every hopping to SIX DECIMALS, so every value in the file is
#      an exact integer multiple of 1 μeV. Storing those integers rather than their
#      decimal text is exact, not approximate — verified on all 1 884 672 of them.
#    * the R-vector list (409 vectors, all in-plane) and the degeneracy table are the same
#      in all 144 files — the Wannierization grid does not move with the registry — so
#      they too are stored once.
#    * the hopping integers are laid out VALUE-MAJOR (stacking index fastest), which puts
#      the same matrix element at all 144 registries next to each other. Neighbouring
#      registries have hoppings of very similar magnitude, so the deflate filter sees long
#      runs of equal high-order bytes: 2.4 MB stacking-major becomes 1.3 MB this way, for
#      free.
#
#  The container is HDF5, which the package already depends on — no new dependency, and
#  `h5_info`-style inspection works on it out of the box.
#
#  Nothing downstream knows about any of this. `ensure_deck!` expands the archive back
#  into the genuine Wannier90 directory tree the first time an example runs, and the
#  regenerated `_hr.dat` files are BYTE-IDENTICAL to the originals (`pack_deck` checks this
#  when it builds the archive). The loader then reads real `_hr.dat` files through
#  `read_wannier90_hr`, exactly as it would on your own DFT output.
#
#  Regenerate an archive from a directory tree:
#      julia --project=. paper/Code/deck.jl pack WSe2 <srcdir> <archive>
#  Expand one by hand (the examples do it for you):
#      julia --project=. paper/Code/deck.jl unpack <archive> <destdir>
# =====================================================================================

using HDF5
using Printf
using LinearAlgebra: det

### The tag written into new archives. `NbSe2_grid_12x12.h5` was committed before the codec
### was generalised and carries the older material-specific tag, so both are accepted on read
### rather than re-committing a 1.4 MB binary for a string change.
const DECK_FORMAT = "ertca-registry-deck"
const DECK_FORMAT_LEGACY = "nbse2-registry-deck"

### The Wannier90 `_hr.dat` fixed-width dialect: five I5 integers then two F12.6 reals.
### `NEG_ZERO` is the encoded stand-in for a printed `-0.000000` (see `_parse_hr`); it is
### `typemin(Int32)`, which no genuine μeV hopping can reach — the largest here is 5.2 eV.
const NEG_ZERO = typemin(Int32)

_hr_line(r1, r2, r3, m, n, re, im) =
    @sprintf("%5d%5d%5d%5d%5d%12.6f%12.6f", r1, r2, r3, m, n, re, im)

"""
    _parse_hr(path) -> (header, num_wann, rvectors, ndegen, values_ueV)

Parse a Wannier90 `_hr.dat`. `rvectors` is `3 × nrpts` Int32, `ndegen` is `nrpts` Int32,
and `values_ueV` is the flat `2·num_wann²·nrpts` Int32 vector of the printed reals in
units of 1 μeV, in file order (real part then imaginary part of each entry).

The μeV conversion is asserted to be exact: Wannier90 prints six decimals, so a value that
does not land on the microvolt grid means the file is in a different dialect than this
codec assumes, and packing it would lose information silently.
"""
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
                                       "`deck.jl` packs losslessly")
            ### Wannier90 prints `-0.000000` for a value in (-0.5 μeV, 0), and that minus
            ### sign is the only thing separating it from `0.000000` in the text. Both round
            ### to the same integer, so signed zero gets its own sentinel; without it the
            ### round trip is exact in VALUE but not byte-for-byte, and 1228 lines of the
            ### first stacking alone would come back changed.
            vals[k += 1] = (q == 0 && startswith(lstrip(s), '-')) ? NEG_ZERO : q
        end
    end
    close(io)
    return header, nw, rvecs, ndegen, vals
end

"""
    _parse_wout(path) -> (lattice_3x3, centres_3x4, spreads_4)

The three numbers the loader needs out of a 77 kB Wannier90 log: the printed lattice
vectors (row `m` is `aₘ`, Å), the final Wannier centres (cartesian Å) and their spreads.
"""
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

"""
    pack_deck(seedname, srcdir, archive; ngrid=12) -> String

Pack a `shift_i<ii>_j<jj>/` tree of Wannier90 output into the compressed archive, and
verify the pack by regenerating every `_hr.dat` in memory and comparing it BYTE FOR BYTE
with the original. Returns the archive path.

`seedname` is the Wannier90 seed, i.e. the files are `<seedname>.wout` and
`<seedname>_hr.dat` inside each stacking directory. It is recorded in the archive so that
[`unpack_deck`](@ref) writes the tree back under the same names.
"""
function pack_deck(seedname::AbstractString, srcdir::AbstractString,
                   archive::AbstractString; ngrid::Int=12)
    ij = [(i, j) for i in 0:(ngrid-1) for j in 0:(ngrid-1)]
    ns = length(ij)
    headers = Vector{String}(undef, ns)
    local centres::Array{Float64,3}
    local spreads::Matrix{Float64}
    lat0, rv0, nd0, nw0 = nothing, nothing, nothing, 0
    values = nothing                       # ns × nval, stacking index FASTEST
    for (s, (i, j)) in pairs(ij)
        dir = joinpath(srcdir, "shift_i" * lpad(i, 2, '0') * "_j" * lpad(j, 2, '0'))
        isdir(dir) || error("missing stacking directory $dir")
        a, cen, spr = _parse_wout(joinpath(dir, "$(seedname).wout"))
        hdr, nw, rv, nd, vals = _parse_hr(joinpath(dir, "$(seedname)_hr.dat"))
        if s == 1
            lat0, rv0, nd0, nw0 = a, rv, nd, nw
            values = Matrix{Int32}(undef, ns, length(vals))
            ### the number of Wannier functions is whatever the deck declares, not a constant
            centres = Array{Float64,3}(undef, 3, nw, ns)
            spreads = Matrix{Float64}(undef, nw, ns)
        else
            a == lat0 || error("$dir prints a different lattice than the first stacking; " *
                               "this codec stores it once")
            (rv == rv0 && nd == nd0 && nw == nw0) ||
                error("$dir has a different R-vector grid than the first stacking; " *
                      "this codec stores it once")
        end
        size(cen, 2) == nw0 || error("$dir declares $(size(cen, 2)) Wannier centres in its " *
                                     ".wout but $nw0 in its _hr.dat")
        headers[s] = hdr
        centres[:, :, s] = cen
        spreads[:, s] = spr
        values[s, :] = vals
    end

    mkpath(dirname(archive))
    h5open(archive, "w") do f
        attrs(f)["format"] = DECK_FORMAT
        attrs(f)["version"] = 1
        attrs(f)["num_wann"] = nw0
        attrs(f)["ngrid"] = ngrid
        attrs(f)["seedname"] = seedname
        attrs(f)["units"] = "hoppings are Int32 in μeV (Wannier90 prints six decimals, so this is exact); lattice, centres and spreads are Å"
        attrs(f)["layout"] = "hoppings[s, v]: stacking index FASTEST, so one matrix element across all registries is contiguous — that is what makes the deflate filter effective"
        f["lattice"] = lat0
        f["rvectors"] = rv0
        f["ndegen"] = nd0
        f["shift_index"] = Int32[c[k] for k in 1:2, c in ij]
        f["centres"] = centres
        f["spreads"] = spreads
        f["headers"] = headers
        ### chunk along the contiguous (stacking) axis; shuffle groups the high-order bytes
        ### of the small integers together, which is where the compression comes from
        d = create_dataset(f, "hoppings_ueV", Int32, size(values);
                           chunk=(size(values, 1), min(1024, size(values, 2))),
                           shuffle=true, deflate=9)
        write(d, values)
    end

    ### the pack is only trustworthy if it round-trips exactly — check every byte
    for (s, (i, j)) in pairs(ij)
        name = "shift_i" * lpad(i, 2, '0') * "_j" * lpad(j, 2, '0')
        got = _render_hr(headers[s], nw0, rv0, nd0, view(values, s, :))
        want = read(joinpath(srcdir, name, "$(seedname)_hr.dat"), String)
        got == want || error("round-trip mismatch for $name — the pack is NOT lossless")
    end
    @info "packed $ns stackings" seedname archive size_MB = round(filesize(archive) / 1e6; digits = 2)
    return archive
end

"""
    _render_hr(header, num_wann, rvectors, ndegen, values_ueV) -> String

Regenerate the exact `_hr.dat` text: the header line, `num_wann`, `nrpts`, the degeneracy
table fifteen entries to a line, then the hoppings in `%5d%5d%5d%5d%5d%12.6f%12.6f`.
"""
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

"""
    _render_wout(lattice, centres, spreads) -> String

A minimal `.wout` carrying exactly the blocks `read_wannier90_wout` needs: the printed
lattice vectors, the `Unit Cell Volume` line that terminates that block, and the
final-state Wannier centres with their spreads. This is NOT the original 77 kB Wannier90
log — the convergence history, the disentanglement report and the run parameters are not
reconstructible and are not needed. The file says so in its own first line.

The volume line carries no information the loader uses; it is emitted because the package
reader delimits the lattice block as `Lattice Vectors (Ang) ... Unit Cell Volume`, exactly
as real Wannier90 output does. Leaving it out would make the reader return no lattice at
all rather than fail, which is the quiet kind of wrong this repository refuses.
"""
function _render_wout(a::AbstractMatrix, cen::AbstractMatrix, spr::AbstractVector)
    io = IOBuffer()
    println(io, " Reconstructed by examples/ertca/deck.jl -- the two blocks the ERTCA")
    println(io, " loader reads. NOT the original Wannier90 log.")
    println(io)
    println(io, "                                    Lattice Vectors (Ang)")
    for m in 1:3
        @printf(io, "                 a_%d %11.6f %11.6f %11.6f\n", m, a[m, 1], a[m, 2], a[m, 3])
    end
    println(io)
    @printf(io, "                   Unit Cell Volume: %14.5f  (Ang^3)\n", abs(det(a)))
    println(io)
    println(io, " Final State")
    for w in axes(cen, 2)
        @printf(io, "  WF centre and spread %5d  ( %10.6f, %10.6f, %10.6f )  %14.8f\n",
                w, cen[1, w], cen[2, w], cen[3, w], spr[w])
    end
    return String(take!(io))
end

"""
    unpack_deck(archive, destdir) -> String

Expand the archive into the `shift_i<ii>_j<jj>/` tree of Wannier90 files, named after the
seed recorded in the archive. The `_hr.dat` files are byte-identical to the originals; the
`.wout` files carry only the two blocks the loader reads (see [`_render_wout`](@ref)).
Returns the destination directory.
"""
function unpack_deck(archive::AbstractString, destdir::AbstractString)
    isfile(archive) || error("no registry-deck archive at $archive")
    h5open(archive, "r") do f
        fmt = attrs(f)["format"]
        fmt in (DECK_FORMAT, DECK_FORMAT_LEGACY) ||
            error("$archive is not an ERTCA registry deck (format = \"$fmt\")")
        nw = attrs(f)["num_wann"]
        ### written since the first archive, but only READ since the codec was generalised:
        ### without it a WSe₂ archive would unpack to NbSe2_* filenames.
        seedname = haskey(attrs(f), "seedname") ? attrs(f)["seedname"] : "NbSe2"
        lat = read(f["lattice"])
        rv = read(f["rvectors"])
        nd = read(f["ndegen"])
        ij = read(f["shift_index"])
        cen = read(f["centres"])
        spr = read(f["spreads"])
        hdr = read(f["headers"])
        vals = read(f["hoppings_ueV"])
        for s in axes(ij, 2)
            name = "shift_i" * lpad(ij[1, s], 2, '0') * "_j" * lpad(ij[2, s], 2, '0')
            dir = joinpath(destdir, name)
            mkpath(dir)
            write(joinpath(dir, "$(seedname)_hr.dat"),
                  _render_hr(hdr[s], nw, rv, nd, view(vals, s, :)))
            write(joinpath(dir, "$(seedname).wout"),
                  _render_wout(lat, view(cen, :, :, s), view(spr, :, s)))
        end
        @info "expanded the $seedname deck" destdir stackings = size(ij, 2)
    end
    return destdir
end

"""
    ensure_deck!(seedname, archive, destdir; ngrid=12) -> String

The directory of Wannier90 stackings, expanding the committed archive on first use.
Idempotent, and cheap once expanded.

Completeness is checked over EVERY stacking, not just one: probing a single file cannot
tell a finished tree from an expansion that died halfway, and the failure mode of getting
that wrong is a silently truncated registry scan. The check is also what protects an
ORIGINAL Wannier90 tree sitting at `destdir` — a complete tree is never rewritten, so the
full `.wout` logs the archive cannot reproduce are left alone.
"""
function ensure_deck!(seedname::AbstractString, archive::AbstractString,
                      destdir::AbstractString; ngrid::Int=12)
    ns = ngrid * ngrid
    present = isdir(destdir) && count(0:(ns-1)) do k
        name = "shift_i" * lpad(k ÷ ngrid, 2, '0') * "_j" * lpad(k % ngrid, 2, '0')
        isfile(joinpath(destdir, name, "$(seedname)_hr.dat"))
    end == ns
    present && return destdir
    @info "expanding the $seedname Wannier90 deck from $(basename(archive)) " *
          "(tens of MB of text, written once; the directory is git-ignored)"
    return unpack_deck(archive, destdir)
end

### Script entry point:
###   julia --project=. paper/Code/deck.jl pack   <seedname> <srcdir> <archive>
###   julia --project=. paper/Code/deck.jl unpack <archive> <destdir>
if abspath(PROGRAM_FILE) == @__FILE__
    cmd = isempty(ARGS) ? "" : ARGS[1]
    if cmd == "pack" && length(ARGS) == 4
        pack_deck(ARGS[2], ARGS[3], ARGS[4])
    elseif cmd == "unpack" && length(ARGS) == 3
        unpack_deck(ARGS[2], ARGS[3])
    else
        error("""usage:
              deck.jl pack   <seedname> <srcdir> <archive>
              deck.jl unpack <archive> <destdir>""")
    end
end
