##################################################################
###### export only the following functions from this module ######
##################################################################

export read_wannier90_centres, write_wannier90_centres

### `registry_scan`'s file path needs `_hr.dat` AND `_centres.xyz`, and this package could read
### the second but never write it -- so a model exported by `write_wannier90_hr` could not be
### round-tripped back through the scanner without hand-writing the centres. This file closes
### that asymmetry and promotes the parser, which lived privately in `hr.jl`, to public API.
###
### The centres matter because the ATOMIC gauge is pinned to them: `read_wannier90_hr` uses
### them to recover each state's wrap shift `W_n` and compensate `R -> R + W_n - W_m`, so a
### file whose centres are lost imports correctly only in the periodic convention.




"""
    read_wannier90_centres(filename; dim=3, atol=1e-6) -> NamedTuple

Parse a Wannier90 `seedname_centres.xyz`, returning `(centres, atoms, comment)`.

`dim` is the dimension to return positions in: an xyz file is always three-dimensional, but a
`D < 3` model wants `D`-component centres, and `read_wannier90_hr`'s `centres=` keyword
requires exactly that. Trailing components must then vanish within `atol` -- a genuinely
out-of-plane centre handed to a two-dimensional cell is a refusal, not a silent projection.

`centres` are the cartesian Wannier centres -- the **leading `X`-tagged** records -- in file
order, and `atoms` are the trailing real-element records as `Symbol => position` pairs.
Wannier90 writes the Wannier functions first and the atomic structure after, so an `X` record
appearing *after* a real element cannot be identified as a centre and is refused rather than
guessed at.

Positions are in the units of the file, which for genuine Wannier90 output is Angstrom. This
package is unit-agnostic: nothing here converts, and the numbers are meaningful only against a
lattice given in the same units.

The record count on line 1 must match the number of records actually present -- a truncated
file is a refusal, not a short list.
"""
function read_wannier90_centres(filename::AbstractString; dim::Integer=3, atol::Float64=1e-6)
    1 <= dim <= 3 || throw(ArgumentError(
        "`read_wannier90_centres`: `dim = $dim` is outside 1:3 -- an xyz file is at most " *
        "three-dimensional."))
    isfile(filename) || throw(ArgumentError(
        "`read_wannier90_centres`: no such file `$filename`."))
    lines = readlines(filename)
    length(lines) >= 2 || throw(ArgumentError(
        "`read_wannier90_centres`: `$filename` has $(length(lines)) lines; an xyz file needs " *
        "at least a count line and a comment line."))
    nrec = tryparse(Int, strip(lines[1]))
    isnothing(nrec) && throw(ArgumentError(
        "`read_wannier90_centres`: line 1 of `$filename` is `$(strip(lines[1]))`, not the " *
        "record count an xyz file starts with."))
    comment = strip(lines[2])
    recs = Tuple{Symbol,SVector{3,Float64}}[]
    for l in lines[3:end]
        isempty(strip(l)) && continue
        f = split(strip(l))
        length(f) >= 4 || throw(ArgumentError(
            "`read_wannier90_centres`: record line `$(strip(l))` in `$filename` has " *
            "$(length(f)) fields; an xyz record needs a tag and three coordinates."))
        v = SVector{3,Float64}(parse(Float64, f[2]), parse(Float64, f[3]), parse(Float64, f[4]))
        push!(recs, (Symbol(f[1]), v))
    end
    length(recs) == nrec || throw(DimensionMismatch(
        "`read_wannier90_centres`: `$filename` declares $nrec records but holds " *
        "$(length(recs)). A truncated or padded centres file is refused rather than read " *
        "short -- the count is what pairs the centres with a model's states."))
    ### the leading-X rule: centres first, atoms after; an X after an atom is not identifiable
    nx = something(findfirst(r -> r[1] !== :X, recs), length(recs) + 1) - 1
    any(r -> r[1] === :X, recs[(nx + 1):end]) && throw(ArgumentError(
        "`read_wannier90_centres`: `$filename` has an `X` record after a real-element " *
        "record, so the leading-`X` rule cannot identify the centres here. Wannier90 writes " *
        "every Wannier function before the atomic structure; a file that interleaves them " *
        "needs splitting by hand."))
    ### project onto `dim`, refusing a centre that genuinely leaves the modelled subspace
    function trim(v::SVector{3,Float64}, what::AbstractString, n::Integer)
        for d in (dim + 1):3
            abs(v[d]) <= atol || throw(ArgumentError(
                "`read_wannier90_centres`: $what $n sits at $(Tuple(v)), whose component " *
                "$d is $(v[d]) -- above `atol = $atol`. Asked for `dim = $dim`, this " *
                "position leaves the modelled subspace, so it is refused rather than " *
                "projected. Read with `dim = 3` if the model is genuinely three-dimensional."))
        end
        return SVector{dim,Float64}(ntuple(d -> v[d], dim))
    end
    return (centres=[trim(recs[i][2], "centre", i) for i in 1:nx],
            atoms=[recs[i][1] => trim(recs[i][2], "atom", i - nx)
                   for i in (nx + 1):length(recs)],
            comment=String(comment))
end

"""
    write_wannier90_centres(filename, cell::UnitCell{D}; atoms=Pair[], note="") -> String

Write a Wannier90 `seedname_centres.xyz` holding the cell's localized state positions as the
`X`-tagged Wannier centres, in the lattice's own units.

This is the file [`read_wannier90_hr`](@ref) wants back through its `centres=` keyword, and the
one [`registry_scan`](@ref)'s file path requires: emitting it makes
`write_wannier90_hr` + `write_wannier90_centres` a complete, re-importable pair, where the
`_hr.dat` alone loses the centres and therefore the atomic gauge.

Positions are written in **state order** (`STATE_ORDER`), one `X` record per state, so the
`n`-th record pairs with the `n`-th column of the exported Hamiltonian. Co-centred states -- a
spinor manifold, say -- legitimately repeat the same position, exactly as Wannier90 does.

Pass `atoms` as `Symbol => position` pairs (cartesian, same units) to append the atomic
structure after the centres; Wannier90 itself writes those, and the reader above returns them.

Refuses a cell carrying `:plane_wave` sites: a plane wave has no centre, its stored position is
gauge-only, and writing it as a Wannier centre would produce a file that imports as a
completely different model.
"""
function write_wannier90_centres(filename::AbstractString, cell::UnitCell{D};
                                 atoms::AbstractVector=Pair{Symbol,Vector{Float64}}[],
                                 note::AbstractString="")::String where {D}
    D <= 3 || throw(ArgumentError(
        "`write_wannier90_centres`: an xyz file is at most three-dimensional, and this cell " *
        "is $D-dimensional."))
    bad = findfirst(k -> k === :plane_wave, cell.site_kind)
    isnothing(bad) || throw(ArgumentError(
        "`write_wannier90_centres`: site `$(cell.site_names[bad])` is `:plane_wave`, whose " *
        "stored position is gauge-only rather than a physical centre. Writing it as a Wannier " *
        "centre would describe a different model on re-import. Export the localized " *
        "subsystem instead (see `submodel`)."))
    ### validate the atom records BEFORE the "w" open truncates the target
    arecs = Tuple{Symbol,SVector{3,Float64}}[]
    for (i, p) in enumerate(atoms)
        p isa Pair || throw(ArgumentError(
            "`write_wannier90_centres`: `atoms[$i]` is a $(typeof(p)); each entry must be a " *
            "`Symbol => position` pair."))
        v = collect(Float64, last(p))
        length(v) <= 3 || throw(ArgumentError(
            "`write_wannier90_centres`: the position of atom `$(first(p))` has $(length(v)) " *
            "components; at most three."))
        push!(arecs, (Symbol(first(p)), SVector{3,Float64}(ntuple(d -> d <= length(v) ? v[d] : 0.0, 3))))
    end
    pos = [fractional_to_real(b.r_frac, cell.lattice) for b in cell.basis]
    v = Base.pkgversion(@__MODULE__)
    stamp = Dates.format(Dates.now(Dates.UTC), Dates.dateformat"yyyy-mm-dd\THH:MM:SS\Z")
    open(filename, "w") do io
        println(io, lpad(length(pos) + length(arecs), 6))
        println(io, "written by SingleParticleModels.jl v", isnothing(v) ? "?" : v,
                " on ", stamp, isempty(note) ? "" : " -- " * String(note))
        for p in pos
            c = ntuple(d -> d <= D ? p[d] : 0.0, 3)
            println(io, rpad("X", 3), lpad(c[1], 20), lpad(c[2], 20), lpad(c[3], 20))
        end
        for (s, c) in arecs
            println(io, rpad(String(s), 3), lpad(c[1], 20), lpad(c[2], 20), lpad(c[3], 20))
        end
    end
    return String(filename)
end
