# Paper-specific common routines.

using HDF5

using Printf

using LinearAlgebra: det

const DECK_FORMAT = "ertca-registry-deck"

const DECK_FORMAT_LEGACY = "nbse2-registry-deck"

const NEG_ZERO = typemin(Int32)

_hr_line(r1, r2, r3, m, n, re, im) =
    @sprintf("%5d%5d%5d%5d%5d%12.6f%12.6f", r1, r2, r3, m, n, re, im)

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

function unpack_deck(archive::AbstractString, destdir::AbstractString)
    isfile(archive) || error("no registry-deck archive at $archive")
    h5open(archive, "r") do f
        fmt = attrs(f)["format"]
        fmt in (DECK_FORMAT, DECK_FORMAT_LEGACY) ||
            error("$archive is not an ERTCA registry deck (format = \"$fmt\")")
        nw = attrs(f)["num_wann"]
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

using ERTCA

using LinearAlgebra

using StaticArrays

using Logging

const ERTCA_DATA_DIR = joinpath(@__DIR__, "..", "Inputs")

const ERTCA_DECK_CACHE = joinpath(@__DIR__, "..", ".cache")

struct CountingLogger <: Logging.AbstractLogger
    count::Base.RefValue{Int}
end

Logging.min_enabled_level(::CountingLogger) = Logging.Warn

Logging.shouldlog(::CountingLogger, args...) = true

Logging.catch_exceptions(::CountingLogger) = false

function Logging.handle_message(l::CountingLogger, level, message, _mod, _grp, _id,
                                _file, _line; kwargs...)
    level >= Logging.Warn && (l.count[] += 1)
    return nothing
end

function quiet(f)
    lg = CountingLogger(Ref(0))
    res = with_logger(f, lg)
    return res, lg.count[]
end

struct ERTCAMaterial
    seedname::String
    ngrid::Int
    archive::String
    deckdir::String
    site_names::Vector{Symbol}
    match_atol::Float64
    filling::Union{Rational{Int},Nothing}
    zshift::Float64
    name::String
end

const NBSE2 = ERTCAMaterial(
    "NbSe2", 12,
    joinpath(ERTCA_DATA_DIR, "NbSe2_grid_12x12_tight.h5"),
    joinpath(ERTCA_DECK_CACHE, "NbSe2_grid_12x12_tight"),
    [:Nb_b, :Nb_t], 0.05, 1 // 2, 0.0, "NbSe2")

const WSE2 = ERTCAMaterial(
    "WSe2", 12,
    joinpath(ERTCA_DATA_DIR, "WSe2_grid_12x12.h5"),
    joinpath(ERTCA_DECK_CACHE, "WSe2_grid_12x12"),
    [:W_b, :W_t], 0.05, nothing, 0.0, "WSe2")

const _DECK_ROOTS = Dict{String,String}()

function deck_root(mat::ERTCAMaterial)
    return get!(_DECK_ROOTS, mat.seedname) do
        ensure_deck!(mat.seedname, mat.archive, mat.deckdir; ngrid=mat.ngrid)
    end
end

function ertca_lattice(printed::BravaisLattice{3})
    a0 = norm(primitive_vector(printed, 1))
    c = primitive_vector(printed, 3)[3]
    return embed_into_higher_dimension(bravais_lattice(:triangular; a=a0), 3;
                                       periodic=true, padding=c)
end

function snap_cocentred(centres::AbstractVector; atol::Real=1e-5)
    reps = eltype(centres)[]
    out = similar(collect(centres))
    for (i, c) in pairs(collect(centres))
        k = findfirst(r -> all(abs.(c .- r) .<= atol), reps)
        if isnothing(k)
            push!(reps, c)
            out[i] = c
        else
            out[i] = reps[k]
        end
    end
    return out
end

function ertca_stacking(mat::ERTCAMaterial, i::Int, j::Int;
                        root::AbstractString=deck_root(mat))
    dir = joinpath(root, "shift_i" * lpad(i, 2, '0') * "_j" * lpad(j, 2, '0'))
    isdir(dir) || error("missing stacking directory $dir")
    w = read_wannier90_wout(joinpath(dir, "$(mat.seedname).wout"))
    centres, spreads = w.centres, w.spreads
    length(centres) == 4 || error("$dir declares $(length(centres)) Wannier functions, " *
                                  "expected 4 (2 layers × 1 orbital × 2 spin)")
    w.lattice === nothing && error(
        "$dir: the `.wout` has no `Lattice Vectors (Ang) ... Unit Cell Volume` block, so it " *
        "was rendered by a deck.jl predating that line. Delete $(dirname(dir)) and re-run; " *
        "`ensure_deck!` will expand the archive again.")
    lat = ertca_lattice(w.lattice)
    centres = snap_cocentred(centres)
    zc = iszero(mat.zshift) ? centres :
         [c + SVector(0.0, 0.0, mat.zshift) for c in centres]
    model = read_wannier90_hr(lat, joinpath(dir, "$(mat.seedname)_hr.dat");
                              centres=zc,
                              schema=InternalSchema(:spin => [:up, :dn]),
                              site_names=mat.site_names,
                              name="$(mat.name) d=($i/$(mat.ngrid), $j/$(mat.ngrid))")
    return (d=[i / mat.ngrid, j / mat.ngrid], model=model, spreads=spreads)
end

function ertca_scan(mat::ERTCAMaterial; stride::Int=1, skip=(),
                    root::AbstractString=deck_root(mat), match::Symbol=:centres,
                    name::AbstractString="$(mat.name) $(mat.ngrid)x$(mat.ngrid) registry scan")
    skipset = Set(Tuple.(skip))
    entries = Pair[]
    for i in 0:stride:(mat.ngrid-1), j in 0:stride:(mat.ngrid-1)
        (i, j) in skipset && continue
        s = ertca_stacking(mat, i, j; root=root)
        push!(entries, s.d => s.model)
    end
    return registry_scan(entries; layers=[:b, :b, :t, :t], shifted_layer=:t,
                         labels=[:b_up, :b_dn, :t_up, :t_dn],
                         match=match, atol=mat.match_atol, name=name)
end

using ERTCA

using LinearAlgebra

using StaticArrays

import ERTCA: _harmonic_value, _rotation2, threaded_chunks, HarmonicBlock

function average_bilayer(h::RegistryHarmonics)
    iG0 = findfirst(iszero, h.gvecs)
    isnothing(iG0) && error("the harmonics carry no Ḡ = 0 term")
    nb = length(h.labels)
    return k -> Hermitian([_harmonic_value(h, iG0, i, j, SVector{2,Float64}(k))
                           for i in 1:nb, j in 1:nb])
end

function spin_align(h::RegistryHarmonics, kbar)
    nb = length(h.labels)
    iG0 = findfirst(iszero, h.gvecs)
    W0 = [_harmonic_value(h, iG0, i, j, SVector{2,Float64}(kbar)) for i in 1:nb, j in 1:nb]
    U = zeros(ComplexF64, nb, nb)
    info = NamedTuple[]
    for lay in (false, true)
        idx = findall(==(lay), h.shifted)
        B = Hermitian(W0[idx, idx])
        F = eigen(B)
        V = F.vectors
        for c in axes(V, 2)
            V[:, c] .*= cis(-angle(V[argmax(abs.(V[:, c])), c]))
        end
        U[idx, idx] = V
        push!(info, (layer=lay ? h.layer_names[2] : h.layer_names[1],
                     mixing_deg=rad2deg(atan(abs(V[2, 2]), abs(V[1, 2]))),
                     splitting_meV=1000 * (F.values[end] - F.values[1]),
                     spinflip_meV=1000 * abs(W0[idx, idx][1, 2])))
    end
    blocks = similar(h.blocks)
    for iG in axes(h.blocks, 1)
        deltas = SVector{2,Float64}[]
        seen = Dict{SVector{2,Float64},Int}()
        for i in 1:nb, j in 1:nb, d in h.blocks[iG, i, j].delta
            haskey(seen, d) && continue
            push!(deltas, d); seen[d] = length(deltas)
        end
        A = zeros(ComplexF64, length(deltas), nb, nb)
        for i in 1:nb, j in 1:nb
            b = h.blocks[iG, i, j]
            for m in eachindex(b.amp)
                A[seen[b.delta[m]], i, j] += b.amp[m]
            end
        end
        for i in 1:nb, j in 1:nb
            amp = ComplexF64[]
            dl = SVector{2,Float64}[]
            for n in eachindex(deltas)
                v = zero(ComplexF64)
                for p in 1:nb, q in 1:nb
                    v += conj(U[p, i]) * A[n, p, q] * U[q, j]
                end
                abs(v) > 1e-14 && (push!(dl, deltas[n]); push!(amp, v))
            end
            blocks[iG, i, j] = HarmonicBlock(dl, amp)
        end
    end
    hh = RegistryHarmonics(h.lattice, copy(h.labels), copy(h.shifted), copy(h.tau),
                           h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                           copy(h.d), h.name * " (spin-aligned)")
    return hh, info
end

function select_orbitals(h::RegistryHarmonics, idx::AbstractVector{Int})
    blocks = [h.blocks[iG, i, j] for iG in axes(h.blocks, 1), i in idx, j in idx]
    return RegistryHarmonics(h.lattice, h.labels[idx], h.shifted[idx], h.tau[idx],
                             h.layer_names, copy(h.gvecs), blocks, h.cond, h.hermiticity,
                             copy(h.d), h.name * " (orbitals $(Tuple(idx)))")
end

function fermi_level(h::RegistryHarmonics; nk::Int=180, filling::Real)
    W0 = average_bilayer(h)
    nb = length(h.labels)
    Es = Float64[]
    for i in 0:nk-1, j in 0:nk-1
        append!(Es, eigvals(W0(SVector((i + 0.5) / nk, (j + 0.5) / nk))))
    end
    sort!(Es)
    m = clamp(round(Int, filling * length(Es)), 1, length(Es) - 1)
    return (Es[m] + Es[m+1]) / 2
end

struct UnfoldMap
    bmT::SMatrix{2,2,Float64}
    bminv::SMatrix{2,2,Float64}
    nb::Int
    layer_of::Vector{Int}                                   # orbital -> 1 (unshifted) | 2 (shifted)
    rot_of::NTuple{2,SMatrix{2,2,Float64}}                  # L_ℓ⁻¹ per layer
    coset::Vector{Dict{Tuple{Int,Int},SVector{2,Int}}}      # per layer: label -> representative G
    Mdata::Vector{Tuple{SMatrix{2,2,Int},Int}}              # per layer: (adj M_ℓ, N)
    state_of::Dict{Tuple{Int,Tuple{Int,Int}},Int}           # (orbital, G) -> state index
    gsets::Vector{Vector{SVector{2,Int}}}
end

function UnfoldMap(h::RegistryHarmonics, tb::TwistedBilayer, model::HamiltonianModel{2};
                   top::Int=2)
    nb = length(h.labels)
    lay_shift = top
    lay_fixed = top == 2 ? 1 : 2
    layer_of = [h.shifted[b] ? lay_shift : lay_fixed for b in 1:nb]
    rot_of = (_rotation2(+tb.theta / 2), _rotation2(-tb.theta / 2))
    coset = [Dict{Tuple{Int,Int},SVector{2,Int}}() for _ in 1:2]
    Mdata = Vector{Tuple{SMatrix{2,2,Int},Int}}(undef, 2)
    gsets = [monolayer_bz_gvectors(tb, l) for l in 1:2]
    for l in 1:2
        M = supercell_matrix(tb, l)
        N = abs(M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        adj = SMatrix{2,2,Int}(M[2, 2], -M[2, 1], -M[1, 2], M[1, 1])
        Mdata[l] = (adj, N)
        for g in gsets[l]
            x = adj * g
            coset[l][(mod(x[1], N), mod(x[2], N))] = g
        end
        length(coset[l]) == N || error(
            "layer $l's tiling set is not one representative per coset ($(length(coset[l])) of $N)")
    end
    state_of = Dict{Tuple{Int,Tuple{Int,Int}},Int}()
    GS = state_g_vectors(model.cell)
    INT = state_internals(model.cell)
    for a in eachindex(GS)
        state_of[(INT[a], (GS[a][1], GS[a][2]))] = a
    end
    return UnfoldMap(SMatrix{2,2,Float64}(transpose(tb.moire.b)),
                     SMatrix{2,2,Float64}(inv(transpose(tb.moire.b))),
                     nb, layer_of, rot_of, coset, Mdata, state_of, gsets)
end

@inline kappa_of(U::UnfoldMap, k::SVector{2,Float64}, G::SVector{2,Int}) =
    U.bmT * (k + SVector{2,Float64}(G))

function carriers(U::UnfoldMap, kappa::SVector{2,Float64})
    v = U.bminv * kappa
    G0 = floor.(Int, v)
    k = v - SVector{2,Float64}(G0)
    idx = Vector{Int}(undef, U.nb)
    for b in 1:U.nb
        l = U.layer_of[b]
        adj, N = U.Mdata[l]
        x = adj * SVector{2,Int}(G0)
        g = U.coset[l][(mod(x[1], N), mod(x[2], N))]
        idx[b] = U.state_of[(b, (g[1], g[2]))]
    end
    return k, idx
end

function bin_to_grid(FS, xlims::Tuple{Real,Real}, ylims::Tuple{Real,Real};
                     bins::Union{Nothing,Int}=nothing, layers=(1, 2))
    span = max(xlims[2] - xlims[1], ylims[2] - ylims[1])
    nb = isnothing(bins) ? max(24, round(Int, span / (1.6 * FS.spacing))) : bins
    xs = range(xlims[1], xlims[2]; length=nb)
    ys = range(ylims[1], ylims[2]; length=nb)
    dx = (xlims[2] - xlims[1]) / (nb - 1)
    dy = (ylims[2] - ylims[1]) / (nb - 1)
    acc = zeros(nb, nb)
    cnt = zeros(Int, nb, nb)
    for n in eachindex(FS.kx)
        FS.layer[n] in layers || continue
        ix = round(Int, (FS.kx[n] - xlims[1]) / dx) + 1
        iy = round(Int, (FS.ky[n] - ylims[1]) / dy) + 1
        (1 <= ix <= nb && 1 <= iy <= nb) || continue
        acc[iy, ix] += FS.weight[n]
        cnt[iy, ix] += 1
    end
    Z = [cnt[i, j] > 0 ? acc[i, j] / cnt[i, j] : 0.0 for i in 1:nb, j in 1:nb]
    empty = [cnt[i, j] == 0 for i in 1:nb, j in 1:nb]
    Zf = copy(Z)
    for i in 1:nb, j in 1:nb
        empty[i, j] || continue
        s = 0.0; c = 0
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            a, b = i + di, j + dj
            (1 <= a <= nb && 1 <= b <= nb && !empty[a, b]) || continue
            s += Z[a, b]; c += 1
        end
        c > 0 && (Zf[i, j] = s / c)
    end
    return xs, ys, Zf
end

function weight_clims(w::AbstractVector; q::Real=0.995)
    nz = [x for x in w if x > 0]
    isempty(nz) && return (0.0, 1.0)
    s = sort(nz)
    return (0.0, s[clamp(ceil(Int, q * length(s)), 1, length(s))])
end

function bz_hexagon(lattice::BravaisLattice{2})
    bT = transpose(lattice.b)
    b1 = bT * SVector(1.0, 0.0); b2 = bT * SVector(0.0, 1.0)
    K = [(b1 + b2) / 3, (2b2 - b1) / 3, (b2 - 2b1) / 3,
         -(b1 + b2) / 3, (b1 - 2b2) / 3, (2b1 - b2) / 3]
    push!(K, K[1])
    return [k[1] for k in K], [k[2] for k in K]
end

function bz_corners(lattice::BravaisLattice{2})
    bT = transpose(lattice.b)
    b1 = bT * SVector(1.0, 0.0); b2 = bT * SVector(0.0, 1.0)
    return [(b1 + b2) / 3, (2b2 - b1) / 3, (b2 - 2b1) / 3,
            -(b1 + b2) / 3, (b1 - 2b2) / 3, (2b1 - b2) / 3]
end

using Printf

const PAPER_FIGURES = normpath(joinpath(@__DIR__, "..", "Figures"))

const GSHELLS = 4

const HARM_ATOL = 1e-6

const MN = (6, 7)

const HBAR2_2M = 3.80998212     # ħ²/(2mₑ) in eV·Å²

function ertca_context(mat::ERTCAMaterial; mn::Tuple{Int,Int}=MN, gshells::Int=GSHELLS,
                       atol::Real=HARM_ATOL, verbose::Bool=true)
    (scan, nwarn) = quiet(() -> ertca_scan(mat))
    h = registry_harmonics(scan; gshells=gshells, atol=atol)
    tb = twisted_bilayer(h.lattice, mn...)
    EF = isnothing(mat.filling) ? nothing : fermi_level(h; filling=mat.filling)
    lat = h.lattice
    bT = transpose(lat.b)
    Lmat = ERTCA._ertca_L(tb.theta / 2)
    moire = BravaisLattice(Matrix(lat.a * transpose(inv(Lmat))))
    if verbose
        @printf("""
        %s ERTCA — %d shifted-bilayer Wannierizations, %d retained harmonics (|F| = %d),
        cond(E) = %.8f, Hermiticity residual %.1e, %d drift warnings (max %.1e, relaxation).
        θ = %.4f°, N = %d cells/layer, |b̄| = %.4f Å⁻¹, |b_m| = %.4f Å⁻¹, E_F = %s.
        """, mat.name, n_stackings(scan), length(h.gvecs), 2 * length(h.gvecs) - 1,
                 h.cond, h.hermiticity, nwarn,
                 maximum(scan.drift), twist_angle(tb), n_monolayer_cells(tb),
                 norm(bT * SVector(1.0, 0.0)),
                 norm(transpose(moire.b) * SVector(1.0, 0.0)),
                 isnothing(EF) ? "not pinned (filling undeclared)" : @sprintf("%.5f eV", EF))
    end
    return (h=h, tb=tb, EF=EF, lat=lat, bT=bT, binv=inv(bT), moire=moire,
            bmT=transpose(moire.b), bminv=inv(transpose(moire.b)), theta=tb.theta)
end

function valley_setup(ctx, which::Symbol)
    which in (:K, :Gamma) || throw(ArgumentError("`which` is :K or :Gamma; got :$which"))
    kbar = which === :K ? SVector(1 / 3, 1 / 3) : SVector(0.0, 0.0)
    kc = ctx.bT * kbar
    hal, info = spin_align(ctx.h, SVector(1 / 3, 1 / 3))
    h1 = select_orbitals(hal, [2, 4])
    th = ctx.theta
    Kb = ERTCA._rotation2(-th / 2) * kc
    Kt = ERTCA._rotation2(+th / 2) * kc
    if which === :K
        kap_b = SVector(2 / 3, 2 / 3)
        expansion = Kb - ctx.bmT * kap_b
        kap_t = ctx.bminv * (Kt - expansion)
    else
        kap_b = SVector(0.0, 0.0)
        expansion = Kb                      # = Kt = Γ: the twist leaves Γ fixed
        kap_t = ctx.bminv * (Kt - expansion)
    end
    return (h=h1, spin=info, kbar=kbar, kc=kc, expansion=expansion,
            kap_b=kap_b, kap_t=kap_t, Kb=Kb, Kt=Kt)
end

function valley_parameters(h1::RegistryHarmonics, kbar, lat::BravaisLattice{2})
    binv = inv(transpose(lat.b))
    iG0 = findfirst(iszero, h1.gvecs)
    Wv(iG, i, j, k) = _harmonic_value(h1, iG, i, j, SVector{2,Float64}(k))
    eps1(kf) = real(Wv(iG0, 1, 1, kf))
    dk = 2e-3
    Hs = zeros(2, 2)
    for a in 1:2, b in 1:2
        ea = SVector(a == 1 ? dk : 0.0, a == 2 ? dk : 0.0)
        eb = SVector(b == 1 ? dk : 0.0, b == 2 ? dk : 0.0)
        Hs[a, b] = (eps1(kbar .+ binv * (ea + eb)) - eps1(kbar .+ binv * (ea - eb))
                    - eps1(kbar .+ binv * (-ea + eb)) + eps1(kbar .+ binv * (-ea - eb))) / (4dk^2)
    end
    ev = eigvals(Symmetric(Hs))
    mstar = -2HBAR2_2M / (sum(ev) / 2)
    star1 = ERTCA.g_star(lat, 1)
    C3 = ERTCA._integer_c3(lat)
    reps = ERTCA._c3_closed_reps(star1, C3)
    function amp(i, j, g)
        hit = ERTCA._harmonic_match(h1, g)
        isnothing(hit) && return nothing
        iG, sg = hit
        sg > 0 ? Wv(iG, i, j, kbar) : conj(Wv(iG, j, i, kbar))
    end
    intra = [amp(1, 1, g) for g in reps]
    V = sum(abs, intra) / length(intra)
    psi = angle(sum(cis(angle(v)) for v in intra) / length(intra))
    w0 = amp(1, 2, SVector(0, 0))
    inter1 = [(g=g, w=amp(1, 2, g)) for g in star1 if !isnothing(amp(1, 2, g))]
    return (edge=eps1(kbar), mstar=mstar, hess=ev,
            anisotropy=abs(ev[1] - ev[2]) / abs(sum(ev) / 2),
            reps=reps, intra=intra, V=V, psi=psi, w0=w0, inter1=inter1)
end
