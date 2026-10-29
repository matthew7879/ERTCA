# NbSe2 Wannier orbital, monolayer Fermi surface, and stacking bands.
# Run from this repository: julia --project=. -t auto Fig1.jl

include(joinpath(@__DIR__, "src", "common.jl"))
import Contour

const FIG = PAPER_FIGURES
const WANNIER_PDF = joinpath(@__DIR__, "Inputs", "NbSe2_Wannier.pdf")
isfile(WANNIER_PDF) || error("Missing VESTA export: $WANNIER_PDF")
const NGRID = 840        # grid for the monolayer Fermi-surface contours
const LIM = 1.30         # Fermi-surface plot half-width (Å⁻¹)
# Supply the monolayer SCF Fermi energy, in the same reference as the hr file.
# Do not substitute the bilayer Fermi energy or the default .wout plotting zero.
# Monolayer SCF output supplied by the user: 1 Oct 2026, PWSCF 7.3.1.
const EF = isempty(ARGS) ? -0.1453 : parse(Float64, ARGS[1])

# Import the monolayer lattice and actual Cartesian Wannier centres.
const INPUTS = joinpath(@__DIR__, "Inputs")
const WOUT = read_wannier90_wout(joinpath(INPUTS, "NbSe2_monolayer.wout"))
const MODEL_AA = read_wannier90_hr(WOUT.lattice,
    joinpath(INPUTS, "NbSe2_monolayer_hr.dat"); centres=WOUT.centres, schema=InternalSchema(:orbital => collect(1:WOUT.num_wann)), atol=0.0)
const LAT = BravaisLattice(Matrix(WOUT.lattice.a[1:2, 1:2]))
const BT = transpose(LAT.b)
const BINV = inv(BT)
const NBANDS = n_states(MODEL_AA.cell)
const NODES = [(0.0,0.0), (0.5,0.0), (1/3,1/3), (0.0,0.0)]
HAA(k) = Hermitian(bloch_hamiltonian(MODEL_AA, SVector(k[1], k[2], 0.0)))
# Spin interpolation for Fig1, in the original periodic Wannier gauge.
# Checkpoint record order: Wannier90 v3.1.0 src/w90chk2chk.F90.
# This reader deliberately supports only the supplied non-disentangled,
# little-endian checkpoint with 4-byte Fortran record markers.
function fig1_spin_data(inputs, wout, model)
    records = Vector{UInt8}[]
    open(joinpath(inputs,"NbSe2_monolayer.chk")) do io
        while !eof(io)
            n = Int(ltoh(read(io,Int32)))
            0 <= n <= 100_000_000 || error("Unsupported checkpoint record format")
            bytes = read(io,n)
            length(bytes)==n && ltoh(read(io,Int32))==n || error("Corrupt checkpoint record")
            push!(records,bytes)
        end
    end
    scalar(i) = only(reinterpret(Int32,records[i]))
    @assert ENDIAN_BOM == 0x04030201
    nb, nk, nw = scalar(2), scalar(7), scalar(11)
    @assert scalar(13)==0 "Disentangled checkpoint requires U_dis support"
    @assert length(records)==17 && nb==nw==2
    mesh = Int.(reinterpret(Int32,records[8]))
    k = reshape(copy(reinterpret(Float64,records[9])),3,nk)
    U = reshape(copy(reinterpret(ComplexF64,records[14])),nw,nw,nk)
    centres = reshape(copy(reinterpret(Float64,records[16])),3,nw)
    @assert maximum(abs.(centres-hcat(wout.centres...))) < 2e-6 "Checkpoint/wout centres differ"
    lattice = reshape(copy(reinterpret(Float64,records[5])),3,3)
    @assert maximum(abs.(lattice-Matrix(wout.lattice.a))) < 2e-6 "Checkpoint lattice mismatch"
    win = read(joinpath(inputs,"NbSe2_monolayer.win"),String)
    kb = match(r"(?is)begin\s+kpoints\s*\n(.*?)end\s+kpoints",win)[1]
    kw = reshape(parse.(Float64,split(kb)),3,nk)
    @assert maximum(abs.(k-kw)) < 1e-8
    spn = read_wannier90_spn(joinpath(inputs,"NbSe2_monolayer.spn"))
    @assert spn.num_bands==nb && spn.num_kpoints==nk && prod(mesh)==nk
    uniterr = maximum(norm(U[:,:,j]'*U[:,:,j]-I) for j in 1:nk)
    @assert uniterr < 1e-10
    Sz = cat([U[:,:,j]'*spn.spin[:,:,3,j]*U[:,:,j] for j in 1:nk]...;dims=3)
    # Read hr directly so both operators use exactly the unwrapped periodic basis.
    lines = readlines(joinpath(inputs,"NbSe2_monolayer_hr.dat"))
    @assert parse(Int,strip(lines[2]))==nw
    nr = parse(Int,strip(lines[3]))
    tok = split(join(lines[4:end]," "))
    deg = parse.(Int,tok[1:nr])
    vals = reshape(parse.(Float64,tok[nr+1:end]),7,nw*nw,nr)
    R = zeros(Int,3,nr); HR = zeros(ComplexF64,nw,nw,nr)
    for r in 1:nr
        R[:,r] = Int.(vals[1:3,1,r])
        for j in 1:nw*nw
            m,n = Int.(vals[4:5,j,r])
            HR[m,n,r] = complex(vals[6,j,r],vals[7,j,r])/deg[r]
        end
    end
    @assert isapprox(sum(1 ./ deg),nk;atol=1e-8)
    phase = cispi.(2 .* (R' * k))
    SR = reshape(reshape(Sz,nw*nw,nk)*conj.(transpose(phase))/nk,nw,nw,nr)
    SR ./= reshape(deg,1,1,nr)
    recon = reshape(reshape(SR,nw*nw,nr)*phase,nw,nw,nk)
    spinerr = maximum(abs.(recon-Sz))
    @assert spinerr < 1e-9 "Spin Fourier reconstruction failed on the NSCF mesh"
    Hmesh = reshape(reshape(HR,nw*nw,nr)*phase,nw,nw,nk)
    gaugeerr = maximum(abs((U[:,:,j]*Hmesh[:,:,j]*U[:,:,j]')[1,2]) for j in 1:nk)
    @assert gaugeerr < 2e-5 "Checkpoint and hr gauges appear inconsistent"
    # Independent spectrum check against the package's imported Hamiltonian.
    for q in (SVector(.17,.29,0.),SVector(1/3,1/3,0.),SVector(2/3,-1/3,0.))
        H = reshape(reshape(HR,nw*nw,nr)*cispi.(2 .* (R'*q)),nw,nw)
        @assert maximum(abs.(eigvals(Hermitian(H))-eigvals(Hermitian(bloch_hamiltonian(model,q))))) < 1e-8
    end
    @printf("Spin audit: U unitarity %.2e; mesh reconstruction %.2e; H off-diagonal in DFT gauge %.2e eV\n",uniterr,spinerr,gaugeerr)
    return (R=R,H=reshape(HR,nw*nw,nr),S=reshape(SR,nw*nw,nr),nw=nw)
end

function fig1_band_spin(data,k,b)
    phase = cispi.(2 .* (data.R' * SVector(k[1],k[2],0.)))
    H = Hermitian(reshape(data.H*phase,data.nw,data.nw))
    S = Hermitian(reshape(data.S*phase,data.nw,data.nw))
    ev = eigen(H)
    # At a true degeneracy an energy-sorted eigenvector has no unique spin label.
    minimum(diff(ev.values)) < 1e-7 && return 0.0
    return real(dot(ev.vectors[:,b],S*ev.vectors[:,b]))
end

const SPIN_DATA = fig1_spin_data(INPUTS, WOUT, MODEL_AA)
const SPIN_SAMPLES = Tuple{Int,Float64,Float64,Float64}[]
@printf("Monolayer: %d bands; E_F = %.8f eV\n", NBANDS, EF)

# -------------------------------------------------------------------------------------
#  The Fermi surface, as contours of each band
# -------------------------------------------------------------------------------------
# The monolayer has a periodic spectrum. Show one hexagonal zone for clarity.
### the Wigner–Seitz cell: |κ·ĝ| ≤ |g|/2 for the six shortest reciprocal vectors
const GS6 = let bT = BT
    v = [bT * SVector(1.0, 0.0), bT * SVector(0.0, 1.0), bT * SVector(1.0, -1.0)]
    vcat(v, [-g for g in v])
end
in_bz(kc) = all(dot(kc, g) <= dot(g, g) / 2 + 1e-9 for g in GS6)

xs = range(-LIM, LIM; length=NGRID)
BND = [fill(NaN, NGRID, NGRID) for _ in 1:NBANDS]
Threads.@threads for ix in 1:NGRID
    for iy in 1:NGRID
        kc = SVector(xs[ix], xs[iy])
        in_bz(kc) || continue
        ev = eigvals(HAA(BINV * kc)) .- EF
        for b in 1:NBANDS; BND[b][iy, ix] = 1000ev[b]; end
    end
end
hx, hy = bz_hexagon(LAT)
crossing = [b for b in 1:NBANDS if any(x -> !isnan(x) && x < 0, BND[b]) &&
                               any(x -> !isnan(x) && x > 0, BND[b])]
@printf("  %d of %d bands cross E_F inside the first zone: %s\n",
        length(crossing), NBANDS, join(crossing, ", "))


# Two-column layout: unchanged left panels, five comparison rows on the right.
const WIDTH_PT = 510
const HEIGHT_PT = 410
const FONT_PT = 16
const BAND_COLORS = ("FF0000", "0072B2") # red, blue
const STEM = "Fig1"

# Crop and atom centres refer to the supplied, unlabelled 1616 × 507 bp VESTA
# PDF, measured from its bottom-left corner. Update these if it is re-exported
# with a different view. Cropping only removes blank margins; the PDF stays vector.
const WANNIER_VIEWPORT = (549.0, 96.0, 1074.0, 478.0)
# Identify each species once, beside the unobscured left-hand Nb and upper Se.
# Entries are (text, source_x, source_y, label_dx_pt, label_dy_pt).
const WANNIER_LABELS = (
    ("Nb", 670.6, 341.4, -9.0, 6.0),
    ("Se", 670.6, 448.4, -11.0, 0.0),
)

function tex_curve(io, x, y; style)
    println(io, "\\addplot[no marks, unbounded coords=jump, ", style, "] coordinates {")
    for (xi, yi) in zip(x, y)
        if isfinite(xi) && isfinite(yi)
            @printf(io, "(%.9g,%.9g)\n", xi, yi)
        else
            println(io, "(nan,nan)")
        end
    end
    println(io, "};")
end

# Colour each contour segment by physical out-of-plane spin, not band index.
function tex_spin_contour(io,cx,cy,b)
    colours = String[]
    for n in 1:length(cx)-1
        kc = SVector((cx[n]+cx[n+1])/2,(cy[n]+cy[n+1])/2)
        if !all(isfinite,kc)
            push!(colours, "")
            continue
        end
        spin = fig1_band_spin(SPIN_DATA, BINV*kc, b)
        push!(SPIN_SAMPLES,(b,kc[1],kc[2],spin))
        push!(colours, abs(spin)<1e-3 ? "black!50" : spin>0 ? "band1" : "band2")
    end
    first = 1
    while first <= length(colours)
        last = first
        while last < length(colours) && colours[last+1]==colours[first]
            last += 1
        end
        isempty(colours[first]) || tex_curve(io,cx[first:last+1],cy[first:last+1];style="$(colours[first]),line width=1pt")
        first = last+1
    end
end

function tex_wannier(io)
    left, bottom, right, top = WANNIER_VIEWPORT
    x, width, centre_y = 25.0, 145.0, 318.0
    scale = width / (right - left)
    y = centre_y - scale * (top - bottom) / 2
    println(io, "\\node[anchor=south west,inner sep=0pt] at ($(x)pt,$(y)pt) {",
            "\\includegraphics[viewport={", join(WANNIER_VIEWPORT, " "),
            "},clip,width=$(width)pt]{\\detokenize{", WANNIER_PDF, "}}};")
    for (element, ax, ay, dx, dy) in WANNIER_LABELS
        px, py = x + scale * (ax - left) + dx, y + scale * (ay - bottom) + dy
        @printf(io, "\\node[anchor=east,inner sep=0pt,text=black] at (%.6fpt,%.6fpt) {%s};\n",
                px, py, element)
    end
end

# Five original bilayer Wannier Hamiltonians, without registry fitting.
# Subtract the same stacking-specific SCF Fermi energy from DFT and Wannier.
const SHIFTS_C = [(0,0), (1,2), (2,4), (3,6), (4,8)]
const FERMI_C = Dict((0,0)=>5.2061, (1,2)=>5.2051, (2,4)=>5.2021,
                     (3,6)=>5.1957, (4,8)=>5.1926)
function read_dft_bands_c(filename)
    txt = read(filename, String)
    header = match(r"(?is)&plot\b.*?/", txt)
    isnothing(header) && error("Missing QE bands header in $filename")
    nb = parse(Int, match(r"nbnd\s*=\s*(\d+)"i, header.match)[1])
    nk = parse(Int, match(r"nks\s*=\s*(\d+)"i, header.match)[1])
    data = parse.(Float64, split(replace(txt[header.offset+ncodeunits(header.match):end], 'D'=>'E')))
    length(data) == (nb+3)*nk || error("Incorrect DFT data length: $filename")
    blocks = reshape(data, nb+3, nk)
    kcart = blocks[1:3, :]
    expected = [0 .5 1/3 0; 0 sqrt(3)/6 sqrt(3)/3 0; 0 0 0 0]
    @assert nk == 151 && maximum(abs.(kcart[:, [1,51,101,151]] - expected)) < 2e-6 "Unexpected DFT path"
    x = vcat(0.0, cumsum([norm(kcart[:, n]-kcart[:, n-1]) for n in 2:nk]))
    return (x=x ./ last(x), E=permutedims(blocks[4:end,:]))
end

const PANELS_C = map(SHIFTS_C) do (i,j)
    tag = @sprintf("i%02d_j%02d", i, j)
    dft = read_dft_bands_c(joinpath(INPUTS, "NbSe2_bands_DFT_$(tag).dat"))
    ef = FERMI_C[(i,j)]
    dft = (x=dft.x, E=dft.E .- ef)
    entry = ertca_stacking(NBSE2, i, j)
    model = entry.model
    bT = transpose(model.cell.lattice.b)
    nodes = [SVector(0.,0.,0.), SVector(.5,0.,0.), SVector(1/3,1/3,0.), SVector(0.,0.,0.)]
    ks = SVector{3,Float64}[]
    dist = Float64[]
    ticks = [0.0]
    for seg in 1:3
        seglen = norm(bT * (nodes[seg+1]-nodes[seg]))
        for t in range(0,1; length=201)[(seg==1 ? 1 : 2):end]
            push!(ks, (1-t)*nodes[seg]+t*nodes[seg+1])
            push!(dist, ticks[end]+t*seglen)
        end
        push!(ticks, ticks[end]+seglen)
    end
    E = permutedims(hcat([eigvals(Hermitian(bloch_hamiltonian(model,k))) for k in ks]...))
    E .-= ef
    x = dist ./ last(dist)
    ticks ./= last(ticks)
    @assert maximum(abs.(dft.x[[1,51,101,151]]-ticks)) < 2e-6
    @printf("Panel c %s: %d DFT bands, %d Wannier bands; Fermi-referenced Wannier range %.4f to %.4f eV\n", tag, size(dft.E,2),size(E,2),minimum(E),maximum(E))
    # Uniform Cartesian path distance, rather than equal samples per segment.
    marker_x = collect(range(0,1; length=61))
    marker_k = map(marker_x) do u
        seg = min(searchsortedlast(ticks,u),3)
        t = (u-ticks[seg])/(ticks[seg+1]-ticks[seg])
        (1-t)*nodes[seg]+t*nodes[seg+1]
    end
    marker_E = permutedims(hcat([eigvals(Hermitian(bloch_hamiltonian(model,k))) for k in marker_k]...)) .- ef
    @assert maximum(abs.(diff(marker_x) .- 1/60)) < 1e-12
    (tag=tag, i=i,j=j,dft=dft,x=x,E=E,ticks=ticks,marker_x=marker_x,marker_E=marker_E)
end
# Display range only; energies retain the stacking-specific SCF Fermi reference.
const C_YMIN = -1.2
const C_YMAX = 1.6

# Projected Se-Nb-Se motifs viewed along the in-plane direction perpendicular
# to AA -> AB. Se has fractional position (1/3,2/3) relative to Nb.
# Horizontal displacement is proportional to the actual d = (i*a1+j*a2)/12;
# vertical separation is schematic. At AB the top Nb is above bottom Se.
function tex_stacking(io, p, y; x0=470.0)
    a = Matrix(LAT.a)
    se = (a' * [1/3, 2/3])
    direction = se / norm(se)
    d = a' * [p.i/12, p.j/12]
    shift = 16 * dot(d, direction) / norm(se)
    # Fixed vertical guides through the bottom-layer Nb and Se columns.
    guides = (p.i,p.j) == (0,0) ? (x0,x0+16) : (p.i,p.j) == (4,8) ? (x0+16,) : ()
    for x in guides
        @printf(io,"\\draw[gray!65,dashed,line width=.5pt] (%.3fpt,%.3fpt)--(%.3fpt,%.3fpt);\n",x,y+3,x,y+49)
    end
    if 0 < p.i < 4
        @printf(io,"\\draw[->,black,line width=.5pt] (%.3fpt,%.3fpt)--(%.3fpt,%.3fpt);\n",x0,y+56,x0+shift,y+56)
    end
    for (x,z) in ((x0,y+11), (x0+shift,y+40))
        for dz in (-7.,7.)
            @printf(io,"\\draw[green!35!black,line width=1pt] (%.3fpt,%.3fpt)--(%.3fpt,%.3fpt);\n",x,z,x+16,z+dz)
        end
        @printf(io,"\\shade[ball color=green!45!black] (%.3fpt,%.3fpt) circle[radius=3.1pt];\n",x,z)
        for dz in (-7.,7.)
            @printf(io,"\\shade[ball color=green!65!yellow] (%.3fpt,%.3fpt) circle[radius=2.5pt];\n",x+16,z+dz)
        end
    end
end

# Opaque parallel plot planes. The same affine projection is applied to every
# plane; it changes presentation only, not energies or sampled momenta.
function tex_bilayer_comparisons(io)
    println(io, raw"\node[anchor=north west,inner sep=0pt] at (166pt,397pt) {c)};")
    println(io, raw"\draw[black!50,line width=.65pt] (278pt,399pt)--(293pt,399pt) node[right,text=black] {DFT};")
    println(io, raw"\draw[band1,line width=1pt,dash pattern=on 3pt off 2pt] (377pt,399pt)--(397pt,399pt);\node[anchor=west] at (401pt,399pt) {Wannier};")
    w, h, dataw = 170., 180., 170.
    ey(e) = h*(e-C_YMIN)/(C_YMAX-C_YMIN)
    # Paint bottom first, top last: no transparent plot backgrounds.
    for row in 5:-1:1
        p = PANELS_C[row]
        # A 77.4 pt projected height and 74 pt spacing give a small overlap.
        y = 310 - 74*(row-1)
        println(io,"\\begin{scope}[cm={1.10,0,-.40,.43,(264pt,$(y)pt)},transform shape]")
        println(io,"\\filldraw[fill=white,draw=black,line width=.55pt] (0pt,0pt) rectangle ($(w)pt,$(h)pt);")
        println(io,"\\begin{scope}\\clip (0pt,0pt) rectangle ($(w)pt,$(h)pt);")
        for t in p.ticks[2:3]
            @printf(io,"\\draw[black!15,line width=.4pt] (%.5fpt,0pt)--(%.5fpt,%.5fpt);\n",dataw*t,dataw*t,h)
        end
        for b in axes(p.dft.E,2)
            v=p.dft.E[:,b]
            minimum(v)<=C_YMAX && maximum(v)>=C_YMIN || continue
            println(io,raw"\draw[black!50,line width=.65pt] plot coordinates {")
            for (x,e) in zip(p.dft.x,v)
                @printf(io,"(%.6fpt,%.6fpt)\n",dataw*x,ey(e))
            end
            println(io,"};")
        end
        for b in axes(p.E,2)
            println(io,raw"\draw[band1,line width=1pt,dash pattern=on 3pt off 2pt] plot coordinates {")
            for (x,e) in zip(p.x,p.E[:,b])
                @printf(io,"(%.6fpt,%.6fpt)\n",dataw*x,ey(e))
            end
            println(io,"};")
        end
        println(io,raw"\end{scope}")
        for e in (-1.,0.,1.)
            @printf(io,"\\draw[black,line width=.4pt] (0pt,%.6fpt)--(3pt,%.6fpt);\n",ey(e),ey(e))
            @printf(io,"\\node[anchor=east,inner sep=2pt,yscale=1.5,font=\\fontsize{10}{12}\\selectfont] at (0pt,%.6fpt) {%.1f};\n",ey(e),e)
        end
        row == 5 && println(io,raw"\node[anchor=east,yscale=2,font=\fontsize{18}{21}\selectfont] at (-20pt,60pt) {$E$ (eV)};")
        if row==5
            for (t,l) in zip(p.ticks,(raw"$\Gamma$",raw"$M$",raw"$K$",raw"$\Gamma$"))
                @printf(io,"\\node[anchor=north,inner sep=3pt,yscale=2,font=\\fontsize{18}{21}\\selectfont] at (%.6fpt,0pt) {%s};\n",dataw*t,l)
            end
        end
        println(io,raw"\end{scope}")
        # Upright diagrams outside the graph remain legible at column width.
        # Their lateral displacements still use the same registry (i,j).
        tex_stacking(io,p,y+7; x0=456.)
        if row==1 || row==5
            println(io,"\\node[font=\\fontsize{12}{14}\\selectfont] at (472pt,$(y+66)pt) {$(row==1 ? "MM" : "MX")};")
        end
    end
end

function write_figure_tex(filename)
    open(filename, "w") do io
        print(io, raw"""
\documentclass[10pt,tikz,border=0pt]{standalone}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
""")
        for (b, color) in pairs(BAND_COLORS)
            println(io, "\\definecolor{band$b}{HTML}{$color}")
        end
        println(io, raw"\begin{document}")
        println(io, "\\begin{tikzpicture}[font=\\fontsize{$FONT_PT}{12}\\selectfont]")
        # A square momentum window keeps the Fermi-surface geometry undistorted.
        print(io, raw"""
\begin{axis}[
  at={(16pt,68pt)}, anchor=south west,
  width=150pt, height=150pt, scale only axis,
  hide axis, enlargelimits=false, clip=false,
""")
        println(io, "  xmin=$(-LIM), xmax=$LIM, ymin=$(-LIM), ymax=$LIM]")
        tex_curve(io, hx, hy; style="black,line width=0.65pt")
        for b in crossing
            # BND stores [y,x], whereas Contour expects [x,y].
            curves = Contour.lines(Contour.contour(xs, xs, permutedims(BND[b]), 0.0))
            for curve in curves
                cx, cy = Contour.coordinates(curve)
                tex_spin_contour(io, cx, cy, b)
            end
        end
        # Use the same fractional Γ, M, K as the band path. The right-hand
        # corner K′ = -K + b₁ is the opposite valley, modulo a reciprocal vector.
        for (kf, label, placement) in (
            (SVector{2,Float64}(NODES[1]), raw"$\Gamma$", "anchor=east,xshift=-4pt"),
            (SVector{2,Float64}(NODES[2]), raw"$M$", "anchor=south west,xshift=4pt,yshift=3pt"),
            (SVector{2,Float64}(NODES[3]), raw"$K$", "anchor=south west,xshift=3pt,yshift=3pt"),
            (SVector(2 / 3, -1 / 3), raw"$K^{\prime}$", "anchor=west,xshift=4pt"),
        )
            kc = BT * kf
            @printf(io, "\\fill[black] (axis cs:%.9g,%.9g) circle[radius=1.2pt];\n", kc...)
            @printf(io, "\\node[%s,inner sep=0pt] at (axis cs:%.9g,%.9g) {%s};\n",
                    placement, kc[1], kc[2], label)
        end
        println(io, raw"\end{axis}")

        tex_wannier(io)
        tex_bilayer_comparisons(io)

        println(io, raw"\node[anchor=north west,inner sep=0pt] at (10pt,397pt) {a)};")
        println(io, raw"\node[anchor=north west,inner sep=0pt] at (10pt,238pt) {b)};")

        # Fixed page bounds; labels are enlarged for reduction to one-column width.
        println(io, raw"\pgfresetboundingbox")
        println(io, "\\path[use as bounding box] (0pt,0pt) rectangle ($(WIDTH_PT)pt,$(HEIGHT_PT)pt);")
        println(io, raw"\end{tikzpicture}")
        println(io, raw"\end{document}")
    end
end

function save_paper_figure()
    # Editor-launched Julia sessions may not inherit the shell's TeX/Homebrew PATH.
    function executable(program)
        path = Sys.which(program)
        if isnothing(path) && Sys.isapple()
            for directory in ("/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin")
                path = Sys.which(joinpath(directory, program))
                isnothing(path) || return path
            end
        end
        isnothing(path) && error("$program is required to render the figure. " *
                                "Install TeX (pdflatex) and Poppler (pdftoppm), " *
                                "or add their executable directories to ENV[\"PATH\"].")
        return path
    end
    pdflatex = executable("pdflatex")
    pdftoppm = executable("pdftoppm")
    mkpath(FIG)
    builddir = joinpath(@__DIR__, ".cache", "figures")
    mkpath(builddir)
    texfile = joinpath(builddir, STEM * ".tex")
    empty!(SPIN_SAMPLES)
    write_figure_tex(texfile)
    open(joinpath(builddir,"Fig1_spin_contours.csv"),"w") do io
        println(io,"energy_band,kx_Ainv,ky_Ainv,sigma_z")
        for r in SPIN_SAMPLES
            println(io,join(r,","))
        end
    end
    for (label,k) in (("K",SVector(1/3,1/3)),("Kprime",SVector(2/3,-1/3)))
        println("Spin at ",label," (lower,upper): ",[fig1_band_spin(SPIN_DATA,k,b) for b in 1:NBANDS])
    end
    logfile = joinpath(builddir, STEM * "_build.log")
    open(logfile, "w") do io
        run(pipeline(`$pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$builddir $texfile`,
                     stdout=io, stderr=io))
    end
    pdf = joinpath(FIG, STEM * ".pdf")
    cp(joinpath(builddir, STEM * ".pdf"), pdf; force=true)
    run(`$pdftoppm -r 300 -png -singlefile $pdf $(joinpath(FIG, STEM))`)
    println("\n    figure → ", pdf)
    println("    preview → ", joinpath(FIG, STEM * ".png"))
end

save_paper_figure()
