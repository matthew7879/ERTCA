# NbSe2 interlayer tunneling and shell convergence.
# Run from this repository: julia --project=. -t auto Fig2.jl

include(joinpath(@__DIR__, "src", "common.jl"))

# The harmonic fit retains at least GSHELLS=4; this selects only what is displayed.
# 1 → G0–G6; 2 → G0–G12; 3 → G0–G18. The fourth shell has twelve vectors.
const PLOT_SHELLS = 1
const NPOINTS = 1600  # smooth curves and narrow minima in the enlarged export
const SPIN_CHANNEL = (β=1, βp=3)  # original imported bottom-up / top-up basis

# Horizontal path layout with enlarged text; final text size scales with inclusion width.
const WIDTH_PT = 510
const HEIGHT_PT = 196
const FONT_PT = 16
const PNG_DPI = 1200  # high-resolution raster export; the PDF remains fully vector
const STEM = "Fig2"

# Dark neutral G0, then six distinguishable colours, all with solid curves.
const CURVE_COLORS = ("252525", "0072B2", "D55E00", "009E73",
                      "CC79A7", "E69F00", "56B4E9")
const CURVE_DASHES = ("solid", "solid", "solid", "solid", "solid", "solid", "solid")

# Actual Wannier-centre phases, all 144 input stackings, no coefficient pruning,
# no spin averaging or additional basis rotation.
scan, _ = quiet(() -> ertca_scan(NBSE2))
h = registry_harmonics(scan; gshells=max(GSHELLS, PLOT_SHELLS), atol=0.0)

const GAMMA = SVector(0.0, 0.0)
const MPOINT = SVector(0.5, 0.0)
const KPOINT = SVector(1 / 3, 1 / 3)
const KPRIME = SVector(2 / 3, -1 / 3)
c3 = @SMatrix [-1.0 -1.0;
               1.0  0.0]

nodes = [
    "Γ"      => GAMMA,
    "M"      => MPOINT,
    "K′"     => KPRIME,
    "Γ"      => GAMMA,
    "C₃M"    => c3 * MPOINT,
    "C₃K′"   => c3 * KPRIME,
    "Γ"      => GAMMA,
    "C₃²M"   => c3^2 * MPOINT,
    "C₃²K′"  => c3^2 * KPRIME,
    "Γ"      => GAMMA,
]

path = kpath_line(h.lattice, nodes; npoints=NPOINTS)
bT = transpose(h.lattice.b)
@assert (bT * KPRIME)[1] > 0 && isapprox((bT * KPRIME)[2], 0.0; atol=1e-12)

# Number each shell counterclockwise starting from the upward direction.
# G_j = n1*b1 + n2*b2. Changing the valley does not rename the harmonics.
selected = [SVector(0, 0),
            SVector(0, 1),    # G1: up = b2
            SVector(-1, 1),   # G2: upper left
            SVector(-1, 0),   # G3: lower left
            SVector(0, -1),   # G4: down
            SVector(1, -1),   # G5: lower right
            SVector(1, 0)]    # G6: upper right = b1


# Only the half of ±G is stored (using Hermiticity)
harmonics = map(selected) do G
    iG, sgn = ERTCA._harmonic_match(h, G)
    (G=G, iG=iG, sgn=sgn)
end
function tunneling_value(h, harmonic, k)
    β, βp = SPIN_CHANNEL.β, SPIN_CHANNEL.βp
    if harmonic.sgn == 1
        return ERTCA._harmonic_value(h, harmonic.iG, β, βp, k)
    end
    return conj(ERTCA._harmonic_value(h, harmonic.iG, βp, β, k))
end
magnitudes = [1000abs(tunneling_value(h, harmonic, k))
              for k in path.kpts, harmonic in harmonics]

# Reconstruct the same matrix element with ALL fitted shells, not just the
# first-shell coefficients displayed in (a). Precompute their complex values
# at fixed K′; only the registry Fourier phase changes along this path.
registry_shell_ids = ERTCA._shell_ids(h)
registry_terms = map(eachindex(h.gvecs)) do iG
    (G=h.gvecs[iG], shell=registry_shell_ids[iG],
     forward=ERTCA._harmonic_value(h, iG, SPIN_CHANNEL.β, SPIN_CHANNEL.βp, KPRIME),
     backward=conj(ERTCA._harmonic_value(h, iG, SPIN_CHANNEL.βp, SPIN_CHANNEL.β, KPRIME)))
end
function fitted_registry_tunneling(d; shells=4)
    value = 0.0 + 0.0im
    for term in registry_terms
        term.shell <= shells || continue
        phase = cispi(-2dot(term.G, d))
        value += phase * term.forward
        iszero(term.G) || (value += conj(phase) * term.backward)
    end
    return value
end
registry_t = collect(range(0.0, 1.0; length=601))
registry_d = [SVector(1-t, t) for t in registry_t]
registry_values = fitted_registry_tunneling.(registry_d)
registry_magnitudes = 1000abs.(registry_values)
registry_shell_values = [fitted_registry_tunneling(d; shells=s) for d in registry_d, s in 1:4]
registry_shell_magnitudes = 1000abs.(registry_shell_values)
@assert registry_shell_values[:,4] ≈ registry_values

# Compare with the original imported matrix element at all 144 fitted registries.
# These are in-sample residuals, not an independent validation set.
registry_raw = map(eachindex(scan.d)) do s
    ip = invperm(scan.permutations[s])
    W = bloch_hamiltonian(scan.models[s], SVector(KPRIME[1], KPRIME[2], 0.0))
    W[ip[SPIN_CHANNEL.β], ip[SPIN_CHANNEL.βp]]
end
registry_errors = map(1:4) do shells
    terms = filter(t -> t.shell <= shells, registry_terms)
    F = vcat([t.G for t in terms], [-t.G for t in terms if !iszero(t.G)])
    coefficients = vcat([t.forward for t in terms], [t.backward for t in terms if !iszero(t.G)])
    E = [cispi(-2dot(g,d)) for d in scan.d, g in F]
    # On this complete regular grid, truncating the four-shell solve is equivalent
    # to independently fitting each smaller set. Verify that equivalence explicitly.
    @assert isapprox(E \ registry_raw, coefficients; atol=1e-10, rtol=1e-8)
    fit = E * coefficients
    mag_errors = 1000 .* (abs.(fit) .- abs.(registry_raw))
    complex_errors = 1000 .* (fit .- registry_raw)
    result = (shells=shells, vectors=length(F),
              magnitude_rmse=sqrt(sum(abs2,mag_errors)/length(mag_errors)),
              magnitude_max=maximum(abs,mag_errors),
              complex_rmse=sqrt(sum(abs2,complex_errors)/length(complex_errors)))
    @printf("Shell cutoff %d (%d vectors): magnitude RMSE %.6f, max %.6f; complex RMSE %.6f meV\n",
            result.shells, result.vectors, result.magnitude_rmse, result.magnitude_max, result.complex_rmse)
    result
end
registry_decay = map(0:4) do shell
    terms = filter(t -> t.shell == shell, registry_terms)
    coefficients = 1000vcat([t.forward for t in terms], [t.backward for t in terms if !iszero(t.G)])
    result = (shell=shell, count=length(coefficients), maximum=maximum(abs,coefficients),
              rms=sqrt(sum(abs2,coefficients)/length(coefficients)), aa_sum=sum(coefficients))
    @printf("Shell %d: %d coefficients, max %.6f, RMS %.6f; MM sum %.6f%+.6fi meV\n",
            result.shell, result.count, result.maximum, result.rms, real(result.aa_sum), imag(result.aa_sum))
    result
end
# The diagonal contains twelve original grid samples, plus the periodic MM endpoint.
registry_sample_t = collect(0:12) ./ 12
registry_sample_magnitudes = map(registry_sample_t) do t
    d = SVector(mod(1-t,1.0), mod(t,1.0))
    index = findfirst(p -> isapprox(p,d; atol=1e-10), scan.d)
    @assert !isnothing(index)
    1000abs(registry_raw[index])
end
# MM: shift_i00_j00; XM: shift_i08_j04; MX: shift_i04_j08 (top-bottom order).
registry_nodes = [("MM", 0.0), ("XM", 1/3), ("MX", 2/3), ("MM", 1.0)]
# Check the optimized scalar sum against the package's full matrix reconstruction.
for (label, t) in registry_nodes
    d = SVector(1-t, t)
    reference = ERTCA._reconstruct_W(h, d, KPRIME)[SPIN_CHANNEL.β, SPIN_CHANNEL.βp]
    @assert isapprox(fitted_registry_tunneling(d), reference; atol=1e-11, rtol=1e-10)
    @printf("Panel (b), %s: d=(%.6f,%.6f), |W(K′,d)|=%.6f meV\n",
            label, d..., 1000abs(reference))
end
@assert isapprox(first(registry_values), last(registry_values); atol=1e-11, rtol=1e-10)

@printf("Fit: %d stackings, %d reciprocal vectors; plotting %d harmonics.\n",
        length(scan.d), 2length(h.gvecs) - 1, length(harmonics))
@printf("K′: fractional (2/3,-1/3); Cartesian (%.9f, %.9f) Å⁻¹.\n", (bT * KPRIME)...)
for (index, harmonic) in pairs(harmonics)
    @printf("  G%d = (%d,%d)\n", index - 1, harmonic.G...)
end



for (label, point) in nodes[1:3]
    vals = [1000abs(tunneling_value(h, harmonic, point)) for harmonic in harmonics]
    @printf("%s: %s meV\n", label, join([@sprintf("%.6f", x) for x in vals], ", "))
end

function tex_curve(io, x, y; style)
    println(io, "\\addplot[no marks, ", style, "] coordinates {")
    for (xi, yi) in zip(x, y)
        @printf(io, "(%.12g,%.12g)\n", xi, yi)
    end
    println(io, "};")
end

function curve_style(index)
    # Additional shells reuse the palette with distinct patterns; the default
    # publication figure contains seven curves with seven different colours.
    color = mod1(index, length(CURVE_COLORS))
    dash = index <= 7 ? CURVE_DASHES[index] :
           (index <= 14 ? "dash pattern=on 2pt off 1.5pt" :
                          "dash pattern=on 4pt off 1.5pt on 1pt off 1.5pt")
    return "harmonic$color,$dash,line width=0.9pt"
end

# Brillouin-zone inset. Derive the highlighted path from the same
# Cartesian momenta used for the curves, preserving the unrotated hexagon axes.
function draw_path_inset(io, centre_x, centre_y, points;
                         radius=18.0, show_labels=true, path_color="red")
    scale = radius / norm(bT * KPRIME)
    xy(k) = (centre_x + scale * (bT * k)[1], centre_y + scale * (bT * k)[2])
    println(io, raw"\draw[black,fill=white,line width=0.6pt,line join=round]")
    for j in 0:5
        @printf(io, "(%.5fpt,%.5fpt) -- ",
                centre_x + radius * cosd(60j), centre_y + radius * sind(60j))
    end
    println(io, "cycle;")
    println(io, "\\path[fill=$path_color,fill opacity=0.05]")
    for k in points[1:3]
        @printf(io, "(%.5fpt,%.5fpt) -- ", xy(k)...)
    end
    println(io, "cycle;")
    println(io, "\\draw[$path_color,line width=0.8pt,line join=round]")
    for (i, k) in enumerate(points)
        @printf(io, "(%.5fpt,%.5fpt)", xy(k)...)
        print(io, i == length(points) ? ";\n" : " -- ")
    end
    # Small open arrowheads halfway along Γ → M → K′ → Γ, rotated with the path.
    for i in 1:3
        a, b = SVector(xy(points[i])...), SVector(xy(points[i+1])...)
        direction = normalize(b - a)
        normal = SVector(-direction[2], direction[1])
        tip = a + 0.57 * (b - a)
        wing1 = tip - 1.5direction + 0.8normal
        wing2 = tip - 1.5direction - 0.8normal
        @printf(io, "\\draw[%s,line width=0.65pt,line cap=round,line join=round] (%.5fpt,%.5fpt) -- (%.5fpt,%.5fpt) -- (%.5fpt,%.5fpt);\n",
                path_color, wing1..., tip..., wing2...)
    end
    @printf(io, "\\fill[black] (%.5fpt,%.5fpt) circle[radius=0.7pt];\n",
            centre_x, centre_y)
    show_labels || return nothing
    labels = (raw"\Gamma", "M", raw"K^{\prime}")
    directions = (-normalize(bT * (points[2] + points[3])),
                  normalize(bT * points[2]), normalize(bT * points[3]))
    for i in 1:3
        d = directions[i]
        p = SVector(xy(points[i])...) + (i == 1 ? 2.0 : 2.5) * d
        anchor = abs(d[1]) < 1e-8 ? (d[2] > 0 ? "south" : "north") :
                 abs(d[2]) < 1e-8 ? (d[1] > 0 ? "west" : "east") :
                 (d[2] > 0 ? "south " : "north ") * (d[1] > 0 ? "west" : "east")
        @printf(io, "\\node[anchor=%s,inner sep=0.4pt,fill=white,font=\\fontsize{14}{16}\\selectfont] at (%.5fpt,%.5fpt) {\$%s\$};\n",
                anchor, p..., labels[i])
    end
end

function draw_xm_zoom(io, registry_ymax)
    # Only the display sampling is refined; the fitted coefficients and all
    # main-panel data are unchanged. The horizontal axis is the same registry t.
    txm = 1 / 3
    halfwidth = 1 / 75
    tmin, tmax = txm - halfwidth, txm + halfwidth
    zoom_ymax = 2.5
    zoom_t = collect(range(tmin, tmax; length=601))
    zoom_magnitudes = [1000abs(fitted_registry_tunneling(SVector(1-t,t); shells=s))
                       for t in zoom_t, s in 1:4]

    # Mark the enlarged region in panel (b), with a light connector to the inset.
    xlo, xhi = 58 + 444tmin, 58 + 444tmax
    ytop = 40 + 138zoom_ymax / registry_ymax
    @printf(io,"\\draw[black!45,line width=0.45pt] (%.5fpt,40pt) rectangle (%.5fpt,%.5fpt);\n",xlo,xhi,ytop)
    @printf(io,"\\draw[black!45,line width=0.45pt,->] (%.5fpt,%.5fpt) to[out=75,in=235] (260pt,116pt);\n",xhi,ytop)
    print(io,raw"""
\begin{axis}[
  at={(247pt,116pt)},anchor=south west,
  width=96pt,height=53pt,scale only axis,
  axis lines=box,axis on top,axis background/.style={fill=white},
  axis line style={black,line width=0.55pt},
  tick align=inside,tick pos=both,tick style={black,line width=0.55pt},
  tick label style={font=\fontsize{10}{12}\selectfont},
  label style={font=\fontsize{10}{12}\selectfont},
  ylabel={},
  xtick={0.3333333333333333},xticklabels={XM},
  xticklabel style={yshift=-1pt},ytick={0,1,2},
  scaled ticks=false,grid=none,enlargelimits=false,
""")
    println(io,"  xmin=$tmin,xmax=$tmax,ymin=0,ymax=$zoom_ymax]")
    tex_curve(io,[txm,txm],[0,zoom_ymax];style="black!14,line width=0.45pt,forget plot")
    for shells in 1:4
        color=("harmonic2","harmonic3","harmonic4","harmonic1")[shells]
        tex_curve(io,zoom_t,zoom_magnitudes[:,shells];style="$color,line width=0.9pt,forget plot")
    end
    println(io,raw"\addplot[only marks,mark=*,black,mark size=1.5pt,forget plot] coordinates {")
    for (t,value) in zip(registry_sample_t,registry_sample_magnitudes)
        tmin <= t <= tmax || continue
        @printf(io,"(%.12g,%.12g)\n",t,value)
    end
    println(io,raw"};")
    println(io,raw"\end{axis}")
end

function write_figure_tex(filename)
    ymax = 1.04maximum(magnitudes)
    height = 440
    open(filename, "w") do io
        print(io,raw"""
\documentclass[10pt,tikz,border=3pt]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}[font=\fontsize{16}{19}\selectfont]
""")
        for (index,color) in pairs(CURVE_COLORS)
            println(io,"\\definecolor{harmonic$index}{HTML}{$color}")
        end
        println(io,raw"\node[anchor=north west,inner sep=0pt] at (5pt,438pt) {a)};")
        print(io,raw"""
\begin{axis}[
  at={(58pt,245pt)},anchor=south west,
  width=444pt,height=146pt,scale only axis,
  axis lines=box,axis on top,
  axis line style={black,line width=0.55pt},
  tick align=inside,tick pos=both,tick style={black,line width=0.55pt},
  tick label style={font=\fontsize{16}{19}\selectfont},
  label style={font=\fontsize{16}{19}\selectfont},
  ylabel={$\lvert W_{b\uparrow,t\uparrow}(\bar k,\bar G)\rvert$ (meV)},
  legend columns=7,legend style={at={(0.5,1.06)},anchor=south,draw=black,line width=0.55pt,fill=white,font=\fontsize{16}{19}\selectfont,column sep=4pt},
  scaled ticks=false,grid=none,enlargelimits=false,
  xticklabels={$\Gamma$,$M$,$K^{\prime}$,$\Gamma$,$\kern-6pt C_3M\kern6pt$,$\kern6pt C_3K^{\prime}\kern-6pt$,$\Gamma$,$\kern-6pt C_3^2M\kern6pt$,$\kern6pt C_3^2K^{\prime}\kern-6pt$,$\Gamma$},
""")
        println(io,"  xmin=0,xmax=$(path.dist[end]),ymin=0,ymax=$ymax,")
        println(io,"  xtick={",join(path.ticks,","),"},ytick={0,50,100,150,200,250}]")
        for (section,first_node) in enumerate((1,4,7))
            color=("harmonic4","harmonic6","harmonic2")[section]
            @printf(io,"\\path[fill=%s,fill opacity=0.055,draw=%s,draw opacity=0.28,line width=0.35pt] (axis cs:%.12g,0) rectangle (axis cs:%.12g,%.12g);\n",color,color,path.ticks[first_node],path.ticks[first_node+3],ymax)
        end
        for t in path.ticks[2:end-1]
            tex_curve(io,[t,t],[0,ymax];style="black!14,line width=0.45pt,forget plot")
        end
        for ig in eachindex(harmonics)
            tex_curve(io,path.dist,magnitudes[:,ig];style=curve_style(ig))
            @printf(io, "\\addlegendentry{\$\\bar{G}_{%d}\$}\n", ig-1)
        end
        println(io,raw"\end{axis}")
        for (section,first_node) in enumerate((1,4,7))
            lo,hi=path.ticks[first_node],path.ticks[first_node+3]
            x=58+444*(lo+0.27*(hi-lo))/path.dist[end]
            points=[last(node) for node in nodes[first_node:first_node+3]]
            color=("harmonic4","harmonic6","harmonic2")[section]
            draw_path_inset(io,x,362.0,points;show_labels=section==1,path_color=color)
        end
        println(io,raw"\node[anchor=north west,inner sep=0pt] at (5pt,213pt) {b)};")
        print(io,raw"""
\begin{axis}[
  at={(58pt,40pt)},anchor=south west,
  width=444pt,height=138pt,scale only axis,
  axis lines=box,axis on top,
  axis line style={black,line width=0.55pt},
  tick align=inside,tick pos=both,tick style={black,line width=0.55pt},
  tick label style={font=\fontsize{16}{19}\selectfont},
  label style={font=\fontsize{16}{19}\selectfont},
  ylabel={$\lvert W_{b\uparrow,t\uparrow}(K^{\prime},\mathbf d)\rvert$ (meV)},
  xlabel={},
  xtick={0,0.3333333333333333,0.6666666666666667,1},
  xticklabels={MM,XM,MX,MM},
  xticklabel style={yshift=-6pt},
  legend columns=4,legend style={at={(0.5,1.06)},anchor=south,draw=black,line width=0.55pt,fill=white,font=\fontsize{14}{16}\selectfont,column sep=4pt},
  scaled ticks=false,grid=none,enlargelimits=false,
""")
        registry_ymax=1.08maximum(registry_shell_magnitudes)
        println(io,"  xmin=0,xmax=1,ymin=0,ymax=$registry_ymax]")
        for t in (1/3,2/3)
            tex_curve(io,[t,t],[0,registry_ymax];style="black!14,line width=0.45pt,forget plot")
        end
        for shells in 1:4
            color=("harmonic2","harmonic3","harmonic4","harmonic1")[shells]
            tex_curve(io,registry_t,registry_shell_magnitudes[:,shells];style="$color,line width=0.9pt")
            println(io,"\\addlegendentry{",shells,shells==1 ? " shell}" : " shells}")
        end
        println(io, raw"\addplot[only marks,mark=*,black,mark size=2.2pt,forget plot] coordinates {")
        for (t, value) in zip(registry_sample_t, registry_sample_magnitudes)
            @printf(io, "(%.12g,%.12g)\n", t, value)
        end
        println(io, raw"};")
        println(io,raw"\end{axis}")
        draw_xm_zoom(io,registry_ymax)
        println(io,raw"\end{tikzpicture}")
        println(io,raw"\end{document}")
    end
end

function executable(program)
    path = Sys.which(program)
    if isnothing(path) && Sys.isapple()
        for directory in ("/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin")
            path = Sys.which(joinpath(directory, program))
            isnothing(path) || return path
        end
    end
    isnothing(path) && error("$program is required; use the TeX/Poppler setup from figure 1.")
    return path
end

function save_paper_figure()
    mkpath(PAPER_FIGURES)
    builddir = joinpath(@__DIR__, ".cache", "figures")
    mkpath(builddir)
    # Keep evaluated data beside the generated TeX for inspection or restyling.
    open(joinpath(builddir, STEM * ".csv"), "w") do io
        println(io, "distance_inv_A,k1,k2,", join(["G$(i-1)_meV" for i in eachindex(harmonics)], ","))
        for (ik, k) in pairs(path.kpts)
            println(io, join(vcat(path.dist[ik], collect(k), magnitudes[ik, :]), ","))
        end
    end
    open(joinpath(builddir, STEM * "_registry_path.csv"), "w") do io
        println(io, "t,d1,d2,real_W_meV,imag_W_meV,abs_W_meV")
        for i in eachindex(registry_t)
            println(io, join((registry_t[i], registry_d[i]...,
                             1000real(registry_values[i]), 1000imag(registry_values[i]),
                             registry_magnitudes[i]), ","))
        end
    end
    texfile = joinpath(builddir, STEM * ".tex")
    open(joinpath(builddir, STEM * "_shell_convergence.csv"), "w") do io
        println(io, "t,d1,d2,one_shell_meV,two_shells_meV,three_shells_meV,four_shells_meV")
        for i in eachindex(registry_t)
            println(io, join((registry_t[i], registry_d[i]..., registry_shell_magnitudes[i,:]...), ","))
        end
    end
    open(joinpath(builddir, STEM * "_shell_errors.csv"), "w") do io
        println(io, "shells,vectors,magnitude_rmse_meV,magnitude_max_error_meV,complex_rmse_meV")
        for r in registry_errors
            println(io, join(values(r), ","))
        end
    end
    open(joinpath(builddir, STEM * "_shell_decay.csv"), "w") do io
        println(io, "shell,count,max_abs_meV,rms_abs_meV,aa_sum_real_meV,aa_sum_imag_meV")
        for r in registry_decay
            println(io, join((r.shell,r.count,r.maximum,r.rms,real(r.aa_sum),imag(r.aa_sum)), ","))
        end
    end
    write_figure_tex(texfile)
    pdflatex = executable("pdflatex")
    pdftoppm = executable("pdftoppm")
    logfile = joinpath(builddir, STEM * "_build.log")
    open(logfile, "w") do io
        cmd = Cmd([pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                   "-output-directory=$builddir", texfile])
        run(pipeline(cmd; stdout=io, stderr=io))
    end
    pdf = joinpath(PAPER_FIGURES, STEM * ".pdf")
    cp(joinpath(builddir, STEM * ".pdf"), pdf; force=true)
    run(Cmd([pdftoppm, "-r", string(PNG_DPI), "-png", "-singlefile", pdf, joinpath(PAPER_FIGURES, STEM)]))
    println("Figure → ", pdf)
    println("Preview → ", joinpath(PAPER_FIGURES, STEM * ".png"))
end

save_paper_figure()

