# Supplementary S1: Nb-Nb interlayer separation.
# Run from this repository: julia --project=. -t auto Fig9.jl

ENV["GKSwstype"] = "100"
using LinearAlgebra
using Printf
import Plots
import NaturalNeighbours

const FIG4_INPUT = normpath(joinpath(@__DIR__, "Inputs", "NbSe2_interlayer_distance.dat"))
const FIG4_OUTDIR = normpath(joinpath(@__DIR__, "Figures"))
const FIG4_COORDINATES = :cartesian  # :indices for n,m; :cartesian for dx,dy in Å
const FIG4_FONT = 7  # Same GR calibration as Fig3: thickness_scaling=6.74/7 for ~10 pt text.
const FIG4_DOT_ALPHA = 0.5
const FIG4_COLORBAR_STEP = 0.1  # Å: half as many labels as the previous 0.05 spacing.

function fig4_data(path=FIG4_INPUT)
    lines = readlines(path)
    function metadata(key)
        prefix = "# " * key * ":"
        k = findfirst(line -> startswith(line, prefix), lines)
        isnothing(k) && error("$path: missing metadata $key")
        return strip(split(lines[k], ':'; limit=2)[2])
    end
    n = parse(Int, metadata("ngrid"))
    n > 1 || error("$path: ngrid must exceed one")
    a1 = parse.(Float64, split(metadata("a_1_Ang")))
    a2 = parse.(Float64, split(metadata("a_2_Ang")))
    length(a1) == length(a2) == 3 || error("$path: expected three components per lattice vector")
    all(isfinite, vcat(a1, a2)) || error("$path: nonfinite lattice")
    maximum(abs, [a1[3], a2[3]]) < 1e-6 || error("$path: lattice is not in the xy plane")
    A = hcat(a1[1:2], a2[1:2])
    abs(det(A)) > 1e-10 || error("$path: singular in-plane lattice")
    rows = [parse.(Float64, split(line)) for line in lines
            if !isempty(strip(line)) && !startswith(strip(line), '#')]
    length(rows) == n^2 || error("$path: expected $(n^2) samples, found $(length(rows))")
    D = fill(NaN, n, n)  # D[j+1,i+1] = vertical separation at d=(i*a1+j*a2)/n
    xs, ys = Float64[], Float64[]
    for row in rows
        length(row) == 9 && all(isfinite, row) || error("$path: expected nine finite columns")
        i, j = Int(row[1]), Int(row[2])
        0 <= i < n && 0 <= j < n || error("$path: shift ($i,$j) outside the grid")
        isnan(D[j+1, i+1]) || error("$path: duplicate shift ($i,$j)")
        u, v, dx, dy, zb, zt, distance = row[3:9]
        maximum(abs, [u-i/n, v-j/n]) < 1e-8 || error("$path: inconsistent fractional shift")
        maximum(abs, A * [i/n, j/n] - [dx, dy]) < 1e-7 || error("$path: inconsistent Cartesian shift")
        distance > 0 && abs(distance-abs(zt-zb)) < 1e-7 || error("$path: inconsistent Nb separation")
        D[j+1, i+1] = distance
        push!(xs, dx)
        push!(ys, dy)
    end
    all(isfinite, D) || error("$path: incomplete shift grid")
    return (; n, A, D, xs, ys)
end

"""Build a periodic natural-neighbour interpolator in Cartesian coordinates."""
function fig4_interpolator(data)
    (; n, A, D) = data
    # Periodic padding supplies neighbours at the edges of the single plotted cell.
    # Use Cartesian Å here: triangulating fractional u,v would use the wrong metric.
    x_t, y_t, distance_t = Float64[], Float64[], Float64[]
    for du in -1:1, dv in -1:1, i in 0:n-1, j in 0:n-1
        x, y = A * [i/n + du, j/n + dv]
        push!(x_t, x)
        push!(y_t, y)
        push!(distance_t, D[j+1, i+1])
    end
    # interpolate constructs the Delaunay triangulation automatically.
    interpolator = NaturalNeighbours.interpolate(x_t, y_t, distance_t)
    method = NaturalNeighbours.Sibson()  # Use Triangle() for piecewise linear interpolation.

    # Wrap fractional shifts into the central cell so opposite boundaries agree.
    function periodic_distance(u, v)
        x, y = A * [mod(u, 1.0), mod(v, 1.0)]
        return interpolator(x, y; method=method, project=false)
    end
    return periodic_distance
end

function fig4_figure(data; resolution=650, coordinates=FIG4_COORDINATES)
    (; A, D, xs, ys) = data
    interp = fig4_interpolator(data)
    coordinates in (:indices, :cartesian) || error("coordinates must be :indices or :cartesian")
    indices = coordinates == :indices
    # Map fractional registry coordinates to either index coordinates or Cartesian Å.
    basis = indices ? Float64(data.n) * Matrix{Float64}(I, 2, 2) : A
    samples = basis * (A \ permutedims(hcat(xs, ys)))
    corners = basis * [0.0 1 1 0 0; 0 0 1 1 0]
    xmin, xmax = extrema(corners[1, :])
    ymin, ymax = extrema(corners[2, :])
    gx = range(xmin, xmax; length=resolution)
    gy = range(ymin, ymax; length=round(Int, resolution*(ymax-ymin)/(xmax-xmin)))
    basis_inv = inv(basis)
    image = fill(NaN, length(gy), length(gx))
    for (ix, x) in enumerate(gx), (iy, y) in enumerate(gy)
        u, v = basis_inv * [x, y]
        if -1e-10 <= u <= 1+1e-10 && -1e-10 <= v <= 1+1e-10
            image[iy, ix] = interp(clamp(u, 0, 1), clamp(v, 0, 1))
        end
    end
    # Include any smooth-interpolation overshoot in the colour scale.
    image_lo, image_hi = extrema(filter(isfinite, image))
    lo, hi = min(minimum(D), image_lo), max(maximum(D), image_hi)
    # Extend color limits only for constant datasets, for which a zero-width bar is undefined.
    clims = lo < hi ? (lo, hi) : (lo-0.005, hi+0.005)
    colorbar_ticks = FIG4_COLORBAR_STEP .* collect(
        ceil(Int, clims[1]/FIG4_COLORBAR_STEP):floor(Int, clims[2]/FIG4_COLORBAR_STEP))
    isempty(colorbar_ticks) && (colorbar_ticks = [(lo+hi)/2])
    formula = raw"$\mathbf{d}=\frac{n}{" * string(data.n) * raw"}\mathbf{a}_1+\frac{m}{" *
              string(data.n) * raw"}\mathbf{a}_2$"
    ticks = collect(0:max(1, data.n ÷ 4):data.n)
    last(ticks) == data.n || push!(ticks, data.n)
    p = Plots.heatmap(gx, gy, image;
        # PDF width 246 points ≈ 86.8 mm: include at \columnwidth in the paper.
        size=indices ? (246, 235) : (246, 145), dpi=600,
        thickness_scaling=6.74/7,
        fontfamily="Computer Modern", tickfontsize=FIG4_FONT, guidefontsize=FIG4_FONT,
        colorbar_tickfontsize=FIG4_FONT, colorbar_titlefontsize=FIG4_FONT,
        xlabel=indices ? raw"$n$" : raw"$d_x\;(\mathrm{\AA})$",
        ylabel=indices ? raw"$m$" : raw"$d_y\;(\mathrm{\AA})$",
        title=indices ? formula : "", titlefontsize=FIG4_FONT,
        xticks=indices ? ticks : :auto, yticks=indices ? ticks : :auto,
        colorbar_title="",  # Caption: Nb–Nb vertical separation in Å.
        color=:jet, clims=clims, colorbar=false,
        aspect_ratio=:equal, framestyle=:box, grid=false, legend=false,
        tick_direction=:in, background_color=:white, foreground_color=:black,
        xlims=indices ? (xmin, xmax) : (xmin-0.15, xmax+0.15),
        ylims=indices ? (ymin, ymax) : (ymin-0.12, ymax+0.12),
        left_margin=0Plots.mm, bottom_margin=0Plots.mm,
        # Reserve room for the narrow colour bar, its labels, and outer padding.
        right_margin=8Plots.mm, top_margin=indices ? 4Plots.mm : 0.5Plots.mm)
    if !indices
        Plots.plot!(p, corners[1, :], corners[2, :]; color=:black, lw=0.8, label="",
                    colorbar_entry=false)
    end
    Plots.scatter!(p, samples[1, :], samples[2, :]; markersize=0.8, markercolor=:white,
                   markerstrokewidth=0, markeralpha=FIG4_DOT_ALPHA, label="", colorbar_entry=false)
    # This GR version ignores custom ticks on its built-in colour bar.
    # A slim inset heatmap uses a normal y axis, so the chosen ticks are respected.
    Plots.plot!(p; inset_subplots=(1, Plots.bbox(1.03, 0.0, 0.04, 1.0)))
    heights = collect(range(clims[1], clims[2]; length=256))
    Plots.heatmap!(p, [0.0, 1.0], heights, hcat(heights, heights);
        subplot=2, color=:jet, clims=clims, colorbar=false,
        xlims=(0, 1), ylims=clims, xaxis=false, ymirror=true,
        yticks=(colorbar_ticks, [@sprintf("%.1f", t) for t in colorbar_ticks]),
        ylabel="", grid=false, legend=false, framestyle=:axes,
        tick_direction=:out, fontfamily="Computer Modern", tickfontsize=FIG4_FONT,
        foreground_color=:black, background_color=:white)
    return p
end

function main_fig4(path=FIG4_INPUT; outdir=FIG4_OUTDIR, coordinates=FIG4_COORDINATES)
    data = fig4_data(path)
    lo, hi = extrema(data.D)
    @printf("Nb–Nb vertical separation: %.5f–%.5f Å (%d registry samples)\n", lo, hi, data.n^2)
    for (label, idx) in (("minimum", argmin(data.D)), ("maximum", argmax(data.D)))
        j, i = Tuple(idx) .- 1
        @printf("  %s: (i,j)=(%d,%d), Δz=%.5f Å\n", label, i, j, data.D[idx])
    end
    fig = fig4_figure(data; coordinates)
    mkpath(outdir)
    for ext in ("pdf", "png")
        output = joinpath(outdir, "Fig9." * ext)
        Plots.savefig(fig, output)
        println("figure → ", output)
    end
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_fig4(isempty(ARGS) ? FIG4_INPUT : ARGS[1])
end
