# Shared WSe2 and NbSe2 valley calculations and figure layouts.

ENV["GKSwstype"] = "100"

const CODE_DIR = @__DIR__

include(joinpath(@__DIR__, "common.jl"))

import Plots

const HARMONIC_SHELLS = 4

const PLANE_WAVE_SHELLS = 4

const NBANDS = 10

const NPOINTS = 200

const EWIN = (-250.0, 15.0)

function gamma_kp_models(ctx)
    V = valley_setup(ctx, :Gamma)
    h = V.h
    P = valley_parameters(h, V.kbar, ctx.lat)
    @assert all(<(0), P.hess) "The Gamma effective mass must describe a band maximum."
    shells = ERTCA._shell_ids(h)
    keep = findall(s -> s <= HARMONIC_SHELLS, shells)

    function build(; momentum_dependent_g0::Bool)
        blocks = map(CartesianIndices(h.blocks[keep, :, :])) do I
            q, i, j = Tuple(I)
            iq = keep[q]
            # Add the parabola and original averaged first-shell intralayer potential below.
            if i == j && shells[iq] <= 1
                return HarmonicBlock(SVector{2,Float64}[], ComplexF64[])
            end
            if momentum_dependent_g0 && iszero(h.gvecs[iq]) && h.shifted[i] != h.shifted[j]
                # Preserve the exact complex Bloch sum, including its phase and anisotropy.
                # No radial averaging, interpolation, or momentum cutoff is introduced here.
                return h.blocks[iq, i, j]
            end
            value = ComplexF64(_harmonic_value(h, iq, i, j, V.kbar))
            return HarmonicBlock([SVector(0.0, 0.0)], [value])
        end
        harmonics = RegistryHarmonics(h.lattice, h.labels, h.shifted, h.tau,
            h.layer_names, h.gvecs[keep], blocks, h.cond, h.hermiticity, h.d,
            momentum_dependent_g0 ? "Gamma k.p, w0(k)" : "Gamma k.p, constant couplings")
        model = ertca_model(harmonics, rad2deg(ctx.theta);
            shells=PLANE_WAVE_SHELLS, expansion=collect(V.expansion))
        potential = vcat(
            [t ⊗ project(:wannier, h.labels[1]) for t in harmonic_potential(ctx.moire, P.V, P.psi)],
            [t ⊗ project(:wannier, h.labels[2]) for t in harmonic_potential(ctx.moire, P.V, -P.psi)])
        return compile(model, [k_quadratic(-HBAR2_2M / P.mstar; name="hole_kinetic"); potential])
    end
    return (; constant=build(momentum_dependent_g0=false),
              g0_dependent=build(momentum_dependent_g0=true), edge=P.edge, valley=V)
end

function gamma_nodes()
    return ["γ" => [0.0, 0.0], "m" => [0.5, 0.0], "κ" => [1/3, 1/3],
             "κ′" => [-1/3, 2/3], "γ" => [0.0, 0.0]]
end

gamma_path(ctx) = kpath_line(ctx.moire, gamma_nodes(); npoints=NPOINTS)

function spectrum_on_path(model, ks)
    return hcat(bz_map((k, r) -> copy(r.values), model, ks; vectors=false)...)
end

function gamma_bz_panel(ctx; nodes=gamma_nodes(), show_angle=true, show_bz_title=true,
        label_size=7, marker_size=1.5, arrow_size=(0.16, 0.08),
        labels=[raw"$\gamma$", raw"$m$", raw"$\kappa$", raw"$\kappa^{\prime}$"])
    B = Matrix(ctx.bmT)
    # A rigid rotation puts b1 horizontally, preserving angles and distances.
    e1 = B[:, 1] / norm(B[:, 1])
    e2 = [-e1[2], e1[1]]
    R = transpose(hcat(e1, e2))
    vertices = [[1/3, 1/3], [-1/3, 2/3], [-2/3, 1/3],
                [-1/3, -1/3], [1/3, -2/3], [2/3, -1/3]]
    radius = norm(B * first(vertices))
    xy(k) = R * B * k / radius
    polygon = hcat(xy.(vertices)...)
    route = hcat([xy(last(node)) for node in nodes]...)
    # Check that these vertices belong to the lattice's Wigner-Seitz cell.
    for v in eachcol(polygon), i in -2:2, j in -2:2
        g = xy([i, j])
        @assert dot(v, g) <= dot(g, g)/2 + 1e-10
    end
    q = Plots.plot(; aspect_ratio=:equal, framestyle=:none, grid=false,
        legend=false, fontfamily="Computer Modern", background_color=:white,
        xlims=(-1.4, 1.4), ylims=(-1.65, 1.65),
        left_margin=0Plots.mm, right_margin=1Plots.mm,
        top_margin=2Plots.mm, bottom_margin=2Plots.mm)
    closed = hcat(polygon, polygon[:, 1])
    Plots.plot!(q, closed[1, :], closed[2, :]; color=:black, lw=0.65, label="")
    route_color = "#8C4CB8"
    Plots.plot!(q, route[1, :], route[2, :]; color=route_color, lw=0.9, label="")
    for i in 1:(length(nodes)-1)
        a, b = route[:, i], route[:, i+1]
        # Draw the arrowhead in reciprocal-space units: GR's native arrows
        # otherwise become oversized in narrow side panels.
        direction = (b-a) / norm(b-a)
        normal = [-direction[2], direction[1]]
        tip = a + 0.65*(b-a)
        base = tip - arrow_size[1]*direction
        wings = hcat(tip, base + arrow_size[2]/2*normal, base - arrow_size[2]/2*normal)
        Plots.plot!(q, Plots.Shape(wings[1, :], wings[2, :]);
            color=route_color, lw=0, label="")
    end
    Plots.scatter!(q, route[1, 1:end-1], route[2, 1:end-1]; color=route_color,
        markersize=marker_size, markerstrokewidth=0, label="")
    for i in 1:(length(nodes)-1)
        r = route[:, i]
        pos = i == 1 ? r + [-0.14, -0.14] : 1.22*r
        Plots.annotate!(q, pos[1], pos[2], Plots.text(labels[i], label_size, :black))
    end
    show_bz_title && Plots.annotate!(q, 0, 1.45, Plots.text(raw"$\mathrm{mBZ}$", 7, :black))
    angle_label = "\$\\theta = $(round(rad2deg(ctx.theta); digits=4))^{\\circ}\$"
    show_angle && Plots.annotate!(q, 0, -1.45, Plots.text(angle_label, 7, :black))
    return q
end

function gamma_comparison_figure(ctx, path, Efull, Econstant, Eg0; include_bz=true, draw_frame=true,
        energy_window=EWIN, nbands::Int=NBANDS,
        ticks=[raw"$\gamma$", raw"$m$", raw"$\kappa$", raw"$\kappa^{\prime}$", raw"$\gamma$"])
    @assert size(Efull, 2) == size(Econstant, 2) == size(Eg0, 2) == length(path.dist)
    @assert all(isfinite, Efull) && all(isfinite, Econstant) && all(isfinite, Eg0)
    # A single common reference; the k.p band-edge constant is restored before this call.
    # This preserves relative shifts between the three models.
    zero_ref = maximum(Efull)
    top(E, n) = 1000 .* (E[max(1, size(E, 1)-n+1):end, :] .- zero_ref)
    red = top(Efull, 2nbands)  # both spin partners
    blue = top(Econstant, nbands)
    black = top(Eg0, nbands)
    xpad = 0.03 * (last(path.dist) - first(path.dist))
    xmin, xmax = first(path.dist)-xpad, last(path.dist)+xpad
    p = Plots.plot(;
        size=(246, 220), dpi=600,
        fontfamily="Computer Modern", tickfontsize=6, guidefontsize=7,
        title="", legend=false, grid=false, framestyle=:axes,
        foreground_color=:black, background_color=:white,
        tick_direction=:in, minorgrid=false,
        xticks=(path.ticks, ticks), ylabel=raw"$E\;(\mathrm{meV})$",
        xlims=(xmin, xmax), ylims=energy_window, yticks=-250:50:0,
        left_margin=2Plots.mm, right_margin=2Plots.mm,
        bottom_margin=2Plots.mm, top_margin=2Plots.mm)
    Plots.plot!(p, path.dist, transpose(blue); color="#0087CC", lw=0.55, ls=:dash, label="")
    Plots.plot!(p, path.dist, transpose(black); color=:black, lw=0.55, ls=:dash, label="")
    Plots.plot!(p, path.dist, transpose(red); color=:red, lw=0.55, ls=:solid, label="")
    # Complete the native bottom/left axes to match the boxed WSe2 reference figure.
    # GR clips half a stroke at the boundary: lw=2 matches the native frame thickness.
    if draw_frame
        Plots.plot!(p, [xmin, xmax, xmax], [energy_window[2], energy_window[2], energy_window[1]];
            color=:black, lw=2, label="", primary=false)
    end
    include_bz || return p
    q = gamma_bz_panel(ctx)
    return Plots.plot(p, q; layout=Plots.grid(1, 2; widths=[0.66, 0.34]),
        size=(390, 220), dpi=600)
end

using Serialization

const WSE2_NB = 7

const WSE2_REF = normpath(joinpath(@__DIR__, "..", "Inputs", "WSe2_DFT_reference"))

const FIG3_FRAME_LINEWIDTH = (0.55 * 72 / 72.27) / (6.74 / 7)

function fig3_box!(panel)
    xmin, xmax = Plots.xlims(panel)
    ymin, ymax = Plots.ylims(panel)
    xticks = first(Plots.xticks(panel)[1])
    yticks = first(Plots.yticks(panel)[1])
    lw = FIG3_FRAME_LINEWIDTH
    Plots.plot!(panel; framestyle=:grid, grid=false, tick_direction=:none)
    # Half the boundary stroke is clipped by GR, so double only the frame width.
    Plots.plot!(panel, [xmin,xmax,xmax,xmin,xmin], [ymin,ymin,ymax,ymax,ymin];
        color=:black, lw=2lw, label="", primary=false)
    x, y = Float64[], Float64[]
    dx, dy = 0.02(xmax-xmin), 0.02(ymax-ymin)
    for t in xticks
        xmin < t < xmax || continue
        append!(x, [t,t,NaN,t,t,NaN])
        append!(y, [ymin,ymin+dy,NaN,ymax,ymax-dy,NaN])
    end
    for t in yticks
        ymin < t < ymax || continue
        append!(x, [xmin,xmin+dx,NaN,xmax,xmax-dx,NaN])
        append!(y, [t,t,NaN,t,t,NaN])
    end
    Plots.plot!(panel, x, y; color=:black, lw=lw, label="", primary=false)
    return panel
end

function fig3_legend_box!(panel)
    xmin, xmax = Plots.xlims(panel)
    ymin, ymax = Plots.ylims(panel)
    # Every edge is at an inset boundary, so compensate for clipping exactly as
    # for the band box. A closed path also keeps all four corners even.
    Plots.plot!(panel,[xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin];
        color=:black,lw=2FIG3_FRAME_LINEWIDTH,label="",primary=false)
    return panel
end

function read_wse2_reference()
    krows = [split(l, ",") for l in readlines(joinpath(WSE2_REF, "kpath.csv"))[2:end]]
    ks = [[parse(Float64, r[2]), parse(Float64, r[3])] for r in krows]
    labels = [r[5] for r in krows]
    dist = [parse(Float64, r[6]) for r in krows]
    tbl = Dict{Tuple{Int,Int},Float64}()
    for l in readlines(joinpath(WSE2_REF, "bands.csv"))[2:end]
        r = split(l, ",")
        tbl[(parse(Int, r[1]), parse(Int, r[2]))] = parse(Float64, r[3])
    end
    bandidx = [6604 - 2 * (n - 1) for n in 1:WSE2_NB]
    E = [tbl[(ik, b)] * 1000 for b in bandidx, ik in eachindex(ks)]      # upper of each pair
    Elo = [tbl[(ik, b - 1)] * 1000 for b in bandidx, ik in eachindex(ks)] # lower of each pair
    return (ks=ks, labels=labels, dist=dist, E=E, Elo=Elo, bandidx=bandidx)
end

const WSE2_PAPER = (a0=3.317, theta=5.08, mstar=0.43, mstar_fit=0.38, V=9.0, psi=128.0, w=18.0)

function wse2_published_model(; shells::Int=GSHELLS, mstar::Real=WSE2_PAPER.mstar)
    th = WSE2_PAPER.theta * pi / 180
    aM = WSE2_PAPER.a0 / (2 * sin(th / 2))
    lat = bravais_lattice(:triangular; a=aM)
    B = recip_matrix(lat)
    kincof = 3809.982 / mstar                          # ħ²/2m* in meV·Å²
    cell = plane_wave_cell(lat; shells=shells, schema=InternalSchema(:layer => [1, -1]),
                           origins=:layer => [1 => Vector(B * [1 / 3, 1 / 3]),
                                              -1 => Vector(B * [2 / 3, 2 / 3])])
    psi = WSE2_PAPER.psi * pi / 180
    pot = vcat([t ⊗ project(:layer, 1) for t in harmonic_potential(lat, WSE2_PAPER.V, psi)],
               [t ⊗ project(:layer, -1) for t in harmonic_potential(lat, WSE2_PAPER.V, -psi)])
    return compile(cell, [k_quadratic(-kincof; name="hole_kinetic"); pot;
                          moire_tunnelling(cell, WSE2_PAPER.w)];
                   name="WLM at the published fit")
end

function wse2_comparison_data()
    ref = read_wse2_reference()
    ctx = ertca_context(WSE2; gshells=HARMONIC_SHELLS)
    V = valley_setup(ctx, :K)
    model = ertca_model(V.h, rad2deg(ctx.theta); shells=PLANE_WAVE_SHELLS,
        expansion=collect(V.expansion), name="ERTCA WSe2 near K")
    Emod = hcat([sort(bands(model, k); rev=true)[1:WSE2_NB] for k in ref.ks]...) .* 1000
    offset = maximum(ref.E) - maximum(Emod)
    Emod .+= offset
    @assert maximum(abs, Emod[1, :] - ref.E[1, :]) < 5.0
    nodes = ["γ" => [0.0, 0.0], "κ₊" => Vector(V.kap_t),
             "m" => [0.5, 0.0], "γ" => [0.0, 0.0]]
    sp = kpath_line(ctx.moire, nodes; npoints=100)
    # Match the blue curve in 00_dft_benchmark.jl: use the fitted mass 0.38 me.
    wlm = wse2_published_model(; mstar=WSE2_PAPER.mstar_fit)
    Ewlm = hcat([sort(bands(wlm, k); rev=true)[1:WSE2_NB] for k in ref.ks]...)
    offset_wlm = maximum(ref.E) - maximum(Ewlm)
    fitted_error = maximum(abs, Ewlm[1, :] .+ offset_wlm .- ref.E[1, :])
    quoted = wse2_published_model(; mstar=WSE2_PAPER.mstar)
    Equoted = hcat([sort(bands(quoted, k); rev=true)[1:WSE2_NB] for k in ref.ks]...)
    offset_quoted = maximum(ref.E) - maximum(Equoted)
    quoted_error = maximum(abs, Equoted[1, :] .+ offset_quoted .- ref.E[1, :])
    # This is the relative-error criterion in 00_dft_benchmark.jl.
    @assert quoted_error < 30.0
    @assert fitted_error < 0.4 * quoted_error
    println("WLM top-band errors: fitted mass = ", fitted_error,
            " meV; quoted mass = ", quoted_error, " meV")
    spw = spectrum_along_line(wlm, nodes; npoints=100)
    Ewdense = sort(spw.energies; rev=true, dims=1)[1:WSE2_NB, :] .+ offset_wlm
    full = ertca_model(ctx.h, ctx.tb; folding=:auto, name="WSe2 full BZ")
    Bfull = transpose(ctx.tb.moire.b)
    ks = [mod.(Bfull \ (V.expansion + ctx.bmT*k), 1.0) for k in sp.kpts]
    println("Computing WSe2 full-BZ bands...")
    all_full = spectrum_on_path(full, ks)
    Efull = reverse(all_full[end-2WSE2_NB+1:end, :]; dims=1) .* 1000 .+ offset
    @assert maximum(abs, Efull[:, 1] - Efull[:, end]) < 1e-5
    return (; ctx, ref, nodes, sp, spw, Efull, Ewdense)
end

function nbse2_comparison_data(; mn::Tuple{Int,Int}=MN)
    ctx = ertca_context(NBSE2; mn=mn, gshells=HARMONIC_SHELLS)
    kp = gamma_kp_models(ctx)
    path = gamma_path(ctx)
    full = ertca_model(ctx.h, ctx.tb; folding=:auto, name="NbSe2 full BZ")
    Bfull = transpose(ctx.tb.moire.b)
    ks = [mod.(Bfull \ (kp.valley.expansion + ctx.bmT*k), 1.0) for k in path.kpts]
    @printf("Computing NbSe2 full-BZ bands at %.6f degrees, (m,n)=(%d,%d), %d path points...\n",
            rad2deg(ctx.theta), mn..., length(ks))
    Efull = spectrum_on_path(full, ks)
    Econstant = spectrum_on_path(kp.constant, path.kpts) .+ kp.edge
    Eg0 = spectrum_on_path(kp.g0_dependent, path.kpts) .+ kp.edge
    @assert maximum(abs, Efull[:, 1] - Efull[:, end]) < 1e-8
    return (; ctx, path, Efull, Econstant, Eg0)
end

function wse2_band_panel(data)
    (; ref, sp, spw, Efull, Ewdense) = data
    zero_ref = maximum(ref.E)
    # Put the DFT and WLM path lengths on the same horizontal scale as ERTCA.
    xref = ref.dist .* (last(sp.dist) / last(ref.dist))
    xwlm = spw.dist .* (last(sp.dist) / last(spw.dist))
    # TeX-style subscript avoids the missing Unicode subscript-plus glyph in GR's font.
    paper_ticks = [raw"$\gamma$", raw"$\kappa_{-}$", raw"$m$", raw"$\gamma$"]
    xpad = 0.03 * (last(sp.dist) - first(sp.dist))
    p1 = Plots.plot(;
        # GR's PDF export has this size in points: 246 pt is approximately 86.8 mm.
        size=(246, 220), dpi=600,
        fontfamily="Computer Modern", tickfontsize=6, guidefontsize=7,
        # A thin frame is added after the final layout and ticks are selected.
        title="", legend=false, grid=false, framestyle=:grid,
        foreground_color=:black, background_color=:white,
        tick_direction=:in, minorgrid=false,
        xticks=(sp.ticks, paper_ticks), ylabel=raw"$E\; (\mathrm{meV})$",
        xlims=(first(sp.dist) - xpad, last(sp.dist) + xpad), ylims=(-250, 15),
        yticks=-250:50:0,
        left_margin=2Plots.mm, right_margin=2Plots.mm,
        bottom_margin=2Plots.mm, top_margin=2Plots.mm)
    
    # Shade between the two DFT energies in each pair, behind the curves and dots.
    for n in 1:WSE2_NB
        Plots.plot!(p1, xref, ref.E[n, :] .- zero_ref;
            fillrange=ref.Elo[n, :] .- zero_ref, fillalpha=0.55, color=:gray65,
            lw=0, label="")
    end
    
    Plots.plot!(p1, sp.dist, transpose(Efull .- zero_ref);
        color=:red, lw=0.55, ls=:solid, label="")
    Plots.plot!(p1, xwlm, transpose(Ewdense .- zero_ref);
        color="#0087CC", lw=0.55, ls=:dash, label="")
    
    # Draw both members of each DFT pair as small dots above the curves.
    Plots.scatter!(p1, repeat(xref, inner=WSE2_NB), vec(ref.E .- zero_ref);
        color=:black, markershape=:circle, markersize=1.2,#0.8,
        markerstrokewidth=0, label="")
    Plots.scatter!(p1, repeat(xref, inner=WSE2_NB), vec(ref.Elo .- zero_ref);
        color=:black, markershape=:circle, markersize=1.2,#0.8,
        markerstrokewidth=0, label="")

    # Highlight the two locations marked in the WSe2 comparison at kappa-minus.
    # These are annotation coordinates in the existing relative-energy scale.
    # Transparent interiors leave all bands and DFT points visible underneath.
    # A Shape gives GR separate fill/stroke opacity; its native circle does not.
    circle_angles = range(0, 2pi; length=97)
    highlight_circle = Plots.Shape(cos.(circle_angles), sin.(circle_angles))
    Plots.scatter!(p1, fill(sp.ticks[2], 2), [-207.0, -225.0];
        markershape=highlight_circle, markersize=6.5,
        markercolor=:white, markeralpha=0,
        markerstrokecolor="#8C4CB8", markerstrokealpha=1,
        markerstrokewidth=0.9, label="", primary=false)
    
    return p1
end

function fig3_display_basis(ctx)
    reciprocal = Matrix(ctx.bmT)
    e1 = reciprocal[:, 1]/norm(reciprocal[:, 1])
    rotation = transpose(hcat(e1, [-e1[2], e1[1]]))
    return rotation*reciprocal/norm(reciprocal*[1/3, 1/3])
end

function fig3_path_panel(ctx; energy_label_x=0.0)
    fig_font = 7  # Combined with thickness_scaling=6.74/7, matches 10pt REVTeX.
    B = fig3_display_basis(ctx)
    @assert norm(B*[1/3,1/3]) ≈ 1.0
    vertices = [[1/3,1/3], [-1/3,2/3], [-2/3,1/3],
                [-1/3,-1/3], [1/3,-2/3], [2/3,-1/3], [1/3,1/3]]
    hex = hcat([B*v for v in vertices]...)
    # The narrow left column holds the small BZ above the shared energy label.
    q = Plots.plot(; aspect_ratio=:equal, framestyle=:none, grid=false,
        legend=false, showaxis=false, ticks=nothing,
        xlims=(-1.65,1.8), ylims=(-7.5,1.65),
        left_margin=0Plots.mm, right_margin=-6Plots.mm,
        top_margin=0Plots.mm, bottom_margin=0Plots.mm)
    Plots.plot!(q, hex[1,:], hex[2,:]; color=:black, lw=0.65, label="")
    # Union of the two plotted paths: red dotted segments, no arrows.
    # Draw shared edges once so the dotted pattern stays uniform.
    nodes = [[0.,0.], [0.5,0.], [1/3,1/3], [-1/3,2/3]]
    for (i,j) in ((1,2), (2,3), (3,4), (4,1), (1,3))
        a, b = B*nodes[i], B*nodes[j]
        Plots.plot!(q, [a[1],b[1]], [a[2],b[2]];
            color=:red, ls=:dot, lw=0.7, label="")
    end
    labels = [raw"$\gamma$", raw"$m$", raw"$\kappa_{-}$", raw"$\kappa_{+}$"]
    for (i,k) in enumerate(nodes)
        r = B*k
        Plots.scatter!(q, [r[1]], [r[2]]; color=:black, markersize=1.7,
            markerstrokewidth=0, label="")
        pos = i == 1 ? r + [-0.32,-0.35] : 1.38*r
        Plots.annotate!(q, pos[1], pos[2], Plots.text(labels[i], fig_font, :black))
    end
    Plots.annotate!(q, energy_label_x, -3.6,
        Plots.text(raw"$E\;(\mathrm{meV})$", fig_font, :black; rotation=90))
    return q
end

function fig3_single_figure(panel, ctx; energy_ticks, trim_left=false, top_legend=false)
    Plots.plot!(panel; title="", legend=false, ylabel="",
        fontfamily="Computer Modern", tickfontsize=7, guidefontsize=7,
        yticks=(energy_ticks, ["\$$(v)\$" for v in energy_ticks]),
        left_margin=0Plots.mm, right_margin=1Plots.mm,
        top_margin=1Plots.mm, bottom_margin=0Plots.mm)
    fig3_box!(panel)
    schematic = fig3_path_panel(ctx; energy_label_x=0.4)
    # Trim about 4 mm from the WSe2 left edge, retaining the band-panel width.
    column_widths = trim_left ? [64, 216] : [76, 216]
    trim_left && Plots.plot!(schematic; xlims=(-1.0, 1.8))
    if top_legend
        # Reserve space above the axes for a legend anchored to the band box.
        Plots.plot!(panel;top_margin=9Plots.mm)
        Plots.plot!(schematic;top_margin=9Plots.mm)
    end
    figure = Plots.plot(schematic, panel;
        layout=Plots.grid(1, 2; widths=column_widths ./ sum(column_widths)),
        size=(sum(column_widths), top_legend ? 214 : 190), dpi=600, fontfamily="Computer Modern",
        thickness_scaling=6.74/7)
    top_legend && fig3_wse2_legend!(figure)
    return figure
end

function fig3_wse2_legend!(figure)
    # Relative to the band plotting area: centered and 98% of its width.
    # Bounds follow the border, preserving its vertical position and height.
    Plots.plot!(figure;inset_subplots=(2,Plots.bbox(0.01,-0.1704,0.98,0.1008)))
    panel = figure[3]
    Plots.plot!(panel;fontfamily="Computer Modern",framestyle=:none,
        axis=false,grid=false,legend=false,xlims=(0,200),ylims=(0.08,0.92),
        margin=0Plots.mm)
    fig3_legend_box!(panel)
    Plots.plot!(panel,[5,18],[0.5,0.5];color=:red,lw=0.8,label="")
    Plots.annotate!(panel,22,0.5,
        Plots.text("ERTCA",6,:black,:left;family="Computer Modern"))
    Plots.plot!(panel,[64,77],[0.5,0.5];color="#0087CC",lw=0.8,ls=:dash,label="")
    Plots.annotate!(panel,81,0.5,
        Plots.text(raw"$k\cdot p$",6,:black,:left;family="Computer Modern"))
    Plots.scatter!(panel,[109],[0.5];color=:black,markersize=1.2,
        markerstrokewidth=0,label="")
    Plots.annotate!(panel,117,0.5,
        Plots.text("Large-scale DFT",6,:black,:left;family="Computer Modern"))
    return figure
end

function fig3_wse2_figure(w)
    return fig3_single_figure(wse2_band_panel(w), w.ctx;
        energy_ticks=collect(-250:50:0), trim_left=true, top_legend=true)
end

function fig3_nbse2_figure(n; energy_window=(-170, 15), nbands::Int=NBANDS,
        figure_size=(236, 190), energy_tick_step=50, energy_label_x=-0.18,
        top_legend=false)
    panel = gamma_comparison_figure(n.ctx, n.path, n.Efull, n.Econstant, n.Eg0;
        include_bz=false, draw_frame=false, energy_window=energy_window, nbands=nbands,
        ticks=[raw"$\gamma$", raw"$m$", raw"$\kappa_{-}$", raw"$\kappa_{+}$", raw"$\gamma$"])
    # A compact band panel with its own energy label, without the hexagon column.
    # Keep ticks at round multiples when requesting a deeper energy window.
    energy_ticks = collect(energy_tick_step*ceil(Int,energy_window[1]/energy_tick_step):
                          energy_tick_step:energy_tick_step*floor(Int,energy_window[2]/energy_tick_step))
    Plots.plot!(panel;
        title="", legend=false, ylabel="",
        yticks=(energy_ticks, ["\$$(v)\$" for v in energy_ticks]),
        fontfamily="Computer Modern", tickfontsize=7, guidefontsize=7,
        size=(figure_size[1],figure_size[2]+(top_legend ? 24 : 0)),
        dpi=600, thickness_scaling=6.74/7,
        left_margin=2Plots.mm, right_margin=1Plots.mm,
        top_margin=(top_legend ? 9 : 1)Plots.mm, bottom_margin=0Plots.mm)
    # Leave a little breathing room beside the three-digit negative tick labels.
    Plots.annotate!(panel, [((energy_label_x, 0.5),
        Plots.text(raw"$E\;(\mathrm{meV})$", 7, :black; rotation=90))])
    fig3_box!(panel)
    top_legend && fig3_nbse2_legend!(panel)
    return panel
end

function fig3_nbse2_legend!(figure)
    # Align both legend edges with the band plotting area, independently of its margins.
    Plots.plot!(figure;inset_subplots=(1,Plots.bbox(0.0,-0.1704,1.0,0.1008)))
    panel = figure[2]
    Plots.plot!(panel;fontfamily="Computer Modern",framestyle=:none,
        axis=false,grid=false,legend=false,xlims=(0,200),ylims=(0.08,0.92),
        margin=0Plots.mm)
    fig3_legend_box!(panel)
    # Shorter samples leave larger gaps between entries without shrinking the text.
    Plots.plot!(panel,[4,12],[0.5,0.5];color=:red,lw=0.8,label="")
    Plots.annotate!(panel,16,0.5,
        Plots.text("ERTCA",6,:black,:left;family="Computer Modern"))
    # Draw two complete dashes so the short samples remain clearly dashed.
    Plots.plot!(panel,[61,64,NaN,66,69],fill(0.5,5);color="#0087CC",lw=0.8,label="")
    Plots.annotate!(panel,73,0.5,
        Plots.text(raw"$k\cdot p$",6,:black,:left;family="Computer Modern"))
    Plots.plot!(panel,[101,104,NaN,106,109],fill(0.5,5);color=:black,lw=0.8,label="")
    Plots.annotate!(panel,113,0.5,
        Plots.text(raw"$k\cdot p + W_{b,t}(\bar{k},\bar{G}_0)$",6,:black,:left;
            family="Computer Modern"))
    return figure
end

function save_fig3(fig; outdir=PAPER_FIGURES, stem="Fig3")
    mkpath(outdir)
    for ext in ("png", "pdf")
        dest = joinpath(outdir, stem * "." * ext)
        Plots.savefig(fig, dest)
        println("figure → ", dest)
    end
end
