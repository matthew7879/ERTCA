# NbSe2 DOS at three twist angles.
# Run from this repository: julia --project=. -t auto Fig5.jl

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "src", "common.jl"))
using Printf
using Serialization
import Plots

# -------------------------- Settings to change ---------------------------------
const TWIST_PAIRS = [(6,7), (5,6), (11,14)] # 5.09, 6.01, 7.93 degrees
# Other choices: (13,16) for 6.84 degrees, or (10,13) for 8.61 degrees.
const NK_MOIRE = 22             # NK_MOIRE^2 momenta in each moire Brillouin zone
const NK_REFERENCE = 300        # 4-band G=0 reference, sampled in the first Brillouin zone
const REGISTRY_SHELLS = 4       # same four-shell ERTCA model used by the paper
const HARMONIC_ATOL = 1e-6      # eV: existing paper Bloch-sum coefficient pruning
# No real-space hopping-radius cutoff is applied.
const REFERENCE_FERMI_EV = nothing # compute half filling of W(k,0); or supply an energy in eV
const SIGMA_EV = 0.012          # Gaussian STANDARD DEVIATION; identical for every curve
const ENERGY_STEP_EV = 0.001    # 1 meV grid; covers the whole sampled manifold
const BAND_POINTS = 160         # points per band-path segment
# Designed for figure* with \includegraphics[width=\textwidth]{...}:
# this landscape layout spans BOTH columns, with readable text at the printed size.
const FIGURE_SIZE = (900,340)   # shorter panels; width and font sizes stay fixed
const TICK_FONT_SIZE = 10
const AXIS_FONT_SIZE = 11
const DOS_AXIS_FONT_SIZE = 11   # one-line DOS labels
const LEGEND_FONT_SIZE = 10
const FEATURE_FONT_SIZE = 10
const ENERGY_LABEL_X = -0.19    # relative to the band panel; 0 is its left edge
# Purple van-Hove peaks, muted-red band energies, and a blue minimum-near-M guide.
# Switch each set on/off independently; these controls affect annotations only.
const SHOW_DOS_PEAK_GUIDES = true
const SHOW_BAND_EDGE_GUIDES = true
const SHOW_M_POINT_GUIDE = false # shallow lower-band minimum just after M toward Kprime
const SHOW_FERMI_GUIDE = false   # optional horizontal E=0 line in all three panels
const DOS_PEAK_GUIDE_COLOR = "#8955A5"
const BAND_EDGE_GUIDE_COLOR = "#B94F55"
const M_POINT_GUIDE_COLOR = "#244F87"
const REFERENCE_COLOR = "#000000" # matching reference bands, DOS outline, and legend line
const REFERENCE_FILL_COLOR = "#444444" # light-grey shading when drawn with alpha=0.12
const REFERENCE_LABEL = raw"Stacking-averaged bilayer ($\bar{G}_0$)"
const BAND_COLOR = REFERENCE_COLOR
# Find the two circled peaks of the BROADENED reference DOS in these energy windows.
# The nearby spin-split saddle energies are not exactly the broadened peak energies.
const DOS_PEAK_WINDOWS_MEV = ((-210.0,-130.0), (-120.0,-40.0))
# Match each angle by color in the DOS and delta-DOS panels; all curves are solid.
const COLORS = ["#244F87", "#D62728", "#228833", "#7C4D91", "#C56C20"]
const FIGURE_DIR = normpath(joinpath(@__DIR__, "Figures"))
const CACHE_DIR = normpath(joinpath(@__DIR__, ".cache", "ERTCA", "NbSe2_DOS_twist_angles"))
const USE_CACHE = true          # saved eigenvalues make later styling changes quick
const STEM = "Fig5"

# Cache identity includes the input archive and the package/loader source code.
# Increment the version if you change the spectral calculation in THIS file.
function input_signature()
    root = @__DIR__
    files = [NBSE2.archive]
    for (directory, _, names) in walkdir(joinpath(root, "src"))
        append!(files, [joinpath(directory,n) for n in names if endswith(n,".jl")])
    end
    # Compare exact input and source bytes.
    contents = [(relpath(file, root), read(file)) for file in sort(files)]
    return (version=2, julia=string(VERSION), contents=contents,
            shells=REGISTRY_SHELLS, atol=HARMONIC_ATOL)
end

function cached_calculation(calculate, name, key)
    path = joinpath(CACHE_DIR, name * ".jls")
    if USE_CACHE && isfile(path)
        saved = deserialize(path)
        if saved.key == key
            println("Using cached ", name)
            return saved.value
        end
    end
    value = calculate()
    mkpath(CACHE_DIR)
    temporary = path * ".$(getpid()).tmp"
    serialize(temporary, (; key, value))
    mv(temporary, path; force=true)
    return value
end

# Equal-weight midpoint mesh covering one reciprocal primitive cell.
# Each column contains ALL eigenvalues at one k point. Spin is already included.
function sample_spectrum(model, nk; slab=false)
    nbands = n_states(model.cell)
    energies = Matrix{Float64}(undef, nbands, nk^2)
    finished = Threads.Atomic{Int}(0)
    ERTCA.threaded_chunks(nk^2) do chunk
        H = zeros(ComplexF64, nbands, nbands)
        for index in chunk
            i, j = mod(index-1,nk), div(index-1,nk)
            kx, ky = (i+0.5)/nk, (j+0.5)/nk
            k = slab ? SVector(kx,ky,0.0) : SVector(kx,ky)
            bloch_hamiltonian!(H, model, k)
            if index == 1
                @assert maximum(abs, H-H') < 1e-9 "Hamiltonian is not Hermitian"
            end
            energies[:,index] = eigvals!(Hermitian(H))
            done = Threads.atomic_add!(finished,1) + 1
            if done % max(1,div(nk^2,8)) == 0 || done == nk^2
                @printf("  %d / %d momenta (%d bands)\n", done, nk^2, nbands)
                flush(stdout)
            end
        end
    end
    @assert all(isfinite, energies)
    return energies
end

# Zero-temperature half filling, found separately for EACH spectrum.
function half_filling_fermi(energies)
    levels = sort(vec(energies))
    @assert iseven(length(levels))
    middle = length(levels) ÷ 2
    return (levels[middle] + levels[middle+1])/2
end

# Standard normalized Gaussian: exp[-(E-e)^2/(2*sigma^2)]/(sqrt(2pi)*sigma).
# The integral is nbands/N = 4 states per primitive bilayer area.
function primitive_dos(energies, ef, energy_grid, N)
    density = zeros(length(energy_grid))
    step = energy_grid[2] - energy_grid[1]
    for level in energies
        e = level - ef
        first_index = max(1, ceil(Int,(e-6SIGMA_EV-energy_grid[1])/step)+1)
        last_index = min(length(density), floor(Int,(e+6SIGMA_EV-energy_grid[1])/step)+1)
        for index in first_index:last_index
            density[index] += exp(-0.5*((energy_grid[index]-e)/SIGMA_EV)^2)
        end
    end
    density ./= sqrt(2pi)*SIGMA_EV*size(energies,2)*N
    return density
end

# W(k, G=0) sampled in the centered, hexagonal first Brillouin zone.
# Its isolated interlayer harmonic need not be reciprocal-periodic, so evaluating
# it in the unfolded [0,1)^2 parallelogram can give a different reference DOS.
function sample_reference_spectrum(h, nk)
    W0 = average_bilayer(h)
    binv = inv(transpose(h.lattice.b))
    energies = Matrix{Float64}(undef,length(h.labels),nk^2)
    finished = Threads.Atomic{Int}(0)
    ERTCA.threaded_chunks(nk^2) do chunk
        for index in chunk
            i, j = mod(index-1,nk), div(index-1,nk)
            kbox = SVector((i+0.5)/nk,(j+0.5)/nk)
            k = binv * fold_to_bz(h.lattice,kbox)
            energies[:,index] = eigvals(W0(k))
            done = Threads.atomic_add!(finished,1) + 1
            if done % max(1,div(nk^2,8)) == 0 || done == nk^2
                @printf("  G=0 reference: %d / %d momenta\n",done,nk^2)
                flush(stdout)
            end
        end
    end
    @assert all(isfinite,energies)
    return energies
end

# Four bands of that SAME G=0 Hamiltonian, using the SAME energy zero as its DOS.
function reference_bands(h, ef)
    W0 = average_bilayer(h)
    B = transpose(h.lattice.b)
    nodes = [SVector(0.0,0.0), SVector(0.5,0.0),
             SVector(2/3,-1/3), SVector(0.0,0.0)]
    x = Float64[]
    kpoints = SVector{2,Float64}[]
    ticks = [0.0]
    for segment in 1:length(nodes)-1
        start, stop = nodes[segment], nodes[segment+1]
        distance = norm(B*(stop-start))
        for t in range(0.0,1.0; length=BAND_POINTS)[(segment==1 ? 1 : 2):end]
            push!(x, ticks[end]+t*distance)
            push!(kpoints, (1-t)*start+t*stop)
        end
        push!(ticks, ticks[end]+distance)
    end
    bands = zeros(length(kpoints),length(h.labels))
    for (index,k) in enumerate(kpoints)
        bands[index,:] = 1000 .* (eigvals(W0(k)) .- ef)
    end
    return (;x, bands, ticks, labels=["Γ","M","K′","Γ"])
end

# The reference DOS supplies the van-Hove PEAK positions. Each broadened peak
# combines nearby spin-split saddles, so its maximum is not an exact saddle energy.
# The band plot supplies the three high-symmetry energies and the Gamma-M band bottom.
function reference_guides(energy_grid, reference_dos, band)
    energy = 1000 .* energy_grid
    peaks = Float64[]
    for (low,high) in DOS_PEAK_WINDOWS_MEV
        candidates = [i for i in 2:length(energy)-1
                      if low <= energy[i] <= high &&
                         reference_dos[i] > reference_dos[i-1] &&
                         reference_dos[i] > reference_dos[i+1]]
        isempty(candidates) && error("No reference DOS peak in ($low,$high) meV; adjust DOS_PEAK_WINDOWS_MEV.")
        index = candidates[argmax(reference_dos[candidates])]
        push!(peaks,energy[index])
    end
    kprime_index = argmin(abs.(band.x .- band.ticks[3]))
    m_index = argmin(abs.(band.x .- band.ticks[2]))
    # Minimum of the lowest plotted band along Gamma-M (same shared energy zero).
    band_bottom = minimum(@view band.bands[1:m_index,1])
    # Shallow minimum just after M: use the lowest band on the M-Kprime segment.
    # This is slightly below its value exactly at M.
    m_minimum = minimum(@view band.bands[m_index:kprime_index,1])
    # Upper member of each Kprime pair (bands 2 and 4), then upper Gamma (band 4).
    # These are evaluated at the marked k points, not estimated from DOS shoulders.
    edges = [band_bottom, band.bands[kprime_index,2], band.bands[kprime_index,4], band.bands[1,4]]
    return (;peaks,edges,band_bottom,m_minimum)
end

function add_energy_guides!(panel, guides; m_label_gap=nothing)
    if SHOW_DOS_PEAK_GUIDES
        Plots.hline!(panel,guides.peaks;color=DOS_PEAK_GUIDE_COLOR,linealpha=0.65,
                     linestyle=:dash,linewidth=0.85,label="")
    end
    if SHOW_BAND_EDGE_GUIDES
        Plots.hline!(panel,guides.edges;color=BAND_EDGE_GUIDE_COLOR,linealpha=0.65,
                     linestyle=:dash,linewidth=0.85,label="")
    end
    if SHOW_M_POINT_GUIDE
        # Leave a short gap behind the vHS1 label in the DOS panel only.
        left, right = Plots.xlims(panel)
        x = isnothing(m_label_gap) ? [left,right] :
            [left,m_label_gap[1],NaN,m_label_gap[2],right]
        Plots.plot!(panel,x,fill(guides.m_minimum,length(x));
                    color=M_POINT_GUIDE_COLOR,linealpha=0.65,
                    linestyle=:dash,linewidth=0.85,label="")
    end
end

# One compact legend row across the full-width figure.
# These positions are in figure-pixel units; only the legend uses these axes.
function compact_legend(moire)
    labels = [REFERENCE_LABEL; [@sprintf("%.2f°",r.angle) for r in moire]]
    colors = [REFERENCE_COLOR; COLORS[1:length(moire)]]
    sample_width, text_gap, entry_gap = 44.0, 8.0, 18.0
    text_widths = [235.0; fill(44.0,length(moire))]
    box_width = sum(text_widths) + length(labels)*(sample_width+text_gap) +
                (length(labels)-1)*entry_gap + 24.0
    left = (FIGURE_SIZE[1]-box_width)/2
    right = left+box_width
    panel = Plots.plot(;fontfamily="Computer Modern",framestyle=:none,
        axis=false,grid=false,legend=false,xlims=(0,FIGURE_SIZE[1]),ylims=(0,1),
        margin=0Plots.mm)
    Plots.plot!(panel,[left,right,right,left,left],[0.08,0.08,0.94,0.94,0.08];
                color=:black,linewidth=0.8,label="")
    x = left+12.0
    for index in eachindex(labels)
        y = 0.5
        if index == 1
            patch = Plots.Shape([x,x+sample_width,x+sample_width,x], [y-0.13,y-0.13,y+0.13,y+0.13])
            Plots.plot!(panel,patch;fillcolor=REFERENCE_FILL_COLOR,fillalpha=0.12,linealpha=0,label="")
        end
        Plots.plot!(panel,[x,x+sample_width],[y,y];color=colors[index],linewidth=1.8,label="")
        Plots.annotate!(panel,x+sample_width+text_gap,y,
                        Plots.text(labels[index],LEGEND_FONT_SIZE,:black,:left;family="Computer Modern"))
        x += sample_width+text_gap+text_widths[index]+entry_gap
    end
    return panel
end

function plot_bands_dos(energy_grid, reference_dos, moire, band, guides; outdir=FIGURE_DIR)
    common = (fontfamily="Computer Modern", framestyle=:box, grid=false,
              foreground_color=:black, tick_direction=:in,
              tickfontsize=TICK_FONT_SIZE, guidefontsize=AXIS_FONT_SIZE,
              legendfontsize=LEGEND_FONT_SIZE)
    energy = 1000 .* energy_grid
    energy_limits = extrema(energy)
    energy_ticks = collect(300ceil(energy[1]/300):300:300floor(energy[end]/300))

    # Left: the angle-independent, registry-averaged reference bands.
    bands_panel = Plots.plot(;common...,legend=false,
        xlims=extrema(band.x),xticks=(band.ticks,band.labels),
        ylims=energy_limits,yticks=energy_ticks)
    # GR's automatic y-label leaves a fixed gap. Position it explicitly to save space.
    Plots.annotate!(bands_panel,(ENERGY_LABEL_X,0.5),
                    Plots.text(raw"$E\;(\mathrm{meV})$",AXIS_FONT_SIZE,:black,:center;
                               family="Computer Modern",rotation=90.0))
    add_energy_guides!(bands_panel,guides)
    Plots.plot!(bands_panel,band.x,band.bands;color=BAND_COLOR,linewidth=1.4,label="")

    # Middle: rotate the DOS coordinates, putting energy on the vertical axis.
    dos_max = maximum(vcat(reference_dos,[r.dos for r in moire]...))
    dos_panel = Plots.plot(;common...,legend=false,
        xlims=(0,1.16dos_max),xticks=[0,4,8],ylims=energy_limits,yticks=energy_ticks,
        xlabel=raw"$\mathrm{DOS}\;(\mathrm{eV}^{-1}A_c^{-1})$",xguidefontsize=DOS_AXIS_FONT_SIZE)
    # A polygon fills horizontally from DOS=0 to the grey reference curve.
    reference_fill = Plots.Shape(vcat(0.0,reference_dos,0.0),
                                 vcat(energy[1],energy,energy[end]))
    Plots.plot!(dos_panel,reference_fill;fillcolor=REFERENCE_FILL_COLOR,fillalpha=0.12,
                linealpha=0,label="")
    add_energy_guides!(dos_panel,guides;
                      m_label_gap=SHOW_DOS_PEAK_GUIDES ? (0.55,2.1) : nothing)
    Plots.plot!(dos_panel,reference_dos,energy;color=REFERENCE_COLOR,linewidth=1.8,
                label=REFERENCE_LABEL)
    for (index,r) in enumerate(moire)
        Plots.plot!(dos_panel,r.dos,energy;color=COLORS[index],linestyle=:solid,
                    linewidth=1.7,label=@sprintf("%.2f°",r.angle))
    end
    # Put the labels outside the pair of guides, so neither subscript touches a line.
    if SHOW_DOS_PEAK_GUIDES
        labels = [raw"$\mathrm{vHS}_1$",raw"$\mathrm{vHS}_2$"]
        for (peak,label,offset,alignment) in zip(guides.peaks,labels,(-6,16),(:top,:bottom))
            Plots.annotate!(dos_panel,0.65,peak+offset,
                            Plots.text(label,FEATURE_FONT_SIZE,DOS_PEAK_GUIDE_COLOR,:left,alignment;
                                       family="Computer Modern"))
        end
    end

    # Right: subtract the same reference DOS at each energy for all three angles.
    delta_max = maximum(abs,vcat([r.dos-reference_dos for r in moire]...))
    delta_limit = 1.12max(delta_max,1e-3)
    delta_panel = Plots.plot(;common...,legend=false,
        xlims=(-delta_limit,delta_limit),xticks=[-2,0,2],ylims=energy_limits,yticks=energy_ticks,
        xlabel=raw"$\Delta\mathrm{DOS}\;(\mathrm{eV}^{-1}A_c^{-1})$",xguidefontsize=DOS_AXIS_FONT_SIZE)
    Plots.vline!(delta_panel,[0.0];color=:black,linewidth=0.5,label="")
    add_energy_guides!(delta_panel,guides)
    for (index,r) in enumerate(moire)
        Plots.plot!(delta_panel,r.dos-reference_dos,energy;color=COLORS[index],
                    linestyle=:solid,linewidth=1.6,label="")
    end
    if SHOW_FERMI_GUIDE
        for panel in (bands_panel,dos_panel,delta_panel)
            Plots.hline!(panel,[0.0];color=:black,linestyle=:dash,linewidth=0.65,label="")
        end
    end
    # The open upper-left region avoids the bands, DOS curves, and energy guides.
    for (panel,label) in ((bands_panel,"a)"),(dos_panel,"b)"),(delta_panel,"c)"))
        Plots.annotate!(panel,(0.12,0.85),
            Plots.text(label,AXIS_FONT_SIZE,:black,:left,:center;family="Computer Modern"))
    end

    legend_panel = compact_legend(moire)
    layout = Plots.@layout [legend{0.10h}; [bands{0.38w} dos{0.32w} delta]]
    figure = Plots.plot(legend_panel,bands_panel,dos_panel,delta_panel;
        layout=layout,size=FIGURE_SIZE,dpi=300,
        left_margin=1Plots.mm,right_margin=0.5Plots.mm,top_margin=-5Plots.mm,bottom_margin=4Plots.mm)
    Plots.plot!(figure[1];left_margin=0Plots.mm,right_margin=0Plots.mm,
                top_margin=0Plots.mm,bottom_margin=0Plots.mm)
    Plots.plot!(figure[2];left_margin=6Plots.mm) # room for the positioned energy label
    # All three data panels use the same explicit energy limits and ticks.
    # Display energy tick labels only on the left; subplot 1 is the legend row.
    Plots.plot!(figure[3];yformatter=(_ -> ""),left_margin=-3Plots.mm)
    Plots.plot!(figure[4];yformatter=(_ -> ""),left_margin=-3Plots.mm)
    mkpath(outdir)
    for extension in ("pdf","png")
        path = joinpath(outdir,STEM*"."*extension)
        Plots.savefig(figure,path)
        println("Figure → ",path)
    end
    return figure
end

function main_nbse2_dos()
    @assert SIGMA_EV > 0 && ENERGY_STEP_EV > 0
    @assert length(TWIST_PAIRS) <= length(COLORS)
    @assert NBSE2.filling == 1//2
    println("Input: ",NBSE2.archive)
    println("Gaussian standard deviation: ",1000SIGMA_EV," meV")
    signature = input_signature()

    # 1. Fit the updated 144-registry archive once.
    h = cached_calculation("registry_harmonics",signature) do
        scan, warnings = quiet(() -> ertca_scan(NBSE2))
        @printf("Loaded %d registries (%d relaxed-centre drift warnings).\n",length(scan.d),warnings)
        registry_harmonics(scan;gshells=REGISTRY_SHELLS,atol=HARMONIC_ATOL)
    end

    # 2. Keep only W(k,G=0). This single four-band reference has no twist angle.
    reference_key = (signature,NK_REFERENCE,:first_bz_v2)
    reference_levels = cached_calculation("G0_firstBZ_nk$(NK_REFERENCE)",reference_key) do
        println("Sampling the G=0 reference in the first Brillouin zone")
        sample_reference_spectrum(h,NK_REFERENCE)
    end
    reference_half_ef = half_filling_fermi(reference_levels)
    reference_ef = isnothing(REFERENCE_FERMI_EV) ? reference_half_ef : REFERENCE_FERMI_EV
    reference_occupation = count(e -> e < reference_ef,reference_levels)/size(reference_levels,2)
    @printf("G=0: using E_F = %.8f eV; half-filling estimate = %.8f eV; occupation = %.6f / 4\n",
            reference_ef,reference_half_ef,reference_occupation)

    # 3. Build each FULL moire model. Their own Fermi levels are recorded, but
    # every curve uses the SAME G=0 Fermi energy as its plotting zero.
    moire = NamedTuple[]
    for mn in TWIST_PAIRS
        tb = twisted_bilayer(h.lattice,mn...)
        N = n_monolayer_cells(tb)
        angle = twist_angle(tb)
        @printf("\n(m,n) = (%d,%d): theta = %.6f degrees, N = %d, %d bands\n",mn...,angle,N,4N)
        levels = cached_calculation("moire_$(mn[1])_$(mn[2])_nk$(NK_MOIRE)",
                                    (signature,mn,NK_MOIRE)) do
            # Keep the library's commensurate folded/wrapped channels enabled.
            # Construction warnings are shown, not hidden.
            model = ertca_model(h,tb)
            @assert n_states(model.cell) == 4N
            sample_spectrum(model,NK_MOIRE)
        end
        ef = half_filling_fermi(levels)
        @printf("  Half-filling E_F = %.8f eV; E_F - E_F,G0 = %+.5f meV\n",ef,1000*(ef-reference_ef))
        push!(moire,(;mn,angle,N,levels,ef,reference_ef))
    end

    # 4. A shared energy grid and identical broadening make each subtraction
    # compare DOS values at the SAME model energy.
    low = min(minimum(reference_levels),minimum(minimum(r.levels) for r in moire))-reference_ef-6SIGMA_EV
    high = max(maximum(reference_levels),maximum(maximum(r.levels) for r in moire))-reference_ef+6SIGMA_EV
    energy_grid = range(floor(low/ENERGY_STEP_EV)*ENERGY_STEP_EV,
                        ceil(high/ENERGY_STEP_EV)*ENERGY_STEP_EV;step=ENERGY_STEP_EV)
    reference_dos = primitive_dos(reference_levels,reference_ef,energy_grid,1)
    moire = [merge(r,(dos=primitive_dos(r.levels,r.reference_ef,energy_grid,r.N),)) for r in moire]
    for (label,density) in vcat([("G=0",reference_dos)],[(string(r.angle),r.dos) for r in moire])
        area = sum(density)*ENERGY_STEP_EV
        @printf("Integral of DOS, %-10s: %.8f states/primitive cell (expected 4)\n",label,area)
        @assert isapprox(area,4.0;atol=1e-5)
    end
    for r in moire
        @assert abs(sum(r.dos-reference_dos)*ENERGY_STEP_EV) < 1e-5
    end

    # 5. Reference bands on the left, DOS in the middle, delta DOS on the right.
    band = reference_bands(h,reference_ef)
    guides = reference_guides(energy_grid,reference_dos,band)
    println("Broadened reference DOS peak guides (meV): ",guides.peaks)
    println("Lowest band minimum along Gamma-M (meV): ",guides.band_bottom)
    println("Blue guide: shallow lower-band minimum near M (meV): ",guides.m_minimum)
    figure = plot_bands_dos(energy_grid,reference_dos,moire,band,guides)
    mkpath(FIGURE_DIR)
    open(joinpath(FIGURE_DIR,STEM*"_parameters.txt"),"w") do io
        println(io,"Input: ",NBSE2.archive)
        println(io,"Normalization: 4 states and half filling = 2 electrons per primitive bilayer area.")
        println(io,"Displayed DOS units: eV^-1 A_c^-1, where A_c is the primitive in-plane unit-cell area; numerical values equal DOS per primitive cell.")
        println(io,"Registry shells: ",REGISTRY_SHELLS,"; coefficient pruning: ",HARMONIC_ATOL," eV; no hopping-radius cutoff.")
        println(io,"Gaussian standard deviation: ",SIGMA_EV," eV; NK_REFERENCE = ",NK_REFERENCE,"; NK_MOIRE = ",NK_MOIRE)
        println(io,"Reference: W(k,G=0), angle-independent registry average, with uniform interlayer coupling retained.")
        println(io,"Reference momentum domain: centered hexagonal first Brillouin zone.")
        println(io,"Energy zero: shared G=0 reference Fermi energy for ALL bands and DOS curves.")
        println(io,"Delta DOS: full twisted DOS per primitive cell minus the SAME G=0 reference DOS.")
        println(io,"Displayed reference name: ",REFERENCE_LABEL)
        println(io,"Horizontal E=0 guide enabled: ",SHOW_FERMI_GUIDE)
        println(io,"Purple dashed peak guides enabled: ",SHOW_DOS_PEAK_GUIDES,"; vHS1, vHS2 (meV): ",guides.peaks)
        println(io,"vHS1 and vHS2 mark the broadened van-Hove DOS peaks; each combines nearby spin-split saddles.")
        println(io,"Muted-red dashed band guides enabled: ",SHOW_BAND_EDGE_GUIDES,"; Gamma-M lowest band minimum, Kprime band 2, Kprime band 4, Gamma band 4 (meV): ",guides.edges)
        println(io,"Blue dashed M guide enabled: ",SHOW_M_POINT_GUIDE,"; shallow lower-band minimum on M-Kprime (meV): ",guides.m_minimum)
        @printf(io,"G=0 E_F used = %.10f eV; half-filling estimate = %.10f eV; occupation = %.8f / 4\n",
                reference_ef,reference_half_ef,reference_occupation)
        println(io,"m n angle_degrees N bands E_F_eV E_F_minus_G0_meV")
        for r in moire
            @printf(io,"%d %d %.8f %d %d %.10f %+.6f\n",r.mn...,r.angle,r.N,4r.N,r.ef,1000*(r.ef-reference_ef))
        end
        println(io,"\nG=0 reference: four band energies relative to its E_F, in meV")
        for (label,tick) in zip(band.labels,band.ticks)
            index = argmin(abs.(band.x .- tick))
            println(io,label,"  ",join([@sprintf("%.6f",e) for e in band.bands[index,:]],"  "))
        end
    end
    println("Finished. Change NK_MOIRE and SIGMA_EV at the top to check DOS convergence.")
    display(figure)
    return (;figure,energy_grid,reference_dos,reference_ef,reference_half_ef,moire,band,guides)
end

dos_results = main_nbse2_dos()
