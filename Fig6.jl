# NbSe2 DOS versus large-scale DFT at 9.43 degrees.
# Run from this repository: julia --project=. -t auto Fig6.jl

module NbSe2DOS943

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "src", "common.jl"))
using Printf
using Serialization
import Plots

# -------------------------- Settings to change ---------------------------------
const TWIST_PAIR = (3, 4)       # theta = 9.4300079 degrees; N = 37; 148 bands
const NK_MOIRE = 22
const NK_REFERENCE = 300
const REGISTRY_SHELLS = 4
const HARMONIC_ATOL = 1e-6      # eV; no real-space hopping-radius cutoff
const REFERENCE_FERMI_EV = nothing # nothing computes half filling of W(k,0)
const SIGMA_EV = 0.012          # Gaussian standard deviation for the two model curves
const ENERGY_STEP_EV = 0.001
const BAND_POINTS = 160         # reference-path sampling for energy guides only
const FIGURE_SIZE = (450, 350) # single-column layout, with a separate legend above
const ENERGY_MIN_MEV = -500.0   # displayed lower energy limit
const SHOW_DOS_PEAK_GUIDES = true
const SHOW_BAND_EDGE_GUIDES = false
const SHOW_M_POINT_GUIDE = false
const SHOW_FERMI_GUIDE = true
const DOS_PEAK_WINDOWS_MEV = ((-210.0, -130.0), (-120.0, -40.0))
const DOS_PEAK_GUIDE_COLOR = "#8955A5"
const BAND_EDGE_GUIDE_COLOR = "#B94F55"
const M_POINT_GUIDE_COLOR = "#244F87"
const TWIST_COLOR = "#244F87"
const DFT_COLOR = "#D62728"
const DFT_LABEL = "Large-scale DFT"
const DFT_FILE = normpath(joinpath(@__DIR__, "Inputs", "NbSe2_9p43deg_DFT_DOS_gauss15meV.dat"))
const ALIGN_DFT_FERMI = true    # align DFT E_F to ERTCA E_F; false keeps tabulated x values
const REFERENCE_LABEL = raw"Stacking-averaged bilayer ($\bar{G}_0$)"
const FIGURE_DIR = normpath(joinpath(@__DIR__, "Figures"))
const STEM = "Fig6"
# Reuse compatible spectra from the original three-panel script.
const CACHE_DIR = normpath(joinpath(@__DIR__, ".cache", "ERTCA", "NbSe2_DOS_twist_angles"))
const USE_CACHE = true

# Same spectral calculation/cache identity as NbSe2_DOS_twist_angles.jl.
# Increment version if the sampling or Hamiltonian calculation changes.
function input_signature()
    root = @__DIR__
    files = [NBSE2.archive]
    for (directory, _, names) in walkdir(joinpath(root, "src"))
        append!(files, [joinpath(directory, n) for n in names if endswith(n, ".jl")])
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

# Equal-weight midpoint mesh over one moire reciprocal primitive cell.
function sample_spectrum(model, nk)
    nbands = n_states(model.cell)
    energies = Matrix{Float64}(undef, nbands, nk^2)
    finished = Threads.Atomic{Int}(0)
    ERTCA.threaded_chunks(nk^2) do chunk
        H = zeros(ComplexF64, nbands, nbands)
        for index in chunk
            i, j = mod(index - 1, nk), div(index - 1, nk)
            k = SVector((i + 0.5) / nk, (j + 0.5) / nk)
            bloch_hamiltonian!(H, model, k)
            if index == 1
                @assert maximum(abs, H - H') < 1e-9 "Hamiltonian is not Hermitian"
            end
            energies[:, index] = eigvals!(Hermitian(H))
            done = Threads.atomic_add!(finished, 1) + 1
            if done % max(1, div(nk^2, 8)) == 0 || done == nk^2
                @printf("  %d / %d momenta (%d bands)\n", done, nk^2, nbands)
                flush(stdout)
            end
        end
    end
    @assert all(isfinite, energies)
    return energies
end

# The isolated G=0 interlayer harmonic need not be reciprocal-periodic:
# sample it in the centered hexagonal first BZ, as in the original figure.
function sample_reference_spectrum(h, nk)
    W0 = average_bilayer(h)
    binv = inv(transpose(h.lattice.b))
    energies = Matrix{Float64}(undef, length(h.labels), nk^2)
    finished = Threads.Atomic{Int}(0)
    ERTCA.threaded_chunks(nk^2) do chunk
        for index in chunk
            i, j = mod(index - 1, nk), div(index - 1, nk)
            kbox = SVector((i + 0.5) / nk, (j + 0.5) / nk)
            k = binv * fold_to_bz(h.lattice, kbox)
            energies[:, index] = eigvals(W0(k))
            done = Threads.atomic_add!(finished, 1) + 1
            if done % max(1, div(nk^2, 8)) == 0 || done == nk^2
                @printf("  G=0 reference: %d / %d momenta\n", done, nk^2)
                flush(stdout)
            end
        end
    end
    @assert all(isfinite, energies)
    return energies
end

function half_filling_fermi(energies)
    levels = sort(vec(energies))
    @assert iseven(length(levels))
    middle = length(levels) ÷ 2
    return (levels[middle] + levels[middle + 1]) / 2
end

# Normalized Gaussian broadening; keep eV^-1 units even when x is shown in meV.
function primitive_dos(energies, ef, energy_grid, N)
    density = zeros(length(energy_grid))
    step = energy_grid[2] - energy_grid[1]
    for level in energies
        e = level - ef
        first_index = max(1, ceil(Int, (e - 6SIGMA_EV - energy_grid[1]) / step) + 1)
        last_index = min(length(density), floor(Int, (e + 6SIGMA_EV - energy_grid[1]) / step) + 1)
        for index in first_index:last_index
            density[index] += exp(-0.5 * ((energy_grid[index] - e) / SIGMA_EV)^2)
        end
    end
    density ./= sqrt(2pi) * SIGMA_EV * size(energies, 2) * N
    return density
end

# Input columns: E-E_F in meV, DOS in states/eV/primitive bilayer cell.
# The file is already divided by 37; do not divide again or normalize to four
# states, since the large-scale DFT DOS can include bands outside our manifold.
function load_dft_dos(path=DFT_FILE)
    energy_meV, density = Float64[], Float64[]
    headers = String[]
    for (line_number, line) in enumerate(eachline(path))
        line = strip(line)
        isempty(line) && continue
        if startswith(line, "#")
            push!(headers, line)
            continue
        end
        fields = split(line)
        length(fields) == 2 || error("$path:$line_number: expected energy and DOS columns")
        e, rho = tryparse.(Float64, fields)
        (isnothing(e) || isnothing(rho)) && error("$path:$line_number: invalid numeric value")
        (isfinite(e) && isfinite(rho) && rho >= 0) ||
            error("$path:$line_number: energy and DOS must be finite, with nonnegative DOS")
        push!(energy_meV, e)
        push!(density, rho)
    end
    length(energy_meV) >= 2 || error("$path: expected at least two DOS samples")
    all(>(0), diff(energy_meV)) || error("$path: energies must be strictly increasing")
    integral = sum(diff(energy_meV) .* (density[1:end-1] .+ density[2:end]) ./ 2) / 1000
    return (; path, energy_meV, dos=density, headers, integral)
end

# Same reference features as the original middle panel, now vertical guides.
# Band-path energies are computed only for these guides; no band panel is plotted.
function reference_guides(h, ef, energy_grid, reference_dos)
    energy = 1000 .* energy_grid
    peaks = Float64[]
    if SHOW_DOS_PEAK_GUIDES
        for (low, high) in DOS_PEAK_WINDOWS_MEV
            candidates = [i for i in 2:length(energy)-1
                          if low <= energy[i] <= high &&
                             reference_dos[i] > reference_dos[i-1] &&
                             reference_dos[i] > reference_dos[i+1]]
            isempty(candidates) && error("No reference DOS peak in ($low,$high) meV; adjust DOS_PEAK_WINDOWS_MEV.")
            push!(peaks, energy[candidates[argmax(reference_dos[candidates])]])
        end
    end
    W0 = average_bilayer(h)
    gamma, m, kp = SVector(0.0, 0.0), SVector(0.5, 0.0), SVector(2/3, -1/3)
    band_energy(k) = 1000 .* (eigvals(W0(k)) .- ef)
    ts = range(0.0, 1.0; length=BAND_POINTS)
    band_bottom = minimum(first(band_energy((1-t)*gamma + t*m)) for t in ts)
    m_minimum = minimum(first(band_energy((1-t)*m + t*kp)) for t in ts)
    kp_energies = band_energy(kp)
    edges = [band_bottom, kp_energies[2], kp_energies[4], band_energy(gamma)[4]]
    return (; peaks, edges, m_minimum)
end

# A boxed two-row legend stays legible at single-column width.
function dos_legend()
    panel = Plots.plot(;fontfamily="Computer Modern",framestyle=:none,
        axis=false,grid=false,legend=false,xlims=(0,450),ylims=(0,1),
        margin=0Plots.mm)
    Plots.plot!(panel,[22,428,428,22,22],[0.06,0.06,0.94,0.94,0.06];
        color=:black,linewidth=0.7,label="")
    # Align the line samples and text starts in the first column across both rows.
    for (x,y,color,label) in ((38.0,0.71,:black,REFERENCE_LABEL),
                            (38.0,0.30,TWIST_COLOR,"ERTCA"),
                            (224.0,0.30,DFT_COLOR,DFT_LABEL))
        if color === :black
            patch = Plots.Shape([x,x+32,x+32,x],[y-0.09,y-0.09,y+0.09,y+0.09])
            Plots.plot!(panel,patch;fillcolor="#444444",fillalpha=0.12,linealpha=0,label="")
        end
        Plots.plot!(panel,[x,x+32],[y,y];color,linewidth=1.8,label="")
        Plots.annotate!(panel,x+40,y,
            Plots.text(label,11,:black,:left;family="Computer Modern"))
    end
    panel
end

function plot_dos(energy_grid, reference_dos, moire_dos, guides, dft; outdir=FIGURE_DIR)
    energy = 1000 .* energy_grid
    dos_max = max(maximum(reference_dos), maximum(moire_dos), maximum(dft.dos))
    energy_limits = (ENERGY_MIN_MEV,
                     10ceil(max(last(energy), last(dft.plot_energy_meV))/10))
    energy_ticks = collect(300ceil(energy_limits[1]/300):300:300floor(energy_limits[2]/300))
    figure = Plots.plot(energy, reference_dos;
        fontfamily="Computer Modern", framestyle=:box, grid=false,
        foreground_color=:black, tick_direction=:in,
        tickfontsize=11, guidefontsize=12, legendfontsize=11,
        xlabel=raw"$E\;(\mathrm{meV})$",
        ylabel=raw"$\mathrm{DOS}\;(\mathrm{eV}^{-1}A_c^{-1})$",
        xlims=energy_limits, xticks=energy_ticks, ylims=(0, 1.16dos_max),
        color=:black, linewidth=1.8, fillrange=0,
        fillcolor="#444444", fillalpha=0.12, label=REFERENCE_LABEL,
        legend=false, background_color_legend=:white,
        size=FIGURE_SIZE, dpi=300,
        left_margin=3Plots.mm, right_margin=1Plots.mm,
        top_margin=1Plots.mm, bottom_margin=2Plots.mm)
    if SHOW_DOS_PEAK_GUIDES
        # Offset labels slightly apart, clear of each other and the E=0 line.
        # Leave a small gap in each dashed guide behind its label.
        ys = [0.0, 0.99dos_max, NaN, 1.11dos_max, 1.16dos_max]
        for (peak, label, offset) in
            zip(guides.peaks, [raw"$\mathrm{vHS}_1$", raw"$\mathrm{vHS}_2$"], (-40.0, 40.0))
            Plots.plot!(figure, fill(peak, length(ys)), ys; color=DOS_PEAK_GUIDE_COLOR,
                        linealpha=0.65, linestyle=:dash, linewidth=0.85, label="")
            Plots.annotate!(figure, peak + offset, 1.05dos_max,
                Plots.text(label, 9, DOS_PEAK_GUIDE_COLOR, :center; family="Computer Modern"))
        end
    end
    if SHOW_BAND_EDGE_GUIDES
        Plots.vline!(figure, guides.edges; color=BAND_EDGE_GUIDE_COLOR,
                     linealpha=0.65, linestyle=:dash, linewidth=0.85, label="")
    end
    if SHOW_M_POINT_GUIDE
        # Leave a gap behind the vHS1 label, as in the original middle panel.
        ys = SHOW_DOS_PEAK_GUIDES ? [0.0, 0.99dos_max, NaN, 1.11dos_max, 1.16dos_max] :
                                   [0.0, 1.16dos_max]
        Plots.plot!(figure, fill(guides.m_minimum, length(ys)), ys;
                    color=M_POINT_GUIDE_COLOR, linealpha=0.65,
                    linestyle=:dash, linewidth=0.85, label="")
    end
    if SHOW_FERMI_GUIDE
        Plots.vline!(figure, [0.0]; color=:black, linestyle=:solid, linewidth=0.65, label="")
    end
    Plots.plot!(figure, energy, moire_dos; color=TWIST_COLOR, linewidth=1.8,
                label="ERTCA")
    Plots.plot!(figure, dft.plot_energy_meV, dft.dos; color=DFT_COLOR,
                linestyle=:solid, linewidth=1.8, label=DFT_LABEL)
    figure = Plots.plot(dos_legend(),figure;
        layout=Plots.grid(2,1;heights=[0.18,0.82]),size=FIGURE_SIZE,dpi=300)
    Plots.plot!(figure[1];top_margin=0Plots.mm,bottom_margin=-4Plots.mm)
    Plots.plot!(figure[2];top_margin=0Plots.mm)
    mkpath(outdir)
    for extension in ("pdf", "png")
        path = joinpath(outdir, STEM * "." * extension)
        Plots.savefig(figure, path)
        println("Figure → ", path)
    end
    return figure
end

function main_nbse2_dos_9p43()
    @assert SIGMA_EV > 0 && ENERGY_STEP_EV > 0
    @assert NK_MOIRE > 0 && NK_REFERENCE > 0
    @assert NBSE2.filling == 1//2
    dft_raw = load_dft_dos()
    println("Input: ", NBSE2.archive)
    println("Gaussian standard deviation: ", 1000SIGMA_EV, " meV")
    println("DFT input: ", dft_raw.path)
    signature = input_signature()
    h = cached_calculation("registry_harmonics", signature) do
        scan, warnings = quiet(() -> ertca_scan(NBSE2))
        @printf("Loaded %d registries (%d relaxed-centre drift warnings).\n", length(scan.d), warnings)
        registry_harmonics(scan; gshells=REGISTRY_SHELLS, atol=HARMONIC_ATOL)
    end
    reference_key = (signature, NK_REFERENCE, :first_bz_v2)
    reference_levels = cached_calculation("G0_firstBZ_nk$(NK_REFERENCE)", reference_key) do
        println("Sampling the G=0 reference in the first Brillouin zone")
        sample_reference_spectrum(h, NK_REFERENCE)
    end
    reference_half_ef = half_filling_fermi(reference_levels)
    reference_ef = isnothing(REFERENCE_FERMI_EV) ? reference_half_ef : REFERENCE_FERMI_EV
    @printf("G=0 E_F used = %.8f eV; half-filling estimate = %.8f eV\n", reference_ef, reference_half_ef)

    tb = twisted_bilayer(h.lattice, TWIST_PAIR...)
    N, angle = n_monolayer_cells(tb), twist_angle(tb)
    @printf("(m,n) = (%d,%d): theta = %.8f degrees, N = %d, %d bands\n", TWIST_PAIR..., angle, N, 4N)
    levels = cached_calculation("moire_$(TWIST_PAIR[1])_$(TWIST_PAIR[2])_nk$(NK_MOIRE)",
                                (signature, TWIST_PAIR, NK_MOIRE)) do
        # Full-BZ tiling, with the default folded/wrapped channels enabled.
        model = ertca_model(h, tb)
        @assert n_states(model.cell) == 4N
        sample_spectrum(model, NK_MOIRE)
    end
    ef = half_filling_fermi(levels)
    @printf("Twisted E_F = %.8f eV; E_F - E_F,G0 = %+.5f meV\n", ef, 1000(ef-reference_ef))
    # E_DFT is already relative to its own E_F. On the current plot the ERTCA
    # Fermi level lies at 1000*(ef-reference_ef), so place DFT E_F there too.
    dft_shift_meV = ALIGN_DFT_FERMI ? 1000(ef-reference_ef) : 0.0
    dft = merge(dft_raw, (plot_energy_meV=dft_raw.energy_meV .+ dft_shift_meV,
                          energy_shift_meV=dft_shift_meV))
    @printf("DFT: %d samples; energy shift = %+.5f meV; supplied-window integral = %.8f states/primitive cell\n",
            length(dft.energy_meV), dft_shift_meV, dft.integral)

    low = min(minimum(reference_levels), minimum(levels)) - reference_ef - 6SIGMA_EV
    high = max(maximum(reference_levels), maximum(levels)) - reference_ef + 6SIGMA_EV
    energy_grid = range(floor(low/ENERGY_STEP_EV)*ENERGY_STEP_EV,
                        ceil(high/ENERGY_STEP_EV)*ENERGY_STEP_EV; step=ENERGY_STEP_EV)
    reference_dos = primitive_dos(reference_levels, reference_ef, energy_grid, 1)
    moire_dos = primitive_dos(levels, reference_ef, energy_grid, N)
    for (label, density) in (("G=0", reference_dos), (@sprintf("%.2f degrees", angle), moire_dos))
        area = sum(density) * ENERGY_STEP_EV
        @printf("Integral of DOS, %-12s: %.8f states/primitive cell (expected 4)\n", label, area)
        @assert all(isfinite, density) && all(>=(0), density)
        @assert isapprox(area, 4.0; atol=1e-5)
    end
    guides = reference_guides(h, reference_ef, energy_grid, reference_dos)
    figure = plot_dos(energy_grid, reference_dos, moire_dos, guides, dft)
    open(joinpath(FIGURE_DIR, STEM * "_parameters.txt"), "w") do io
        println(io, "Input: ", NBSE2.archive)
        println(io, "Model: full commensurate ERTCA, both layers and spins, default folded/wrapped channels.")
        println(io, "Normalization: 4 states per primitive in-plane area A_c; moire DOS divided by N.")
        println(io, "x axis: E in meV relative to shared G=0 reference E_F; y axis: DOS in eV^-1 A_c^-1.")
        println(io, "Displayed lower energy limit: ", ENERGY_MIN_MEV, " meV.")
        println(io, "Reference: W(k,G=0) sampled in the centered hexagonal first Brillouin zone.")
        println(io, "Registry shells: ", REGISTRY_SHELLS, "; coefficient pruning: ", HARMONIC_ATOL, " eV; no hopping-radius cutoff.")
        println(io, "Gaussian standard deviation: ", SIGMA_EV, " eV; energy step: ", ENERGY_STEP_EV, " eV.")
        println(io, "DFT input: ", dft.path)
        println(io, "DFT columns: E-E_F in meV; DOS already in states/eV/primitive bilayer cell.")
        println(io, "DFT DOS used as supplied: no rescaling, no additional broadening, no four-state normalization.")
        println(io, "DFT broadening: 15 meV Gaussian as reported in input header; model standard deviation remains ", 1000SIGMA_EV, " meV.")
        println(io, "DFT Fermi level aligned to twisted ERTCA Fermi level: ", ALIGN_DFT_FERMI)
        println(io, "DFT plotted energy = tabulated (E-E_F) + ", dft.energy_shift_meV, " meV.")
        println(io, "DFT supplied-window integral: ", dft.integral, " states/primitive cell.")
        println(io, "DFT source header:")
        foreach(line -> println(io, "  ", line), dft.headers)
        println(io, "NK_REFERENCE = ", NK_REFERENCE, "; NK_MOIRE = ", NK_MOIRE)
        @printf(io, "G=0 E_F used = %.10f eV; half-filling estimate = %.10f eV\n", reference_ef, reference_half_ef)
        println(io, "m n angle_degrees N bands E_F_eV E_F_minus_G0_meV")
        @printf(io, "%d %d %.8f %d %d %.10f %+.6f\n", TWIST_PAIR..., angle, N, 4N, ef, 1000(ef-reference_ef))
        println(io, "Purple peak guides enabled: ", SHOW_DOS_PEAK_GUIDES, "; energies (meV): ", guides.peaks)
        println(io, "Muted-red band guides enabled: ", SHOW_BAND_EDGE_GUIDES, "; energies (meV): ", guides.edges)
        println(io, "Blue minimum-near-M guide enabled: ", SHOW_M_POINT_GUIDE, "; energy (meV): ", guides.m_minimum)
        println(io, "Vertical E=0 guide enabled: ", SHOW_FERMI_GUIDE)
    end
    isinteractive() && display(figure)
    return (; figure, energy_grid, reference_dos, moire_dos, reference_ef, reference_half_ef,
            angle, N, ef, guides, dft)
end

end # module NbSe2DOS943

dos_9p43_results = NbSe2DOS943.main_nbse2_dos_9p43()
