# Supplementary S2: interlayer harmonic decay.
# Run from this repository: julia --project=. -t auto Fig10.jl

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "src", "common.jl"))
using Printf
using FFTW: ifft
import Plots

const NGRID = 12
const SPIN_CHANNEL = (β=1, βp=3)  # original imported bottom-up / top-up basis
const KPRIME = SVector(0,0)#(2 / 3, -1 / 3)
const HOPPING_RADIUS = Inf  # No cutoff: use all imported hoppings. Set e.g. 15.0 for 15 Å.
# Full 3D Wannier-centre distance |R + τ_j - τ_i|, including layer separation.
# Applied before the k-space sum; both plots and their reference dots use this cutoff.
const MAX_SHELLS = :all  # :all uses all 144 coefficients; set e.g. 6 for a shorter comparison
const OUTDIR = normpath(joinpath(@__DIR__, "Figures"))

function truncate_hoppings(model, radius)
    radius >= 0 || error("HOPPING_RADIUS must be nonnegative or Inf")
    radius == Inf && return model
    truncated = deepcopy(model)  # Keep the imported model intact.
    A = transpose(model.cell.lattice.a)
    for e in eachindex(truncated.couplings.w)
        # model.delta contains R + τ_j - τ_i in fractional coordinates.
        distance = norm(A * model.delta[e])  # Cartesian distance in Å
        if distance > radius
            truncated.couplings.w[e] = 0
        end
    end
    return truncated
end

function shortest_g_representatives(B, n)
    # -n:n covers the first reciprocal sampling cell for this triangular lattice.
    candidates = [SVector(p, q) for p in -n:n for q in -n:n]
    sort!(candidates; by=g -> (round(sum(abs2, B*g); digits=10), g[1], g[2]))
    seen = Set{Tuple{Int,Int}}()
    gs = SVector{2,Int}[]
    for g in candidates
        residue = (mod(g[1], n), mod(g[2], n))
        residue in seen && continue
        push!(seen, residue)
        push!(gs, g)
        length(gs) == n^2 && break
    end
    @assert length(gs) == n^2
    return gs
end


# Inverse sum at any fractional registry d=(d1,d2). There is NO 1/144 here.
# Pass only the G vectors and coefficients that belong to the desired shell cutoff.
function reconstruct_registry(d, gs, coefficients)
    value = 0.0 + 0.0im
    for (g, coefficient) in zip(gs, coefficients)
        phase = cispi(-2dot(g, d)) # exp(-i G⋅d), since a_l⋅b_m=2πδ_lm
        value += phase * coefficient
    end
    return value
end

function save_figure(fig, stem, outdir)
    for ext in ("pdf", "png")
        dest = joinpath(outdir, stem * "." * ext)
        Plots.savefig(fig, dest)
        println("figure → ", dest)
    end
end

# Shared |G|a ticks for the separate plots and the Γ/K′ comparison.
function harmonic_ticks()
    # Shell radii are (4π/√3)√(p²+pq+q²), with nonuniform spacing.
    # Include shells near 6π, 10π, and 14π (s = 7, 19, 37).
    values = [0.0, 4pi/sqrt(3), 4pi, 4pi*sqrt(7/3), 8pi,
              4pi*sqrt(19/3), 12pi, 4pi*sqrt(37/3), 16pi]
    labels = [raw"$0$", raw"$\frac{4\pi}{\sqrt{3}}$", raw"$4\pi$",
              raw"$\frac{4\pi\sqrt{7}}{\sqrt{3}}$", raw"$8\pi$",
              raw"$\frac{4\pi\sqrt{19}}{\sqrt{3}}$", raw"$12\pi$",
              raw"$\frac{4\pi\sqrt{37}}{\sqrt{3}}$", raw"$16\pi$"]
    return values, labels
end

function main_harmonic_decay(; kpoint=KPRIME, outdir=OUTDIR)
    kname, klabel = if isapprox(kpoint, SVector(0.0, 0.0); atol=1e-12)
        ("Gamma", raw"\Gamma")
    elseif isapprox(kpoint, SVector(2/3, -1/3); atol=1e-12)
        ("Kprime", raw"K^{\prime}")
    else
        error("Choose Γ=(0,0) or K′=(2/3,-1/3)")
    end
    stem = "NbSe2_harmonic_decay_$(kname)_FFT"
    println("Wannier hopping radius: ", HOPPING_RADIUS == Inf ? "off (all hoppings)" : "$(HOPPING_RADIUS) Å (3D centre distance)")
    # 1. Construct W_13(k,d), applying the optional hopping cutoff first.
    # Atomic gauge + true Wannier centres, exactly as in 2Interlayer_tunnelings.jl.
    scan, nwarn = quiet(() -> ertca_scan(NBSE2))
    @assert NBSE2.ngrid == NGRID && length(scan.d) == NGRID^2
    Wgrid = zeros(ComplexF64, NGRID, NGRID)
    occupied = falses(NGRID, NGRID)
    for s in eachindex(scan.d)
        i, j = round.(Int, NGRID * scan.d[s])
        @assert 0 <= i < NGRID && 0 <= j < NGRID
        @assert norm(scan.d[s] - SVector(i, j)/NGRID) < 1e-10
        @assert !occupied[i+1,j+1] "Repeated registry sample"
        ip = invperm(scan.permutations[s]) # reference orbital → this model's state
        model = truncate_hoppings(scan.models[s], HOPPING_RADIUS)
        W = bloch_hamiltonian(model, SVector(kpoint[1], kpoint[2], 0.0))
        Wgrid[i+1,j+1] = W[ip[SPIN_CHANNEL.β], ip[SPIN_CHANNEL.βp]]
        occupied[i+1,j+1] = true
    end
    @assert all(occupied) && all(isfinite, Wgrid)

    # 2. FFT of the COMPLEX registry samples. ifft uses the required positive
    # exponential and divides by 12×12; do not divide by 144 again.
    B = transpose(scan.lattice.b) # columns b1,b2 in Å⁻¹, already containing 2π
    gs = shortest_g_representatives(B, NGRID)
    # Wcoeff = direct_harmonics(Wgrid, gs)  # Original direct sum (uncomment its function too).
    Wfft_grid = ifft(Wgrid)  # transforms BOTH grid dimensions
    Wcoeff = [Wfft_grid[mod(g[1], NGRID)+1, mod(g[2], NGRID)+1] for g in gs]
    Gabs = [norm(B*g) for g in gs]
    a_mono = norm(primitive_vector(scan.lattice, 1)) # Å; 3.45 for this NbSe2 lattice
    Ga = a_mono .* Gabs                            # dimensionless

    # Group vectors into shells by their radius. For this triangular lattice,
    # |G|a = (4π/√3)√s, where s=p²+pq+q² takes values 0,1,3,4,7,9,12,... .
    # Using the integer s avoids splitting a shell through floating-point roundoff.
    shell_metric = [g[1]^2 + g[1]*g[2] + g[2]^2 for g in gs]
    shell_levels = sort(unique(shell_metric))
    shell_ids = [searchsortedfirst(shell_levels, s) - 1 for s in shell_metric]
    # There are 18 nonzero radius groups among our 144 chosen representatives.
    # The outer groups need not contain a complete reciprocal-lattice shell:
    # aliased vectors were removed when choosing one G per Fourier class.
    nshells = MAX_SHELLS === :all ? maximum(shell_ids) : MAX_SHELLS
    @assert isapprox(Ga, (4pi/sqrt(3)) .* sqrt.(shell_metric); atol=1e-10)
    @assert [count(==(s), shell_ids) for s in 0:6] == [1, 6, 6, 6, 12, 6, 6]
    @assert 1 <= nshells <= maximum(shell_ids)

    # 3. Reconstruct the 144 samples to check the FFT sign, indexing and normalization.
    Wback = [reconstruct_registry(SVector(i/NGRID, j/NGRID), gs, Wcoeff)
             for i in 0:NGRID-1, j in 0:NGRID-1]
    reconstruction_error = maximum(abs, Wback - Wgrid)
    @assert reconstruction_error < 1e-10
    @assert isapprox(sum(abs2, Wcoeff), sum(abs2, Wgrid)/NGRID^2; rtol=1e-10, atol=1e-20)
    @printf("%d shifts → %d FFT coefficients at %s=(%.6g,%.6g), channel (%d,%d)\n",
            length(Wgrid), length(gs), kname, kpoint[1], kpoint[2], SPIN_CHANNEL.β, SPIN_CHANNEL.βp)
    @printf("Inverse reconstruction: max error %.3e eV\n", reconstruction_error)
    @printf("Import: %d centre-drift warnings (relaxed stackings)\n", nwarn)
    @printf("Horizontal axis: |G|*a, monolayer a=%.6f Å\n", a_mono)
    @printf("G=0: W=%.9f%+.9fi meV, |W|=%.9f meV\n",
            real(1000Wcoeff[1]), imag(1000Wcoeff[1]), 1000abs(Wcoeff[1]))

    return (;gs, Ga, Wcoeff)
end

# 8. Compare the SAME interlayer component at Γ and K′ on one logarithmic axis.
# Each dot is one G vector; coefficients are not averaged over a shell.
function plot_harmonic_comparison(gamma, kprime; outdir=OUTDIR)
    @assert gamma.gs == kprime.gs
    @assert isapprox(gamma.Ga, kprime.Ga; atol=1e-10)
    radii = round.(gamma.Ga; digits=10)
    gamma_meV = 1000abs.(gamma.Wcoeff)
    kprime_meV = 1000abs.(kprime.Wcoeff)
    positive_values = filter(>(0), vcat(gamma_meV, kprime_meV))
    isempty(positive_values) && error("All coefficients are zero; cannot plot a log scale")
    xlimits = (-1.5, maximum(radii)+1.5)
    ylimits = (0.7minimum(positive_values), 1.8maximum(positive_values))
    log_ticks = 10.0 .^ (ceil(Int, log10(ylimits[1])):floor(Int, log10(ylimits[2])))
    gamma_color = "#0072B2"   # muted blue
    kprime_color = "#D55E00"  # vermilion

    fig = Plots.plot(;
        xlabel=raw"$|\!\bar{G}\,|a$",  # Balance GR's spacing inside the absolute-value bars.
        ylabel=raw"$|W_{b\uparrow,t\uparrow}(\bar{k},\bar{G})|\;(\mathrm{meV})$",
        yscale=:log10, xlims=xlimits, ylims=ylimits,
        xticks=harmonic_ticks(), yticks=log_ticks,
        size=(350, 225), dpi=600, fontfamily="Computer Modern",
        tickfontsize=7, guidefontsize=7, legendfontsize=7,
        thickness_scaling=6.74/7, legend=:topright,
        foreground_color_legend=nothing, background_color_legend=nothing,
        grid=false, framestyle=:axes, tick_direction=:in,
        left_margin=2Plots.mm, bottom_margin=1Plots.mm,
        right_margin=1Plots.mm, top_margin=1Plots.mm)
    for (magnitudes, label) in ((gamma_meV, raw"$\Gamma$"),
                               (kprime_meV, raw"$K^{\prime}$"))
        keep = magnitudes .> 0  # Exact zeros cannot be shown on a logarithmic axis.
        count(!, keep) > 0 && @info "Exact zeros omitted from comparison" label count=count(!, keep)
    end
    # Fourth nonzero shell: p² + pq + q² = 7. Draw behind the markers.
    Plots.vline!(fig, [4pi*sqrt(7/3)]; color=:grey70, ls=:dash, lw=0.6,
        label="", primary=false)
    Plots.hline!(fig, [0.1]; color=:black, ls=:dot, lw=0.55,
        label="", primary=false)  # 0.1 meV reference, behind the markers.
    # Hollow Γ circles remain visible around smaller filled K′ diamonds.
    # Both series keep their true |G|a coordinates, with no horizontal offsets.
    gamma_keep, kprime_keep = gamma_meV .> 0, kprime_meV .> 0
    # GR applies markeralpha to both the circle's fill and outline.
    Plots.scatter!(fig, radii[gamma_keep], gamma_meV[gamma_keep];
        label=raw"$\Gamma$", color=gamma_color, marker=:circle, markersize=3.5,
        markercolor=:white, markeralpha=0.8, markerstrokecolor=gamma_color,
        markerstrokealpha=0.8, markerstrokewidth=2.0)
    Plots.scatter!(fig, radii[kprime_keep], kprime_meV[kprime_keep];
        label=raw"$K^{\prime}$", color=kprime_color, marker=:diamond, markersize=2.3,
        markerstrokewidth=0)
    Plots.plot!(fig, [xlimits[1], xlimits[2], xlimits[2]],
        [ylimits[2], ylimits[2], ylimits[1]];
        color=:black, lw=2, label="", primary=false)
    mkpath(outdir)
    save_figure(fig, "Fig10", outdir)
    return fig
end

function main_harmonic_comparison(; outdir=OUTDIR)
    gamma = main_harmonic_decay(; kpoint=SVector(0.0, 0.0), outdir)
    kprime = main_harmonic_decay(; kpoint=SVector(2/3, -1/3), outdir)
    fig = plot_harmonic_comparison(gamma, kprime; outdir)
    return (; gamma, kprime, fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_harmonic_comparison()
end
