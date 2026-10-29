# NbSe2 momentum-resolved spectral weight.
# Run from this repository: julia --project=. -t auto Fig7.jl

module NbSe2SpectralFunction

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "src", "common.jl"))
using SHA
using Serialization, DelimitedFiles
import Plots

const TWIST_PAIR = (6,7)        # 5.08584781 degrees, 127 monolayer areas per moire cell
const NK = 22                  # NK^2 moire momenta
const NK_REFERENCE = 300       # exactly the reference mesh in the DOS figure
const REGISTRY_SHELLS = 4
const HARMONIC_ATOL = 1e-6      # same Bloch-sum coefficient pruning as the DOS figure
const SIG = 0.006              # eV: spectral window exp[-((E-E*)/SIG)^2]
const DOS_SIGMA_EV = 0.012     # eV: STANDARD DEVIATION for selecting the DOS peak
const ENERGY_STEP_EV = 0.001   # same 1 meV energy grid as the DOS figure
const EWIN = (-0.140,-0.060)   # locate vHS2 of the reference in this relative-energy window
const SLICE_ENERGY_MEV = nothing # or specify an energy relative to reference EF, e.g. -81.0
const LIM = 1.30               # axes half-width in inverse Angstrom
const STEM = "Fig7"
const DEFAULT_PALETTE = :Blues
const FIGURE_DIR = normpath(joinpath(@__DIR__, "Figures"))
# Read the raw archive afresh, including its Wannier90 deck. The explicit root
# bypasses the loader's persistent extracted-deck cache. Temporary files are
# removed automatically after the harmonic construction, including on errors.
function fresh_harmonics()
    isfile(NBSE2.archive) || error("Required input archive is missing: $(NBSE2.archive)")
    mktempdir() do directory
        println("Unpacking fresh input: ", NBSE2.archive)
        unpack_deck(NBSE2.archive,directory)
        scan,nwarn = quiet(() -> ertca_scan(NBSE2;root=directory))
        @printf("Loaded %d stackings (%d construction warnings).\n",length(scan.d),nwarn)
        registry_harmonics(scan;gshells=REGISTRY_SHELLS,atol=HARMONIC_ATOL)
    end
end

# Same reference sampling and half-filling definition as the DOS plotting script.
# Fold the uniform fractional mesh into the centered hexagonal first BZ before
# evaluating W(k,G=0); the isolated G=0 harmonic need not be reciprocal-periodic.
function sample_reference_spectrum(h,nk)
    W0 = average_bilayer(h)
    binv = inv(transpose(h.lattice.b))
    levels = Matrix{Float64}(undef,length(h.labels),nk^2)
    finished = Threads.Atomic{Int}(0)
    SingleParticleModels.threaded_chunks(nk^2) do chunk
        for index in chunk
            i,j = mod(index-1,nk),div(index-1,nk)
            k = binv*fold_to_bz(h.lattice,SVector((i+0.5)/nk,(j+0.5)/nk))
            levels[:,index] = eigvals(W0(k))
            done = Threads.atomic_add!(finished,1)+1
            if done % max(1,nk^2÷8) == 0 || done == nk^2
                @printf("  Reference DOS: %d / %d momenta\n",done,nk^2); flush(stdout)
            end
        end
    end
    @assert all(isfinite,levels)
    levels
end

function half_filling_fermi(levels)
    ordered = sort(vec(levels))
    @assert iseven(length(ordered))
    m = length(ordered) ÷ 2
    (ordered[m]+ordered[m+1])/2
end

function slice_energy(levels,ef)
    SLICE_ENERGY_MEV === nothing || return Float64(SLICE_ENERGY_MEV)/1000
    grid = collect(range(EWIN[1],EWIN[2];step=ENERGY_STEP_EV))
    dos = zeros(length(grid))
    for level in levels
        e = level-ef
        first_index = max(1,ceil(Int,(e-6DOS_SIGMA_EV-grid[1])/ENERGY_STEP_EV)+1)
        last_index = min(length(grid),floor(Int,(e+6DOS_SIGMA_EV-grid[1])/ENERGY_STEP_EV)+1)
        for index in first_index:last_index
            dos[index] += exp(-0.5*((grid[index]-e)/DOS_SIGMA_EV)^2)
        end
    end
    dos ./= sqrt(2pi)*DOS_SIGMA_EV*size(levels,2)
    peak = argmax(dos)
    @assert 1 < peak < length(grid) "Peak is at the search boundary; adjust EWIN."
    @printf("Reference DOS peak: E-EF = %+.1f meV, DOS = %.6f states/eV/cell\n",1000grid[peak],dos[peak])
    grid[peak]
end

# Inside the centered monolayer hexagon: nearest-reciprocal-neighbor half planes.
function inside_bz(kc,neighbors)
    all(g -> abs(dot(kc,g)) <= dot(g,g)/2+1e-10, neighbors)
end

# All panels use exactly the same momenta, including zero-weight samples.
# For each laboratory kappa, locate ONE carrier per orbital in the twisted model;
# different layers can use different representatives of their momentum labels.
# The map routines use these indices to sum both layers and spins before binning.
function common_sampling(h,U)
    B = transpose(h.lattice.b)
    binv = inv(B)
    neighbors = (B*SVector(1,0),B*SVector(0,1),B*SVector(1,-1))
    hx,hy = bz_hexagon(h.lattice)
    bounds = [U.bminv*SVector(x,y) for (x,y) in zip(hx,hy)]
    lo = floor.(Int,reduce((a,b)->min.(a,b),bounds)) .- 1
    hi = ceil.(Int,reduce((a,b)->max.(a,b),bounds)) .+ 1
    candidates = [SVector(i,j) for i in lo[1]:hi[1], j in lo[2]:hi[2]]
    groups = map(1:NK^2) do index
        i,j = mod(index-1,NK),div(index-1,NK)
        k = SVector((i+0.5)/NK,(j+0.5)/NK)
        points = NamedTuple{(:kc,:kf,:idx),Tuple{SVector{2,Float64},SVector{2,Float64},SVector{4,Int}}}[]
        for G in candidates
            kc = U.bmT*(k+G)
            inside_bz(kc,neighbors) || continue
            kcheck,idx = carriers(U,kc)
            @assert norm(kcheck-k) < 1e-9
            @assert length(idx) == 4 && length(unique(idx)) == 4
            push!(points,(;kc,kf=SVector{2,Float64}(binv*kc),idx=SVector{4,Int}(idx)))
        end
        (;k,points)
    end
    @assert all(g->!isempty(g.points),groups)
    groups,neighbors
end

# Evaluate every requested energy using the same freshly computed eigenvalues.
function reference_maps(h,groups,energies)
    W0 = average_bilayer(h)
    rows = Vector{Matrix{Float64}}(undef,length(groups))
    finished = Threads.Atomic{Int}(0)
    SingleParticleModels.threaded_chunks(length(groups)) do chunk
        for index in chunk
            points = groups[index].points
            values = Matrix{Float64}(undef,length(points),length(energies))
            for (ip,p) in pairs(points)
                levels = eigvals(W0(p.kf))
                for (ie,E0) in pairs(energies)
                    values[ip,ie] = sum(exp(-((e-E0)/SIG)^2) for e in levels)
                end
            end
            rows[index] = values
            done = Threads.atomic_add!(finished,1)+1
            if done % max(1,length(groups)÷8) == 0 || done == length(groups)
                @printf("  G=0 reference: %d / %d sampling groups\n",done,length(groups)); flush(stdout)
            end
        end
    end
    reduce(vcat,rows)
end

# Keep the single-energy interface available for checks and interactive use.
reference_map(h,groups,E0) = reference_maps(h,groups,[E0])[:,1]

function twisted_maps(model,groups,energies)
    ns = n_states(model.cell)
    rows = Vector{Matrix{Float64}}(undef,length(groups))
    orthogonality = zeros(length(groups))
    hermiticity = zeros(length(groups))
    finished = Threads.Atomic{Int}(0)
    SingleParticleModels.threaded_chunks(length(groups)) do chunk
        H = zeros(ComplexF64,ns,ns)
        for index in chunk
            group = groups[index]
            bloch_hamiltonian!(H,model,group.k)
            hermiticity[index] = maximum(abs,H-H')
            @assert hermiticity[index] < 1e-9
            F = eigen(Hermitian(H))
            probabilities = abs2.(F.vectors)
            orthogonality[index] = maximum(abs,vec(sum(probabilities;dims=2)).-1)
            @assert orthogonality[index] < 1e-10
            values = Matrix{Float64}(undef,length(group.points),length(energies))
            for (ie,E0) in pairs(energies)
                weights = probabilities*exp.(-((F.values.-E0)./SIG).^2)
                values[:,ie] = [sum(weights[p.idx]) for p in group.points]
            end
            rows[index] = values
            done = Threads.atomic_add!(finished,1)+1
            if done % max(1,length(groups)÷8) == 0 || done == length(groups)
                @printf("  Full ERTCA: %d / %d moire momenta (%d bands)\n",done,length(groups),ns); flush(stdout)
            end
        end
    end
    (;weight=reduce(vcat,rows),max_hermiticity_eV=maximum(hermiticity),
      max_row_norm_error=maximum(orthogonality))
end

function twisted_map(model,groups,E0)
    result = twisted_maps(model,groups,[E0])
    (;result...,weight=result.weight[:,1])
end

function binned_maps(h,U,neighbors,kx,ky,refweight,twistweight)
    spacing = norm(U.bmT*SVector(1.0,0.0))/NK
    function bin(weight)
        # Each sample already includes both layers and spins. layer=1 is only
        # the binning helper's tag; there is no extra layer average or factor 2.
        FS = (;kx,ky,weight,layer=ones(Int,length(weight)),spacing)
        x,y,Z = bin_to_grid(FS,(-LIM,LIM),(-LIM,LIM))
        for ix in eachindex(x),iy in eachindex(y)
            inside_bz(SVector(x[ix],y[iy]),neighbors) || (Z[iy,ix]=NaN)
        end
        x,y,Z
    end
    x,y,Z0 = bin(refweight)
    x1,y1,Z1 = bin(twistweight)
    @assert x == x1 && y == y1
    x,y,Z0,Z1
end

# Matplotlib's afmhot uses RGB = clamp.((2u, 2u-0.5, 2u-1), 0, 1).
# Reverse u for afmhot_r; these breakpoints preserve its piecewise-linear colors.
# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py
function spectral_palette(palette)
    if palette == :reference_blue
        # Low-to-high interior colorbar samples from the user's Fig. 4 screenshot.
        return Plots.cgrad(["#C8E2F2","#AED2E6","#88BFDD","#6CAED0","#489CC1",
                            "#2E86B3","#146CA0","#005287","#00376F"])
    end
    if palette == :afmhot_r
        return Plots.cgrad([Plots.RGB(clamp(2u,0,1),clamp(2u-0.5,0,1),clamp(2u-1,0,1))
                           for u in range(1,0;length=5)])
    end
    if palette isa Symbol && endswith(string(palette),"_r")
        return Plots.cgrad(Symbol(chop(string(palette);tail=2));rev=true)
    end
    Plots.cgrad(palette)
end

function spectral_light_theme(palette)
    low = spectral_palette(palette)[0.0]
    0.2126low.r + 0.7152low.g + 0.0722low.b > 0.55
end

function spectral_panel(h,x,y,Z,title,color_max;palette=DEFAULT_PALETTE)
    light = spectral_light_theme(palette)
    hx,hy = bz_hexagon(h.lattice)
    p = Plots.heatmap(x,y,clamp.(Z,0,color_max);c=spectral_palette(palette),clims=(0,color_max),
        colorbar=false,
        aspect_ratio=:equal,framestyle=:box,legend=false,grid=false,
        xlims=(-LIM,LIM),ylims=(-LIM,LIM),xticks=[-1,0,1],yticks=[-1,0,1],
        xlabel=raw"$\kappa_x\;(\mathrm{\AA}^{-1})$",ylabel=raw"$\kappa_y\;(\mathrm{\AA}^{-1})$",
        title=title,titlefontsize=10,guidefontsize=10,tickfontsize=9,
        fontfamily="Computer Modern",background_color_inside=light ? :white : :black)
    Plots.plot!(p,hx,hy;color=light ? :black : :white,lw=0.8,alpha=0.7,label="")
    p
end

function plot_maps(h,x,y,Z0,Z1,angle,eslice;
                   color_max=nothing,stem=STEM,panel_titles=nothing,palette=DEFAULT_PALETTE)
    light = spectral_light_theme(palette)
    annotation_color = light ? :gray25 : :gray80
    if color_max === nothing
        color_max = weight_clims(vcat(vec(Z0),vec(Z1));q=0.99)[2]
    end
    @assert isfinite(color_max) && color_max > 0
    titles = isnothing(panel_titles) ?
        ("Stacking-averaged\nbilayer " * raw"$(\bar{G}_0)$",
         "\n" * raw"$\theta = " * @sprintf("%.2f",angle) * raw"^\circ$") : panel_titles
    p0 = spectral_panel(h,x,y,Z0,titles[1],color_max;palette)
    p1 = spectral_panel(h,x,y,Z1,titles[2],color_max;palette)
    Plots.plot!(p1;ylabel="",yticks=([-1,0,1],["","",""]))
    # Use the actual commensurate model lattice in the laboratory frame:
    # the BZ boundary is not the set of reciprocal-vector endpoints.
    tb = twisted_bilayer(h.lattice,TWIST_PAIR...)
    @assert isapprox(twist_angle(tb),angle;atol=1e-8)
    bm = transpose(tb.moire.b)
    # The coincidence lattice may use an acute or obtuse primitive basis.
    # Intersect bisectors of its six shortest reciprocal vectors to obtain
    # the Wigner-Seitz hexagon without assuming either basis convention.
    candidates = [bm*SVector(i,j) for i in -2:2 for j in -2:2 if (i,j) != (0,0)]
    shortest = minimum(norm,candidates)
    neighbors = filter(g->isapprox(norm(g),shortest;rtol=1e-8),candidates)
    @assert length(neighbors) == 6
    sort!(neighbors;by=g->mod(atan(g[2],g[1])+1e-8,2pi))
    corners = map(1:6) do j
        g1,g2 = neighbors[j],neighbors[mod1(j+1,6)]
        vcat(transpose(g1),transpose(g2)) \ [dot(g1,g1)/2,dot(g2,g2)/2]
    end
    @assert norm(sum(corners)) < 1e-8
    @assert all(isapprox(norm(v),shortest/sqrt(3);rtol=1e-8) for v in corners)
    area = abs(sum(corners[j][1]*corners[mod1(j+1,6)][2] -
                   corners[j][2]*corners[mod1(j+1,6)][1] for j in 1:6))/2
    @assert isapprox(area,abs(det(bm));rtol=1e-8)
    boundary = hcat(corners...,corners[1])
    Plots.plot!(p1,boundary[1,:],boundary[2,:];color=light ? :gray40 : :gray65,lw=0.7,label="")
    for j in 1:2
        g = neighbors[j] # an equivalent shortest primitive pair, 60 degrees apart
        perpendicular = [-g[2],g[1]]
        Plots.plot!(p1,[0,g[1]],[0,g[2]];color=annotation_color,lw=0.7,label="")
        # Arrowheads sized relative to the true reciprocal-vector length.
        head = hcat(0.78g+0.10perpendicular,g,0.78g-0.10perpendicular)
        Plots.plot!(p1,head[1,:],head[2,:];color=annotation_color,lw=0.7,label="")
        label_position = 1.5g
        j == 1 && (label_position = label_position + SVector(0.06,0.0))
        Plots.annotate!(p1,label_position[1],label_position[2],
            Plots.text("\$\\mathbf{g}_{$j}\$",7,"Computer Modern",annotation_color))
    end
    Plots.scatter!(p1,[0.0],[0.0];color=annotation_color,ms=1.2,msw=0,label="")
    # Label the panels in the empty upper-left corners of the momentum maps.
    for (panel,label) in ((p0,"a)"),(p1,"b)"))
        Plots.annotate!(panel,-0.92LIM,0.92LIM,
            Plots.text(label,10,"Computer Modern",light ? :black : :white,:left,:top))
    end
    # Anchor the scale to the second map's plotting area. GR fits the square
    # map inside that area, so give the narrow inset a matching aspect ratio:
    # its rendered height is the map width, with the same vertical center.
    # Reserve only the space needed for the inset scale and its tick labels.
    layout = Plots.grid(1,2;widths=[0.50,0.50])
    # Match the canvas height to the square panels so aspect-ratio padding does
    # not leave a large gap between the subplot titles and the maps.
    figure = Plots.plot(p0,p1;layout=layout,size=(450,255),dpi=300,
        top_margin=1Plots.mm,left_margin=[1 -3].*Plots.mm,right_margin=[-2 12].*Plots.mm,
        bottom_margin=2Plots.mm)
    scale = collect(range(0,color_max;length=256))
    Plots.heatmap!(figure,[0.0,1.0],scale,hcat(scale,scale);
        inset=(2,Plots.bbox(1.06,0.0,0.055,1.0)),subplot=3,c=spectral_palette(palette),
        clims=(0,color_max),colorbar=false,legend=false,grid=false,
        xlims=(0,1),ylims=(0,color_max),xticks=false,ymirror=true,
        aspect_ratio=1/(0.055*color_max),
        framestyle=:box,title="Weight\n(a.u.)",titlefontsize=8,
        tickfontsize=8,fontfamily="Computer Modern")
    mkpath(FIGURE_DIR)
    for extension in ("png","pdf")
        output = joinpath(FIGURE_DIR,stem*"."*extension)
        Plots.savefig(figure,output)
        println("Figure → ",output)
    end
    figure,color_max
end

# Reuse the original CSV weights, with only geometry and the original color
# limit from the saved companion calculation. Its recomputed weights are unused.
function main()
    @assert SIG > 0 && NK > 0 && DOS_SIGMA_EV > 0
    println("Input: ",NBSE2.archive)
    println("Computing all model and spectral data from scratch (no caches).")
    h = fresh_harmonics()
    println("Sampling the DOS reference in the centered first BZ")
    reference_levels = sample_reference_spectrum(h,NK_REFERENCE)
    ef = half_filling_fermi(reference_levels)
    eslice = slice_energy(reference_levels,ef)
    @printf("Shared reference EF = %.10f eV; slice E-EF = %+.1f meV\n",ef,1000eslice)
    tb = twisted_bilayer(h.lattice,TWIST_PAIR...)
    angle = twist_angle(tb)
    model,nwarn = quiet(() -> ertca_model(h,tb))
    @assert n_states(model.cell) == 4n_monolayer_cells(tb)
    @printf("Full ERTCA: theta = %.8f degrees, %d bands; %d construction warnings\n",angle,n_states(model.cell),nwarn)
    U = UnfoldMap(h,tb,model)
    groups,neighbors = common_sampling(h,U)
    points = reduce(vcat,[g.points for g in groups])
    kx,ky = [p.kc[1] for p in points],[p.kc[2] for p in points]
    @printf("Both panels: %d identical momenta in the centered monolayer first BZ\n",length(points))
    refweight = reference_map(h,groups,ef + eslice)
    twisted = twisted_map(model,groups,ef + eslice)
    @assert length(refweight) == length(twisted.weight) == length(points)
    @assert all(w->isfinite(w) && 0 <= w <= 4+1e-10,vcat(refweight,twisted.weight))
    x,y,Z0,Z1 = binned_maps(h,U,neighbors,kx,ky,refweight,twisted.weight)
    figure,color_max = plot_maps(h,x,y,Z0,Z1,angle,eslice)
    datadir = joinpath(FIGURE_DIR,"data")
    mkpath(datadir)
    open(joinpath(datadir,STEM*"_samples.csv"),"w") do io
        println(io,"kx_inv_A,ky_inv_A,untwisted_G0_weight,full_ERTCA_weight")
        for i in eachindex(kx)
            println(io,join((kx[i],ky[i],refweight[i],twisted.weight[i]),","))
        end
    end
    open(joinpath(FIGURE_DIR,STEM*"_parameters.txt"),"w") do io
        println(io,"Input: ",NBSE2.archive)
        println(io,"Input SHA256: ",bytes2hex(sha256(read(NBSE2.archive))))
        println(io,"Reference: untwisted angle-independent average_bilayer(h) = W(k,G=0); both layers and spins, including uniform interlayer coupling.")
        println(io,"Reference sampling: centered monolayer first BZ, nk = ",NK_REFERENCE,"; half filling = 2 of 4 states.")
        @printf(io,"Reference EF = %.10f eV; slice E-EF = %.6f meV\n",ef,1000eslice)
        @printf(io,"Full ERTCA: (m,n) = (%d,%d); angle = %.8f degrees; %d bands; %d harmonic shells\n",TWIST_PAIR...,angle,n_states(model.cell),REGISTRY_SHELLS)
        println(io,"Spectral broadening: exp[-((E-E*)/SIG)^2], SIG = ",SIG," eV; standard deviation = ",SIG/sqrt(2)," eV.")
        println(io,"DOS peak-selection Gaussian standard deviation = ",DOS_SIGMA_EV," eV; energy step = ",ENERGY_STEP_EV," eV.")
        println(io,"Moire mesh = ",NK," x ",NK,"; common first-BZ momentum samples = ",length(points))
        println(io,"Both layers and spins summed at each kappa before identical spatial binning; bins contain mean weight. Shared color scale capped at 99th percentile = ",color_max)
        println(io,"No lambda_m interpolation, no difference panel, and no hopping-radius cutoff.")
        println(io,"Maximum Hamiltonian Hermiticity residual (eV): ",twisted.max_hermiticity_eV)
        println(io,"Maximum eigenvector row-normalization error: ",twisted.max_row_norm_error)
        println(io,"Computed from the raw HDF5 archive on this run; no DOS, spectral, or extracted-deck cache was used.")
    end
    println("Parameters → ",joinpath(FIGURE_DIR,STEM*"_parameters.txt"))
    display(figure)
    (;figure,reference_ef=ef,eslice,angle,kx,ky,refweight,twisted,
      h,x,y,Z0,Z1,color_max)
end

end # module

NbSe2SpectralFunction.main()
