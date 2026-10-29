# WSe2 two-state unfolded spectral weights.
# Run from this repository: julia --project=. -t auto Fig4.jl

module WSe2UnfoldedThreeBands

include(joinpath(@__DIR__, "src", "valleys.jl"))

const STEM = "Fig4"
const DATA_DIR = joinpath(PAPER_FIGURES, "data")
const POCKET_RADIUS = 0.30 # inverse Angstrom; used only for summary diagnostics

# Reciprocal relabelling, including the Bloch phase of each Wannier centre:
# |k+G_layer,beta> = exp(i G_layer.r_beta) |k,beta>.
# This compares states in a common basis across the fractional-BZ box seam.
function carry_permutation(ctx, Umap, full, carry)
    gs, internal = state_g_vectors(full.cell), state_internals(full.cell)
    perm = Vector{Int}(undef, length(gs))
    phases = Vector{ComplexF64}(undef, length(gs))
    for a in eachindex(gs)
        beta = internal[a]
        layer = Umap.layer_of[beta]
        adj, nc = Umap.Mdata[layer]
        c = adj * (gs[a] + carry)
        rep = Umap.coset[layer][(mod(c[1], nc), mod(c[2], nc))]
        perm[a] = Umap.state_of[(beta, Tuple(rep))]
        M = supercell_matrix(ctx.tb, layer)
        delta = gs[a] + carry - rep
        wrap = round.(Int, M \ delta)
        @assert M * wrap == delta
        phases[a] = cispi(2dot(wrap, ctx.h.tau[beta]))
    end
    @assert sort(perm) == collect(eachindex(gs))
    return (; perm, phases)
end

function transport_vectors(ctx, Umap, full, q, raw, vectors, transports)
    carry = round.(Int, q-raw)
    transport = get!(transports, Tuple(carry)) do
        carry_permutation(ctx, Umap, full, carry)
    end
    out = similar(vectors)
    out[transport.perm,:] = transport.phases .* vectors
    return out
end

# Resolve narrow avoided crossings locally instead of trusting a large endpoint
# overlap or repeatedly doubling the entire path mesh.
function advance_pair(ctx, full, Umap, direction, transports, previous, t0, t1, eig;
                      depth=0)
    N = n_states(full.cell)
    raw = t1*direction
    q = mod.(raw,1.0)
    transported = transport_vectors(ctx,Umap,full,q,raw,eig.vectors,transports)
    overlaps = abs2.(previous' * transported)
    choices = [(j,k) for j in axes(overlaps,2), k in axes(overlaps,2) if j != k]
    j,k = choices[argmax([overlaps[1,j]+overlaps[2,k] for (j,k) in choices])]
    current = transported[:,[j,k]]
    subspace_overlap = minimum(svdvals(previous' * current))^2
    if subspace_overlap < 0.90 && depth < 12
        tm = (t0+t1)/2
        qm = mod.(tm*direction,1.0)
        mid = diagonalize(bloch_hamiltonian(full,qm);nev=24,which=:largest)
        a = advance_pair(ctx,full,Umap,direction,transports,previous,t0,tm,mid;depth=depth+1)
        b = advance_pair(ctx,full,Umap,direction,transports,a.current,tm,t1,eig;depth=depth+1)
        return (; b.current, b.chosen, trace=vcat(a.trace,b.trace))
    end
    row = (; t=t1, q1=q[1], q2=q[2], index13=N-24+j, index14=N-24+k,
        energy13_eV=eig.values[j],energy14_eV=eig.values[k],
        overlap13=overlaps[1,j],overlap14=overlaps[2,k],subspace_overlap)
    return (; current,chosen=[j,k],trace=[row])
end

function track_gamma_pair(ctx, full, Umap, gamma; nsteps=60)
    N = length(gamma.values)
    V = valley_setup(ctx, :K)
    direction = Umap.bminv * (ctx.bmT * V.kap_t)
    ts = collect(range(0.0, 1.0; length=nsteps+1))
    raw = [t * direction for t in ts]
    qs = [mod.(q, 1.0) for q in raw]
    qs[1] = SVector(0.0, 0.0)
    carries = [round.(Int, qs[i] - raw[i]) for i in eachindex(qs)]
    permutations = Dict(Tuple(c) => carry_permutation(ctx, Umap, full, c)
                        for c in unique(carries))
    println("Following the Gamma u13/u14 pair to kappa-minus on ", nsteps, " steps...")
    flush(stdout)
    done = Threads.Atomic{Int}(0)
    outlock = ReentrantLock()
    spectra = bz_map(full, qs; nev=24, which=:largest, vectors=true) do q, result
        count = Threads.atomic_add!(done, 1) + 1
        if count % 20 == 0 || count == length(qs)
            lock(outlock) do
                println("  eigenstates: ", count, " / ", length(qs))
                flush(stdout)
            end
        end
        (; values=copy(result.values), vectors=copy(result.vectors))
    end
    previous = gamma.vectors[:, [N-12, N-13]]
    trace = NamedTuple[]
    chosen = [12, 11] # ascending positions in the highest-24 window at Gamma
    for i in 2:length(qs)
        eig = spectra[i]
        step = advance_pair(ctx,full,Umap,direction,permutations,previous,ts[i-1],ts[i],eig)
        append!(trace,step.trace)
        chosen = step.chosen
        previous = step.current
    end
    last_eig = last(spectra)
    return (; q=last(qs), indices=N-24 .+ chosen,
        vectors=last_eig.vectors[:,chosen], values=last_eig.values[chosen],
        endpoint=last_eig, trace)
end

function aggregate(points, probabilities)
    acc = Dict{Tuple{Int,Int},Tuple{SVector{2,Float64},Float64}}()
    for (k,w) in zip(points, probabilities)
        key = Tuple(round.(Int, k ./ 1e-9))
        old = get(acc, key, (k, 0.0))
        acc[key] = (old[1], old[2] + w)
    end
    entries = sort(collect(values(acc)); by=x -> (x[1][1],x[1][2]))
    (; k=[x[1] for x in entries], weight=[x[2] for x in entries])
end

function unfolded_weights(ctx, full, Umap, q, u)
    G = state_g_vectors(full.cell)
    beta = state_internals(full.cell)
    probability = abs2.(u)
    @assert sum(probability) ≈ 1.0 atol=1e-10
    lab = [kappa_of(Umap, SVector{2,Float64}(q), g) for g in G]
    mono = map(eachindex(G)) do a
        layer = Umap.layer_of[beta[a]]
        k = Umap.rot_of[layer] * lab[a]
        folded = fold_to_bz(ctx.lat, ctx.binv * k)
        shift = ctx.binv * (k-folded)
        @assert maximum(abs, shift-round.(shift)) < 1e-8
        # Check membership in the first monolayer BZ, including its boundary.
        for n1 in -2:2, n2 in -2:2
            g = ctx.bT * SVector(n1,n2)
            @assert 2dot(folded,g) <= dot(g,g) + 1e-8
        end
        folded
    end
    monomap = aggregate(mono, probability)
    labmap = aggregate(lab, probability)
    @assert sum(monomap.weight) ≈ 1.0 atol=1e-10
    @assert sum(labmap.weight) ≈ 1.0 atol=1e-10
    return (; mono=monomap, lab=labmap, G, beta, probability,
        component_mono=mono, component_lab=lab)
end

function pocket_weights(ctx, map)
    K = ctx.bT * SVector(1/3,1/3)
    distance(k, center) = minimum(norm(k-center-ctx.bT*SVector(a,b))
                                 for a in -2:2, b in -2:2)
    weights = [sum(w for (k,w) in zip(map.k,map.weight)
                   if distance(k,center) <= POCKET_RADIUS; init=0.0)
               for center in (SVector(0.0,0.0), K, -K)]
    (; Gamma=weights[1], K=weights[2], Kprime=weights[3], other=1-sum(weights))
end

function calculate(; ctx=nothing, edge_eV=nothing, nsteps=60)
    BLAS.set_num_threads(1)
    ctx = isnothing(ctx) ? ertca_context(WSE2; gshells=HARMONIC_SHELLS) : ctx
    @assert length(ctx.h.labels) == 4 && n_monolayer_cells(ctx.tb) == 127
    if isnothing(edge_eV)
        V = valley_setup(ctx, :K)
        near = ertca_model(V.h, rad2deg(ctx.theta); shells=PLANE_WAVE_SHELLS,
            expansion=collect(V.expansion))
        edge_eV = maximum(maximum(bands(near,k)) for k in read_wse2_reference().ks)
    end
    full = ertca_model(ctx.h, ctx.tb; folding=:auto, name="WSe2 full BZ")
    Umap = UnfoldMap(ctx.h, ctx.tb, full)
    gamma = diagonalize(bloch_hamiltonian(full, [0.0,0.0]))
    N = length(gamma.values)
    tracked = track_gamma_pair(ctx, full, Umap, gamma; nsteps)
    ep = tracked.endpoint
    seeds = (("Highest red band", N, ep.values[end], ep.vectors[:,end]),
             ("Second red band", N-1, ep.values[end-1], ep.vectors[:,end-1]),
             ("Branch from -204 meV at Gamma", tracked.indices[1],
              tracked.values[1], tracked.vectors[:,1]))
    H = bloch_hamiltonian(full, tracked.q)
    states = map(seeds) do (label,index,energy,u)
        residual = norm(H*u-energy*u)
        @assert residual < 1e-9
        weights = unfolded_weights(ctx, full, Umap, tracked.q, u)
        pockets = pocket_weights(ctx, weights.mono)
        @printf("%s: index %d, rank %d, E=%.6f meV; Gamma %.4f%%, K %.4f%%, Kprime %.4f%%\n",
            label, index, N-index+1, 1000(energy-edge_eV),
            100pockets.Gamma, 100pockets.K, 100pockets.Kprime)
        (; label, index, rank=N-index+1, energy_eV=energy,
            energy_meV=1000(energy-edge_eV), u, residual, weights, pockets)
    end
    origin = unfolded_weights(ctx, full, Umap, [0.,0.], gamma.vectors[:,N-12])
    return (; states, q=tracked.q, theta_deg=rad2deg(ctx.theta), lat=ctx.lat,
        B=ctx.bT, theta=ctx.theta, trace=tracked.trace, tracked_indices=tracked.indices,
        gamma_index=N-12, gamma_energy_meV=1000(gamma.values[N-12]-edge_eV),
        gamma_pockets=pocket_weights(ctx,origin.mono), gamma_weights=origin,
        labels=ctx.h.labels, shifted=ctx.h.shifted, pocket_radius=POCKET_RADIUS)
end

function weight_panel(data, state; frame=:mono, logarithmic=false, panel_index=1,
                      palette=:viridis, show_layer_bz=true)
    map = getproperty(state.weights, frame)
    corners = bz_corners(data.lat)
    radius = maximum(norm,corners)
    limit = frame == :mono ? 1.33radius : max(1.33radius,1.08maximum(norm,map.k))
    p = Plots.plot(; aspect_ratio=:equal, xlims=(-limit,limit), ylims=(-limit,limit),
        xlabel=raw"$k_x\;(\mathrm{\AA}^{-1})$", ylabel=raw"$k_y\;(\mathrm{\AA}^{-1})$",
        # Keep text readable without overwhelming the three compact panels.
        guidefontsize=11, tickfontsize=10, legend=false,
        xticks=[-1,0,1],yticks=[-1,0,1],minorticks=false,
        grid=false, background_color=:white, foreground_color=:black,
        fontfamily="Computer Modern", framestyle=:box,
        bottom_margin=3Plots.mm)
    if hasproperty(state,:sample_label)
        Plots.plot!(p;title=state.sample_label,titlefontsize=9,top_margin=3Plots.mm)
    end
    if panel_index > 1
        Plots.plot!(p;ylabel="",yticks=([-1,0,1],["","",""]))
    end
    closed = vcat(corners,[first(corners)])
    Plots.plot!(p, first.(closed), last.(closed);color=:black,lw=0.9,label="")
    if frame == :lab && show_layer_bz
        # Dashed blue/red: bottom/top. Solid black: unrotated reference BZ.
        for (angle,color,style) in ((-data.theta/2,"#0072B2",:dash),(data.theta/2,"#D62728",:dash))
            boundary = [_rotation2(angle)*k for k in closed]
            Plots.plot!(p,first.(boundary),last.(boundary);color,lw=1.1,ls=style,label="")
        end
    end
    Plots.scatter!(p,first.(map.k),last.(map.k); color="#D5D9DE",markersize=1.0,
        markerstrokewidth=0,label="")
    visible = findall(>=(1e-8),map.weight)
    order = sort(visible; by=i -> map.weight[i])
    probabilities = map.weight[order]
    marker_values = logarithmic ? log10.(probabilities) : probabilities
    marker_sizes = 0.675 .* (logarithmic ? 2 .+ 4sqrt.(probabilities) : 8sqrt.(probabilities))
    color_limits = logarithmic ? (-8,0) : (0,1)
    Plots.scatter!(p,first.(map.k[order]),last.(map.k[order]);
        marker_z=marker_values,clims=color_limits,color=palette,
        markersize=marker_sizes,markerstrokewidth=0,
        colorbar=false,label="")
    Plots.annotate!(p,-.91limit,.91limit,
        Plots.text(string(Char(Int('a')+panel_index-1),")"),11,"Computer Modern",:black,:left,:top))
    gamma_position = hasproperty(state,:gamma_label_position) ? state.gamma_label_position : (.28,.23)
    Plots.annotate!(p,gamma_position...,Plots.text(raw"$\Gamma$",10,"Computer Modern",:black))
    # Keep only the left K and bottom-left K-prime corners.
    labelled_corners = (corners[argmin(first.(corners))],
        corners[argmin([k[1]+2k[2] for k in corners])])
    for k in labelled_corners
        frac = data.B \ k - SVector(1/3,1/3)
        label = maximum(abs,frac-round.(frac)) < 1e-6 ? raw"$K$" : raw"$K'$"
        pos = 1.17k
        # Separate the left-corner label from the horizontal y=0 tick.
        abs(k[2]) < 1e-8 && (pos = pos + SVector(0.0,0.18))
        Plots.annotate!(p,pos[1],pos[2],Plots.text(label,10,"Computer Modern",:black))
    end
    return p
end

# Trim the exported page, including all labels/colorbar, without changing the
# plot's coordinates or scale. Keep a 2-point safety margin and render the PNG
# from the same cropped vector PDF so the two exports have matching bounds.
function trim_unfolding_exports(pdf; padding=2, dpi=240)
    cropper=Sys.which("pdfcrop")
    isnothing(cropper) && isfile("/Library/TeX/texbin/pdfcrop") && (cropper="/Library/TeX/texbin/pdfcrop")
    renderer=Sys.which("pdftoppm")
    isnothing(cropper) && error("Tight export requires pdfcrop on PATH.")
    isnothing(renderer) && error("Tight export requires pdftoppm on PATH.")
    mktempdir() do tmp
        cropped=joinpath(tmp,"cropped.pdf")
        run(`$cropper --margins $padding $pdf $cropped`)
        prefix=joinpath(tmp,"cropped")
        run(`$renderer -r $dpi -png -singlefile $cropped $prefix`)
        mv(cropped,pdf;force=true)
        mv(prefix*".png",splitext(pdf)[1]*".png";force=true)
    end
end

function plot_data(data; stem=STEM, frames=(:mono,:lab), scales=(false,true),
                   palette=:viridis,show_layer_bz=true,tight=false)
    mkpath(PAPER_FIGURES)
    npanels=length(data.states)
    left_margins=reshape(vcat(5,fill(-3,npanels-1)),1,npanels).*Plots.mm
    right_margins=reshape(vcat(zeros(Int,npanels-1),20),1,npanels).*Plots.mm
    for frame in frames, logarithmic in scales
        panels = [weight_panel(data,s;frame,logarithmic,panel_index=i,palette,show_layer_bz) for (i,s) in enumerate(data.states)]
        fig = Plots.plot(panels...;layout=Plots.grid(1,npanels;widths=fill(1/npanels,npanels)),
            size=(npanels==2 ? 480 : 225npanels,260),dpi=240,top_margin=1Plots.mm,
            left_margin=left_margins,right_margin=right_margins)
        color_limits = logarithmic ? (-8.0,0.0) : (0.0,1.0)
        color_label = logarithmic ? raw"$\log_{10} P(\mathbf{k})$" : raw"$P(\mathbf{k})$"
        scale = collect(range(color_limits...;length=256))
        # One shared scale, anchored to the final panel and matching its height.
        Plots.heatmap!(fig,[0.0,1.0],scale,hcat(scale,scale);
            inset=(npanels,Plots.bbox(1.04,0.0,0.055,1.0)),subplot=npanels+1,
            c=palette,clims=color_limits,colorbar=false,legend=false,grid=false,
            xlims=(0,1),ylims=color_limits,xticks=false,ymirror=true,
            yticks=logarithmic ? [-8,-4,0] : [0,.5,1],minorticks=false,
            ylabel=color_label,guidefontsize=10,tickfontsize=9,
            aspect_ratio=1/(0.055*(color_limits[2]-color_limits[1])),
            framestyle=:box,fontfamily="Computer Modern")
        output_stem = stem
        for ext in ("png","pdf")
            dest = joinpath(PAPER_FIGURES,output_stem*"."*ext)
            Plots.savefig(fig,dest)
            println("Saved ",dest)
        end
        tight && trim_unfolding_exports(joinpath(PAPER_FIGURES,output_stem*".pdf"))
    end
end

function save_data(data; stem=STEM)
    mkpath(DATA_DIR)
    serialize(joinpath(DATA_DIR,stem*".jls"),data)
    open(joinpath(DATA_DIR,stem*"_components.csv"),"w") do io
        println(io,"panel,band_index,component,G1,G2,orbital,layer,k_lab_x,k_lab_y,k_mono_x,k_mono_y,u_real,u_imag,probability")
        for (panel,state) in enumerate(data.states), a in eachindex(state.u)
            w = state.weights
            beta = w.beta[a]
            row = (panel,state.index,a,w.G[a]...,data.labels[beta],data.shifted[beta] ? 2 : 1,
                   w.component_lab[a]...,w.component_mono[a]...,real(state.u[a]),imag(state.u[a]),w.probability[a])
            println(io,join(row,','))
        end
    end
    open(joinpath(DATA_DIR,stem*"_weights.csv"),"w") do io
        println(io,"panel,band_index,frame,kx_Ainv,ky_Ainv,total_probability")
        for (panel,state) in enumerate(data.states), frame in (:mono,:lab)
            map = getproperty(state.weights,frame)
            for (k,w) in zip(map.k,map.weight)
                println(io,join((panel,state.index,frame,k...,w),','))
            end
        end
    end
    open(joinpath(DATA_DIR,stem*"_tracking.csv"),"w") do io
        println(io,join(keys(first(data.trace)),','))
        for row in data.trace
            println(io,join(values(row),','))
        end
    end
    open(joinpath(DATA_DIR,stem*"_summary.txt"),"w") do io
        println(io,"Red full-BZ ERTCA, four coupling-harmonic shells, both spins; theta = ",data.theta_deg," degrees.")
        if hasproperty(data,:sampling_description)
            println(io,data.sampling_description)
            println(io,"Both panels use the plotted kappa-minus; full-model fractional q = ",Tuple(data.q))
            println(io,"Color scale: pale blue at low probability, dark navy at high probability.")
        elseif hasproperty(data,:sample_q)
            println(io,"Panels a/b: exact midpoint of m -> Gamma. Panel c: original tracked state at kappa-minus.")
            for (i,q) in enumerate(data.sample_q)
                println(io,"Panel ",i," full-model fractional q = ",Tuple(q))
            end
            println(io,"Top-band splitting at midpoint = ",data.states[1].energy_meV-data.states[2].energy_meV," meV.")
        else
            println(io,"All three panels use the plotted kappa-minus; full-model fractional q = ",Tuple(data.q))
        end
        println(io,"Original frame: inverse-rotate each layer, fold into its original monolayer BZ, sum probabilities at coincident coordinates.")
        println(io,"Lab companion: sum probabilities at identical physical q+G, with the original and rotated layer BZ outlines.")
        if hasproperty(data,:sample_q)
            println(io,"Revised mono/lab figures: log10 probability colors (-8 to 0), with a minimum marker size to expose weak tails.")
        else
            println(io,"Main figures: linear probability colors (0 to 1), with marker area proportional to probability.")
            println(io,"Separate _log figures: log10 probability colors (-8 to 0), with a minimum marker size to expose weak tails.")
        end
        println(io,"Gray dots mark sampled momenta. Colored markers require probability >= 1e-8; all weights are retained in the data and normalization. Every panel sums to one.")
        println(io,"Pocket weights use periodic distances within ",data.pocket_radius," inverse Angstrom of Gamma/K/Kprime.")
        println(io,"Gamma starting index ",data.gamma_index,", E = ",data.gamma_energy_meV," meV; pockets = ",data.gamma_pockets)
        println(io,"Tracked pair endpoint indices: ",data.tracked_indices)
        println(io,"Minimum consecutive pair-subspace overlap: ",minimum(r.subspace_overlap for r in data.trace))
        for s in data.states
            println(io,s.label,": index ",s.index,", rank from top ",s.rank,", E = ",s.energy_meV," meV; ",s.pockets)
            println(io,"  normalization = ",sum(s.weights.probability),"; eigenpair residual = ",s.residual)
        end
    end
end

function main_two_panels(;frames=(:mono,))
    original=calculate()
    gamma_label_position=(0.10,0.32)
    first_state=merge(original.states[1],(;gamma_label_position))
    gamma_state=merge(original.states[3],(;gamma_label_position))
    data=merge(original,(;states=(first_state,gamma_state),sample_q=(original.q,original.q),
        sampling_description="Panel a: single highest band (original a). Panel b: unchanged Gamma-derived state (original c)."))
    for (new,old) in zip(data.states,(original.states[1],original.states[3]))
        @assert new.u==old.u && new.weights==old.weights && new.energy_eV==old.energy_eV
        @assert sum(new.weights.probability) ≈ 1.0 atol=1e-10
    end
    stem="Fig4"
    save_data(data;stem)
    palette=Plots.cgrad(["#D5E7F3","#83B6D5","#347FAD","#08306B"])
    plot_data(data;stem,frames,scales=(true,),palette,show_layer_bz=false,tight=true)
    return data
end


end # module
WSe2UnfoldedThreeBands.main_two_panels()
