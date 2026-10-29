# NbSe2 Gamma-valley bands.
# Run from this repository: julia --project=. -t auto Fig8.jl

include(joinpath(@__DIR__, "src", "valleys.jl"))
BLAS.set_num_threads(1)
data = nbse2_comparison_data()
save_fig3(fig3_nbse2_figure(data; top_legend=true); stem="Fig8")
