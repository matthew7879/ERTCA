# WSe2 moire band benchmark.
# Run from this repository: julia --project=. -t auto Fig3.jl

include(joinpath(@__DIR__, "src", "valleys.jl"))
BLAS.set_num_threads(1)
data = wse2_comparison_data()
save_fig3(fig3_wse2_figure(data); stem="Fig3")
