##################################################################
###### export only the following functions from this module ######
##################################################################

### nothing is exported: this file is the compiler's private view of a folded cell

### ------------------------------------------------------------------------------------
### The FINE (parent) lattice view of a folded cell
### ------------------------------------------------------------------------------------
###
### A cell folded by `M` stores its states on the coarse lattice and records `M` as the period
### of each localized site. Real-space terms, however, are written against the PARENT lattice:
### a folded comb has to find its fine-lattice neighbours, and the translations the emitters
### produce are fine-lattice integers converted by the exact rational `M^-T`.
###
### These two functions are that view -- the fine neighbour table, and the integer wrap shifts
### that keep the table's `[0,1)^D`-wrapped positions consistent with the folded cell's stored
### ones. They are pure functions of a `UnitCell`, and their only consumers are the compiler
### (`core/models/model.jl` and `core/models/terms.jl`).
###
### They lived in `transforms/fold.jl` until 2026-09-16. That is where the fold is PERFORMED,
### but it sits above `core/models/` in the layering, so the compiler reaching back down for
### them was an upward edge -- the only one `core/models/` had.

"""
    _fold_fine_neighbor_table(cell, deepest; rcut=nothing) -> NeighborTable

The neighbour table of a folded cell, built on the FINE (parent) lattice: `a_s = M⁻¹ a`, site
positions `r_s = Mᵀ r` (which are exactly the parent's positions), same site ids. Its integer
fine translations are converted to model coordinates by the emitters through the exact rational
`M⁻ᵀ` on the `CompileContext`.
"""
function _fold_fine_neighbor_table(cell::UnitCell{D}, deepest::Int;
                                   rcut::Union{Nothing,Real}=nothing) where {D}
    s = findfirst(k -> cell.site_kind[k] === :localized &&
                       abs(det(cell.site_period[k])) != 1, eachindex(cell.site_kind))
    M = SMatrix{D,D,Int}(cell.site_period[s])
    fine_a = inv(SMatrix{D,D,Float64}(M)) * SMatrix{D,D,Float64}(cell.lattice.a)
    fine_lat = BravaisLattice(Matrix(fine_a); force=true)
    Mtf = SMatrix{D,D,Float64}(transpose(M))
    fine_pos = [canonicalize_fracpos(SVector{D,Float64}(Mtf * p)) for p in cell.site_positions]
    proxy_basis = [BasisState{D}(fine_pos[k], zero(SVector{D,Int}), 1, TRIVIAL_SCHEMA)
                   for k in eachindex(fine_pos)]
    proxy = UnitCell{D}(fine_lat, TRIVIAL_SCHEMA, copy(cell.site_names), fine_pos, proxy_basis;
                        atol=cell.atol, warn_empty_sites=false)
    return isnothing(rcut) ? neighbor_table_by_shell(proxy, deepest) :
                             neighbor_table(proxy, rcut)
end


"""
    _fold_fine_wrapshifts(cell) -> Vector{SVector{D,Int}}

The integer wrap shifts `ρ_s = wrap(Mᵀτ_s) − Mᵀτ_s` of the fine-table site positions built by
`_fold_fine_neighbor_table`. The fine table stores its positions wrapped into `[0,1)ᴰ`
of the fine cell, so a fine translation `ΔR_fine` between fine representatives converts to model
coordinates as `M⁻ᵀ(ΔR_fine + ρ_j − ρ_i)` -- dropping the shifts leaves the emitted translation
inconsistent with the folded cell's stored positions by exactly `M⁻ᵀ(ρ_j − ρ_i)` (invisible to
the spectrum, wrong for every position-dependent quantity). One entry per site, matching the
position arithmetic of the table builder exactly.
"""
function _fold_fine_wrapshifts(cell::UnitCell{D})::Vector{SVector{D,Int}} where {D}
    s = findfirst(k -> cell.site_kind[k] === :localized &&
                       abs(det(cell.site_period[k])) != 1, eachindex(cell.site_kind))
    M = SMatrix{D,D,Int}(cell.site_period[s])
    Mtf = SMatrix{D,D,Float64}(transpose(M))
    return [canonicalize_int(canonicalize_fracpos(SVector{D,Float64}(Mtf * p)) - Mtf * p)
            for p in cell.site_positions]
end
