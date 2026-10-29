##################################################################
###### export only the following functions from this module ######
##################################################################

export monkhorst_pack, density_of_states




#############################################################
###### Brillouin-zone sampling                          #####
#############################################################

"""
    monkhorst_pack(lattice, nk; shift=0.5) -> Vector{SVector{D,Float64}}

A uniform `nk × nk × ...` grid of fractional momenta covering the Brillouin zone once:
`k_i = (m + shift) / nk` with `m = 0 ... nk-1` along each axis.

Args:
-----
- `nk`: points per axis -- one `Int` for an isotropic grid, or an `NTuple{D,Int}`
- `shift`: offset in grid units. The default `0.5` is the Monkhorst-Pack choice: it avoids Γ
  and the zone boundary, where band degeneracies and van-Hove singularities live, so smooth
  integrands converge fastest. Use `shift=0` when the grid must CONTAIN the high-symmetry
  points instead.

Notes:
------
Fractional coordinates make the grid lattice-agnostic: it covers the primitive reciprocal cell
`[0,1)^D`, which tiles k-space exactly like the first Brillouin zone (same points, different
fundamental domain), so every BZ integral can be sampled on it without Wigner-Seitz geometry.
"""
function monkhorst_pack(lattice::BravaisLattice{D}, nk::NTuple{D,Int};
                        shift::Real=0.5) where {D}
    all(>=(1), nk) || throw(DomainError(nk, "Every grid dimension must be at least 1."))
    ### aperiodic directions have no Brillouin zone: the grid collapses to k = 0 there,
    ### whatever was requested (one point, weight-neutral in every per-cell normalization)
    P = nperiodic(lattice)
    nkeff = ntuple(a -> a <= P ? nk[a] : 1, D)
    kpts = Vector{SVector{D,Float64}}(undef, prod(nkeff))
    idx = 1
    for m in Iterators.product(ntuple(a -> 0:(nkeff[a] - 1), D)...)
        kpts[idx] = SVector{D,Float64}(ntuple(a ->
            a <= P ? (m[a] + shift) / nkeff[a] : 0.0, D))
        idx += 1
    end
    return kpts
end

monkhorst_pack(lattice::BravaisLattice{D}, nk::Int; kwargs...) where {D} =
    monkhorst_pack(lattice, ntuple(_ -> nk, D); kwargs...)




#############################################################
###### Density of states                                #####
#############################################################

"""
    density_of_states(model; nk=20, nE=400, broadening=nothing, window=nothing,
                      method=:gaussian, shift=0.5, kwargs...)
        -> (energies, dos, integrated)

The density of states per unit cell,

    g(E) = (1/N_k) Σ_{k,n} δ_broadened(E - E_n(k)),

sampled on a `monkhorst_pack` grid and broadened with a normalized Gaussian or Lorentzian, so
that `∫ g(E) dE = n_states(cell)` (each band contributes one state per cell).

Args:
-----
- `nk`: k-points per axis (`Int` or `NTuple{D,Int}`) -- the convergence parameter
- `nE`: energy-mesh resolution (bins of the returned `energies`). Named `nE`, not `npoints`,
  because `npoints` means k-points along a path everywhere else in the analysis stack; the old
  `npoints=` spelling still works for now but warns and will be removed.
- `broadening`: kernel width. The default is Silverman's density-estimation bandwidth over the
  sampled eigenvalue cloud, `1.06·std(E)·N^(-1/5)` with `N` the number of sampled eigenvalues:
  it shrinks with the grid at the consistency rate, in any dimension. Converge by raising `nk`
  and lowering `broadening` together; sharp features (van Hove singularities) may want an
  explicit smaller value once `nk` is large.
- `window`: `(emin, emax)` restricting the energy mesh (default: the sampled spectrum plus a
  margin of five broadenings)
- `method`: `:gaussian` or `:lorentzian` kernel
- `shift`: forwarded to `monkhorst_pack`
- `kregion`: sample a fractional box `(kmin, kmax)` instead of the Brillouin zone (required on
  models with `NoZone()`). The `1/N_k` weight then means "per sampled k-point of
  the box", NOT per unit cell: the integral of `g` is `n_states` regardless of the box size,
  so compare boxes by shape, not absolute magnitude.
- `kwargs...`: the iterative-solver controls of [`diagonalize`](@ref) (`tol`, `maxiter`,
  `krylovdim`, `hermitian_atol`); with `nev=..., sigma=...` this computes a PARTIAL DOS around
  a target energy for continuum models too large to diagonalize fully (then only the captured
  bands integrate to their count, and `window` should match). Any other keyword is refused by
  name -- Plots attributes belong on [`plot_dos`](@ref).

Returns:
--------
- `energies::Vector{Float64}`: the energy mesh
- `dos::Vector{Float64}`: `g(E)`, states per energy per unit cell
- `integrated::Vector{Float64}`: `∫^E g` -- the counting function, `n_states(cell)` at the top
  of the spectrum

Notes:
------
The k-loop is parallel over the Julia threads (see [`threaded_chunks`](@ref)). A Lorentzian kernel
has heavy tails that shift spectral weight far from band edges; the Gaussian default is the
right choice unless the broadening is meant to model a physical linewidth.
"""
function density_of_states(model::HamiltonianModel{D};
                           nk::Union{Int,NTuple{D,Int}}=20,
                           nE::Int=400,
                           npoints::Union{Nothing,Int}=nothing,
                           broadening::Union{Nothing,Real}=nothing,
                           window::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                           method::Symbol=:gaussian,
                           shift::Real=0.5,
                           kregion=nothing,
                           nev::Union{Nothing,Int}=nothing,
                           which::Symbol=:smallest,
                           sigma::Union{Nothing,Real}=nothing,
                           matrix::Symbol=:dense,
                           kwargs...) where {D}
    if !isnothing(npoints)
        @warn "`npoints` of `density_of_states` is deprecated: it names the ENERGY mesh here " *
              "but k-points along a path everywhere else. Use `nE=` instead." maxlog = 1
        nE = npoints
    end
    ### unknown keywords are refused HERE by name: silently riding into the k-loop they would
    ### surface as a MethodError on `diagonalize`, an internal the caller never typed
    for key in keys(kwargs)
        key in (:tol, :maxiter, :krylovdim, :hermitian_atol) || throw(ArgumentError(
            "Unknown keyword `$key` for `density_of_states`. The pass-through set is the " *
            "iterative-solver controls of `diagonalize` (tol, maxiter, krylovdim, " *
            "hermitian_atol); Plots attributes belong on `plot_dos`."))
    end
    method in (:gaussian, :lorentzian) || throw(ArgumentError(
        "Unknown broadening kernel `$method`; expected `:gaussian` or `:lorentzian`."))
    nE >= 2 || throw(DomainError(nE, "`nE` must be at least 2."))

    kpts = if isnothing(kregion)
        _require_zone(kperiodicity(model), "`density_of_states` tiles the Brillouin zone",
            "Pass `kregion=(kmin, kmax)`, a fractional box to sample.")
        monkhorst_pack(model.cell.lattice, nk; shift=shift)
    else
        _kregion_grid(Val(D), kregion, nk; shift=shift)
    end
    nkt = length(kpts)
    n = n_states(model.cell)
    nb = isnothing(nev) ? n : nev
    evals = Matrix{Float64}(undef, nb, nkt)
    check = has_kdependence(model)      # constant lists are Hermitian by construction; see `bands`
    _check_matrix_mode(matrix, nev)

    ### one Hamiltonian buffer (or sparse workspace) per chunk: no per-k-point matrix allocation
    threaded_chunks(nkt) do chunk
        solve = bloch_solver(model; matrix=matrix)
        for idx in chunk
            r = solve(kpts[idx]; nev=nev, which=which, sigma=sigma, vectors=false,
                      check=check, kwargs...)
            evals[:, idx] = r.values
        end
    end

    sigma_E = isnothing(broadening) ? _default_broadening(evals) : Float64(broadening)
    sigma_E > 0 || throw(DomainError(sigma_E, "The broadening must be positive."))

    emin, emax = isnothing(window) ?
        (minimum(evals) - 5 * sigma_E, maximum(evals) + 5 * sigma_E) :
        (Float64(window[1]), Float64(window[2]))
    emin < emax || throw(ArgumentError("The energy window ($emin, $emax) is empty."))
    energies = collect(range(emin, emax; length=nE))

    dos = zeros(Float64, nE)
    if method === :gaussian
        pref = 1 / (sigma_E * sqrt(2π) * nkt)
        for e in evals
            _accumulate_gaussian!(dos, energies, e, sigma_E, pref)
        end
    else
        pref = sigma_E / (π * nkt)
        for e in evals
            @inbounds for m in eachindex(energies)
                dos[m] += pref / ((energies[m] - e)^2 + sigma_E^2)
            end
        end
    end

    dE = energies[2] - energies[1]
    integrated = cumsum(dos) .* dE
    return (energies=energies, dos=dos, integrated=integrated)
end

"""
    _default_broadening(evals) -> Float64

A broadening matched to the SIZE of the eigenvalue cloud: Silverman's kernel-density rule,
`σ = 1.06 · std(E) · N^(-1/5)` over all `N = n_bands · n_k` sampled eigenvalues.

Notes:
------
Tying the default to the SPECTRUM's total width (the obvious `(Emax - Emin)/100`) misestimates
in both directions. The former default -- the median k-to-k level shift along the flattened
column order -- measured the resolution of ONE k-line and therefore over-broadened a
`D`-dimensional grid by roughly `nk^(D-1)`: a 3D bulk DOS at its own defaults came out as a
featureless Gaussian with every van Hove singularity erased. Silverman's rule is the standard
density-estimation bandwidth: it sees the whole cloud, shrinks with `N` at the rate that keeps
the estimate consistent, and stays generous enough on a coarse two-band grid that the delta
cloud does not show as spikes.
"""
function _default_broadening(evals::Matrix{Float64})::Float64
    N = length(evals)
    N >= 2 || return 1e-3
    mu = sum(evals) / N
    sd = sqrt(sum(x -> (x - mu)^2, evals) / N)
    sd > 0 || return 1e-3
    return 1.06 * sd * N^(-1 / 5)
end


"""
    _accumulate_gaussian!(dos, energies, e, sigma, pref) -> Nothing

Add one normalized Gaussian centred at `e`. Only the mesh window within `±6σ` is touched:
beyond it the kernel is below 1e-8 of its peak, and skipping it turns the accumulation from
O(n_E) into O(σ/ΔE) per eigenvalue -- the difference between seconds and minutes on a dense
continuum spectrum.
"""
function _accumulate_gaussian!(dos::Vector{Float64}, energies::Vector{Float64},
                               e::Float64, sigma::Float64, pref::Float64)::Nothing
    lo = searchsortedfirst(energies, e - 6 * sigma)
    hi = searchsortedlast(energies, e + 6 * sigma)
    @inbounds for m in lo:hi
        dos[m] += pref * exp(-((energies[m] - e) / sigma)^2 / 2)
    end
    return nothing
end


"""
    _kregion_grid(::Val{D}, kregion, nk; shift=0.5) -> Vector{SVector{D,Float64}}

A uniform k-grid over the box `kregion = (kmin, kmax)` (fractional coordinates), `nk` points per
axis -- the sampling used instead of a Brillouin-zone tiling on models with no torus.
"""
function _kregion_grid(::Val{D}, kregion, nk::Union{Int,NTuple{D,Int}};
                       shift::Real=0.5) where {D}
    klo = SVector{D,Float64}(kregion[1])
    khi = SVector{D,Float64}(kregion[2])
    all(khi .> klo) || throw(ArgumentError("The k-region ($klo, $khi) is empty."))
    nks = nk isa Int ? ntuple(_ -> nk, D) : nk
    all(>=(1), nks) || throw(DomainError(nks, "Every axis needs at least one k-point."))
    out = SVector{D,Float64}[]
    for idx in Iterators.product(ntuple(d -> 0:(nks[d] - 1), D)...)
        push!(out, klo + SVector{D,Float64}(ntuple(d ->
            (idx[d] + shift) / nks[d] * (khi[d] - klo[d]), D)))
    end
    return out
end
