##################################################################
###### export only the following functions from this module ######
##################################################################

export BZMesh, bz_mesh, optimal_bz_mesh
export k_index, grid_coordinates, k_fractional, q_fractional, kshift, momentum_conservation




#############################################################
###### The mesh: a twisted uniform Brillouin-zone grid  #####
#############################################################

"""
    BZMesh{D}

A uniform, possibly twisted, discretization of the Brillouin zone -- the momentum content of a
finite cluster `L = M·a` with twisted boundary conditions. Build it with [`bz_mesh`](@ref).

The momenta, before wrapping, are

    k(n) = Uᵀ ((n .+ twist) ./ N),        nₐ ∈ 0 … Nₐ-1,

in the model lattice's OWN fractional frame -- `U` only selects which momenta are sampled, so
no consumer ever sees a "twisted frame" and no reciprocal label ever needs re-indexing. A
boundary flux `φ` (in units of 2π) on a cluster axis is a rigid fractional shift of the whole
grid; it never touches the model, because [`bloch_hamiltonian!`](@ref) accepts any fractional
momentum.

Fields:
-------
- `lattice::BravaisLattice{D}`: the UNIT-CELL lattice whose Brillouin zone is discretized (not
  the cluster). A mesh is pure momentum geometry -- one mesh serves every model on this lattice;
  the drivers check the pairing.
- `N::NTuple{D,Int}`: grid points per axis; `length(mesh) == prod(N)`.
- `U::SMatrix{D,D,Int}`: unimodular (|det| = 1) axis mix -- the cluster-shape freedom beyond a
  diagonal grid. `I` for a plain Monkhorst-Pack-style grid.
- `twist::SVector{D,Float64}`: per-axis offset in GRID-STEP units, canonical in `[0, 1)` -- the
  twisted boundary condition. `0` (the default of `bz_mesh`) is the Γ-containing grid.
- `kpts::Vector{SVector{D,Float64}}`: the momenta, wrapped into `[0,1)^D` (aperiodic axes
  carry 0), in K-ORDER: linear index 1-based, FIRST grid axis fastest (`LinearIndices(N)`).
- `g0::Vector{SVector{D,Int}}`: the exact integer each raw momentum was reduced by,
  `k_raw(n) = kpts[i] + g0[i]`. Derived FROM the canonicalized `kpts` (never an independent
  `floor`), which is what keeps [`kshift`](@ref)'s reciprocal carries exact at the zone
  boundary.

Two invariants every consumer may rely on:

- **k-differences are twist-independent.** `k_raw(n + dn) - k_raw(n) = Uᵀ(dn ./ N)` carries no
  `twist`, so [`q_fractional`](@ref) is bit-identical across twists, and so is the destination
  INDEX of [`kshift`](@ref) / [`momentum_conservation`](@ref) (pure index arithmetic modulo
  `N`) -- only the k's move. Their reciprocal CARRY `G` does NOT inherit this once `U ≠ I`: it
  is a difference of stored `g0`, which records which side of the zone boundary each raw
  momentum landed on, and a sheared grid has momenta outside `[0,1)^D` that a twist slides
  across that boundary. With `U = I` no raw momentum ever leaves the box, `g0 ≡ 0` identically,
  and the carries are twist-independent -- which is why the plain-grid intuition holds there
  and only there. A flux-averaging loop may therefore reuse the q-lattice and the index tables
  across twists, but must recompute `G` (and, of course, the eigenvectors) at each one.
- **Fundamental domain, no seam.** Each momentum appears once (no duplicated zone boundary);
  relating a wrapped `k + q` back to a stored point is the consumer's job, through the exact
  integer carry returned by `kshift` and, on `:exact` models, [`wrap_unitary`](@ref).
The type's second parameter is StaticArrays' length `L = D²`: `SMatrix{D,D,Int}` alone is an
incomplete (abstract) field type, and every access through it would box -- the index
arithmetic here is a hot-loop primitive and must stay allocation-free.
"""
struct BZMesh{D,L}
    lattice::BravaisLattice{D}
    N::NTuple{D,Int}
    U::SMatrix{D,D,Int,L}
    twist::SVector{D,Float64}
    kpts::Vector{SVector{D,Float64}}
    g0::Vector{SVector{D,Int}}
end

Base.length(mesh::BZMesh) = prod(mesh.N)
Base.size(mesh::BZMesh) = mesh.N
Base.eachindex(mesh::BZMesh) = Base.OneTo(length(mesh))

### identity is the generator data -- `kpts`/`g0` are derived from it deterministically
Base.:(==)(m1::BZMesh{D1}, m2::BZMesh{D2}) where {D1,D2} =
    D1 == D2 && m1.lattice == m2.lattice && m1.N == m2.N && m1.U == m2.U && m1.twist == m2.twist
Base.hash(m::BZMesh{D}, h::UInt) where {D} =
    hash(:BZMesh, hash(D, hash(m.lattice, hash(m.N, hash(m.U, hash(m.twist, h))))))

function Base.show(io::IO, mesh::BZMesh{D}) where {D}
    print(io, "BZMesh{", D, "} ", join(mesh.N, "×"), " (", length(mesh), " k-points)")
    mesh.U == one(SMatrix{D,D,Int}) || print(io, ", U = ", Tuple.(eachrow(mesh.U)))
    all(iszero, mesh.twist) || print(io, ", twist = ", Tuple(mesh.twist))
    return nothing
end




#############################################################
###### Constructors                                     #####
#############################################################

"""
    bz_mesh(lattice, N; twist=0.0, U=I)              -> BZMesh{D}
    bz_mesh(lattice, M::AbstractMatrix{<:Integer}; fluxes=zeros(D)) -> BZMesh{D}
    bz_mesh(lattice, cluster::BravaisLattice; fluxes=zeros(D))      -> BZMesh{D}
    bz_mesh(model, args...; kwargs...)               -> BZMesh{D}

A uniform (twisted) Brillouin-zone mesh -- see [`BZMesh`](@ref) for the momentum formula and
the invariants.

Args:
-----
- `lattice`: the unit-cell lattice (or a `model`, whose `cell.lattice` is taken).
- `N`: grid points per axis -- an `NTuple{D,Int}`, or one `Int` applied to every periodic axis.
- `M`: the integer supercell matrix of the CLUSTER, `L = M·a` -- "give the large lattice": its
  `|det M|` allowed momenta `k = M⁻¹(m + fluxes)` become the mesh. `cluster` may be given as an
  explicit `BravaisLattice` instead; the integer relation is derived (and refused if the
  cluster is not a supercell of `lattice`).
- `twist`: per-axis grid offset in GRID-STEP units (scalar or per-axis), canonicalized into
  `[0,1)`. The default 0 gives the Γ-containing grid -- deliberately NOT `monkhorst_pack`'s
  0.5 convention: here the offset IS the boundary condition, given per run.
- `U`: unimodular axis mix (see `BZMesh.U`); identity by default.
- `fluxes`: boundary-condition twist angles in units of 2π, one per cluster axis --
  `fluxes = [1/3, 0]` means a wrapped electron picks up `e^{2πi/3}` around the first cluster
  vector. `k = M⁻¹(m + fluxes)` exactly; on a diagonal cluster `M = diag(N)` this is
  `twist = fluxes` per axis.

Returns:
--------
- `BZMesh{D}` with `prod(N) = |det M|` momenta.

Notes:
------
The `M` entry point reduces to `(N, U, twist)` exactly: if `M = diag(d)·V` with `V` unimodular
(each `d` the gcd of a row of `M`), then `N = d`, `U = V⁻ᵀ`, `twist = fluxes` -- the intuitive
grid, e.g. `M = diag(2,3)` gives a 2×3 mesh. Otherwise the exact unimodular diagonalization
`Λ = L·M·R` gives `N = diag(Λ)`, `U = Rᵀ`, `twist = mod.(L·fluxes, 1)` -- the same momentum
SET (a test pins it against `coset_representatives(M; space=:reciprocal)` at zero flux), on a
rectangular index grid.

Refusals, all named: non-invertible or non-square `M`; non-unimodular `U`; `N < 1`; on a
partially periodic lattice, anything but `N = 1`, `twist = 0`, blockdiagonal `U`/`M` on the
aperiodic axes (mirroring [`supercell`](@ref)'s guard -- an aperiodic direction has no momentum
to discretize, and collapsing silently would corrupt every downstream index).

Example:
--------
    lat  = bravais_lattice(:triangular; a=1.0)
    mesh = bz_mesh(lat, (6, 6); twist=[0.1, 0.0])   # 36 momenta, flux 2π/10 on the first cycle
    mesh = bz_mesh(lat, [6 3; 0 6])                 # a sheared 36-site cluster
"""
function bz_mesh(lattice::BravaisLattice{D}, N::NTuple{D,Int};
                 twist::Union{Real,AbstractVector{<:Real}}=0.0,
                 U::AbstractMatrix{<:Integer}=one(SMatrix{D,D,Int})) where {D}
    all(N .>= 1) || throw(ArgumentError(
        "`bz_mesh`: every grid dimension must be at least 1; got N = $N."))
    size(U) == (D, D) || throw(DimensionMismatch(
        "`bz_mesh`: `U` must be $D×$D; got size $(size(U))."))
    Us = SMatrix{D,D,Int}(U)
    _int_det_abs(Us) == 1 || throw(ArgumentError(
        "`bz_mesh`: `U` must be unimodular (|det U| = 1); got det = " *
        "$(_int_det(Us)). A non-unimodular axis mix changes the " *
        "momentum content -- pass it as the cluster matrix `M` instead."))
    tw = twist isa Real ? SVector{D,Float64}(ntuple(_ -> Float64(twist), Val(D))) :
         (length(twist) == D ? SVector{D,Float64}(twist) : throw(DimensionMismatch(
            "`bz_mesh`: `twist` must have one entry per axis ($D); got $(length(twist)).")))
    tw = wrap01(tw)
    P = nperiodic(lattice)
    if P < D
        for m in (P + 1):D
            N[m] == 1 || throw(ArgumentError(
                "`bz_mesh`: this lattice is periodic only along its first $P " *
                "direction$(P == 1 ? "" : "s"); direction $m has no momentum to discretize, " *
                "so its grid dimension must be 1 (got $(N[m]))."))
            tw[m] == 0.0 || throw(ArgumentError(
                "`bz_mesh`: a twist along the aperiodic direction $m is not a boundary " *
                "condition -- there is no cycle to thread flux through. Set it to 0."))
        end
        (all(iszero, Us[1:P, (P + 1):D]) && all(iszero, Us[(P + 1):D, 1:P]) &&
         Us[(P + 1):D, (P + 1):D] == one(SMatrix{D - P,D - P,Int})) || throw(ArgumentError(
            "`bz_mesh`: this lattice is periodic only along its first $P " *
            "direction$(P == 1 ? "" : "s"); `U` must be blockdiag(U_P, I) on it. Got U = $Us."))
    end
    kpts, g0 = _mesh_points(lattice, N, Us, tw)
    return BZMesh(lattice, N, Us, tw, kpts, g0)
end

function bz_mesh(lattice::BravaisLattice{D}, N::Integer; kwargs...) where {D}
    P = nperiodic(lattice)
    return bz_mesh(lattice, ntuple(m -> m <= P ? Int(N) : 1, Val(D)); kwargs...)
end

function bz_mesh(lattice::BravaisLattice{D}, M::AbstractMatrix{<:Integer};
                 fluxes::AbstractVector{<:Real}=zeros(D)) where {D}
    size(M) == (D, D) || throw(DimensionMismatch(
        "`bz_mesh`: the cluster matrix `M` must be $D×$D; got size $(size(M))."))
    length(fluxes) == D || throw(DimensionMismatch(
        "`bz_mesh`: `fluxes` must have one entry per axis ($D); got $(length(fluxes))."))
    Ms = SMatrix{D,D,Int}(M)
    _int_det_abs(Ms) == 0 && throw(ArgumentError(
        "`bz_mesh`: the cluster matrix must be invertible; got det = 0 for M = $Ms."))
    P = nperiodic(lattice)
    if P < D
        (all(iszero, Ms[1:P, (P + 1):D]) && all(iszero, Ms[(P + 1):D, 1:P]) &&
         Ms[(P + 1):D, (P + 1):D] == one(SMatrix{D - P,D - P,Int})) || throw(ArgumentError(
            "`bz_mesh`: this lattice is periodic only along its first $P " *
            "direction$(P == 1 ? "" : "s"); a cluster matrix must be blockdiag(M_P, I) on it. " *
            "Got M = $Ms."))
        for m in (P + 1):D
            fluxes[m] == 0 || throw(ArgumentError(
                "`bz_mesh`: a flux along the aperiodic direction $m is not a boundary " *
                "condition -- there is no cycle to thread it through. Set it to 0."))
        end
    end
    phi = SVector{D,Float64}(fluxes)
    ### fast path M = diag(d)·V with V unimodular: the intuitive grid (diag(2,3) -> a 2×3 mesh).
    ### k = M⁻¹(m + φ) = V⁻¹((m + φ) ./ d), so N = d, Uᵀ = V⁻¹, twist = φ.
    d = ntuple(i -> reduce(gcd, Tuple(Ms[i, :])), Val(D))
    V = Ms .÷ SVector{D,Int}(d)                        # exact: dᵢ divides row i by construction
    if _int_det_abs(V) == 1
        Vinv = _int_inv(V)                                 # exact: adj(V)/det, |det| = 1 here
        return bz_mesh(lattice, d; twist=phi, U=transpose(Vinv))
    end
    ### general path, exact: Λ = L·M·R (unimodular L, R), so k = M⁻¹(m + φ) = R((m' + Lφ)./λ)
    ### with m' = L·m ranging over ℤᴰ -- a rectangular diag(Λ) grid sheared by U = Rᵀ and
    ### shifted by twist = L·φ. Same momentum set, exact by construction.
    L, Lam, R = _unimodular_diagonalize(Ms; nblock=P)
    return bz_mesh(lattice, ntuple(i -> Lam[i, i], Val(D));
                   twist=SVector{D,Float64}(L * phi), U=transpose(R))
end

function bz_mesh(lattice::BravaisLattice{D}, cluster::BravaisLattice{D};
                 fluxes::AbstractVector{<:Real}=zeros(D)) where {D}
    M = _supercell_integer_matrix(Matrix(cluster.a), Matrix(lattice.a))
    return bz_mesh(lattice, M; fluxes=fluxes)
end

bz_mesh(model::HamiltonianModel{D}, args...; kwargs...) where {D} =
    bz_mesh(model.cell.lattice, args...; kwargs...)

"""
    _mesh_points(lattice, N, U, twist) -> (kpts, g0)

Materialize the wrapped momenta and their exact integer reduction offsets, in K-ORDER (first
grid axis fastest). `g0` is derived FROM the canonicalized point -- `g0 = k_raw - kpts`, never
an independent `floor` -- so a raw momentum within rounding of the zone boundary cannot end up
with a wrap and an offset that disagree by ±1.
"""
function _mesh_points(lattice::BravaisLattice{D}, N::NTuple{D,Int}, U::SMatrix{D,D,Int},
                      twist::SVector{D,Float64}) where {D}
    P = nperiodic(lattice)
    Ut = transpose(U)
    Nv = SVector{D,Float64}(N)
    kpts = Vector{SVector{D,Float64}}(undef, prod(N))
    g0 = Vector{SVector{D,Int}}(undef, prod(N))
    idx = 1
    for n in Iterators.product(ntuple(a -> 0:(N[a] - 1), Val(D))...)
        raw = SVector{D,Float64}(Ut * ((SVector{D,Float64}(n) + twist) ./ Nv))
        kp = canonicalize_fracpos(raw; nperiodic=P)
        kpts[idx] = kp
        g0[idx] = canonicalize_int(raw - kp)
        idx += 1
    end
    return kpts, g0
end




#############################################################
###### Index arithmetic and the exact k+q wrap          #####
#############################################################

@inline _linear_index(N::NTuple{D,Int}, n::SVector{D,Int}) where {D} =
    LinearIndices(N)[Tuple(n .+ 1)...]

"""
    k_index(mesh, n) -> Int

The 1-based linear index of the grid point with 0-based grid coordinates `n` (first axis
fastest -- `LinearIndices` order).

Out-of-range coordinates are REFUSED, never silently wrapped: a wrap is a physical statement
(it carries a reciprocal vector), and the place that makes it is [`kshift`](@ref), which
returns the carry explicitly.
"""
function k_index(mesh::BZMesh{D}, n::AbstractVector{<:Integer})::Int where {D}
    length(n) == D || throw(DimensionMismatch(
        "`k_index`: grid coordinates must have length $D; got $(length(n))."))
    nv = SVector{D,Int}(n)
    all(0 .<= nv .< SVector{D,Int}(mesh.N)) || throw(ArgumentError(
        "`k_index`: grid coordinates $(Tuple(nv)) are outside the mesh " *
        "(0-based, bounds $(mesh.N)). Out-of-range coordinates are not wrapped silently -- " *
        "a wrap carries a reciprocal vector; use `kshift` to get both."))
    return _linear_index(mesh.N, nv)
end
k_index(mesh::BZMesh{D}, n::Tuple) where {D} = k_index(mesh, SVector{D,Int}(n))

"""
    grid_coordinates(mesh, i) -> SVector{D,Int}

The 0-based grid coordinates of linear k-index `i` -- the inverse of [`k_index`](@ref).
"""
function grid_coordinates(mesh::BZMesh{D}, i::Integer)::SVector{D,Int} where {D}
    1 <= i <= length(mesh) || throw(ArgumentError(
        "`grid_coordinates`: k-index $i is outside 1:$(length(mesh))."))
    return SVector{D,Int}(Tuple(CartesianIndices(mesh.N)[i])) .- 1
end

"""
    k_fractional(mesh, i) -> SVector{D,Float64}

The fractional momentum of k-index `i`, wrapped into `[0,1)^D` -- `mesh.kpts[i]` with a bounds
check.
"""
function k_fractional(mesh::BZMesh{D}, i::Integer)::SVector{D,Float64} where {D}
    1 <= i <= length(mesh) || throw(ArgumentError(
        "`k_fractional`: k-index $i is outside 1:$(length(mesh))."))
    return mesh.kpts[i]
end

"""
    _q_carry(mesh, dn) -> (q, gq0)

The wrapped momentum transfer of a grid-step difference `dn`, and the exact integer it was
reduced by: `Uᵀ(dn ./ N) = q + gq0`. Exact over the common denominator `lcm(N)` (integer
numerators), with `gq0` derived FROM the canonicalized `q` -- the same discipline as
`_mesh_points`, and for the same boundary-rounding reason.
"""
@inline function _q_carry(mesh::BZMesh{D}, dn::SVector{D,Int}) where {D}
    P = nperiodic(mesh.lattice)
    if P < D
        for m in (P + 1):D
            dn[m] == 0 || throw(ArgumentError(
                "grid-step difference $(Tuple(dn)) is non-zero along the aperiodic " *
                "direction $m, where there is no momentum. Set it to 0."))
        end
    end
    Nv = SVector{D,Int}(mesh.N)
    ### unrolled lcm: `reduce(lcm, ::NTuple)` takes an allocating generic path
    L = 1
    for Na in mesh.N
        L = lcm(L, Na)
    end
    p = transpose(mesh.U) * (dn .* (L .÷ Nv))
    raw = SVector{D,Float64}(p) / L
    q = canonicalize_fracpos(raw; nperiodic=P)
    return q, canonicalize_int(raw - q)
end

"""
    q_fractional(mesh, dn) -> SVector{D,Float64}

The wrapped momentum transfer between two grid points separated by `dn` grid steps, in
`[0,1)^D`.

TWIST-INDEPENDENT, bit-exactly: `k_raw(n + dn) - k_raw(n) = Uᵀ(dn ./ N)` carries no twist, so
the q-lattice of a twisted mesh is identical to the untwisted one and a flux loop may index its
transfers by `dn` throughout. (A test pins the bit-equality, on a sheared mesh too.) The
reciprocal CARRIES of [`kshift`](@ref) / [`momentum_conservation`](@ref) do not inherit this
once `U ≠ I` -- see [`BZMesh`](@ref) for why.
"""
q_fractional(mesh::BZMesh{D}, dn::AbstractVector{<:Integer}) where {D} =
    _q_carry(mesh, SVector{D,Int}(dn))[1]

"""
    kshift(mesh, i, dn) -> (index::Int, G::SVector{D,Int})
    kshift(mesh, i, iq::Integer)

Shift k-point `i` by `dn` grid steps (or by the grid coordinates of k-point `iq`), returning
the wrapped destination index and the EXACT reciprocal carry `G`:

    k_fractional(mesh, index) + G == k_fractional(mesh, i) + q_fractional(mesh, dn)

with `G` exact integers and the float identity holding to `2 * LATTICE_ATOL`. This is the one
sanctioned way to leave the zone: the carry is what a consumer hands to
[`wrap_unitary`](@ref) (`:exact` models) or accounts for explicitly (`:truncated` models) --
never a silent `mod`.

Notes:
------
All-integer and allocation-free: with `n` the coordinates of `i` and `s = n + dn`, the wrap is
`sw = mod.(s, N)`, `o = fld.(s, N)` (an index wrap by `Nₐ` along axis `a` shifts the raw
momentum by the exact reciprocal vector `Uᵀeₐ`), and

    G = g0[index] + Uᵀ·o - g0[i] - gq0(dn)

is a sum of stored exact integers. In the `U = I`, `twist = 0` limit with `dn` in the grid box
this reduces to the textbook `G = fld.(n + dn, N) ∈ {0,1}ᴰ`.
"""
function kshift(mesh::BZMesh{D}, i::Integer, dn::AbstractVector{<:Integer}) where {D}
    length(dn) == D || throw(DimensionMismatch(
        "`kshift`: the grid-step difference must have length $D; got $(length(dn))."))
    dnv = SVector{D,Int}(dn)
    _, gq0 = _q_carry(mesh, dnv)
    return _kshift_at(mesh, Int(i), dnv, gq0)
end

"""
    _kshift_at(mesh, i, dn, gq0) -> (index, G)

[`kshift`](@ref)'s per-k arithmetic, with the per-`dn` piece `gq0` supplied by the caller.

`gq0` is `_q_carry(mesh, dn)[2]`, which depends on `dn` ALONE -- an lcm reduction over the
grid shape, a `Uᵀ` matrix-vector product and a canonicalization. A driver that sweeps every
k-point at a fixed transfer (`form_factors`, and every consumer built on it) recomputed all of
that once per (k, q) pair through the public entry point; hoisting it to once per q is what
this split is for. `kshift` itself is this function with `gq0` computed on the spot, so there
is ONE implementation of the wrap, not two that can drift.
"""
@inline function _kshift_at(mesh::BZMesh{D}, i::Int, dn::SVector{D,Int},
                            gq0::SVector{D,Int}) where {D}
    n = grid_coordinates(mesh, i)                       # bounds-checks i
    Nv = SVector{D,Int}(mesh.N)
    s = n + dn
    sw = mod.(s, Nv)
    o = fld.(s, Nv)
    iw = _linear_index(mesh.N, sw)
    G = mesh.g0[iw] + transpose(mesh.U) * o - mesh.g0[i] - gq0
    return (index=iw, G=G)
end
kshift(mesh::BZMesh{D}, i::Integer, iq::Integer) where {D} =
    kshift(mesh, i, grid_coordinates(mesh, iq))

"""
    momentum_conservation(mesh, i1, i2, i3) -> (index::Int, G::SVector{D,Int})

The fourth momentum of a two-body vertex, `k₄ ≡ k₁ + k₂ - k₃` folded onto the mesh, with the
EXACT umklapp carry:

    k_fractional(i1) + k_fractional(i2) - k_fractional(i3) - k_fractional(index) == G

(`G` exact integers; float identity to `2 * LATTICE_ATOL`). The twist enters the sum exactly
once (`+k₁ +k₂ -k₃` carries one net `+twist`), so it lands on `k₄` and the identity above holds
exactly at every twist. The carry itself is a difference of the stored `g0`, so it is
twist-independent only where those are -- on a `U = I` mesh. On a sheared mesh a twist can slide
a momentum across the zone boundary and move `G` by a reciprocal vector; see [`BZMesh`](@ref).
"""
function momentum_conservation(mesh::BZMesh{D}, i1::Integer, i2::Integer,
                               i3::Integer) where {D}
    s = grid_coordinates(mesh, i1) + grid_coordinates(mesh, i2) - grid_coordinates(mesh, i3)
    Nv = SVector{D,Int}(mesh.N)
    sw = mod.(s, Nv)
    o = fld.(s, Nv)
    i4 = _linear_index(mesh.N, sw)
    G = mesh.g0[i4] + transpose(mesh.U) * o - mesh.g0[i1] - mesh.g0[i2] + mesh.g0[i3]
    return (index=i4, G=G)
end




#############################################################
###### Optimal cluster shapes                           #####
#############################################################

"""
    optimal_bz_mesh(lattice, nk_total; fluxes=zeros(D)) -> BZMesh{D}

The most isotropic `nk_total`-point cluster of this lattice: enumerate every distinct cluster
of that size (one Hermite-normal-form matrix per sublattice of index `nk_total`), reduce each
cluster basis, and keep the one whose SHORTEST cluster vector is longest -- the length that
controls the finite-size gap of a many-body calculation on the cluster. Ties fall to the
SMALLEST next length (fixed volume and λ₁, a shorter λ₂ is the more orthogonal, more isotropic
cluster -- the aspect-ratio criterion), then to enumeration order (deterministic).

Args:
-----
- `nk_total`: the number of k-points (= cluster cells = |det M|).
- `fluxes`: boundary twist angles in units of 2π, forwarded to [`bz_mesh`](@ref).

Notes:
------
The search is exact and dependency-free: sublattices of index `n` in `ℤᴰ` are in bijection
with lower-triangular Hermite forms of determinant `n` (`σ(n)` of them in 2D), and the basis
reduction is Semaev's greedy reduction (`_reduced_norms` below), which realizes the
successive minima for `D ≤ 3` -- a pairwise (Lagrange) reduction would not, and in 3D its
overestimates of `λ₁` are exactly what wins this argmax. Defined for fully periodic lattices;
on a partially periodic lattice there is no cluster shape to optimize -- pass `N` per periodic
axis to `bz_mesh` directly.

Example:
--------
    lat = bravais_lattice(:square; a=1.0)
    optimal_bz_mesh(lat, 16)          # the 4×4 cluster: shortest cluster vector 4a
"""
function optimal_bz_mesh(lattice::BravaisLattice{D}, nk_total::Integer;
                         fluxes::AbstractVector{<:Real}=zeros(D)) where {D}
    nk_total >= 1 || throw(ArgumentError(
        "`optimal_bz_mesh`: the cluster size must be positive; got $nk_total."))
    nperiodic(lattice) == D || throw(ArgumentError(
        "`optimal_bz_mesh`: the cluster-shape search is defined for fully periodic lattices; " *
        "this one is periodic along $(nperiodic(lattice)) of $D directions. Pass the grid " *
        "size per periodic axis to `bz_mesh` directly."))
    a = lattice.a
    best_M = nothing
    best = ntuple(_ -> -Inf, Val(D))
    for M in _hnf_candidates(Val(D), Int(nk_total))
        lens = _reduced_norms(SMatrix{D,D,Float64}(M) * a)
        ### lexicographic max: λ₁ first, then the NEGATED further lengths -- ties on the
        ### finite-size length fall to the most isotropic cluster, not the most skewed one
        score = ntuple(j -> j == 1 ? lens[1] : -lens[j], Val(D))
        if score > best
            best = score
            best_M = M
        end
    end
    return bz_mesh(lattice, best_M::SMatrix{D,D,Int}; fluxes=fluxes)
end

"""
    _hnf_candidates(Val(D), n) -> Vector{SMatrix{D,D,Int}}

One lower-triangular Hermite-form representative per sublattice of index `n` in `ℤᴰ`:
positive diagonal with product `n`, sub-diagonal entries reduced modulo the diagonal entry of
their column. Deterministic enumeration order (the tie-break of `optimal_bz_mesh`).
"""
function _hnf_candidates(::Val{1}, n::Int)
    return [SMatrix{1,1,Int}(n)]
end
function _hnf_candidates(::Val{2}, n::Int)
    out = SMatrix{2,2,Int}[]
    for a in 1:n
        n % a == 0 || continue
        d = n ÷ a
        for c in 0:(a - 1)
            push!(out, SMatrix{2,2,Int}(a, c, 0, d))    # rows (a, 0), (c, d)
        end
    end
    return out
end
function _hnf_candidates(::Val{3}, n::Int)
    out = SMatrix{3,3,Int}[]
    for a in 1:n
        n % a == 0 || continue
        for c in 1:(n ÷ a)
            (n ÷ a) % c == 0 || continue
            f = n ÷ (a * c)
            for b in 0:(a - 1), d in 0:(a - 1), e in 0:(c - 1)
                push!(out, SMatrix{3,3,Int}(a, b, d, 0, c, e, 0, 0, f))
            end
        end
    end
    return out
end
_hnf_candidates(::Val{D}, n::Int) where {D} = throw(ArgumentError(
    "`optimal_bz_mesh` enumerates cluster shapes for D ≤ 3; got D = $D."))

### The lattice reduction `_reduced_norms` runs on lives in `core/unit_cell/reduction.jl`: it
### is metric geometry, not mesh code, and the moire coincidence search and the Bravais
### classifier need the same routine.



#############################################################
###### Small-vector arithmetic of a grid step           #####
#############################################################

### A grid step `dn` moves the fractional momentum by `U'(dn ./ N)`, not by `dn ./ N`: the mesh
### is `k(n) = U'((n + twist) ./ N)`, so the axis-mixing matrix rides along. It is the identity
### on a plain Monkhorst-Pack grid, which is why dropping it only shows up on a mesh built from
### a cluster matrix -- silently, with the wrong `b`.
@inline function _b_fractional(mesh::BZMesh{D}, dn::AbstractVector{<:Integer}) where {D}
    step = SVector{D,Float64}(ntuple(d -> dn[d] / mesh.N[d], D))
    return SVector{D,Float64}(transpose(mesh.U) * step)
end

### The umklapp between a grid step's SMALL vector and its stored transfer. `q_fractional`
### wraps into the zone, so `q = b + m` for an integer reciprocal vector `m`, nonzero as soon
### as `dn` has mixed signs -- the hexagonal `(1, -1)` step, say.
###
### The distinction is not cosmetic: a link at the wrapped transfer puts the ket at `k + q`,
### while the small vector puts it at `k + b`, and those two eigenvectors differ by the
### zone-boundary relabelling `W(m)`. The axis steps `±e_a` all have `m = 0`, so a Chern number
### or a Wilson loop never notices; a Wannier90 stencil, which needs the diagonal steps too,
### does -- and reading the ket at the wrong momentum there produces a deck that runs and whose
### spread is wrong.
@inline function _shift_umklapp(mesh::BZMesh{D}, dn::AbstractVector{<:Integer}) where {D}
    d = q_fractional(mesh, dn) .- _b_fractional(mesh, dn)
    m = round.(Int, d)
    maximum(abs.(d .- m)) <= 2 * LATTICE_ATOL || throw(ErrorException(
        "`_shift_umklapp`: the wrapped transfer of $(Tuple(dn)) differs from its small " *
        "vector by the non-integer $(Tuple(d)). This is an internal mesh inconsistency."))
    return SVector{D,Int}(m)
end
