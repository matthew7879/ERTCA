##################################################################
###### export only the following functions from this module ######
##################################################################

export diagonalize, threaded_chunks, foreach_threaded




#############################################################
###### The master diagonalization entry point           #####
#############################################################

### Below this size a dense solve beats building a Krylov space regardless of `nev`: threaded
### LAPACK diagonalizes a 2048×2048 Hermitian matrix in about a second, and the iterative paths
### only win once the matrix-vector product is the sole affordable operation -- which for the
### dense Bloch Hamiltonians this package builds means appreciably larger than this.
const _DENSE_CUTOFF::Int = 2048

"""
    diagonalize(H; method=:auto, nev=nothing, which=:smallest, sigma=nothing,
                vectors=true, tol=1e-12, maxiter=300, krylovdim=nothing,
                hermitian_atol=HERMITICITY_ATOL, check=true)
        -> (values::Vector{Float64}, vectors::Union{Nothing,Matrix{ComplexF64}})

Diagonalize a Hermitian matrix, choosing the right algorithm for the request. This is the ONE
entry point every analysis routine goes through, so the algorithm choice, the Hermiticity check
and the ordering convention live in a single place.

Methods:
--------
- `:dense`        full spectrum by LAPACK (`eigen(Hermitian(...))`). Multithreaded through BLAS.
- `:lanczos`      `nev` extremal eigenpairs by a Krylov iteration (KrylovKit.jl), never forming
                  more than matrix-vector products -- the method for large sparse or
                  structured `H` when only band edges are needed.
- `:shift_invert` `nev` eigenpairs CLOSEST to the shift `sigma`: Lanczos on `(H - σI)⁻¹` via one
                  LU factorization, mapped back through `λ = σ + 1/θ`. The method for interior
                  bands -- e.g. the flat bands of a moiré model, which sit mid-spectrum where
                  plain Lanczos converges last.
- `:auto`         `:dense` when the full spectrum is wanted or the matrix is small
                  (n ≤ $(_DENSE_CUTOFF)); `:shift_invert` when `sigma` is given; `:lanczos`
                  otherwise.

Args:
-----
- `H`: Hermitian `AbstractMatrix` (dense `Matrix`, `Hermitian`, or a sparse matrix -- anything
  supporting `*` for `:lanczos` and `lu` for `:shift_invert`)
- `nev`: how many eigenpairs (`nothing` = all; forces a dense solve)
- `which`: `:smallest` or `:largest` spectrum end for `nev` (ignored when `sigma` is given)
- `sigma`: target energy for `:shift_invert` -- "the `nev` states nearest this energy"
- `vectors`: also return eigenvectors (`false` saves the back-transform and the storage)
- `overwrite`: allow the dense path to DESTROY `H` (LAPACK works in place) instead of copying
  it first. This is what lets a k-loop reuse one buffer with zero per-point matrix allocations:
  refill, diagonalize with `overwrite=true`, repeat. With `vectors=true` the returned vectors
  may alias the buffer -- copy out what must survive the next refill.
- `tol`, `maxiter`, `krylovdim`: Krylov iteration controls (ignored by `:dense`)
- `check`, `hermitian_atol`: verify Hermiticity before trusting one triangle (see Notes)

Returns:
--------
- `values::Vector{Float64}`: ascending, always -- also for `:shift_invert`, whose natural order
  (by distance to `sigma`) is re-sorted so every caller sees one convention
- `vectors`: `n × length(values)` matrix, column `i` belonging to `values[i]`, or `nothing`

Multithreading:
---------------
The dense path is multithreaded by BLAS/LAPACK -- check `LinearAlgebra.BLAS.get_num_threads()`.
Batch drivers that call this once per k-point (`spectrum_along_line`, `density_of_states`,
`berry_curvature`) additionally parallelize over k with `Threads.@threads`; they use
[`threaded_chunks`](@ref), which rebalances BLAS threads so the two levels do not oversubscribe
each other. Start Julia with `-t auto` (or set `JULIA_NUM_THREADS`) to enable the k-level.

Notes:
------
Hermiticity is verified BEFORE wrapping in `Hermitian`: the wrapper reads only one triangle, so
applying it blindly would silently symmetrize away a genuine asymmetry (the same policy as
`bands`). Pass `check=false` only when the caller has already verified it.

KrylovKit was chosen over Arpack for the iterative paths: pure Julia (no binary dependency to
break), works on any vector/matrix type, and its `:LM` mode on the inverted map is exactly the
shift-invert transform.
"""
function diagonalize(H::AbstractMatrix;
                     method::Symbol=:auto,
                     nev::Union{Nothing,Int}=nothing,
                     which::Symbol=:smallest,
                     sigma::Union{Nothing,Real}=nothing,
                     vectors::Bool=true,
                     overwrite::Bool=false,
                     tol::Float64=1e-12,
                     maxiter::Int=300,
                     krylovdim::Union{Nothing,Int}=nothing,
                     hermitian_atol::Float64=HERMITICITY_ATOL,
                     check::Bool=true)
    n = size(H, 1)
    size(H, 2) == n || throw(DimensionMismatch("`diagonalize` needs a square matrix; got $(size(H))."))
    which in (:smallest, :largest) || throw(ArgumentError(
        "`which` must be `:smallest` or `:largest`; got `$which`. Use `sigma=` to target an " *
        "interior energy."))
    isnothing(nev) || 1 <= nev <= n || throw(ArgumentError(
        "`nev = $nev` is outside 1:$n."))
    check && _assert_hermitian(H, hermitian_atol)

    meth = method === :auto ? _auto_method(n, nev, sigma) : method
    if meth === :dense
        return _diag_dense(H, nev, which, sigma, vectors, overwrite)
    elseif meth === :lanczos
        isnothing(nev) && throw(ArgumentError(
            "`method = :lanczos` computes a partial spectrum; give `nev` (or use `:dense`)."))
        ### `sigma` was accepted and then ignored, so a caller asking for interior states got
        ### the extremal ones instead -- silently, and only on this branch. `:shift_invert`
        ### refuses the mirror-image contradiction (a shift with no `sigma`), so refuse here too.
        isnothing(sigma) || throw(ArgumentError(
            "`method = :lanczos` computes an EXTREMAL partial spectrum and cannot target " *
            "`sigma = $sigma`; it would ignore the shift and return the extremal states. Use " *
            "`method = :shift_invert` (or `:auto`, which picks it when `sigma` is given) for " *
            "interior states, or drop `sigma`."))
        return _diag_lanczos(H, nev, which, vectors; tol=tol, maxiter=maxiter, krylovdim=krylovdim)
    elseif meth === :shift_invert
        isnothing(sigma) && throw(ArgumentError(
            "`method = :shift_invert` targets the spectrum near a shift; give `sigma`."))
        isnothing(nev) && throw(ArgumentError(
            "`method = :shift_invert` computes a partial spectrum; give `nev`."))
        return _diag_shift_invert(H, nev, Float64(sigma), vectors;
                                  tol=tol, maxiter=maxiter, krylovdim=krylovdim)
    else
        throw(ArgumentError(
            "Unknown method `$meth`; expected `:auto`, `:dense`, `:lanczos` or `:shift_invert`."))
    end
end

function _auto_method(n::Int, nev::Union{Nothing,Int}, sigma::Union{Nothing,Real})::Symbol
    isnothing(nev) && return :dense                    # full spectrum has exactly one algorithm
    n <= _DENSE_CUTOFF && return :dense                # small: dense wins even for one eigenvalue
    isnothing(sigma) ? :lanczos : :shift_invert
end

"""
    _assert_hermitian(H, atol) -> Nothing

The allocation-free upper-triangle sweep of `model.jl`'s `_check_hermitian`, generalized to any
matrix type this module accepts. Kept separate because `diagonalize` must not depend on the
model layer.
"""
function _assert_hermitian(H::AbstractSparseMatrix, atol::Float64)::Nothing
    ### the dense triangle sweep below would call sparse getindex n²/2 times, and
    ### `maximum(abs, H - adjoint(H))` would MATERIALIZE a whole new sparse matrix per call --
    ### this runs once per k-point in the sparse batch drivers. Instead: every stored (i, j)
    ### is compared against the stored (j, i) read through the CSC column directly. A missing
    ### partner contributes |H[i,j]| (H[j,i] reads as an exact structural zero).
    S = SparseMatrixCSC(H)
    rows = rowvals(S)
    vals = nonzeros(S)
    dev = 0.0
    ### EVERY stored entry is judged against its transpose partner, read through the CSC index.
    ### Skipping `i > j` (as this once did) means an entry stored ONLY in the lower triangle is
    ### never visited from either side -- `S[j, i]` reads as a structural zero from the upper
    ### sweep that never happens -- so a matrix asymmetric only below the diagonal passed the
    ### check and was then silently symmetrized by `Hermitian(H)` into a different spectrum.
    @inbounds for j in axes(S, 2), idx in nzrange(S, j)
        i = rows[idx]
        d = i == j ? abs(imag(vals[idx])) : abs(vals[idx] - conj(S[j, i]))
        d > dev && (dev = d)
    end
    dev <= atol || throw(DomainError(dev,
        "`diagonalize` expects a Hermitian matrix: max |H - H'| = $dev > atol = $atol. " *
        "Pass `check=false` only if you know the asymmetry is float noise."))
    return nothing
end

function _assert_hermitian(H::AbstractMatrix, atol::Float64)::Nothing
    dev = 0.0
    n = size(H, 1)
    if H isa Hermitian
        return nothing                                  # true by type
    end
    @inbounds for j in 1:n
        d = abs(imag(H[j, j]))
        d > dev && (dev = d)
        for i in 1:(j - 1)
            d = abs(H[i, j] - conj(H[j, i]))
            d > dev && (dev = d)
        end
    end
    dev <= atol || throw(DomainError(dev,
        "`diagonalize` expects a Hermitian matrix: max |H - H'| = $dev > atol = $atol. " *
        "Pass `check=false` only if you know the asymmetry is float noise."))
    return nothing
end




#############################################################
###### The three algorithm backends                     #####
#############################################################

function _diag_dense(H::AbstractMatrix, nev::Union{Nothing,Int}, which::Symbol,
                     sigma::Union{Nothing,Real}, vectors::Bool, overwrite::Bool)
    ### By default `Matrix(H)` copies, so the caller's H survives and `eigen!` works on the copy.
    ### With `overwrite=true` and a plain dense complex matrix the copy is skipped -- the batch
    ### drivers rely on this to keep the per-k-point matrix allocation count at zero.
    Hd = (overwrite && H isa Matrix{ComplexF64}) ? Hermitian(H) :
                                                   Hermitian(Matrix{ComplexF64}(H))
    ### An END of the ascending spectrum is a CONTIGUOUS INDEX RANGE, which LAPACK's `heevr`
    ### takes natively (`range = 'I'`): it runs the same reduction but back-transforms only the
    ### `nev` wanted eigenvectors instead of all `n` and then discarding `n - nev` of them.
    ### That is the whole cost of an eigenvector solve beyond the tridiagonalization, and it is
    ### also `n²` of garbage per k-point that the batch drivers never look at. `sigma` picks a
    ### set that is contiguous in ENERGY, not in index, so it keeps the full solve below.
    irange = _dense_irange(size(Hd, 1), nev, which, sigma)
    if !isnothing(irange)
        vectors || return (values=eigvals!(Hd, irange), vectors=nothing)
        E = eigen!(Hd, irange)
        return (values=E.values, vectors=E.vectors)
    end
    if vectors
        E = eigen!(Hd)                                  # LAPACK, ascending, BLAS-threaded
        vals, vecs = E.values, E.vectors
    else
        vals, vecs = eigvals!(Hd), nothing
    end
    idx = _select_indices(vals, nev, which, sigma)
    isnothing(idx) && return (values=vals, vectors=vecs)
    return (values=vals[idx], vectors=isnothing(vecs) ? nothing : vecs[:, idx])
end

"""
    _dense_irange(n, nev, which, sigma) -> Union{Nothing,UnitRange{Int}}

The `1:nev` / `n-nev+1:n` index window a `nev` + `which` request names, or `nothing` when the
request is not an index range (`nev` absent, `nev >= n`, or a `sigma` window, which is
contiguous in energy but not in index). This is the range handed to LAPACK's index mode.
"""
@inline function _dense_irange(n::Int, nev::Union{Nothing,Int}, which::Symbol,
                               sigma::Union{Nothing,Real})::Union{Nothing,UnitRange{Int}}
    (isnothing(nev) || !isnothing(sigma) || nev >= n) && return nothing
    return which === :smallest ? (1:nev) : ((n - nev + 1):n)
end

### The dense path serves `nev`/`sigma` requests too (it is the `:auto` choice for small
### matrices), so the subset selection has to exist here as well: `which` slices an end of the
### ascending spectrum, `sigma` picks the `nev` values nearest the shift, re-sorted ascending.
function _select_indices(vals::Vector{<:Real}, nev::Union{Nothing,Int}, which::Symbol,
                         sigma::Union{Nothing,Real})
    isnothing(nev) && return nothing
    if !isnothing(sigma)
        return sort!(partialsortperm(abs.(vals .- sigma), 1:nev))
    end
    return which === :smallest ? (1:nev) : (length(vals) - nev + 1):length(vals)
end

function _diag_lanczos(H::AbstractMatrix, nev::Int, which::Symbol, vectors::Bool;
                       tol::Float64, maxiter::Int, krylovdim::Union{Nothing,Int})
    n = size(H, 1)
    kdim = isnothing(krylovdim) ? min(n, max(30, 2 * nev + 10)) : krylovdim
    selector = which === :smallest ? :SR : :LR
    blk = Block(_start_block(n, nev))
    alg = BlockLanczos(; krylovdim=min(n, max(kdim, 2 * nev + 10)), maxiter=maxiter, tol=tol)
    vals, vecs, info = eigsolve(H, blk, nev, selector, alg)
    _require_krylov_count(vals, nev, info, "Lanczos", "")
    return _package_krylov(vals, vecs, nev, vectors; transform=identity)
end

function _diag_shift_invert(H::AbstractMatrix, nev::Int, sigma::Float64, vectors::Bool;
                            tol::Float64, maxiter::Int, krylovdim::Union{Nothing,Int})
    n = size(H, 1)
    ### One factorization, reused for every Krylov step. `lu` covers both worlds: LAPACK for a
    ### dense matrix, UMFPACK for a sparse one -- and raises on a singular shift, i.e. `sigma`
    ### hitting an eigenvalue exactly, where the user should nudge the shift.
    F = lu(H - sigma * I)
    ### The eigenvalues of the map θ = 1/(λ - σ) are LARGEST for λ nearest σ, so shift-invert is
    ### Lanczos in `:LM` mode on the inverted map -- this transform is the whole method.
    invmap = x -> F \ x
    kdim = isnothing(krylovdim) ? min(n, max(30, 2 * nev + 10)) : krylovdim
    blk = Block(_start_block(n, nev))
    alg = BlockLanczos(; krylovdim=min(n, max(kdim, 2 * nev + 10)), maxiter=maxiter, tol=tol)
    thetas, vecs, info = eigsolve(invmap, blk, nev, :LM, alg)
    _require_krylov_count(thetas, nev, info, "Shift-invert", "")
    ### `lu` does NOT reliably raise on an exactly singular shift: UMFPACK happily factorizes
    ### on ~1e-16 pivots when `sigma` sits dead on an eigenvalue, the inverted map is then
    ### dominated by that null space, and eigsolve converges in θ-space to pairs that are
    ### GARBAGE in the original problem (spurious multiplicity, O(1) residuals) -- silently.
    ### The honest check is the back-transformed residual ‖Hv − λv‖ itself.
    ### The eigenvalue comes from the RAYLEIGH QUOTIENT of the returned vector, not from
    ### back-transforming θ. The transform σ + 1/θ inherits the solver's error in θ linearly,
    ### and near a shift deliberately placed close to the spectrum that showed as ~4e-7 absolute
    ### and would not improve with `tol`, `krylovdim` or `maxiter` -- it is a conditioning floor
    ### of the inverted map, not a convergence setting. The Rayleigh quotient is variationally
    ### accurate (its error is second order in the eigenvector residual), and on that same case
    ### lands at 3e-13. Vectors are always available here: `eigsolve` returns them whether or
    ### not the caller asked for them.
    lams = Float64[real(dot(v, H * v) / dot(v, v)) for v in vecs]
    for i in 1:min(nev, length(thetas))
        lam = lams[i]
        v = vecs[i]
        resid = norm(H * v - lam * v) / max(norm(v), eps())
        ### RELATIVE, and deliberately loose. The guard exists to catch `sigma` sitting ON an
        ### eigenvalue, where the inverted map is dominated by a numerical null space and the
        ### back-transformed residuals are O(1) relative. A healthy solve at a shift that is
        ### merely CLOSE to the spectrum -- `sigma = λ + 1e-4`, which `test_analysis.jl` uses on
        ### purpose -- is ill-conditioned and lands near 5e-8 relative. An absolute bar tuned to
        ### the single-vector solver's luck sat between those and started refusing the healthy
        ### case as soon as the block solver converged to a slightly different set. 1e-4 keeps
        ### ~3 orders of margin above the healthy regime and ~4 below the garbage one.
        resid <= SHIFT_INVERT_RESID_RTOL * max(1.0, abs(lam)) || throw(ArgumentError(
            "Shift-invert returned an eigenpair with residual ‖Hv − λv‖ = $resid at " *
            "λ = $lam: the shift sigma = $sigma (almost) coincides with an eigenvalue, so " *
            "(H − σI) is numerically singular and the inverted map is meaningless. Nudge " *
            "`sigma` off the spectrum (any offset large against the eigenvalue spacing works)."))
    end
    return _package_krylov(lams, vecs, nev, vectors; transform=identity)
end

### KrylovKit needs a dense start vector; a deterministic non-symmetric one keeps runs
### reproducible (Random is deliberately not a dependency) while avoiding the classic trap of a
### symmetric start that has zero overlap with an antisymmetric ground state.
_start_vector(n::Int) = ComplexF64[cispi(2 * i / max(n, 2) + i^2 / (3n)) for i in 1:n]

### A BLOCK of deterministic, orthonormal start vectors -- the reason the Krylov paths can see
### degeneracy at all.
###
### A Krylov space built from ONE start vector contains at most one direction from each
### degenerate eigenspace, so single-vector Lanczos returns one eigenvalue per DISTINCT level
### and calls it `nev`. On a spin- or valley-doubled model -- ordinary physics here, not a
### corner -- "the 2 smallest" came back as eigenvalues 1 and 3, and a Chern number of -2 came
### back 0, with `info.converged == nev` so nothing warned. A block of width `w` resolves
### multiplicity up to `w`, and the `nev` smallest can contain at most `nev` copies of one
### eigenvalue, so `w = nev` is exactly enough.
###
### The columns must be DENSE and generic, not Cartesian axes. A sparse start is not merely
### inelegant: a basis vector can have accidentally zero overlap with part of the degenerate
### eigenspace, and then the block resolves no more of it than a single vector did. (Tried;
### `nev = 2` on a spin-doubled model still returned eigenvalues 1 and 3.) These are the same
### deterministic phase ramp as `_start_vector`, decorrelated per column and orthonormalized --
### reproducible across machines and runs, with no dependence on `Random`.
function _start_block(n::Int, w::Int)
    w = clamp(w, 1, n)
    A = Matrix{ComplexF64}(undef, n, w)
    for c in 1:w, i in 1:n
        A[i, c] = cispi(2 * i * c / max(n, 2) + (i + c)^2 / (3n) + 0.37 * c)
    end
    Q = Matrix(qr(A).Q)                      # orthonormal columns, deterministic
    return [Vector{ComplexF64}(view(Q, :, c)) for c in 1:w]
end

"""
    _require_krylov_count(vals, nev, info, what, hint) -> Nothing

Refuse, by name, when a Krylov solve is about to return fewer than the `nev` eigenvalues it was
asked for. This used to be an `@warn`, which is the wrong policy for a partial spectrum: every
caller above treats the result as "the nev extremal states", and a short return silently becomes
a different band selection -- a Chern number over the wrong subspace, a form factor over the
wrong bands. The package refuses rather than warns everywhere else it matters, and this is one
of the places it matters most.
"""
function _require_krylov_count(vals, nev::Int, info, what::AbstractString, hint::AbstractString)::Nothing
    length(vals) >= nev || throw(ArgumentError(
        "$what returned $(length(vals)) of the $nev requested eigenvalues" *
        (isempty(hint) ? "" : " ($hint)") *
        ". A partial spectrum that is short is not the spectrum that was asked for -- every " *
        "caller here reads it as `nev` extremal states, so returning fewer would silently " *
        "change the band selection. Raise `maxiter`/`krylovdim`, loosen `tol`, or use " *
        "`matrix = :dense` for this size."))
    return nothing
end

function _package_krylov(raw_vals, raw_vecs, nev::Int, vectors::Bool; transform)
    m = min(nev, length(raw_vals))                     # KrylovKit may return extra converged pairs
    vals = Float64[real(transform(raw_vals[i])) for i in 1:m]
    order = sortperm(vals)
    vals = vals[order]
    vectors || return (values=vals, vectors=nothing)
    n = length(raw_vecs[1])
    vecs = Matrix{ComplexF64}(undef, n, m)
    for (c, i) in enumerate(order)
        vecs[:, c] = raw_vecs[i]
    end
    return (values=vals, vectors=vecs)
end




#############################################################
###### Threaded k-point batches                         #####
#############################################################

"""
    threaded_chunks(f, n) -> Nothing

Split `1:n` into one contiguous chunk per Julia thread and run `f(chunk::UnitRange)` on each,
in parallel.

Purpose:
--------
Every batch driver in the analysis layer -- a band structure, a DOS grid, a Berry-curvature
grid -- is one independent diagonalization per k-point, which is embarrassingly parallel. This
helper is the single place that parallelism lives, and it is public so a CUSTOM analysis (a
spin texture, a Wilson loop, an overlap map) can reuse it instead of re-deriving the pattern:

    threaded_chunks(nk) do chunk
        solve = bloch_solver(model)          # ONE buffer per chunk -- see `bloch_solver`
        for i in chunk
            r = solve(kpts[i]; nev=4)
            ...
        end
    end

It manages two footguns at once:

- **Oversubscription**: each iteration typically ends in LAPACK, which spawns its own BLAS
  threads; `T` Julia threads each spawning `B` BLAS threads floods the machine with `T×B`
  workers and runs SLOWER than serial. For the duration of the batch, BLAS is throttled to
  `max(1, B ÷ T)` threads and restored afterwards.
- **Per-point allocation**: handing `f` a whole chunk (instead of one index) lets it allocate
  its Hamiltonian buffer ONCE per chunk and refill it per k-point -- with `diagonalize(...;
  overwrite=true)` that brings the matrix allocations of a batch from 2 per k-point to 1 per
  thread. One buffer per chunk is also what makes the pattern race-free: chunks never share it.

With a single Julia thread this degenerates to one plain chunk and BLAS keeps its full thread
count -- the dense path is then still multithreaded, just at the LAPACK level.
"""
function threaded_chunks(f, n::Int)::Nothing
    nt = min(Threads.nthreads(), n)
    if nt <= 1
        n >= 1 && f(1:n)
        return nothing
    end
    ### contiguous, near-equal chunks; the first `rem` chunks carry one extra index
    len, rem = divrem(n, nt)
    bounds = cumsum(vcat(0, fill(len + 1, rem), fill(len, nt - rem)))
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(max(1, nb ÷ nt))
    try
        Threads.@threads for c in 1:nt
            f((bounds[c] + 1):bounds[c + 1])
        end
    catch err
        ### a refusal raised inside a task surfaces as CompositeException(TaskFailedException):
        ### programmatic handlers (and the surfaced TYPE) should not depend on the thread
        ### count, so the ORIGINAL exception is unwrapped and rethrown
        rethrow(_unwrap_threaded(err))
    finally
        BLAS.set_num_threads(nb)
    end
    return nothing
end

_unwrap_threaded(e) = e
_unwrap_threaded(e::TaskFailedException) = _unwrap_threaded(e.task.exception)
function _unwrap_threaded(e::CompositeException)
    isempty(e.exceptions) && return e
    return _unwrap_threaded(first(e.exceptions))
end

"""
    foreach_threaded(f, n) -> Nothing

Run `f(i)` for `i in 1:n` in parallel -- the per-index convenience over `threaded_chunks`,
for batches whose iterations carry no reusable buffer.
"""
foreach_threaded(f, n::Int)::Nothing = threaded_chunks(chunk -> foreach(f, chunk), n)
