# Paper-specific numerics routines.

const ROUND_SLACK_DECADES::Int = 2

const LATTICE_ROUND_DIGITS::Int = 14

const LATTICE_ATOL::Float64 = exp10(-LATTICE_ROUND_DIGITS + ROUND_SLACK_DECADES)

const INT_ROUND_DIGITS::Int = 13

const INT_ROUND_ATOL::Float64 = exp10(-INT_ROUND_DIGITS + ROUND_SLACK_DECADES)

const AMPLITUDE_ATOL::Float64 = 1e-12

const HERMITICITY_ATOL::Float64 = 1e-8

const LINDEP_RTOL::Float64 = 1e-10

const SHIFT_INVERT_RESID_RTOL::Float64 = 1e-4

const DISPLAY_DIGITS::Int = 6

const MAX_INTERNAL::Int = 256

const AUTO_SITE_PREFIX::String = "site"

const GAUGE_MODES = (:atomic, :periodic)

const DEFAULT_GAUGE::Symbol = :atomic

const BRAVAIS_RTOL::Float64 = 1e-8

const SHELL_RTOL::Float64 = 1e-8

const SHELL_MAX_GROW::Int = 8

const SHELL_DEGENERACY_FACTOR::Float64 = 10.0

@inline round_atol(digits::Integer)::Float64 = exp10(-digits + ROUND_SLACK_DECADES)

function wrap01(x::Real; digits::Integer=LATTICE_ROUND_DIGITS)::Float64
    y = round(mod(float(x), 1.0), digits=digits)
    return isapprox(y, 1.0; atol=round_atol(digits)) ? 0.0 : y
end

@inline function wrap01(v::SVector{D,<:Real}; digits::Integer=LATTICE_ROUND_DIGITS)::SVector{D,Float64} where {D}
    return SVector{D,Float64}(wrap01.(v; digits=digits))
end

function wrap01(v::AbstractVector{<:Real}; kwargs...)
    return wrap01(SVector{length(v),Float64}(v); kwargs...)
end

@inline function canonicalize_pos(v::SVector{D,<:Real};
                                  digits::Integer=LATTICE_ROUND_DIGITS)::SVector{D,Float64} where {D}
    return SVector{D,Float64}(round.(v, digits=digits))
end

function canonicalize_pos(v::AbstractVector{<:Real}; kwargs...)
    return canonicalize_pos(SVector{length(v),Float64}(v); kwargs...)
end

@inline function canonicalize_fracpos(v::SVector{D,<:Real};
                                      digits::Integer=LATTICE_ROUND_DIGITS,
                                      nperiodic::Integer=D)::SVector{D,Float64} where {D}
    nperiodic == D && return canonicalize_pos(wrap01(v; digits=digits); digits=digits)
    (0 <= nperiodic <= D) || throw(ArgumentError(
        "`nperiodic` must lie in 0:$D; got $nperiodic."))
    return SVector{D,Float64}(ntuple(
        m -> m <= nperiodic ? wrap01(v[m]; digits=digits) : round(Float64(v[m]); digits=digits),
        D))
end

function canonicalize_fracpos(v::AbstractVector{<:Real}; kwargs...)
    return canonicalize_fracpos(SVector{length(v),Float64}(v); kwargs...)
end

@inline function canonicalize_int(v::SVector{D,<:Real};
                                  atol::Float64=INT_ROUND_ATOL)::SVector{D,Int64} where {D}
    rounded = round.(v)
    isapprox(v, rounded; atol=atol) || throw(DomainError(v,
        "Cannot convert to integer with tolerance atol=$atol; nearest integers are $rounded."
    ))
    return SVector{D,Int64}(rounded)
end

@inline function canonicalize_int(v::SVector{D,<:Integer};
                                  atol::Float64=INT_ROUND_ATOL)::SVector{D,Int64} where {D}
    return SVector{D,Int64}(v)
end

function canonicalize_int(v::AbstractVector{<:Real}; kwargs...)
    return canonicalize_int(SVector{length(v),Float64}(v); kwargs...)
end

function canonicalize_int(v::AbstractVector{<:Integer}; kwargs...)
    return SVector{length(v),Int64}(v)
end

_int_adjugate(::SMatrix{1,1,Int}) = SMatrix{1,1,Int}(1)

_int_adjugate(A::SMatrix{2,2,Int}) = SMatrix{2,2,Int}(A[2, 2], -A[2, 1], -A[1, 2], A[1, 1])

function _int_adjugate(A::SMatrix{3,3,Int})
    c(i1, i2, j1, j2) = A[i1, j1] * A[i2, j2] - A[i1, j2] * A[i2, j1]
    return SMatrix{3,3,Int}(
        c(2, 3, 2, 3), -c(2, 3, 1, 3), c(2, 3, 1, 2),
        -c(1, 3, 2, 3), c(1, 3, 1, 3), -c(1, 3, 1, 2),
        c(1, 2, 2, 3), -c(1, 2, 1, 3), c(1, 2, 1, 2))
end

function _int_adjugate(A::SMatrix{D,D,Int}) where {D}
    throw(ArgumentError("`_int_adjugate` is implemented for D ≤ 3; got D = $D."))
end

@inline _coset_label(adjA::SMatrix{D,D,Int}, det_abs::Int, v::SVector{D,Int}) where {D} =
    SVector{D,Int}(mod.(adjA * v, det_abs))

@inline function _int_det(A::SMatrix{D,D,Int})::Int where {D}
    adj = _int_adjugate(A)
    return sum(ntuple(k -> adj[1, k] * A[k, 1], Val(D)))
end

@inline _int_det_abs(A::SMatrix{D,D,Int}) where {D} = round(Int, abs(det(SMatrix{D,D,Float64}(A))))

@inline _int_invT(A::SMatrix{D,D,Int}) where {D} =
    SMatrix{D,D,Rational{Int}}(transpose(_int_adjugate(A)) .// _int_det(A))

@inline function _int_solve(adjA::SMatrix{D,D,Int}, detA::Int,
                            v::SVector{D,Int})::Union{Nothing,SVector{D,Int}} where {D}
    num = adjA * v
    all(ntuple(m -> mod(num[m], detA) == 0, Val(D))) || return nothing
    return SVector{D,Int}(ntuple(m -> num[m] ÷ detA, Val(D)))
end

@inline function bloch_phase(kq::SVector{D,Float64}, δ::SVector{D,Float64})::ComplexF64 where {D}
    return cispi(2 * dot(kq, δ))
end

@inline function bloch_phase(k::SVector{D,Float64}, q::SVector{D,Float64},
                             δ::SVector{D,Float64})::ComplexF64 where {D}
    return bloch_phase(k + q, δ)
end

const _DENSE_CUTOFF::Int = 2048

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

function _assert_hermitian(H::AbstractSparseMatrix, atol::Float64)::Nothing
    S = SparseMatrixCSC(H)
    rows = rowvals(S)
    vals = nonzeros(S)
    dev = 0.0
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

function _diag_dense(H::AbstractMatrix, nev::Union{Nothing,Int}, which::Symbol,
                     sigma::Union{Nothing,Real}, vectors::Bool, overwrite::Bool)
    Hd = (overwrite && H isa Matrix{ComplexF64}) ? Hermitian(H) :
                                                   Hermitian(Matrix{ComplexF64}(H))
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

@inline function _dense_irange(n::Int, nev::Union{Nothing,Int}, which::Symbol,
                               sigma::Union{Nothing,Real})::Union{Nothing,UnitRange{Int}}
    (isnothing(nev) || !isnothing(sigma) || nev >= n) && return nothing
    return which === :smallest ? (1:nev) : ((n - nev + 1):n)
end

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
    F = lu(H - sigma * I)
    invmap = x -> F \ x
    kdim = isnothing(krylovdim) ? min(n, max(30, 2 * nev + 10)) : krylovdim
    blk = Block(_start_block(n, nev))
    alg = BlockLanczos(; krylovdim=min(n, max(kdim, 2 * nev + 10)), maxiter=maxiter, tol=tol)
    thetas, vecs, info = eigsolve(invmap, blk, nev, :LM, alg)
    _require_krylov_count(thetas, nev, info, "Shift-invert", "")
    lams = Float64[real(dot(v, H * v) / dot(v, v)) for v in vecs]
    for i in 1:min(nev, length(thetas))
        lam = lams[i]
        v = vecs[i]
        resid = norm(H * v - lam * v) / max(norm(v), eps())
        resid <= SHIFT_INVERT_RESID_RTOL * max(1.0, abs(lam)) || throw(ArgumentError(
            "Shift-invert returned an eigenpair with residual ‖Hv − λv‖ = $resid at " *
            "λ = $lam: the shift sigma = $sigma (almost) coincides with an eigenvalue, so " *
            "(H − σI) is numerically singular and the inverted map is meaningless. Nudge " *
            "`sigma` off the spectrum (any offset large against the eigenvalue spacing works)."))
    end
    return _package_krylov(lams, vecs, nev, vectors; transform=identity)
end

function _start_block(n::Int, w::Int)
    w = clamp(w, 1, n)
    A = Matrix{ComplexF64}(undef, n, w)
    for c in 1:w, i in 1:n
        A[i, c] = cispi(2 * i * c / max(n, 2) + (i + c)^2 / (3n) + 0.37 * c)
    end
    Q = Matrix(qr(A).Q)                      # orthonormal columns, deterministic
    return [Vector{ComplexF64}(view(Q, :, c)) for c in 1:w]
end

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

function threaded_chunks(f, n::Int)::Nothing
    nt = min(Threads.nthreads(), n)
    if nt <= 1
        n >= 1 && f(1:n)
        return nothing
    end
    len, rem = divrem(n, nt)
    bounds = cumsum(vcat(0, fill(len + 1, rem), fill(len, nt - rem)))
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(max(1, nb ÷ nt))
    try
        Threads.@threads for c in 1:nt
            f((bounds[c] + 1):bounds[c + 1])
        end
    catch err
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
