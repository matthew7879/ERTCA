# Paper-specific hamiltonian routines.

@inline _endpoint_q(Gi::SVector{D,Int}, Gj::SVector{D,Int},
                    k0i::SVector{D,Float64}, k0j::SVector{D,Float64}) where {D} =
    (SVector{D,Float64}(Gi) + SVector{D,Float64}(Gj)) / 2 - (k0i + k0j) / 2

struct ConstantKernel end

@inline function is_canonical_coupling(i::Int, j::Int, dR::SVector{D,Rational{Int}})::Bool where {D}
    for m in 1:D
        dR[m] > 0 && return true
        dR[m] < 0 && return false
    end
    return i <= j
end

@inline _is_self_conjugate(i::Int, j::Int, dR::SVector{D,Rational{Int}}) where {D} =
    (i == j) && iszero(dR)

mutable struct CouplingBuffer{D}
    i::Vector{Int}
    j::Vector{Int}
    dR::Vector{SVector{D,Rational{Int}}}
    q::Vector{SVector{D,Float64}}
    w::Vector{ComplexF64}
    kernel::Vector{Int}
    kernels::Vector{Any}
    term::Vector{Int}
    term_names::Vector{String}
    term_amps::Vector{ComplexF64}
    current_term::Int
end

CouplingBuffer{D}() where {D} =
    CouplingBuffer{D}(Int[], Int[], SVector{D,Rational{Int}}[], SVector{D,Float64}[],
                      ComplexF64[], Int[], Any[ConstantKernel()],
                      Int[], String[], ComplexF64[], 0)

Base.length(buf::CouplingBuffer) = length(buf.w)

function Base.sizehint!(buf::CouplingBuffer, n::Integer)
    sizehint!(buf.i, n); sizehint!(buf.j, n); sizehint!(buf.dR, n)
    sizehint!(buf.q, n); sizehint!(buf.w, n); sizehint!(buf.kernel, n)
    sizehint!(buf.term, n)
    return buf
end

function begin_term!(buf::CouplingBuffer, name::AbstractString, amp::ComplexF64)::Int
    push!(buf.term_names, String(name))
    push!(buf.term_amps, amp)
    buf.current_term = length(buf.term_names)
    return buf.current_term
end

function adopt_provenance!(buf::CouplingBuffer, cl)::Nothing
    isempty(buf.term_names) || throw(ArgumentError(
        "The buffer already carries a term registry; adopt provenance before registering terms."))
    append!(buf.term_names, cl.term_names)
    append!(buf.term_amps, cl.term_amps)
    return nothing
end

@inline function push_coupling!(buf::CouplingBuffer{D}, i::Int, j::Int,
                                dR::SVector{D,Rational{Int}}, q::SVector{D,Float64},
                                w::ComplexF64, kid::Int,
                                tid::Int=buf.current_term)::Nothing where {D}
    push!(buf.i, i); push!(buf.j, j); push!(buf.dR, dR)
    push!(buf.q, q); push!(buf.w, w); push!(buf.kernel, kid)
    push!(buf.term, tid)
    return nothing
end

@inline push_coupling!(buf::CouplingBuffer{D}, i::Int, j::Int, dR::SVector{D,Int},
                       q::SVector{D,Float64}, w::ComplexF64, kid::Int,
                       tid::Int=buf.current_term) where {D} =
    push_coupling!(buf, i, j, SVector{D,Rational{Int}}(dR), q, w, kid, tid)

function kernel_id!(buf::CouplingBuffer, kern)::Int
    kern isa ConstantKernel && return 1
    for (κ, k) in pairs(buf.kernels)
        κ == 1 && continue
        if kern isa CallbackKernel
            if kern.f isa AbstractEnvelope
                (typeof(k) === typeof(kern) && k.parity == kern.parity && k.f == kern.f) &&
                    return κ
            else
                k === kern && return κ
            end
        elseif typeof(k) === typeof(kern) && k == kern
            return κ
        end
    end
    push!(buf.kernels, kern)
    return length(buf.kernels)
end

struct CouplingList{D}
    n::Int
    i::Vector{Int}
    j::Vector{Int}
    dR::Vector{SVector{D,Rational{Int}}}
    q::Vector{SVector{D,Float64}}
    w::Vector{ComplexF64}
    kernel::Vector{Int}
    kernels::Vector{Any}
    groups::Vector{UnitRange{Int}}
    term::Vector{Int}
    term_names::Vector{String}
    term_amps::Vector{ComplexF64}
    pattern_id::UInt

    function CouplingList{D}(n::Int, i::Vector{Int}, j::Vector{Int},
                             dR::Vector{SVector{D,Rational{Int}}},
                             q::Vector{SVector{D,Float64}}, w::Vector{ComplexF64},
                             kernel::Vector{Int}, kernels::Vector{Any},
                             groups::Vector{UnitRange{Int}},
                             term::Vector{Int}=zeros(Int, length(i)),
                             term_names::Vector{String}=String[],
                             term_amps::Vector{ComplexF64}=ComplexF64[],
                             pattern_id::UInt=hash((n, i, j, dR, kernel))) where {D}
        len = length(i)
        (length(j) == len && length(dR) == len && length(q) == len &&
         length(w) == len && length(kernel) == len && length(term) == len) ||
            throw(DimensionMismatch(
            "CouplingList component arrays have inconsistent lengths: i=$(length(i)), " *
            "j=$(length(j)), dR=$(length(dR)), q=$(length(q)), w=$(length(w)), " *
            "kernel=$(length(kernel)), term=$(length(term))."
        ))
        n >= 0 || throw(DomainError(n, "CouplingList dimension must be non-negative."))
        isempty(kernels) && throw(ArgumentError("The kernel table must at least hold ConstantKernel()."))
        kernels[1] isa ConstantKernel || throw(ArgumentError(
            "Kernel id 1 must be ConstantKernel(); got $(typeof(kernels[1]))."))
        length(groups) == length(kernels) || throw(DimensionMismatch(
            "Got $(length(groups)) kernel groups for $(length(kernels)) kernels."))
        length(term_names) == length(term_amps) || throw(DimensionMismatch(
            "Got $(length(term_names)) term names for $(length(term_amps)) term amplitudes."))
        for e in eachindex(i)
            (1 <= i[e] <= n) || throw(BoundsError(1:n, i[e]))
            (1 <= j[e] <= n) || throw(BoundsError(1:n, j[e]))
            (1 <= kernel[e] <= length(kernels)) || throw(BoundsError(kernels, kernel[e]))
            (abs(term[e]) <= length(term_names)) || throw(ArgumentError(
                "Coupling $e carries term id $(term[e]); the registry holds " *
                "$(length(term_names)) terms (the sign encodes orientation)."))
        end
        return new{D}(n, i, j, dR, q, w, kernel, kernels, groups, term, term_names, term_amps,
                      pattern_id)
    end
end

@inline n_couplings(c::CouplingList)::Int = length(c.w)

Base.length(c::CouplingList) = length(c.w)

has_kdependence(c::CouplingList)::Bool =
    any(κ -> !(c.kernels[κ] isa ConstantKernel) && !isempty(c.groups[κ]),
        eachindex(c.kernels))

function finalize_couplings(buf::CouplingBuffer{D}, nstates::Integer;
                            atol::Float64=AMPLITUDE_ATOL,
                            prune::Bool=false)::CouplingList{D} where {D}
    n = Int(nstates)
    m = length(buf)

    # --- 1. canonicalize direction
    ci = Vector{Int}(undef, m); cj = Vector{Int}(undef, m)
    cR = Vector{SVector{D,Rational{Int}}}(undef, m)
    cq = Vector{SVector{D,Float64}}(undef, m)
    cw = Vector{ComplexF64}(undef, m)
    ck = Vector{Int}(undef, m)
    ct = Vector{Int}(undef, m)
    for e in 1:m
        i, j, dR, q, w, κ = buf.i[e], buf.j[e], buf.dR[e], buf.q[e], buf.w[e], buf.kernel[e]
        isfinite(w) || throw(DomainError(w,
            "Coupling ($i, $j, $(Tuple(dR))) carries a non-finite amplitude $w. If the term " *
            "uses a callback amplitude, it returned NaN or Inf for this bond."
        ))
        if is_canonical_coupling(i, j, dR)
            ci[e] = i; cj[e] = j; cR[e] = dR; cw[e] = w
            ct[e] = buf.term[e]
        else
            ci[e] = j; cj[e] = i; cR[e] = -dR; cw[e] = conj(w)
            ct[e] = -buf.term[e]
        end
        cq[e] = q; ck[e] = κ
    end

    # --- 2. sort and coalesce on (kernel, term, i, j, ΔR, q); q compares exactly -- entries
    #        that should merge were computed by identical arithmetic and are bit-equal. The
    #        term id in the key keeps provenance sound for `reweight`: two DIFFERENT terms
    #        hitting the same (i, j, ΔR) stay separate entries (the assembly sums them at
    #        evaluation time, so H(k) is unchanged; the list is marginally longer).
    #        Keys are materialized ONCE: `sortperm(...; by=)` re-derives the key on every
    #        comparison, which dominated large-supercell compiles.
    ckey = Vector{Tuple{Int,Int,Int,Int,NTuple{D,Rational{Int}},NTuple{D,Float64}}}(undef, m)
    for e in 1:m
        ckey[e] = (ck[e], ct[e], ci[e], cj[e], Tuple(cR[e]), Tuple(cq[e]))
    end
    order = sortperm(ckey)
    ki = Int[]; kj = Int[]; kR = SVector{D,Rational{Int}}[]
    kq = SVector{D,Float64}[]; kw = ComplexF64[]; kk = Int[]; kt = Int[]
    for v in (ki, kj, kR, kq, kw, kk, kt)
        sizehint!(v, m)
    end
    for e in order
        if !isempty(ki) && kk[end] == ck[e] && kt[end] == ct[e] && ki[end] == ci[e] &&
           kj[end] == cj[e] && kR[end] == cR[e] && kq[end] == cq[e]
            kw[end] += cw[e]
        else
            push!(ki, ci[e]); push!(kj, cj[e]); push!(kR, cR[e])
            push!(kq, cq[e]); push!(kw, cw[e]); push!(kk, ck[e]); push!(kt, ct[e])
        end
    end

    # --- 3. self-conjugate entries must be real (checked after coalescing). NOTE: the
    #        coalesce key includes the term id, so two DIFFERENT terms contributing complex
    #        on-site amplitudes that only sum to a real total are refused here -- each term's
    #        on-site contribution must be real on its own, which is what a Hermitian term means.
    for e in eachindex(ki)
        if _is_self_conjugate(ki[e], kj[e], kR[e]) && abs(imag(kw[e])) > atol
            throw(DomainError(kw[e],
                "On-site amplitude on basis state $(ki[e]) (kernel $(kk[e])) is not real " *
                "(imaginary part $(imag(kw[e])) exceeds atol=$atol). An entry with i == j and " *
                "ΔR == 0 is its own Hermitian partner, so a complex prefactor would make H(k) " *
                "non-Hermitian."
            ))
        end
    end

    # --- 4. materialise Hermitian partners
    fi = Int[]; fj = Int[]; fR = SVector{D,Rational{Int}}[]
    fq = SVector{D,Float64}[]; fw = ComplexF64[]; fk = Int[]; ft = Int[]
    for v in (fi, fj, fR, fq, fw, fk, ft)
        sizehint!(v, 2 * length(ki))
    end
    for e in eachindex(ki)
        if prune && abs(kw[e]) <= atol
            continue
        end
        selfconj = _is_self_conjugate(ki[e], kj[e], kR[e])
        push!(fi, ki[e]); push!(fj, kj[e]); push!(fR, kR[e]); push!(fq, kq[e])
        push!(fw, selfconj ? ComplexF64(real(kw[e])) : kw[e]); push!(fk, kk[e])
        push!(ft, kt[e])
        if !selfconj
            push!(fi, kj[e]); push!(fj, ki[e]); push!(fR, -kR[e]); push!(fq, kq[e])
            push!(fw, conj(kw[e])); push!(fk, kk[e]); push!(ft, -kt[e])
        end
    end

    # --- 5. final sort: kernel-major (one dispatch per contiguous group), column-fast within
    #        (keys materialized once, for the same reason as the coalesce sort above)
    fkey = Vector{Tuple{Int,Int,Int,NTuple{D,Rational{Int}}}}(undef, length(fi))
    for e in eachindex(fi)
        fkey[e] = (fk[e], fj[e], fi[e], Tuple(fR[e]))
    end
    forder = sortperm(fkey)
    fi = fi[forder]; fj = fj[forder]; fR = fR[forder]
    fq = fq[forder]; fw = fw[forder]; fk = fk[forder]; ft = ft[forder]

    kernels = copy(buf.kernels)
    groups = _kernel_groups(fk, length(kernels))
    return CouplingList{D}(n, fi, fj, fR, fq, fw, fk, kernels, groups,
                           ft, copy(buf.term_names), copy(buf.term_amps))
end

function _kernel_groups(fk::Vector{Int}, nkernels::Int)::Vector{UnitRange{Int}}
    groups = Vector{UnitRange{Int}}(undef, nkernels)
    pos = 1
    for κ in 1:nkernels
        start = pos
        while pos <= length(fk) && fk[pos] == κ
            pos += 1
        end
        groups[κ] = start:(pos - 1)
    end
    pos == length(fk) + 1 || throw(ErrorException(
        "Kernel ids are not sorted kernel-major; this is an internal inconsistency."))
    return groups
end

function each_canonical(cl::CouplingList{D}) where {D}
    return ((; e, i=cl.i[e], j=cl.j[e], dR=cl.dR[e], q=cl.q[e], w=cl.w[e],
             kernel=cl.kernel[e], term=cl.term[e])
            for e in eachindex(cl.i) if is_canonical_coupling(cl.i[e], cl.j[e], cl.dR[e]))
end

n_canonical(cl::CouplingList)::Int =
    count(e -> is_canonical_coupling(cl.i[e], cl.j[e], cl.dR[e]), eachindex(cl.i))

function adopted_buffer(cl::CouplingList{D};
                        per_canonical::Integer=1)::Tuple{CouplingBuffer{D},Vector{Int}} where {D}
    buf = CouplingBuffer{D}()
    adopt_provenance!(buf, cl)
    kids = [kernel_id!(buf, kern) for kern in cl.kernels]
    sizehint!(buf, n_canonical(cl) * Int(per_canonical))
    return buf, kids
end

function is_hermitian(c::CouplingList{D}; atol::Float64=AMPLITUDE_ATOL)::Bool where {D}
    index = Dict{Tuple{Int,Int,Int,SVector{D,Rational{Int}},NTuple{D,Float64}},ComplexF64}()
    for e in eachindex(c.i)
        key = (c.kernel[e], c.i[e], c.j[e], c.dR[e], Tuple(c.q[e]))
        index[key] = get(index, key, zero(ComplexF64)) + c.w[e]
    end
    for ((κ, i, j, dR, q), w) in index
        partner = get(index, (κ, j, i, -dR, q), nothing)
        isnothing(partner) && return false
        isapprox(partner, conj(w); atol=atol) || return false
    end
    return true
end

function _show_dR(dR::SVector{D,Rational{Int}})::String where {D}
    return "(" * join((denominator(x) == 1 ? string(numerator(x)) : string(x) for x in dR),
                      ", ") * ")"
end

function Base.show(io::IO, ::MIME"text/plain", c::CouplingList{D}) where {D}
    m = n_couplings(c)
    print(io, "CouplingList{", D, "}: ", m, " coupling", m == 1 ? "" : "s",
          " (both directions stored) on ", c.n, " states, ",
          length(c.kernels), " kernel", length(c.kernels) == 1 ? "" : "s")
    nshow = min(m, 8)
    for e in 1:nshow
        print(io, "\n  H[", c.i[e], ", ", c.j[e], "]  dR=", _show_dR(c.dR[e]))
        iszero(c.q[e]) || print(io, "  q=", Tuple(c.q[e]))
        print(io, "  w=", c.w[e], "  κ=", c.kernel[e])
        t = c.term[e]
        t == 0 || print(io, "  term=", c.term_names[abs(t)], t > 0 ? "" : "'")
    end
    m > nshow && print(io, "\n  … ", m - nshow, " more")
    return nothing
end

Base.show(io::IO, c::CouplingList{D}) where {D} =
    print(io, "CouplingList{", D, "}(n=", c.n, ", couplings=", n_couplings(c),
          ", kernels=", length(c.kernels), ")")

abstract type AbstractHamiltonianTerm end

abstract type BondTerm       <: AbstractHamiltonianTerm end

abstract type KDependentTerm <: AbstractHamiltonianTerm end

abstract type AbstractBondSelector end

struct OnSiteSelector <: AbstractBondSelector
    sites::Vector{Symbol}
end

OnSiteSelector() = OnSiteSelector(Symbol[])

struct TranslationSelector <: AbstractBondSelector
    dR::Vector{Int}
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}
end

struct ShellSelector <: AbstractBondSelector
    shell::Int
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}

    function ShellSelector(shell::Int, from::Vector{Symbol}, to::Vector{Symbol})
        shell >= 1 || throw(DomainError(shell, "Shell index must be at least 1 (1 = nearest neighbour)."))
        return new(shell, from, to)
    end
end

struct RadiusSelector <: AbstractBondSelector
    rcut::Float64
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}

    function RadiusSelector(rcut::Real, from::Vector{Symbol}, to::Vector{Symbol})
        (isfinite(rcut) && rcut > 0) || throw(DomainError(rcut,
            "Cutoff radius `rcut` must be positive and finite (a cartesian distance, in the " *
            "units of the lattice vectors)."))
        return new(Float64(rcut), from, to)
    end
end

struct DeltaGSelector <: AbstractBondSelector
    dG::Vector{Int}
    sites_from::Vector{Symbol}
    sites_to::Vector{Symbol}
end

struct ExplicitBondSelector <: AbstractBondSelector
    i::Vector{Int}
    j::Vector{Int}
    dR::Vector{Vector{Int}}

    function ExplicitBondSelector(i::Vector{Int}, j::Vector{Int}, dR::Vector{Vector{Int}})
        (length(i) == length(j) == length(dR)) || throw(DimensionMismatch(
            "ExplicitBondSelector arrays must have equal length; got $(length(i)), $(length(j)), $(length(dR))."
        ))
        if !isempty(dR)
            d1 = length(first(dR))
            all(r -> length(r) == d1, dR) || throw(DimensionMismatch(
                "ExplicitBondSelector translations must all have the same length."
            ))
        end
        return new(i, j, dR)
    end
end

max_shell(::AbstractBondSelector) = 0

max_shell(s::ShellSelector) = s.shell

max_radius(::AbstractBondSelector)::Float64 = 0.0

max_radius(s::RadiusSelector)::Float64 = s.rcut

has_intrinsic_direction(::AbstractBondSelector) = true

has_intrinsic_direction(::ShellSelector) = false

has_intrinsic_direction(::RadiusSelector) = false

has_intrinsic_direction(::OnSiteSelector) = false

_what_string(what::AbstractString) = what

_what_string(what) = what()

function _as_svec_int(v::Vector{Int}, D::Int, what, name::AbstractString)
    length(v) == D || throw(DimensionMismatch(
        "Term `$name`: `$(_what_string(what))` has length $(length(v)) but the cell is " *
        "$D-dimensional."
    ))
    return SVector{D,Int}(v)
end

function _materialize_k0(k0::Vector{Float64}, D::Int, name::AbstractString)
    isempty(k0) && return zero(SVector{D,Float64})
    length(k0) == D || throw(DimensionMismatch(
        "Term `$name`: `k0` has length $(length(k0)) but the cell is $D-dimensional."
    ))
    return SVector{D,Float64}(k0)
end

function _materialize_c(c::Vector{Float64}, D::Int, name::AbstractString)
    length(c) == D || throw(DimensionMismatch(
        "Term `$name`: `c` has length $(length(c)) but the cell is $D-dimensional."
    ))
    return SVector{D,Float64}(c)
end

function _materialize_c(c::Pair{Int,Float64}, D::Int, name::AbstractString)
    1 <= c.first <= D || throw(DimensionMismatch(
        "Term `$name`: direction $(c.first) does not exist in a $D-dimensional cell."
    ))
    return SVector{D,Float64}(ntuple(m -> m == c.first ? c.second : 0.0, D))
end

function _materialize_M(M::Matrix{Float64}, D::Int, name::AbstractString)
    size(M) == (D, D) || throw(DimensionMismatch(
        "Term `$name`: `M` is $(size(M)) but the cell is $D-dimensional."
    ))
    return SMatrix{D,D,Float64}(M)
end

_materialize_M(lam::Float64, D::Int, ::AbstractString) = lam * one(SMatrix{D,D,Float64})

_site_list(s::Symbol) = Symbol[s]

_site_list(v::AbstractVector{Symbol}) = Vector{Symbol}(v)

abstract type AbstractBondWeight end

struct UnitWeight <: AbstractBondWeight end

struct Chirality <: AbstractBondWeight end

struct ChiralityPhase <: AbstractBondWeight
    phi::Float64
end

struct ExponentialDecay <: AbstractBondWeight
    lambda::Float64
    d0::Float64

    function ExponentialDecay(lambda::Real, d0::Real=0.0)
        lambda > 0 || throw(DomainError(lambda, "Decay length must be positive."))
        return new(Float64(lambda), Float64(d0))
    end
end

is_symmetric_weight(::AbstractBondWeight) = true

is_symmetric_weight(::Chirality) = false

is_symmetric_weight(::ChiralityPhase) = false

struct CompileContext{D,K<:KPeriodicity}
    cell::UnitCell{D,K}
    tables::CellTables{D}
    neighbors::Union{Nothing,NeighborTable{D}}
    aT::SMatrix{D,D,Float64}
    bT::SMatrix{D,D,Float64}
    fold_MinvT::Union{Nothing,SMatrix{D,D,Rational{Int}}}
    fold_wrapshifts::Union{Nothing,Vector{SVector{D,Int}}}
end

_ctx_kpar(::CompileContext{D,K}) where {D,K} = K

function CompileContext(cell::UnitCell{D,K};
                        tables::CellTables{D}=cell_tables(cell),
                        neighbors::Union{Nothing,NeighborTable{D}}=nothing) where {D,K}
    MinvT = _fold_MinvT(cell)
    return CompileContext{D,K}(cell, tables, neighbors,
                             SMatrix{D,D,Float64}(transpose(cell.lattice.a)),
                             SMatrix{D,D,Float64}(transpose(cell.lattice.b)),
                             MinvT,
                             isnothing(MinvT) ? nothing : _fold_fine_wrapshifts(cell))
end

function _fold_MinvT(cell::UnitCell{D}) where {D}
    is_folded(cell) || return nothing
    s = findfirst(k -> cell.site_kind[k] === :localized &&
                       abs(det(cell.site_period[k])) != 1, eachindex(cell.site_kind))
    return _int_invT(SMatrix{D,D,Int}(cell.site_period[s]))
end

function _require_neighbors(ctx::CompileContext{D}, what::AbstractString)::NeighborTable{D} where {D}
    isnothing(ctx.neighbors) && throw(ArgumentError(
        "$what requires a neighbour table, but none was built. This is an internal inconsistency: " *
        "`compile` should have scanned the terms with `max_shell` before assembling the context."
    ))
    return ctx.neighbors
end

@inline bond_weight(::UnitWeight, ::CompileContext, ::Int, ::Int, ::SVector, ::Float64) = one(ComplexF64)

bond_weight(w::ExponentialDecay, ::CompileContext, ::Int, ::Int, ::SVector, dist::Float64) =
    ComplexF64(exp(-(dist - w.d0) / w.lambda))

function bond_weight(::Chirality, ctx::CompileContext{D}, si::Int, sj::Int,
                     dR::SVector{D,Int}, ::Float64)::ComplexF64 where {D}
    return ComplexF64(_chirality(ctx, si, sj, dR))
end

function bond_weight(w::ChiralityPhase, ctx::CompileContext{D}, si::Int, sj::Int,
                     dR::SVector{D,Int}, ::Float64)::ComplexF64 where {D}
    return cis(w.phi * _chirality(ctx, si, sj, dR))
end

bond_weight(::Union{Chirality,ChiralityPhase}, ::CompileContext{D}, ::Int, ::Int,
            ::SVector{D,Rational{Int}}, ::Float64) where {D} = throw(ArgumentError(
    "`Chirality`/`ChiralityPhase` weights are not supported on a folded cell; compile the " *
    "parent model and `fold` it instead."))

@inline _bond_weight(w::AbstractBondWeight, ctx::CompileContext, si::Int, sj::Int,
                     dR::SVector, dist::Float64) = bond_weight(w, ctx, si, sj, dR, dist)

validate_weight(::AbstractBondWeight, ::CompileContext, ::String)::Nothing = nothing

function _chirality(ctx::CompileContext{D}, si::Int, sj::Int, dR::SVector{D,Int})::Float64 where {D}
    D == 2 || throw(ArgumentError(
        "`Chirality` is defined only in D = 2 (it is the sign of a 2D cross product); got D = $D."
    ))
    nt = _require_neighbors(ctx, "`Chirality`")
    isempty(nt.radii) && throw(ArgumentError("`Chirality` needs a nearest-neighbour shell, but the neighbour table has none."))

    t = ctx.tables
    r_i = ctx.aT * t.site_positions[si]
    r_j = ctx.aT * (SVector{D,Float64}(dR) + t.site_positions[sj])
    nnj = Set{Tuple{Int,SVector{D,Int}}}()
    for e in shell_entries(nt, 1)
        nt.site_i[e] == sj && push!(nnj, (nt.site_j[e], dR + nt.dR[e]))
        nt.site_j[e] == sj && push!(nnj, (nt.site_i[e], dR - nt.dR[e]))
    end
    commons = SVector{D,Float64}[]
    for e in shell_entries(nt, 1)
        nt.site_i[e] == si && (nt.site_j[e], nt.dR[e]) in nnj &&
            push!(commons, ctx.aT * (SVector{D,Float64}(nt.dR[e]) + t.site_positions[nt.site_j[e]]))
        nt.site_j[e] == si && (nt.site_i[e], -nt.dR[e]) in nnj &&
            push!(commons, ctx.aT * (SVector{D,Float64}(-nt.dR[e]) + t.site_positions[nt.site_i[e]]))
    end

    length(commons) == 1 || throw(ArgumentError(
        "`Chirality` requires exactly one common nearest neighbour between the two endpoints of a " *
        "bond, found $(length(commons)) for sites ($si, $sj) at ΔR = $(Tuple(dR)). " *
        "On a square lattice, second-shell bonds have two common neighbours and the chirality is " *
        "ambiguous; select those bonds with an explicit `TranslationSelector` instead."
    ))

    d1 = commons[1] - r_i
    d2 = r_j - commons[1]
    cross2 = d1[1] * d2[2] - d1[2] * d2[1]
    abs(cross2) > LATTICE_ATOL || throw(ArgumentError(
        "`Chirality` is degenerate: the two hops for sites ($si, $sj) at ΔR = $(Tuple(dR)) are collinear."
    ))
    return sign(cross2)
end

struct HoppingTerm <: BondTerm
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    weight::AbstractBondWeight
    t::Union{ComplexF64,Function}
    name::String

    function HoppingTerm(selector::AbstractBondSelector, rule::AbstractQnumRule,
                         weight::AbstractBondWeight, t::Union{ComplexF64,Function},
                         name::String)
        if t isa ComplexF64 && !iszero(imag(t)) &&
           is_symmetric_weight(weight) && !has_intrinsic_direction(selector)
            @warn "HoppingTerm `$name` has a complex amplitude ($t) on an unoriented selector " *
                  "($(typeof(selector))) with a direction-symmetric weight $(typeof(weight)). " *
                  "The phase would be applied in whichever direction the basis ordering makes " *
                  "canonical, which is not physics. Use a direction-antisymmetric weight " *
                  "(e.g. `Chirality()`) for a directed phase."
        end
        return new(selector, rule, weight, t, name)
    end
end

max_shell(term::HoppingTerm) = max_shell(term.selector)

max_radius(term::HoppingTerm)::Float64 = max_radius(term.selector)

max_radius(term::KDependentTerm)::Float64 = max_radius(_kselector(term))

max_radius(::AbstractHamiltonianTerm)::Float64 = 0.0

term_name(term::AbstractHamiltonianTerm) = term.name

_has_zero_amplitude(term::HoppingTerm) = (term.t isa ComplexF64) && iszero(term.t)

_has_zero_amplitude(term::KDependentTerm) = iszero(_kcoefficient(term))

_has_zero_amplitude(::AbstractHamiltonianTerm) = false

function _has_zero_rule(term::AbstractHamiltonianTerm, schema)
    static, directional = _split_directional(_term_rule(term))
    isempty(directional) || return false
    return iszero(qnum_matrix(static, schema))
end

@inline _amplitude(t::ComplexF64, ::BasisState, ::BasisState, ::SVector) = t

@inline _amplitude(f::Function, bs_i::BasisState, bs_j::BasisState, dR::SVector) =
    ComplexF64(f(bs_i, bs_j, dR))

@inline _checked_amplitude(t::ComplexF64, ::BasisState, ::BasisState, ::SVector, ::String) = t

function _checked_amplitude(f::Function, bs_i::BasisState, bs_j::BasisState,
                            dR::SVector, name::String)::ComplexF64
    fwd = ComplexF64(f(bs_i, bs_j, dR))
    rev = ComplexF64(f(bs_j, bs_i, -dR))
    (isfinite(fwd) && isfinite(rev)) || return fwd
    isapprox(rev, conj(fwd); atol=AMPLITUDE_ATOL) || throw(ArgumentError(
        "Term `$name`: its callback amplitude is not conjugate-symmetric on a pair whose " *
        "Hermitian partner is supplied automatically: forward $fwd, reverse $rev ≠ " *
        "conj(forward). Only one orientation of this element is stored, so a non-symmetric " *
        "callback would be silently Hermitized. Make the callback satisfy " *
        "t(j, i, -ΔR) = conj(t(i, j, ΔR)), or select the bonds with an oriented selector " *
        "(`translation_hopping`, `delta_g`, `explicit_bonds`), where one-directional values " *
        "are the contract."
    ))
    return fwd
end

@inline _diagonal_checked_amplitude(t::ComplexF64, ::BasisState, ::SVector, ::String) = t

function _diagonal_checked_amplitude(f::Function, bs::BasisState, dR::SVector,
                                     name::String)::ComplexF64
    a = ComplexF64(f(bs, bs, dR))
    b = ComplexF64(f(bs, bs, dR))
    (isfinite(a) && isfinite(b)) || return a
    isapprox(a, b; atol=AMPLITUDE_ATOL) || throw(ArgumentError(
        "Term `$name`: its callback amplitude is not deterministic -- two evaluations on the " *
        "same on-site element returned $a and then $b. A per-call random draw compiles into " *
        "an irreproducible disorder realization; draw the realization into a table first and " *
        "let the callback read it (the worked recipe is " *
        "examples/hamiltonianconstruction/04_disorder_and_defects.jl)."
    ))
    return a
end

@inline _site_ok(allowed::Vector{Symbol}, site::Symbol)::Bool = isempty(allowed) || (site in allowed)

function _validate_selector(sel::AbstractBondSelector, ctx::CompileContext, name::String)::Nothing
    for (field, sites) in _selector_site_lists(sel)
        for s in sites
            s in ctx.tables.site_names || throw(ArgumentError(
                "Term `$name` selects $field = :$s, but the cell declares no such site (its " *
                "sites are $(ctx.tables.site_names)). No pair would match, so the term would " *
                "silently emit nothing."
            ))
        end
    end
    sel isa TranslationSelector && _check_dR_periodic(sel.dR, ctx, name)
    sel isa DeltaGSelector && _check_dR_periodic(sel.dG, ctx, name)
    return nothing
end

function _validate_selector(sel::ExplicitBondSelector, ctx::CompileContext, name::String)::Nothing
    n = n_states(ctx.cell)
    for e in eachindex(sel.i)
        (1 <= sel.i[e] <= n && 1 <= sel.j[e] <= n) || throw(ArgumentError(
            "Term `$name` names the explicit bond ($(sel.i[e]), $(sel.j[e]), " *
            "$(Tuple(sel.dR[e]))), but the cell has only $n states."
        ))
        _check_dR_periodic(sel.dR[e], ctx, name)
    end
    return nothing
end

function _check_dR_periodic(v::AbstractVector{<:Integer}, ctx::CompileContext,
                            name::String)::Nothing
    P = nperiodic(ctx.cell.lattice)
    D = ndim(ctx.cell.lattice)
    P == D && return nothing
    for m in (P+1):min(D, length(v))
        v[m] == 0 || throw(ArgumentError(
            "Term `$name` carries the integer vector $(Tuple(v)), non-zero along direction " *
            "$m -- but the lattice is periodic only along its first $P direction" *
            "$(P == 1 ? "" : "s"), so there is no translation (or reciprocal vector) there."))
    end
    return nothing
end

_selector_site_lists(sel::OnSiteSelector) = (("sites", sel.sites),)

_selector_site_lists(sel::Union{TranslationSelector,ShellSelector,RadiusSelector,DeltaGSelector}) =
    (("sites_from", sel.sites_from), ("sites_to", sel.sites_to))

_candidate_sites(sel::OnSiteSelector, t::CellTables) =
    [s for s in 1:n_sites(t) if _site_ok(sel.sites, t.site_names[s])]

_candidate_sites(sel::Union{TranslationSelector,ShellSelector,RadiusSelector,DeltaGSelector},
                 t::CellTables) =
    [s for s in 1:n_sites(t)
     if _site_ok(sel.sites_from, t.site_names[s]) || _site_ok(sel.sites_to, t.site_names[s])]

_candidate_sites(sel::ExplicitBondSelector, t::CellTables) =
    unique(vcat([t.state_site[i] for i in sel.i], [t.state_site[j] for j in sel.j]))

function _validate_kinds(term::AbstractHamiltonianTerm, sel::AbstractBondSelector,
                         ctx::CompileContext)::Nothing
    t = ctx.tables
    if term isa KDependentTerm
        for s in _candidate_sites(sel, t)
            t.site_kind[s] === :plane_wave || throw(ArgumentError(
                "k-dependent term `$(term_name(term))` addresses site `$(t.site_names[s])`, " *
                "which is :localized -- a momentum kernel on a localized (Wannier-like) orbital " *
                "is meaningless (`p` on a comb): its matrix elements are given as bonds, not " *
                "computed from a dispersion. Use `hopping`/`onsite`/`explicit_bonds` there, or " *
                "restrict the term with `sites=` to the :plane_wave sites."))
        end
    elseif sel isa DeltaGSelector
        for s in _candidate_sites(sel, t)
            t.site_kind[s] === :plane_wave ||
                (t.site_kind[s] === :localized && abs(det(t.site_period[s])) != 1) ||
                throw(ArgumentError(
                    "Term `$(term_name(term))` uses a ΔG selector on site " *
                    "`$(t.site_names[s])`, which is :localized -- a real-space orbital carries " *
                    "no plane-wave label to shift. ΔG couplings apply to :plane_wave sites, or " *
                    "to the coset labels of a folded cell (`fold`). For tight-binding couplings " *
                    "use `onsite`/`hopping`/`translation_hopping`."))
        end
    end
    return nothing
end

struct EmitSink{D,K<:KPeriodicity}
    buf::CouplingBuffer{D}
    ctx::CompileContext{D,K}
    Mrule::Matrix{ComplexF64}
    static_rule::AbstractQnumRule
    directional::Vector{DirectionalQnum}
    weight::AbstractBondWeight
    amp::Union{ComplexF64,Function}
    kid::Int
    k0::SVector{D,Float64}
    name::String
end

@inline _is_directional(sink::EmitSink) = !isempty(sink.directional)

@inline _term_selector(t::HoppingTerm) = t.selector

@inline _term_selector(t::KDependentTerm) = _kselector(t)

@inline _term_rule(t::HoppingTerm) = t.rule

@inline _term_rule(t::KDependentTerm) = _krule(t)

@inline _term_weight(t::HoppingTerm) = t.weight

@inline _term_weight(::KDependentTerm) = UnitWeight()

@inline _term_amp(t::HoppingTerm) = t.t

@inline _term_amp(t::KDependentTerm) = _kcoefficient(t)

@inline _term_k0raw(::HoppingTerm) = Float64[]

@inline _term_k0raw(t::KDependentTerm) = _k0_raw(t)

_term_kernel(::HoppingTerm, ::Val{D}) where {D} = ConstantKernel()

_term_kernel(t::KDependentTerm, ::Val{D}) where {D} = _make_kernel(t, Val(D))

function _check_kernel_periodic(kern, lattice::BravaisLattice{D},
                                name::AbstractString)::Nothing where {D}
    kern isa ConstantKernel && return nothing
    P = nperiodic(lattice)
    P == D && return nothing
    for m in (P+1):D
        pad = primitive_vector(lattice, m)
        for t in (0.7, -1.3)
            dead = _kvalue(kern, SVector{D,Float64}(t .* pad)) - _kvalue(kern, zero(SVector{D,Float64}))
            abs(dead) <= AMPLITUDE_ATOL || throw(ArgumentError(
                "Term `$name` carries a momentum kernel that responds along aperiodic " *
                "direction $m of this partially periodic lattice (probe response $dead). " *
                "Momentum is identically zero there, so that part of the kernel would be a " *
                "silently dead term -- restrict the kernel to the periodic directions."))
        end
    end
    return nothing
end

function _split_directional(rule::AbstractQnumRule)
    rule isa DirectionalQnum && return (AnyQnum(), DirectionalQnum[rule])
    rule isa ProductQnumRule || return (rule, DirectionalQnum[])
    directional = DirectionalQnum[]
    static = AbstractQnumRule[]
    for r in rule.rules
        s, d = _split_directional(r)
        s isa AnyQnum || push!(static, s)
        append!(directional, d)
    end
    stat = isempty(static) ? AnyQnum() :
           length(static) == 1 ? static[1] : ProductQnumRule(static)
    return (stat, directional)
end

function _validate_directional_selector(sel::AbstractBondSelector, dr::DirectionalQnum,
                                        name::String)::Nothing
    sel isa OnSiteSelector && throw(ArgumentError(
        "Term `$name` carries a directional rule on axis `$(dr.key)`, but its on-site " *
        "selector pairs states on one site: a bond-direction-dependent matrix has no " *
        "direction on an on-site pair. Put the directional physics on a bond term " *
        "(`hopping`, `translation_hopping` at ΔR ≠ 0, `explicit_bonds`)."))
    sel isa DeltaGSelector && throw(ArgumentError(
        "Term `$name` carries a directional rule on axis `$(dr.key)`, but its ΔG selector " *
        "couples plane-wave labels at ΔR = 0 -- there is no bond direction to evaluate " *
        "`f(d)` at. Directional rules apply to real-space bond selectors."))
    return nothing
end

function _probe_directional(dr::DirectionalQnum, ::Val{D}, s::InternalSchema,
                            name::AbstractString)::Nothing where {D}
    for (s1, s2) in ((1.0, 0.0), (-0.611, 0.788), (0.5, -0.207))
        d = SVector{D,Float64}(ntuple(m -> s1 * cospi(0.29m + 0.13) + s2 * sinpi(0.41m - 0.17), D))
        _directional_matrix(dr, d / norm(d), s, name)
    end
    return nothing
end

function _directional_matrix(dr::DirectionalQnum, d::SVector{D,Float64}, s::InternalSchema,
                             name::AbstractString)::Matrix{ComplexF64} where {D}
    Mraw = dr.f(d)
    Mraw isa AbstractMatrix || throw(ArgumentError(
        "Term `$name`: the directional rule on axis `$(dr.key)` returned a " *
        "$(typeof(Mraw)) at d = $(Tuple(d)); `f(d_cart)` must return an AbstractMatrix."))
    n = _order_rank(dr.order, dr, s)
    size(Mraw) == (n, n) || throw(DimensionMismatch(
        "Term `$name`: the directional rule on axis `$(dr.key)` returned a " *
        "$(size(Mraw, 1))×$(size(Mraw, 2)) matrix at d = $(Tuple(d)), but " *
        _order_rank_reason(dr.order, dr, s, n)))
    M = Matrix{ComplexF64}(Mraw)
    all(isfinite, M) || throw(DomainError(M,
        "Term `$name`: the directional rule on axis `$(dr.key)` returned a non-finite " *
        "entry at d = $(Tuple(d)). A hand-written direction cosine often divides by " *
        "`norm(d)` -- normalize once, inside `f`."))
    return M
end

_order_rank(::DeclaredOrder, dr::DirectionalQnum, s::InternalSchema) =
    n_levels(s, _rule_axis(s, dr.key, dr))

_order_rank(::TwoLevelOrder, ::DirectionalQnum, ::InternalSchema) = 2

_order_rank(o::NamedOrder, ::DirectionalQnum, ::InternalSchema) = length(o.levels)

_order_rank_reason(::LevelOrder, dr::DirectionalQnum, s::InternalSchema, n::Int) =
    "the axis declares $n levels ($(levels(s, _rule_axis(s, dr.key, dr))))."

_order_rank_reason(o::NamedOrder, ::DirectionalQnum, ::InternalSchema, n::Int) =
    "its level labels $(o.levels) require $n×$n."

_directional_inner(dr::DirectionalQnum{DeclaredOrder}, M::Matrix{ComplexF64},
                   ::InternalSchema)::AbstractQnumRule = MatrixQnum(dr.key, M)

_directional_inner(dr::DirectionalQnum{TwoLevelOrder}, M::Matrix{ComplexF64},
                   ::InternalSchema)::AbstractQnumRule =
    TwoLevelMatrixQnum(dr.key, M, dr.order.up, dr.order.down)

function _directional_inner(dr::DirectionalQnum{NamedOrder}, M::Matrix{ComplexF64},
                            s::InternalSchema)::AbstractQnumRule
    a = _rule_axis(s, dr.key, dr)
    lv = s.levels[a]
    pos = [findfirst(l -> isequal(l, v), dr.order.levels)::Int for v in lv]
    return MatrixQnum(dr.key, ComplexF64[M[pos[r], pos[c]] for r in eachindex(lv),
                                                               c in eachindex(lv)])
end

function _pair_rule_matrix(sink::EmitSink{D}, si::Int, sj::Int,
                           dR::Union{SVector{D,Int},SVector{D,Rational{Int}}},
                           dist::Float64)::Matrix{ComplexF64} where {D}
    _is_directional(sink) || return sink.Mrule
    t = sink.ctx.tables
    d = sink.ctx.aT * (t.site_positions[si] - t.site_positions[sj] - SVector{D,Float64}(dR))
    nd = norm(d)
    nd > distance_atol(sink.ctx.cell.lattice) || throw(ArgumentError(
        "Term `$(sink.name)` carries a directional rule, but sites ($si, $sj) at " *
        "ΔR = $(Tuple(dR)) form a zero-length pair: a bond-direction-dependent matrix has " *
        "no direction on an on-site pair. Restrict the selector (site filters, or a " *
        "ΔR ≠ 0 translation) so only genuine bonds remain."))
    s = sink.ctx.cell.schema
    rule = sink.static_rule
    for dr in sink.directional
        rule = compose(rule, _directional_inner(dr, _directional_matrix(dr, d, s, sink.name), s))
    end
    return qnum_matrix(rule, s)
end

_throw_directional_asymmetry(name::String, i::Int, j::Int, dR, fwd::ComplexF64,
                             rev::ComplexF64) = throw(ArgumentError(
    "Term `$name`: its directional rule is not conjugate-symmetric on the bond pair " *
    "(states $i, $j, ΔR = $(Tuple(dR))): forward element amp·M(d)[a,b] = $fwd, reverse " *
    "element amp·M(-d)[b,a] = $rev ≠ conj(forward). Only one orientation of this element " *
    "is stored and the partner is supplied automatically, so a non-symmetric directional " *
    "rule would be silently Hermitized. Make the full product satisfy " *
    "amp·f(-d)[b,a] = conj(amp·f(d)[a,b]) (for a real amplitude: f(-d) = f(d)†, e.g. an " *
    "odd anti-Hermitian f), or select the bonds with an oriented selector " *
    "(`translation_hopping`, `explicit_bonds`), where one-directional values are the contract."))

function emit_term!(buf::CouplingBuffer{D}, term::AbstractHamiltonianTerm,
                    ctx::CompileContext{D})::Nothing where {D}
    sel = _term_selector(term)
    # Validation runs even for zero-amplitude terms: a typo does not stop being a typo because
    # the coupling is currently swept through zero.
    _validate_selector(sel, ctx, term_name(term))
    _validate_kinds(term, sel, ctx)
    validate_weight(_term_weight(term), ctx, term_name(term))
    static, directional = _split_directional(_term_rule(term))
    for dr in directional
        validate_rule(dr, ctx.cell.schema)
        _validate_directional_selector(sel, dr, term_name(term))
        _probe_directional(dr, Val(D), ctx.cell.schema, term_name(term))
    end
    # A literally-zero amplitude emits nothing at all: keeping a pattern alive would charge the
    # full per-k cost to a term swept through zero, which is routine in a phase diagram.
    _has_zero_amplitude(term) && return nothing

    Mrule = qnum_matrix(static, ctx.cell.schema)
    kern = _term_kernel(term, Val(D))
    _check_kernel_periodic(kern, ctx.cell.lattice, term_name(term))
    kid = kernel_id!(buf, kern)
    k0 = _materialize_k0(_term_k0raw(term), D, term_name(term))
    if !iszero(k0) && any(!iszero, ctx.tables.k0)
        @warn "Term `$(term_name(term))` carries a per-term `k0` while the cell declares " *
              "per-state momentum origins; the two COMPOSE " *
              "(q = (G_i+G_j)/2 − (K_i+K_j)/2 − k0_term). Prefer origins on the cell -- the " *
              "per-term keyword is sugar for cells without them." maxlog = 1
    end
    sink = EmitSink{D,_ctx_kpar(ctx)}(buf, ctx, Mrule, static, directional, _term_weight(term),
                       _term_amp(term), kid, k0, term_name(term))
    _emit!(sink, sel)
    return nothing
end

@inline function _push_entry!(sink::EmitSink{D}, i::Int, j::Int,
                              dR::Union{SVector{D,Int},SVector{D,Rational{Int}}},
                              amp::ComplexF64)::Nothing where {D}
    iszero(amp) && return nothing
    t = sink.ctx.tables
    q = _endpoint_q(t.G[i], t.G[j], t.k0[i], t.k0[j]) - sink.k0
    push_coupling!(sink.buf, i, j, dR, q, amp, sink.kid)
    return nothing
end

function _emit!(sink::EmitSink{D}, sel::OnSiteSelector)::Nothing where {D}
    t = sink.ctx.tables
    zeroR = zero(SVector{D,Int})
    for s in 1:n_sites(t)
        _site_ok(sel.sites, t.site_names[s]) || continue
        w = _bond_weight(sink.weight, sink.ctx, s, s, zeroR, 0.0)
        _emit_unordered_pairs!(sink, s, s, w, w, true)
    end
    return nothing
end

function _emit_unordered_pairs!(sink::EmitSink{D},
                                si::Int, sj::Int,
                                w::ComplexF64, wrev::ComplexF64, allow_rev::Bool,
                                seen::Set{Tuple{Int,Int}}=Set{Tuple{Int,Int}}();
                                Mfwd::Matrix{ComplexF64}=sink.Mrule,
                                Mrev::Matrix{ComplexF64}=sink.Mrule)::Nothing where {D}
    t = sink.ctx.tables
    for i in t.site_states[si]
        for j in states_on_site_with_G(t, sj, t.G[i])
            key = i <= j ? (i, j) : (j, i)
            key in seen && continue
            push!(seen, key)
            _emit_hermitized_pair!(sink, i, j, w, wrev, allow_rev; Mfwd=Mfwd, Mrev=Mrev)
        end
    end
    return nothing
end

function _emit_hermitized_pair!(sink::EmitSink{D}, i::Int, j::Int,
                                w::ComplexF64, wrev::ComplexF64, allow_rev::Bool;
                                Mfwd::Matrix{ComplexF64}=sink.Mrule,
                                Mrev::Matrix{ComplexF64}=sink.Mrule)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    zeroR = zero(SVector{D,Int})
    f = @inbounds Mfwd[t.internal[i], t.internal[j]]
    g = @inbounds Mrev[t.internal[j], t.internal[i]]
    if _is_directional(sink)
        (iszero(f) && iszero(g)) && return nothing
        Afwd = iszero(f) ? zero(ComplexF64) :
               _amplitude(sink.amp, basis[i], basis[j], zeroR) * f * w
        Arev = iszero(g) ? zero(ComplexF64) :
               _amplitude(sink.amp, basis[j], basis[i], zeroR) * g * wrev
        if allow_rev && i != j && !isapprox(Arev, conj(Afwd); atol=AMPLITUDE_ATOL)
            _throw_directional_asymmetry(sink.name, i, j, zeroR, Afwd, Arev)
        end
        _push_entry!(sink, i, j, zeroR, Afwd)
        return nothing
    end
    if !iszero(f)
        if allow_rev && i != j && !iszero(g) && !isapprox(g, conj(f); atol=AMPLITUDE_ATOL)
            throw(ArgumentError(
                "Term `$(sink.name)`: its quantum-number rule is non-Hermitian on the unordered " *
                "pair (states $i, $j): forward factor $f, reverse factor $g ≠ conj(forward). " *
                "Such a pair is stored once and Hermitized, so only conjugate-symmetric " *
                "or one-sided (ladder) rules are meaningful there."
            ))
        end
        amp0 = if allow_rev && i != j
            _checked_amplitude(sink.amp, basis[i], basis[j], zeroR, sink.name)
        elseif i == j
            _diagonal_checked_amplitude(sink.amp, basis[i], zeroR, sink.name)
        else
            _amplitude(sink.amp, basis[i], basis[j], zeroR)
        end
        amp = amp0 * f * w
        _push_entry!(sink, i, j, zeroR, amp)
        return nothing
    end
    # the rule may only accept the reverse ordering (a ladder operator)
    (allow_rev && !iszero(g)) || return nothing
    amp = (i != j ? _checked_amplitude(sink.amp, basis[j], basis[i], zeroR, sink.name) :
                    _amplitude(sink.amp, basis[j], basis[i], zeroR)) * g * wrev
    _push_entry!(sink, j, i, zeroR, amp)
    return nothing
end

function _emit!(sink::EmitSink{D}, sel::ShellSelector)::Nothing where {D}
    nt = _require_neighbors(sink.ctx, "`ShellSelector` (shell $(sel.shell))")
    sel.shell <= n_shells(nt) || throw(ArgumentError(
        "Term `$(sink.name)` asks for shell $(sel.shell) but the neighbour table resolved only " *
        "$(n_shells(nt)) shells out to radius $(nt.rcut)."
    ))
    MinvT = sink.ctx.fold_MinvT
    rho = sink.ctx.fold_wrapshifts
    for e in shell_entries(nt, sel.shell)
        si, sj, dRf, d = nt.site_i[e], nt.site_j[e], nt.dR[e], nt.dist[e]
        if isnothing(MinvT)
            _emit_site_pair!(sink, si, sj, dRf, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        else
            dRm = SVector{D,Rational{Int}}(
                MinvT * SVector{D,Rational{Int}}(dRf + rho[sj] - rho[si]))
            _emit_site_pair!(sink, si, sj, dRm, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        end
    end
    return nothing
end

function _emit!(sink::EmitSink{D}, sel::RadiusSelector)::Nothing where {D}
    nt = _require_neighbors(sink.ctx, "`RadiusSelector` (rcut $(sel.rcut))")
    datol = distance_atol(sink.ctx.cell.lattice)
    sel.rcut <= nt.rcut + datol || throw(ArgumentError(
        "Term `$(sink.name)` asks for bonds out to rcut = $(sel.rcut), but the neighbour " *
        "table was searched only to radius $(nt.rcut). This is an internal inconsistency: " *
        "`compile` should have scanned the terms with `max_radius` before building the table."
    ))
    MinvT = sink.ctx.fold_MinvT
    rho = sink.ctx.fold_wrapshifts
    for e in eachindex(nt.dist)
        d = nt.dist[e]
        d <= sel.rcut + datol || continue
        si, sj, dRf = nt.site_i[e], nt.site_j[e], nt.dR[e]
        if isnothing(MinvT)
            _emit_site_pair!(sink, si, sj, dRf, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        else
            dRm = SVector{D,Rational{Int}}(
                MinvT * SVector{D,Rational{Int}}(dRf + rho[sj] - rho[si]))
            _emit_site_pair!(sink, si, sj, dRm, d, sel.sites_from, sel.sites_to,
                             has_intrinsic_direction(sel))
        end
    end
    return nothing
end

function _emit!(sink::EmitSink{D}, sel::TranslationSelector)::Nothing where {D}
    t = sink.ctx.tables
    dR = _as_svec_int(sel.dR, D, "dR", sink.name)
    if iszero(dR)
        zeroR = zero(SVector{D,Int})
        seen = Set{Tuple{Int,Int}}()
        for si in 1:n_sites(t), sj in 1:n_sites(t)
            _site_ok(sel.sites_to, t.site_names[si]) || continue      # si holds the targets (bra)
            _site_ok(sel.sites_from, t.site_names[sj]) || continue    # sj holds the sources (ket)
            t.site_kind[si] === t.site_kind[sj] || continue           # no cross-kind couplings
            d = norm(sink.ctx.aT * (t.site_positions[sj] - t.site_positions[si]))
            w = _bond_weight(sink.weight, sink.ctx, si, sj, zeroR, d)
            wrev = is_symmetric_weight(sink.weight) ? w :
                   _bond_weight(sink.weight, sink.ctx, sj, si, zeroR, d)
            allow_rev = _site_ok(sel.sites_to, t.site_names[sj]) &&
                        _site_ok(sel.sites_from, t.site_names[si])
            Mf = _pair_rule_matrix(sink, si, sj, zeroR, d)
            Mr = _is_directional(sink) ? _pair_rule_matrix(sink, sj, si, zeroR, d) : Mf
            _emit_unordered_pairs!(sink, si, sj, w, wrev, allow_rev, seen; Mfwd=Mf, Mrev=Mr)
        end
        return nothing
    end

    for si in 1:n_sites(t), sj in 1:n_sites(t)
        d = norm(sink.ctx.aT * (SVector{D,Float64}(dR) + t.site_positions[sj] - t.site_positions[si]))
        _emit_site_pair!(sink, si, sj, dR, d, sel.sites_from, sel.sites_to,
                         has_intrinsic_direction(sel))
    end
    return nothing
end

function _emit_site_pair!(sink::EmitSink{D}, si::Int, sj::Int,
                          dR::Union{SVector{D,Int},SVector{D,Rational{Int}}}, dist::Float64,
                          from::Vector{Symbol}, to::Vector{Symbol},
                          _directed::Bool)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    if _directed
        _site_ok(to, t.site_names[si]) || return nothing
        _site_ok(from, t.site_names[sj]) || return nothing
    else
        (_site_ok(to, t.site_names[si]) && _site_ok(from, t.site_names[sj])) ||
            (_site_ok(from, t.site_names[si]) && _site_ok(to, t.site_names[sj])) ||
            return nothing
    end
    t.site_kind[si] === t.site_kind[sj] || return nothing

    w = _bond_weight(sink.weight, sink.ctx, si, sj, dR, dist)

    # At ΔR = 0 on one site, (i,j) and (j,i) are Hermitian partners, so each unordered pair may be
    # emitted only once. At ΔR ≠ 0 they are independent matrix elements and both are needed.
    if si == sj && iszero(dR)
        _is_directional(sink) && _pair_rule_matrix(sink, si, sj, dR, dist)
        _emit_unordered_pairs!(sink, si, sj, w, w, true)
        return nothing
    end

    wrev = _directed || is_symmetric_weight(sink.weight) ? w :
           _bond_weight(sink.weight, sink.ctx, sj, si, -dR, dist)
    Mfwd = _pair_rule_matrix(sink, si, sj, dR, dist)
    dirn = _is_directional(sink)
    Mrev = (dirn && !_directed) ? _pair_rule_matrix(sink, sj, si, -dR, dist) : Mfwd

    for i in t.site_states[si]
        for j in states_on_site_with_G(t, sj, t.G[i])
            f = @inbounds Mfwd[t.internal[i], t.internal[j]]
            if _directed
                iszero(f) && continue
                amp = _amplitude(sink.amp, basis[i], basis[j], dR) * f * w
                _push_entry!(sink, i, j, dR, amp)
                continue
            end
            g = @inbounds Mrev[t.internal[j], t.internal[i]]
            if dirn
                (iszero(f) && iszero(g)) && continue
                Afwd = iszero(f) ? zero(ComplexF64) :
                       _amplitude(sink.amp, basis[i], basis[j], dR) * f * w
                Arev = iszero(g) ? zero(ComplexF64) :
                       _amplitude(sink.amp, basis[j], basis[i], -dR) * g * wrev
                isapprox(Arev, conj(Afwd); atol=AMPLITUDE_ATOL) ||
                    _throw_directional_asymmetry(sink.name, i, j, dR, Afwd, Arev)
                _push_entry!(sink, i, j, dR, Afwd)
                continue
            end
            if !iszero(f)
                if !iszero(g) && !isapprox(g, conj(f); atol=AMPLITUDE_ATOL)
                    throw(ArgumentError(
                        "Term `$(sink.name)`: its quantum-number rule is non-Hermitian on the " *
                        "bond pair (states $i, $j, ΔR = $(Tuple(dR))): forward factor $f, " *
                        "reverse factor $g ≠ conj(forward). The two orientations of an " *
                        "undirected bond are one Hermitian element, so only " *
                        "conjugate-symmetric or one-sided (ladder) rules are meaningful there."
                    ))
                end
                amp = _checked_amplitude(sink.amp, basis[i], basis[j], dR, sink.name) * f * w
                _push_entry!(sink, i, j, dR, amp)
            elseif !iszero(g)
                # the rule accepts only the reversed orientation: emit the reversed-bond element
                amp = _checked_amplitude(sink.amp, basis[j], basis[i], -dR, sink.name) * g * wrev
                _push_entry!(sink, j, i, -dR, amp)
            end
        end
    end
    return nothing
end

function _emit!(sink::EmitSink{D}, sel::DeltaGSelector)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    zeroR = zero(SVector{D,Int})
    dG = _as_svec_int(sel.dG, D, "dG", sink.name)
    w = _bond_weight(sink.weight, sink.ctx, 1, 1, zeroR, 0.0)

    # ΔG = 0 makes the coupling its own reverse, so each unordered pair may be emitted only once.
    # For ΔG ≠ 0 the reverse never arises: it would need G_j = G_i + ΔG as well as G_i = G_j + ΔG.
    unordered = iszero(dG)
    seen = unordered ? Set{Tuple{Int,Int}}() : nothing
    folded = !isnothing(sink.ctx.fold_MinvT)
    coset_states = folded ? _folded_coset_index(t) : nothing

    for j in 1:n_states(t)                                  # j is the source (ket)
        sj = t.state_site[j]
        _site_ok(sel.sites_from, t.site_names[sj]) || continue

        if folded && t.site_kind[sj] === :localized
            _site_ok(sel.sites_to, t.site_names[sj]) || continue     # same-site only
            M = SMatrix{D,D,Int}(t.site_period[sj])
            lbl = _coset_label(_int_adjugate(M), _int_det_abs(M), t.G[j] + dG)
            for i in get(coset_states, (sj, lbl), EMPTY_INDICES)
                gwrap = dG + t.G[j] - t.G[i]                # ∈ M ℤᴰ by construction
                phase = cispi(2 * dot(SVector{D,Float64}(gwrap), t.site_positions[sj]))
                if unordered && iszero(gwrap)
                    key = i <= j ? (i, j) : (j, i)
                    key in seen && continue
                    push!(seen, key)
                    _emit_hermitized_pair!(sink, i, j, w, w, true)
                    continue
                end
                f = @inbounds sink.Mrule[t.internal[i], t.internal[j]]
                iszero(f) && continue
                amp = _amplitude(sink.amp, basis[i], basis[j], zeroR) * f * w * phase
                if i == j
                    _push_entry!(sink, i, i, zeroR, amp + conj(amp))
                    continue
                end
                _push_entry!(sink, i, j, zeroR, amp)
            end
            continue
        end

        for i in states_with_G(t, t.G[j] + dG)              # i is the target (bra), exact integer key
            _site_ok(sel.sites_to, t.site_names[t.state_site[i]]) || continue
            t.site_kind[t.state_site[i]] === t.site_kind[sj] || continue
            if unordered
                key = i <= j ? (i, j) : (j, i)
                key in seen && continue
                push!(seen, key)
                allow_rev = _site_ok(sel.sites_to, t.site_names[t.state_site[j]]) &&
                            _site_ok(sel.sites_from, t.site_names[t.state_site[i]])
                _emit_hermitized_pair!(sink, i, j, w, w, allow_rev)
                continue
            end
            f = @inbounds sink.Mrule[t.internal[i], t.internal[j]]
            iszero(f) && continue
            amp = _amplitude(sink.amp, basis[i], basis[j], zeroR) * f * w
            _push_entry!(sink, i, j, zeroR, amp)
        end
    end
    return nothing
end

function _folded_coset_index(t::CellTables{D}) where {D}
    out = Dict{Tuple{Int,SVector{D,Int}},Vector{Int}}()
    for s in 1:n_sites(t)
        t.site_kind[s] === :localized || continue
        M = SMatrix{D,D,Int}(t.site_period[s])
        nd = _int_det_abs(M)
        nd == 1 && continue
        adjM = _int_adjugate(M)
        for i in t.site_states[s]
            push!(get!(Vector{Int}, out, (s, _coset_label(adjM, nd, t.G[i]))), i)
        end
    end
    return out
end

function _emit!(sink::EmitSink{D}, sel::ExplicitBondSelector)::Nothing where {D}
    t = sink.ctx.tables
    basis = sink.ctx.cell.basis
    n = n_states(t)
    for e in eachindex(sel.i)
        i, j = sel.i[e], sel.j[e]
        dR = _as_svec_int(sel.dR[e], D, () -> "dR[$e]", sink.name)
        (1 <= i <= n) || throw(BoundsError(1:n, i))
        (1 <= j <= n) || throw(BoundsError(1:n, j))
        t.G[i] == t.G[j] || throw(ArgumentError(
            "Term `$(sink.name)` was explicitly asked to couple states $i and $j, which carry " *
            "different plane-wave labels ($(Tuple(t.G[i])) and $(Tuple(t.G[j]))). For a " *
            "deliberate ΔG ≠ 0 coupling use `delta_g`/`DeltaGSelector` (`dG = ...`), which " *
            "fixes the momentum-offset convention."
        ))
        d = norm(sink.ctx.aT * (SVector{D,Float64}(dR) + t.tau[j] - t.tau[i]))
        Mp = _pair_rule_matrix(sink, t.state_site[i], t.state_site[j], dR, d)
        f = @inbounds Mp[t.internal[i], t.internal[j]]
        iszero(f) && continue
        w = _bond_weight(sink.weight, sink.ctx, t.state_site[i], t.state_site[j], dR, d)
        amp = _amplitude(sink.amp, basis[i], basis[j], dR) * f * w
        _push_entry!(sink, i, j, dR, amp)
    end
    return nothing
end

struct KLinearTerm <: KDependentTerm
    c::Union{Vector{Float64},Pair{Int,Float64}}
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KLinearTerm(c::Union{Vector{Float64},Pair{Int,Float64}}, k0::Vector{Float64},
                         selector::AbstractBondSelector, rule::AbstractQnumRule,
                         coefficient::ComplexF64, name::String)
        if c isa Pair{Int,Float64}
            first(c) >= 1 || throw(DomainError(first(c),
                "`KLinearTerm` `$name`: the direction index must be at least 1."))
            isfinite(last(c)) || throw(DomainError(last(c),
                "`KLinearTerm` `$name`: the coefficient along direction $(first(c)) is not finite."))
        else
            all(isfinite, c) || throw(DomainError(c,
                "`KLinearTerm` `$name`: the coefficient vector contains a non-finite entry."))
        end
        all(isfinite, k0) || throw(DomainError(k0,
            "`KLinearTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KLinearTerm` `$name`: the prefactor is not finite."))
        return new(c, k0, selector, rule, coefficient, name)
    end
end

struct KQuadraticTerm <: KDependentTerm
    M::Union{Matrix{Float64},Float64}
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KQuadraticTerm(M::Union{Matrix{Float64},Float64}, k0::Vector{Float64},
                            selector::AbstractBondSelector, rule::AbstractQnumRule,
                            coefficient::ComplexF64, name::String)
        if M isa Matrix{Float64}
            issymmetric(M) || throw(ArgumentError(
                "`KQuadraticTerm` `$name` requires a symmetric M; got $M. An antisymmetric part " *
                "would make the k-dependent scalar non-real and break Hermitization."
            ))
            all(isfinite, M) || throw(DomainError(M,
                "`KQuadraticTerm` `$name`: M contains a non-finite entry."))
        else
            isfinite(M) || throw(DomainError(M, "`KQuadraticTerm` `$name`: M is not finite."))
        end
        all(isfinite, k0) || throw(DomainError(k0,
            "`KQuadraticTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KQuadraticTerm` `$name`: the prefactor is not finite."))
        return new(M, k0, selector, rule, coefficient, name)
    end
end

struct KEnvelopeTerm <: KDependentTerm
    f::Function
    parity::Symbol
    k0_frac::Vector{Float64}
    selector::AbstractBondSelector
    rule::AbstractQnumRule
    coefficient::ComplexF64
    name::String

    function KEnvelopeTerm(f::Function, parity::Symbol, k0::Vector{Float64},
                           selector::AbstractBondSelector, rule::AbstractQnumRule,
                           coefficient::ComplexF64, name::String)
        parity in (:even, :odd, :none) || throw(ArgumentError(
            "`KEnvelopeTerm` `$name`: `parity` must be :even, :odd or :none; got :$parity."))
        all(isfinite, k0) || throw(DomainError(k0,
            "`KEnvelopeTerm` `$name`: `k0` contains a non-finite entry."))
        isfinite(coefficient) || throw(DomainError(coefficient,
            "`KEnvelopeTerm` `$name`: the prefactor is not finite."))
        return new(f, parity, k0, selector, rule, coefficient, name)
    end
end

max_shell(term::KDependentTerm) = max_shell(_kselector(term))

@inline _k0_raw(term::KLinearTerm) = term.k0_frac

@inline _k0_raw(term::KQuadraticTerm) = term.k0_frac

@inline _k0_raw(term::KEnvelopeTerm) = term.k0_frac

@inline _kcoefficient(term::KLinearTerm) = term.coefficient

@inline _kcoefficient(term::KQuadraticTerm) = term.coefficient

@inline _kcoefficient(term::KEnvelopeTerm) = term.coefficient

@inline _kselector(term::KLinearTerm) = term.selector

@inline _kselector(term::KQuadraticTerm) = term.selector

@inline _kselector(term::KEnvelopeTerm) = term.selector

@inline _krule(term::KLinearTerm) = term.rule

@inline _krule(term::KQuadraticTerm) = term.rule

@inline _krule(term::KEnvelopeTerm) = term.rule

struct KLinearKernel{D}
    c::SVector{D,Float64}
end

struct KQuadraticKernel{D,L}
    M::SMatrix{D,D,Float64,L}   # the implicit constructor infers {D,L} from the field
end

struct CallbackKernel{F}
    f::F
    parity::Int
end

struct KPolynomialKernel{D,NM,L}
    c::SVector{NM,Float64}
    alpha::SMatrix{NM,D,Int,L}
end

function kvalue end

@inline kvalue(::ConstantKernel, ::SVector) = 1.0

@inline kvalue(kern::KLinearKernel{D}, p::SVector{D,Float64}) where {D} = dot(kern.c, p)

@inline kvalue(kern::KQuadraticKernel{D}, p::SVector{D,Float64}) where {D} = dot(p, kern.M * p)

@inline kvalue(kern::CallbackKernel, p::SVector) = kern.f(p)

@inline function kvalue(kern::KPolynomialKernel{D,NM}, p::SVector{D,Float64}) where {D,NM}
    acc = 0.0
    @inbounds for m in 1:NM
        t = kern.c[m]
        for d in 1:D
            e = kern.alpha[m, d]
            e == 0 && continue
            t *= p[d]^e
        end
        acc += t
    end
    return acc
end

@inline _kvalue(kern, p) = kvalue(kern, p)

kernel_parity(k) = throw(ArgumentError(
    "`nambu_doubling` needs the parity of the k-kernel `$(typeof(k))` under p -> -p; " *
    "define `kernel_parity(::$(typeof(k)))` (+1 even, -1 odd)."))

kernel_parity(::ConstantKernel) = 1

kernel_parity(::KLinearKernel) = -1

kernel_parity(::KQuadraticKernel) = 1

function kernel_parity(k::CallbackKernel)
    k.parity == 0 && throw(ArgumentError(
        "`nambu_doubling` needs the parity of a k-envelope under p -> -p, and this one was " *
        "built with `parity=:none`. Rebuild the term with `k_envelope(f; parity=:even)` or " *
        "`:odd` (split a mixed-parity envelope into its even and odd parts first)."))
    return k.parity
end

kernel_parity(kern::KPolynomialKernel) = _kernel_parity(kern)

abstract type AbstractEnvelope <: Function end

@inline kvalue(env::AbstractEnvelope, p::SVector) = env(p)

struct GaussianEnvelope <: AbstractEnvelope
    sigma::Float64
    p0::Float64

    function GaussianEnvelope(sigma::Real; p0::Real=0.0)
        (isfinite(sigma) && sigma > 0) || throw(DomainError(sigma,
            "`GaussianEnvelope` needs a positive, finite width `sigma`."))
        isfinite(p0) || throw(DomainError(p0, "`GaussianEnvelope`: `p0` is not finite."))
        return new(Float64(sigma), Float64(p0))
    end
end

(env::GaussianEnvelope)(p::AbstractVector) = exp(-(norm(p) - env.p0)^2 / (2 * env.sigma^2))

kernel_parity(::GaussianEnvelope) = 1

struct TabulatedEnvelope <: AbstractEnvelope
    r::Vector{Float64}
    v::Vector{Float64}
    extrapolation::Symbol

    function TabulatedEnvelope(r::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                               extrapolation::Symbol=:zero)
        length(r) == length(v) || throw(DimensionMismatch(
            "`TabulatedEnvelope`: the radial grid `r` has $(length(r)) nodes but `v` holds " *
            "$(length(v)) values; each node needs exactly one value."))
        length(r) >= 2 || throw(ArgumentError(
            "`TabulatedEnvelope` interpolates linearly and needs at least two grid nodes; " *
            "got $(length(r))."))
        all(isfinite, r) && all(isfinite, v) || throw(DomainError((r, v),
            "`TabulatedEnvelope`: the grid and its values must be finite."))
        all(m -> r[m] < r[m+1], 1:(length(r) - 1)) || throw(ArgumentError(
            "`TabulatedEnvelope`: the radial grid `r` must be STRICTLY increasing (it holds " *
            "values of |p|); sort the table and merge duplicate nodes first."))
        extrapolation in (:zero, :flat) || throw(ArgumentError(
            "`TabulatedEnvelope`: unknown extrapolation kind `:$extrapolation`; supported " *
            "kinds are :zero (the envelope vanishes outside the grid) and :flat (the " *
            "nearest endpoint value is held)."))
        return new(collect(Float64, r), collect(Float64, v), extrapolation)
    end
end

function (env::TabulatedEnvelope)(p::AbstractVector)
    x = norm(p)
    r = env.r
    if x < r[1] || x > r[end]
        env.extrapolation === :flat || return 0.0
        return x < r[1] ? env.v[1] : env.v[end]
    end
    m = searchsortedlast(r, x)
    m == length(r) && return env.v[end]
    t = (x - r[m]) / (r[m+1] - r[m])
    return (1 - t) * env.v[m] + t * env.v[m+1]
end

kernel_parity(::TabulatedEnvelope) = 1

Base.:(==)(a::TabulatedEnvelope, b::TabulatedEnvelope) =
    a.r == b.r && a.v == b.v && a.extrapolation === b.extrapolation

Base.hash(env::TabulatedEnvelope, h::UInt) =
    hash((TabulatedEnvelope, env.r, env.v, env.extrapolation), h)

Base.show(io::IO, env::GaussianEnvelope) =
    print(io, "GaussianEnvelope(", env.sigma, iszero(env.p0) ? "" : "; p0=$(env.p0)", ")")

Base.show(io::IO, ::MIME"text/plain", env::GaussianEnvelope) = show(io, env)

Base.show(io::IO, env::TabulatedEnvelope) =
    print(io, "TabulatedEnvelope(", env.r, ", ", env.v,
          env.extrapolation === :zero ? "" : "; extrapolation=$(repr(env.extrapolation))", ")")

Base.show(io::IO, ::MIME"text/plain", env::TabulatedEnvelope) = show(io, env)

struct BlochSumEnvelope <: AbstractEnvelope
    Q::SMatrix{2,2,Float64,4}
    p0::SVector{2,Float64}
    delta::Vector{SVector{2,Float64}}
    amp::Vector{ComplexF64}
    component::Symbol
    wmap::SMatrix{2,2,Float64,4}
    wshift::SVector{2,Float64}
    wradius::Float64
    wwidth::Float64
    dx::Vector{Float64}          # the distinct δ¹ values
    dy::Vector{Float64}          # the distinct δ² values
    xptr::Vector{Int32}          # CSR over `dx`: group a spans xptr[a]:xptr[a+1]-1
    iy::Vector{Int32}            # δ² index of each term (`delta`/`amp` are stored x-grouped)
    fast::Bool                   # do the tables pay for themselves?

    function BlochSumEnvelope(Q::AbstractMatrix{<:Real}, p0::AbstractVector{<:Real},
                              delta::AbstractVector{<:AbstractVector{<:Real}},
                              amp::AbstractVector{<:Number}, component::Symbol;
                              wmap::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                              wshift::AbstractVector{<:Real}=SVector(0.0, 0.0),
                              wradius::Real=Inf, wwidth::Real=0.0)
        size(Q) == (2, 2) || throw(DimensionMismatch(
            "`BlochSumEnvelope`: the momentum map `Q` must be 2×2; got $(size(Q)). The " *
            "envelope is two-dimensional -- momenta of 2D plane-wave models."))
        length(p0) == 2 || throw(DimensionMismatch(
            "`BlochSumEnvelope`: the offset `p0` must have 2 components; got $(length(p0))."))
        length(delta) == length(amp) || throw(DimensionMismatch(
            "`BlochSumEnvelope`: $(length(delta)) separations but $(length(amp)) amplitudes; " *
            "each δ needs exactly one amplitude."))
        isempty(delta) && throw(ArgumentError(
            "`BlochSumEnvelope` needs at least one (δ, amplitude) entry; an identically zero " *
            "envelope is a term you simply do not add."))
        all(d -> length(d) == 2, delta) || throw(DimensionMismatch(
            "`BlochSumEnvelope`: every separation in `delta` must have 2 components."))
        all(isfinite, Q) && all(isfinite, p0) &&
            all(d -> all(isfinite, d), delta) && all(isfinite, amp) || throw(DomainError(
            "BlochSumEnvelope", "`BlochSumEnvelope`: Q, p0, delta and amp must be finite."))
        component in (:re, :im) || throw(ArgumentError(
            "`BlochSumEnvelope`: unknown component `:$component`; supported components are " *
            ":re and :im (a complex Bloch sum enters a Hermitian model as two real terms)."))
        wm = isnothing(wmap) ? one(SMatrix{2,2,Float64}) : SMatrix{2,2,Float64}(wmap)
        length(wshift) == 2 || throw(DimensionMismatch(
            "`BlochSumEnvelope`: the window shift must have 2 components; got $(length(wshift))."))
        (wradius > 0 && !isnan(wradius)) || throw(DomainError(wradius,
            "`BlochSumEnvelope`: the window radius must be positive (Inf = no window)."))
        (isfinite(wwidth) && wwidth >= 0) || throw(DomainError(wwidth,
            "`BlochSumEnvelope`: the window roll-off width must be finite and non-negative."))
        (wwidth < wradius || !isfinite(wradius)) || throw(ArgumentError(
            "`BlochSumEnvelope`: the roll-off width $wwidth reaches the origin at radius " *
            "$wradius (need wwidth < wradius). The window would be below 1 everywhere, " *
            "attenuating the envelope at every momentum rather than bounding where it is " *
            "trusted; narrow `wwidth` or widen `wradius`."))
        all(isfinite, wm) && all(isfinite, wshift) || throw(DomainError(
            "BlochSumEnvelope", "`BlochSumEnvelope`: wmap and wshift must be finite."))
        dl, av, dx, dy, xptr, iy, fast =
            _bloch_tables([SVector{2,Float64}(d) for d in delta], collect(ComplexF64, amp))
        return new(SMatrix{2,2,Float64}(Q), SVector{2,Float64}(p0),
                   dl, av, component, wm, SVector{2,Float64}(wshift), Float64(wradius),
                   Float64(wwidth), dx, dy, xptr, iy, fast)
    end
end

function _bloch_tables(delta::Vector{SVector{2,Float64}}, amp::Vector{ComplexF64})
    n = length(delta)
    xid = Dict{Float64,Int32}()
    yid = Dict{Float64,Int32}()
    dx = Float64[]
    dy = Float64[]
    ix = Vector{Int32}(undef, n)
    iy0 = Vector{Int32}(undef, n)
    for m in 1:n
        d = delta[m]
        ix[m] = get!(xid, d[1]) do
            push!(dx, d[1]); Int32(length(dx))
        end
        iy0[m] = get!(yid, d[2]) do
            push!(dy, d[2]); Int32(length(dy))
        end
    end
    if length(dx) + length(dy) > (7 * n) ÷ 10
        return delta, amp, Float64[], Float64[], Int32[], Int32[], false
    end
    ndx = length(dx)
    xptr = zeros(Int32, ndx + 1)
    xptr[1] = 1
    for m in 1:n
        xptr[ix[m]+1] += Int32(1)
    end
    for a in 1:ndx
        xptr[a+1] += xptr[a]
    end
    pos = xptr[1:ndx]
    iy = Vector{Int32}(undef, n)
    dl = Vector{SVector{2,Float64}}(undef, n)
    av = Vector{ComplexF64}(undef, n)
    for m in 1:n
        a = ix[m]
        q = pos[a]
        pos[a] = q + Int32(1)
        iy[q] = iy0[m]
        dl[q] = delta[m]
        av[q] = amp[m]
    end
    return dl, av, dx, dy, xptr, iy, true
end

@inline function _bloch_sum_direct(env::BlochSumEnvelope, u::SVector{2,Float64})::ComplexF64
    s = zero(ComplexF64)
    @inbounds for m in eachindex(env.amp)
        s += env.amp[m] * cispi(2 * dot(u, env.delta[m]))
    end
    return s
end

@inline function _bloch_sum_fast(env::BlochSumEnvelope, u::SVector{2,Float64},
                                 tx::Vector{ComplexF64},
                                 ty::Vector{ComplexF64})::ComplexF64
    u1, u2 = u[1], u[2]
    @inbounds for a in eachindex(env.dx)
        tx[a] = cispi(2 * u1 * env.dx[a])
    end
    @inbounds for b in eachindex(env.dy)
        ty[b] = cispi(2 * u2 * env.dy[b])
    end
    s = zero(ComplexF64)
    @inbounds for a in eachindex(env.dx)
        t = zero(ComplexF64)
        for m in env.xptr[a]:(env.xptr[a+1]-Int32(1))
            t += env.amp[m] * ty[env.iy[m]]
        end
        s += tx[a] * t
    end
    return s
end

@inline _bloch_scratch(env::BlochSumEnvelope) =
    (Vector{ComplexF64}(undef, length(env.dx)), Vector{ComplexF64}(undef, length(env.dy)))

_has_window(env::BlochSumEnvelope)::Bool = isfinite(env.wradius)

@inline function _bloch_window(env::BlochSumEnvelope, u::SVector{2,Float64})::Float64
    isfinite(env.wradius) || return 1.0
    r = norm(env.wmap * u + env.wshift)
    r <= env.wradius - env.wwidth && return 1.0
    r >= env.wradius + env.wwidth && return 0.0
    env.wwidth == 0 && return r <= env.wradius ? 1.0 : 0.0
    return 0.5 * (1 + cospi((r - env.wradius + env.wwidth) / (2 * env.wwidth)))
end

function (env::BlochSumEnvelope)(p::AbstractVector)
    length(p) == 2 || throw(DimensionMismatch(
        "`BlochSumEnvelope` is two-dimensional but was evaluated at a momentum with " *
        "$(length(p)) components."))
    u = env.Q * SVector{2,Float64}(p[1], p[2]) + env.p0
    win = _bloch_window(env, u)
    win == 0.0 && return 0.0
    s = if env.fast
        tx, ty = _bloch_scratch(env)
        _bloch_sum_fast(env, u, tx, ty)
    else
        _bloch_sum_direct(env, u)
    end
    return win * (env.component === :re ? real(s) : imag(s))
end

@inline function _bloch_value(env::BlochSumEnvelope, p::SVector{2,Float64},
                              tx::Vector{ComplexF64}, ty::Vector{ComplexF64})::Float64
    u = env.Q * p + env.p0
    win = _bloch_window(env, u)
    win == 0.0 && return 0.0
    s = env.fast ? _bloch_sum_fast(env, u, tx, ty) : _bloch_sum_direct(env, u)
    return win * (env.component === :re ? real(s) : imag(s))
end

kernel_parity(::BlochSumEnvelope) = 0

Base.:(==)(a::BlochSumEnvelope, b::BlochSumEnvelope) =
    a.Q == b.Q && a.p0 == b.p0 && a.delta == b.delta && a.amp == b.amp &&
    a.component === b.component && a.wmap == b.wmap && a.wshift == b.wshift &&
    isequal(a.wradius, b.wradius) && a.wwidth == b.wwidth

Base.hash(env::BlochSumEnvelope, h::UInt) =
    hash((BlochSumEnvelope, env.Q, env.p0, env.delta, env.amp, env.component,
          env.wmap, env.wshift, env.wradius, env.wwidth), h)

Base.show(io::IO, env::BlochSumEnvelope) =
    print(io, "BlochSumEnvelope(", length(env.amp), " δ-terms, component=",
          repr(env.component),
          _has_window(env) ? ", window r=$(env.wradius)±$(env.wwidth)" : "", ")")

Base.show(io::IO, ::MIME"text/plain", env::BlochSumEnvelope) = show(io, env)

_make_kernel(term::KLinearTerm, ::Val{D}) where {D} =
    KLinearKernel{D}(_materialize_c(term.c, D, term.name))

_make_kernel(term::KQuadraticTerm, ::Val{D}) where {D} =
    KQuadraticKernel(_materialize_M(term.M, D, term.name))

function _make_kernel(term::KEnvelopeTerm, ::Val{D}) where {D}
    p = term.parity === :even ? 1 : term.parity === :odd ? -1 : 0
    _check_envelope_real(term.f, Val(D), p, term.name)
    return CallbackKernel(term.f, p)
end

function _check_envelope_real(f::Function, ::Val{D}, parity::Int,
                              name::AbstractString)::Nothing where {D}
    probes = SVector{D,Float64}[]
    for scale in (0.317, -1.234, 2.718)
        push!(probes, SVector{D,Float64}(ntuple(m -> scale * (0.4321 + 0.317 * m), D)))
    end
    for (s1, s2) in ((0.911, -0.407), (-0.286, 1.618), (1.207, 0.093))
        push!(probes, SVector{D,Float64}(ntuple(
            m -> s1 * cospi(0.37m + 0.11) + s2 * sinpi(0.53m - 0.29), D)))
    end
    for p in probes
        v = f(p)
        v isa Real || throw(ArgumentError(
            "k-envelope term `$name`: the callback returned $(typeof(v)) at p = $(Tuple(p)); " *
            "the k-dependent scalar must be REAL -- put every phase in `coefficient` or in the " *
            "quantum-number rule. This is what keeps H(k) Hermitian by construction."))
        isfinite(v) || throw(DomainError(v,
            "k-envelope term `$name`: the callback returned a non-finite value at p = $(Tuple(p))."))
        if parity != 0
            vm = f(-p)
            isapprox(Float64(vm), parity * Float64(v);
                     atol=AMPLITUDE_ATOL * max(1.0, abs(v))) || throw(ArgumentError(
                "k-envelope term `$name` declares parity :$(parity == 1 ? "even" : "odd"), " *
                "but f(-p) = $vm while " *
                "$(parity == 1 ? "f(p)" : "-f(p)") = $(parity * v) at p = $(Tuple(p)). A " *
                "mis-declared parity gives a silently wrong hole block under " *
                "`nambu_doubling` -- declare the parity the callback actually has (split a " *
                "mixed-parity envelope into its even and odd parts)."))
        end
    end
    return nothing
end

otimes(term::HoppingTerm, rule::AbstractQnumRule) =
    HoppingTerm(term.selector, compose(term.rule, rule), term.weight, term.t, term.name)

otimes(term::KLinearTerm, rule::AbstractQnumRule) =
    KLinearTerm(term.c, term.k0_frac, term.selector, compose(term.rule, rule),
                term.coefficient, term.name)

otimes(term::KQuadraticTerm, rule::AbstractQnumRule) =
    KQuadraticTerm(term.M, term.k0_frac, term.selector, compose(term.rule, rule),
                   term.coefficient, term.name)

otimes(term::KEnvelopeTerm, rule::AbstractQnumRule) =
    KEnvelopeTerm(term.f, term.parity, term.k0_frac, term.selector, compose(term.rule, rule),
                  term.coefficient, term.name)

_to_amplitude(t::Number) = ComplexF64(t)

_to_amplitude(f::Function) = f

function delta_g(t; dG::AbstractVector{<:Integer},
                 sites_from=Symbol[], sites_to=Symbol[],
                 name::AbstractString="delta_g")
    return HoppingTerm(DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to)),
                       AnyQnum(), UnitWeight(), _to_amplitude(t), String(name))
end

function k_quadratic(M::Union{Real,AbstractMatrix{<:Real}}; coefficient::Number=1.0,
                     dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                     k0::AbstractVector{<:Real}=Float64[],
                     sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                     name::AbstractString="k_quadratic")
    sel = isnothing(dG) ? OnSiteSelector(_site_list(sites)) :
          DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to))
    Mstored = M isa Real ? Float64(M) : Matrix{Float64}(M)
    return KQuadraticTerm(Mstored, Vector{Float64}(k0), sel, AnyQnum(),
                          ComplexF64(coefficient), String(name))
end

function k_envelope(f::Function; coefficient::Number=1.0, parity::Symbol=:none,
                    dG::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                    k0::AbstractVector{<:Real}=Float64[],
                    sites=Symbol[], sites_from=Symbol[], sites_to=Symbol[],
                    name::AbstractString="k_envelope")
    if parity === :none && f isa AbstractEnvelope
        pf = kernel_parity(f)
        parity = pf == 1 ? :even : pf == -1 ? :odd : :none
    end
    sel = isnothing(dG) ? OnSiteSelector(_site_list(sites)) :
          DeltaGSelector(Vector{Int}(dG), _site_list(sites_from), _site_list(sites_to))
    return KEnvelopeTerm(f, parity, Vector{Float64}(k0), sel, AnyQnum(),
                         ComplexF64(coefficient), String(name))
end

function _show_callable(f::Function)::String
    f isa AbstractEnvelope && return repr(f)
    s = string(nameof(f))
    return Base.isidentifier(s) ? s : "var\"$s\""
end

_show_amp(t::ComplexF64)::String = repr(t)

_show_amp(f::Function)::String = _show_callable(f)

function _push_fromto!(kw::Vector{String}, from::Vector{Symbol}, to::Vector{Symbol})
    isempty(from) || push!(kw, "sites_from=$(repr(from))")
    isempty(to) || push!(kw, "sites_to=$(repr(to))")
    return kw
end

function _print_call(io::IO, head::String, arg::String, kw::Vector{String})
    print(io, head, "(", arg)
    isempty(kw) || print(io, "; ", join(kw, ", "))
    print(io, ")")
    return nothing
end

_show_rule_suffix(::IO, ::AnyQnum) = nothing

_show_rule_suffix(io::IO, r::AbstractQnumRule) = print(io, " ⊗ ", r)

function Base.show(io::IO, term::HoppingTerm)
    sel = term.selector
    kw = String[]
    if sel isa OnSiteSelector
        head, default = "onsite", "onsite"
        isempty(sel.sites) || push!(kw, "sites=$(repr(sel.sites))")
    elseif sel isa ShellSelector
        head, default = "hopping", "hopping"
        push!(kw, "shell=$(sel.shell)")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa RadiusSelector
        head, default = "hopping", "hopping"
        push!(kw, "rcut=$(sel.rcut)")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa TranslationSelector
        head, default = "translation_hopping", "translation"
        push!(kw, "dR=$(repr(sel.dR))")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa DeltaGSelector
        head, default = "delta_g", "delta_g"
        push!(kw, "dG=$(repr(sel.dG))")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa ExplicitBondSelector
        head, default = "explicit_bonds", "explicit"
        push!(kw, "i=$(repr(sel.i))", "j=$(repr(sel.j))", "dR=$(repr(sel.dR))")
    else
        head, default = "HoppingTerm", term.name
        push!(kw, "selector=$(repr(sel))")
    end
    term.weight isa UnitWeight || push!(kw, "weight=$(repr(term.weight))")
    term.name == default || push!(kw, "name=$(repr(term.name))")
    _print_call(io, head, _show_amp(term.t), kw)
    _show_rule_suffix(io, term.rule)
    return nothing
end

function _kterm_kwargs(term::KDependentTerm, default_name::String;
                       coefficient::Bool=true)::Vector{String}
    kw = String[]
    coefficient && _kcoefficient(term) != one(ComplexF64) &&
        push!(kw, "coefficient=$(repr(_kcoefficient(term)))")
    sel = _kselector(term)
    if sel isa DeltaGSelector
        push!(kw, "dG=$(repr(sel.dG))")
        _push_fromto!(kw, sel.sites_from, sel.sites_to)
    elseif sel isa OnSiteSelector
        isempty(sel.sites) || push!(kw, "sites=$(repr(sel.sites))")
    end
    isempty(_k0_raw(term)) || push!(kw, "k0=$(repr(_k0_raw(term)))")
    term_name(term) == default_name || push!(kw, "name=$(repr(term_name(term)))")
    return kw
end

function Base.show(io::IO, term::KLinearTerm)
    if term.c isa Pair{Int,Float64}
        kw = String["direction=$(term.c.first)"]
        append!(kw, _kterm_kwargs(term, "dirac"; coefficient=false))
        _print_call(io, "dirac", repr(term.c.second), kw)
    else
        _print_call(io, "k_linear", repr(term.c), _kterm_kwargs(term, "k_linear"))
    end
    _show_rule_suffix(io, term.rule)
    return nothing
end

function Base.show(io::IO, term::KQuadraticTerm)
    _print_call(io, "k_quadratic", repr(term.M), _kterm_kwargs(term, "k_quadratic"))
    _show_rule_suffix(io, term.rule)
    return nothing
end

function Base.show(io::IO, term::KEnvelopeTerm)
    kw = String[]
    _kcoefficient(term) != one(ComplexF64) &&
        push!(kw, "coefficient=$(repr(_kcoefficient(term)))")
    term.parity === :none || push!(kw, "parity=$(repr(term.parity))")
    append!(kw, _kterm_kwargs(term, "k_envelope"; coefficient=false))
    _print_call(io, "k_envelope", _show_callable(term.f), kw)
    _show_rule_suffix(io, term.rule)
    return nothing
end

Base.show(io::IO, ::MIME"text/plain",
          term::Union{HoppingTerm,KLinearTerm,KQuadraticTerm,KEnvelopeTerm}) = show(io, term)

_show_qlevel(x)::String = repr(x)

Base.show(io::IO, ::AnyQnum) = print(io, "AnyQnum()")

Base.show(io::IO, r::ConserveQnum) =
    print(io, "conserve(", join((_show_qlevel(k) for k in r.keys), ", "), ")")

function _print_two_level(io::IO, head::String, key::Symbol, up, down)
    if isequal(up, 1) && isequal(down, -1)
        print(io, head, "(", repr(key), ")")
    else
        print(io, head, "(", repr(key), "; up=", _show_qlevel(up),
              ", down=", _show_qlevel(down), ")")
    end
    return nothing
end

function Base.show(io::IO, r::DiagonalQnum)
    vals = sort!(collect(r.values); by = kv -> _show_qlevel(kv[1]))
    if length(vals) == 1 && vals[1][2] == one(ComplexF64)
        print(io, "project(", repr(r.key), ", ", _show_qlevel(vals[1][1]), ")")
    elseif length(vals) == 2 && Set(last.(vals)) == Set([one(ComplexF64), -one(ComplexF64)])
        up = vals[findfirst(kv -> kv[2] == one(ComplexF64), vals)][1]
        down = vals[findfirst(kv -> kv[2] == -one(ComplexF64), vals)][1]
        _print_two_level(io, "pauli_z", r.key, up, down)
    elseif length(vals) == 2 && all(kv -> kv[2] == one(ComplexF64), vals)
        up, down = vals[1][1], vals[2][1]
        (isequal(down, 1) || isequal(up, -1)) && ((up, down) = (down, up))
        _print_two_level(io, "pauli_0", r.key, up, down)
    else
        print(io, "DiagonalQnum(", repr(r.key), ", Dict(",
              join(("$(_show_qlevel(l)) => $(repr(f))" for (l, f) in vals), ", "), "))")
    end
    return nothing
end

function Base.show(io::IO, r::OffDiagonalQnum)
    prs = sort!(collect(r.pairs); by = kv -> _show_qlevel(kv[1][1]))
    if length(prs) == 1 && prs[1][2] == one(ComplexF64)
        (to, from) = prs[1][1]
        if isequal(to, 1) && isequal(from, -1)
            print(io, "raise(", repr(r.key), ")")
        elseif isequal(to, -1) && isequal(from, 1)
            print(io, "lower(", repr(r.key), ")")
        else
            print(io, "raise(", repr(r.key), "; from=", _show_qlevel(from),
                  ", to=", _show_qlevel(to), ")")
        end
        return nothing
    end
    if length(prs) == 2 && prs[1][1] == reverse(prs[2][1])
        (k1, v1), (k2, v2) = prs
        if v1 == one(ComplexF64) && v2 == one(ComplexF64)
            up, down = k1
            (isequal(down, 1) || isequal(up, -1)) && ((up, down) = (down, up))
            _print_two_level(io, "pauli_x", r.key, up, down)
            return nothing
        end
        if Set([v1, v2]) == Set([ComplexF64(im), ComplexF64(-im)])
            up, down = v1 == -im ? k1 : k2            # the (up, down) element of σ_y is -i
            _print_two_level(io, "pauli_y", r.key, up, down)
            return nothing
        end
    end
    print(io, "OffDiagonalQnum(", repr(r.key), ", Dict(",
          join(("($(_show_qlevel(a)), $(_show_qlevel(b))) => $(repr(f))"
                for ((a, b), f) in prs), ", "), "))")
    return nothing
end

Base.show(io::IO, r::MatrixQnum) =
    print(io, "matrix_rule(", repr(r.key), ", ", repr(r.M), ")")

Base.show(io::IO, r::DirectionalQnum{DeclaredOrder}) =
    print(io, "directional_rule(", _show_callable(r.f), ", ", repr(r.key), ")")

Base.show(io::IO, r::DirectionalQnum) =
    print(io, "DirectionalQnum(", repr(r.key), ", ", _show_callable(r.f), ", ",
          repr(r.order), ")")

Base.show(io::IO, ::DeclaredOrder) = print(io, "DeclaredOrder()")

Base.show(io::IO, o::TwoLevelOrder) =
    print(io, "TwoLevelOrder(", repr(o.up), ", ", repr(o.down), ")")

Base.show(io::IO, o::NamedOrder) = print(io, "NamedOrder(", repr(o.levels), ")")

Base.show(io::IO, r::TwoLevelMatrixQnum) =
    print(io, "TwoLevelMatrixQnum(", repr(r.key), ", ", repr(r.M), ", ",
          repr(r.up), ", ", repr(r.down), ")")

Base.show(io::IO, r::ProductQnumRule) = join(io, r.rules, " ⊗ ")

struct HamiltonianModel{D,K<:KPeriodicity}
    cell::UnitCell{D,K}
    couplings::CouplingList{D}
    delta::Vector{SVector{D,Float64}}
    gauge::Symbol
    name::String

    function HamiltonianModel{D,K}(cell::UnitCell{D,K}, couplings::CouplingList{D},
                                   delta::Vector{SVector{D,Float64}}, gauge::Symbol,
                                   name::String) where {D,K<:KPeriodicity}
        gauge in GAUGE_MODES || throw(ArgumentError(
            "Unknown gauge `$gauge`; expected one of $GAUGE_MODES."
        ))
        couplings.n == n_states(cell) || throw(DimensionMismatch(
            "CouplingList dimension $(couplings.n) does not match n_states(cell) = $(n_states(cell))."
        ))
        length(delta) == n_couplings(couplings) || throw(DimensionMismatch(
            "Phase-displacement table has length $(length(delta)) but the coupling list has " *
            "$(n_couplings(couplings))."
        ))
        return new{D,K}(cell, couplings, delta, gauge, name)
    end
end

function HamiltonianModel(cell::UnitCell{D}, buf::CouplingBuffer{D};
                          gauge::Symbol=DEFAULT_GAUGE, name::AbstractString="",
                          atol::Float64=AMPLITUDE_ATOL,
                          prune::Bool=false)::HamiltonianModel{D} where {D}
    couplings = finalize_couplings(buf, n_states(cell); atol=atol, prune=prune)
    delta = _phase_displacements(cell, couplings)
    return HamiltonianModel{D,_kpar(cell)}(cell, couplings, delta, gauge,
                               String(name))
end

function HamiltonianModel(cell::UnitCell{D}, cl::CouplingList{D};
                          gauge::Symbol=DEFAULT_GAUGE, name::AbstractString="",
                          atol::Float64=AMPLITUDE_ATOL)::HamiltonianModel{D} where {D}
    is_hermitian(cl; atol=atol) || throw(ArgumentError(
        "The coupling list is not Hermitian: some (i, j, ΔR, q) entry lacks or contradicts " *
        "its (j, i, -ΔR, q) partner. A finished `CouplingList` must carry both directions of " *
        "every Hermitian pair exactly -- a corrupt or hand-edited file usually lands here. " *
        "Fix the list, or push one direction per pair into a `CouplingBuffer` and build " *
        "`HamiltonianModel(cell, buffer)`, whose finalize materializes the partners."))
    delta = _phase_displacements(cell, cl)
    return HamiltonianModel{D,_kpar(cell)}(cell, cl, delta, gauge,
                               String(name))
end

Base.length(m::HamiltonianModel) = n_states(m.cell)

ndim(::HamiltonianModel{D}) where {D} = D

has_kdependence(m::HamiltonianModel)::Bool = has_kdependence(m.couplings)

kperiodicity(::HamiltonianModel{D,K}) where {D,K} = K()

_model_kpar(::HamiltonianModel{D,K}) where {D,K} = K

function _phase_displacements(cell::UnitCell{D},
                              c::CouplingList{D})::Vector{SVector{D,Float64}} where {D}
    tau = _effective_positions(cell)
    delta = Vector{SVector{D,Float64}}(undef, n_couplings(c))
    for e in eachindex(c.i)
        delta[e] = SVector{D,Float64}(c.dR[e]) + tau[c.j[e]] - tau[c.i[e]]
    end
    return delta
end

function _effective_positions(cell::UnitCell{D})::Vector{SVector{D,Float64}} where {D}
    z = zero(SVector{D,Float64})
    return [cell.site_kind[cell.state_site[i]] === :plane_wave ? z : b.r_frac
            for (i, b) in pairs(cell.basis)]
end

function compile(cell::UnitCell{D}, terms::Union{AbstractVector,Tuple};
                 gauge::Symbol=DEFAULT_GAUGE,
                 name::AbstractString="",
                 rcut::Union{Nothing,Real}=nothing,
                 atol::Float64=AMPLITUDE_ATOL,
                 prune::Bool=false)::HamiltonianModel{D} where {D}
    gauge in GAUGE_MODES || throw(ArgumentError("Unknown gauge `$gauge`; expected one of $GAUGE_MODES."))
    termvec = _flatten_terms(terms)

    tables = cell_tables(cell)

    # Build the neighbour table only when some term actually needs one. On a FOLDED cell the
    # search runs on the fine (parent) lattice, where the physical neighbours live.
    # `max_radius` is the metric analog of `max_shell` (RadiusSelector / `hopping(t; rcut=)`):
    # when only radius terms consume the table it is built UNGROUPED, so the shell-degeneracy
    # warnings cannot fire on the quasi-continuous distance spectra radius terms exist for.
    deepest = isempty(termvec) ? 0 : maximum(max_shell, termvec)
    rmax = isempty(termvec) ? 0.0 : maximum(max_radius, termvec)
    neighbors = if deepest == 0 && rmax == 0.0
        nothing
    elseif is_folded(cell)
        _fold_fine_neighbor_table(cell, deepest;
                                  rcut=rmax > 0.0 ? max(rmax, something(rcut, 0.0)) : rcut)
    elseif rmax > 0.0
        r = max(rmax, something(rcut, 0.0))
        if deepest > 0
            nt = neighbor_table(cell, r; tables=tables, warn_degenerate=false)
            if length(nt.radii) > deepest
                _warn_near_degenerate(nt.radii, distance_atol(cell.lattice); rtol=SHELL_RTOL)
                nt
            else
                rs = neighbor_table_by_shell(cell, deepest; tables=tables,
                                             warn_degenerate=false).rcut
                neighbor_table(cell, max(r, rs); tables=tables)
            end
        else
            neighbor_table(cell, r; tables=tables, group_shells=false)
        end
    elseif isnothing(rcut)
        neighbor_table_by_shell(cell, deepest; tables=tables)
    else
        neighbor_table(cell, rcut; tables=tables)
    end

    ctx = CompileContext(cell; tables=tables, neighbors=neighbors)

    buf = CouplingBuffer{D}()
    for term in termvec
        before = length(buf)
        begin_term!(buf, term_name(term), _scalar_amplitude(term))
        emit_term!(buf, term, ctx)
        # A term that matched nothing *structurally* is a modelling bug worth naming. A term
        # whose amplitude -- or whose rule MATRIX -- is literally zero is not: sweeping a
        # parameter through 0 is routine, and warning there would make every phase diagram
        # noisy (and misread a U = 0 endpoint as a spelling error).
        if length(buf) == before && !_has_zero_amplitude(term) &&
           !_has_zero_rule(term, cell.schema)
            @warn "Term `$(term_name(term))` emitted no couplings. This usually means a " *
                  "quantum-number rule rejected every pair -- for example a rule on `:spin` " *
                  "applied to a cell that carries no `:spin` quantum number."
        end
    end
    return HamiltonianModel(cell, buf; gauge=gauge, name=name, atol=atol, prune=prune)
end

compile(cell::UnitCell{D}, terms::AbstractHamiltonianTerm...; kwargs...) where {D} =
    compile(cell, collect(terms); kwargs...)

function compile(model::HamiltonianModel{D}, terms::Union{AbstractVector,Tuple};
                 atol::Float64=AMPLITUDE_ATOL,
                 prune::Bool=false)::HamiltonianModel{D} where {D}
    termvec = _flatten_terms(terms)
    c = model.couplings
    buf, kids = adopted_buffer(c)
    for cpl in each_canonical(c)
        push_coupling!(buf, cpl.i, cpl.j, cpl.dR, cpl.q, cpl.w, kids[cpl.kernel], cpl.term)
    end
    if !isempty(termvec)
        for term in termvec
            tn = term_name(term)
            tn in c.term_names && @warn(
                "compile: the new term `$tn` shares its name with a term adopted from the " *
                "parent model, so `reweight(m, \"$tn\" => ...)` will co-address BOTH (every " *
                "term with a matching name rescales together). Give the new term a distinct " *
                "`name=` to keep them separately sweepable.")
        end
        em = compile(model.cell, termvec; gauge=model.gauge, atol=atol)
        ec = em.couplings
        ekids = [kernel_id!(buf, kern) for kern in ec.kernels]
        tbase = length(buf.term_names)
        append!(buf.term_names, ec.term_names)
        append!(buf.term_amps, ec.term_amps)
        for cpl in each_canonical(ec)
            tid = cpl.term == 0 ? 0 : (cpl.term > 0 ? cpl.term + tbase : cpl.term - tbase)
            push_coupling!(buf, cpl.i, cpl.j, cpl.dR, cpl.q, cpl.w, ekids[cpl.kernel], tid)
        end
    end

    return HamiltonianModel(model.cell, buf; gauge=model.gauge, name=model.name,
                            atol=atol, prune=prune)
end

compile(model::HamiltonianModel{D}, terms::AbstractHamiltonianTerm...; kwargs...) where {D} =
    compile(model, collect(terms); kwargs...)

term_names(model::HamiltonianModel)::Vector{String} = unique(model.couplings.term_names)

gauge(model::HamiltonianModel)::Symbol = model.gauge

n_couplings(model::HamiltonianModel)::Int = n_couplings(model.couplings)

_scalar_amplitude(term::BondTerm)::ComplexF64 =
    term.t isa Function ? ComplexF64(NaN, NaN) : ComplexF64(term.t)

_scalar_amplitude(term::KDependentTerm)::ComplexF64 = ComplexF64(term.coefficient)

_scalar_amplitude(::AbstractHamiltonianTerm)::ComplexF64 = ComplexF64(NaN, NaN)

function _flatten_terms(terms)::Vector{AbstractHamiltonianTerm}
    out = AbstractHamiltonianTerm[]
    _flatten_terms!(out, terms)
    return out
end

function _flatten_terms!(out::Vector{AbstractHamiltonianTerm}, x)
    if x isa AbstractHamiltonianTerm
        push!(out, x)
    elseif x isa Union{AbstractVector,Tuple}
        for y in x
            _flatten_terms!(out, y)
        end
    else
        throw(ArgumentError(
            "`compile` expected Hamiltonian terms (or nested collections of them); got $(typeof(x))."
        ))
    end
    return out
end

function change_gauge(model::HamiltonianModel{D}, gauge::Symbol)::HamiltonianModel{D} where {D}
    gauge in GAUGE_MODES || throw(ArgumentError("Unknown gauge `$gauge`; expected one of $GAUGE_MODES."))
    gauge === model.gauge && return model
    return HamiltonianModel{D,_kpar(model.cell)}(model.cell, model.couplings, model.delta,
                                                 gauge, model.name)
end

@inline function _check_k_periodic(lat::BravaisLattice{D}, k::SVector{D,Float64})::Nothing where {D}
    P = nperiodic(lat)
    P == D && return nothing
    for m in (P+1):D
        abs(k[m]) <= LATTICE_ATOL || throw(ArgumentError(
            "k = $(Tuple(k)) has a non-zero component along direction $m, but the lattice is " *
            "periodic only along its first $P direction$(P == 1 ? "" : "s") -- momentum is " *
            "not defined there. Set the trailing component(s) to 0."))
    end
    return nothing
end

function _gauge_phases(cell::UnitCell{D}, k::SVector{D,Float64})::Vector{ComplexF64} where {D}
    return [cell.site_kind[cell.state_site[s]] === :plane_wave ? one(ComplexF64) :
            cispi(2 * dot(k + SVector{D,Float64}(cell.basis[s].G), cell.basis[s].r_frac))
            for s in eachindex(cell.basis)]
end

function bloch_hamiltonian!(H::AbstractMatrix{ComplexF64}, model::HamiltonianModel{D},
                            k_frac::AbstractVector{<:Real}) where {D}
    n = n_states(model.cell)
    size(H) == (n, n) || throw(DimensionMismatch("H is $(size(H)), expected ($n, $n)."))
    length(k_frac) == D || throw(DimensionMismatch("k_frac has length $(length(k_frac)), expected $D."))
    k = SVector{D,Float64}(k_frac)
    _check_k_periodic(model.cell.lattice, k)

    fill!(H, zero(ComplexF64))
    c = model.couplings
    bT = SMatrix{D,D,Float64}(transpose(model.cell.lattice.b))
    for κ in eachindex(c.kernels)
        isempty(c.groups[κ]) && continue
        _accumulate!(H, c.kernels[κ], c, c.groups[κ], k, bT, model.delta)
    end
    if model.gauge === :periodic
        Ws = _gauge_phases(model.cell, k)
        @inbounds for j in axes(H, 2)
            wj = conj(Ws[j])                 # hoisted: `Ws` and `H` could alias for LLVM
            for i in axes(H, 1)
                H[i, j] *= Ws[i] * wj
            end
        end
    end
    return H
end

@inline function _accumulate!(H::AbstractMatrix{ComplexF64}, ::ConstantKernel,
                              c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                              ::SMatrix{D,D,Float64}, δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        H[c.i[e], c.j[e]] += c.w[e] * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

@inline function _accumulate!(H::AbstractMatrix{ComplexF64},
                              kern::CallbackKernel{BlochSumEnvelope},
                              c::CouplingList{2}, r::UnitRange{Int}, k::SVector{2,Float64},
                              bT::SMatrix{2,2,Float64}, δ::Vector{SVector{2,Float64}})::Nothing
    env = kern.f
    tx, ty = _bloch_scratch(env)
    @inbounds for e in r
        kq = k + c.q[e]
        H[c.i[e], c.j[e]] += c.w[e] * _bloch_value(env, bT * kq, tx, ty) *
                             bloch_phase(kq, δ[e])
    end
    return nothing
end

@inline function _accumulate!(H::AbstractMatrix{ComplexF64}, kern,
                              c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                              bT::SMatrix{D,D,Float64}, δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        p = bT * (k + c.q[e])
        H[c.i[e], c.j[e]] += c.w[e] * _kvalue(kern, p) * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

function bloch_hamiltonian(model::HamiltonianModel{D},
                           k_frac::AbstractVector{<:Real})::Matrix{ComplexF64} where {D}
    n = n_states(model.cell)
    return bloch_hamiltonian!(Matrix{ComplexF64}(undef, n, n), model, k_frac)
end

function bands(model::HamiltonianModel{D}, k_frac::AbstractVector{<:Real};
               atol::Float64=HERMITICITY_ATOL, cartesian::Bool=false)::Vector{Float64} where {D}
    k = cartesian ? reciprocal_to_fractional(k_frac, model.cell.lattice) : k_frac
    H = bloch_hamiltonian(model, k)
    has_kdependence(model) && _check_hermitian(H, k_frac, atol)
    return eigvals!(Hermitian(H))
end

function _check_hermitian(H::AbstractMatrix{ComplexF64}, k_frac, atol::Float64)
    # Allocation-free upper-triangle sweep; `abs.(H - H')` would materialize two n×n temporaries
    # per k-point, which at continuum basis sizes is megabytes of garbage per band-structure run.
    dev = 0.0
    n = size(H, 1)
    @inbounds for j in 1:n
        d = abs(imag(H[j, j]))
        d > dev && (dev = d)
        for i in 1:(j - 1)
            d = abs(H[i, j] - conj(H[j, i]))
            d > dev && (dev = d)
        end
    end
    dev <= atol || throw(DomainError(dev,
        "H(k) is not Hermitian at k_frac = $(collect(k_frac)): max |H - H'| = $dev > atol = $atol. " *
        "This is a bug in a term, not a numerical artefact -- the compiled coupling list is " *
        "Hermitian by construction, so check any k-dependent term's prefactors (a k-envelope " *
        "returning a complex value would land here)."
    ))
    return nothing
end

function bands!(buf::AbstractMatrix{Float64}, H::AbstractMatrix{ComplexF64},
                model::HamiltonianModel{D}, kpath; atol::Float64=HERMITICITY_ATOL) where {D}
    check = has_kdependence(model)   # constant lists are Hermitian by construction; see `bands`
    for (idx, k) in enumerate(kpath)
        bloch_hamiltonian!(H, model, k)
        check && _check_hermitian(H, k, atol)
        buf[:, idx] = eigvals!(Hermitian(H))
    end
    return buf
end

function bands(model::HamiltonianModel{D}, kpath::AbstractVector{<:AbstractVector{<:Real}};
               atol::Float64=HERMITICITY_ATOL, cartesian::Bool=false)::Matrix{Float64} where {D}
    n = n_states(model.cell)
    ks = cartesian ? [reciprocal_to_fractional(k, model.cell.lattice) for k in kpath] :
                     collect(kpath)
    buf = Matrix{Float64}(undef, n, length(ks))
    threaded_chunks(length(ks)) do chunk
        H = zeros(ComplexF64, n, n)
        bands!(view(buf, :, chunk), H, model, view(ks, chunk); atol=atol)
    end
    return buf
end

function Base.show(io::IO, ::MIME"text/plain", m::HamiltonianModel{D}) where {D}
    c = m.couplings
    ngrp = count(!isempty, c.groups)
    nc = n_couplings(c)
    print(io, "HamiltonianModel{", D, "}")
    isempty(m.name) || print(io, " ", repr(m.name))
    print(io, ": ", n_states(m.cell), " states on ", n_sites(m.cell), " sites, ",
          nc, " coupling", nc == 1 ? "" : "s",
          " in ", ngrp, " kernel group", ngrp == 1 ? "" : "s",
          ", gauge ", repr(m.gauge), ", k-periodicity ", repr(kperiodicity(m)))
    if isempty(c.term_names)
        print(io, "\n  terms: none recorded (imported or legacy model; `reweight` unavailable)")
        return nothing
    end
    print(io, "\n  terms:")
    nterms = length(c.term_names)
    nshow = nterms > 11 ? 10 : nterms
    for t in 1:nshow
        amp = c.term_amps[t]
        print(io, "\n    ", c.term_names[t], " => ",
              isfinite(amp) ? repr(amp) : "(callback amplitude)")
    end
    nterms > nshow && print(io, "\n    … ", nterms - nshow,
                            " more (see `term_amplitudes`)")
    return nothing
end

Base.show(io::IO, m::HamiltonianModel{D}) where {D} =
    print(io, "HamiltonianModel{", D, "}(", repr(m.name), ", n=", n_states(m.cell),
          ", couplings=", n_couplings(m.couplings), ")")

struct SparseBlochWorkspace{D}
    S::SparseMatrixCSC{ComplexF64,Int}
    slot::Vector{Int}
    source::UInt
end

function sparse_bloch_workspace(model::HamiltonianModel{D})::SparseBlochWorkspace{D} where {D}
    n = n_states(model.cell)
    c = model.couplings
    isempty(c.i) && throw(ArgumentError(
        "The model has no couplings at all; a sparse H(k) would be identically zero."))
    S = sparse(copy(c.i), copy(c.j), zeros(ComplexF64, n_couplings(c)), n, n)
    slotof(i, j) = begin
        r = searchsortedfirst(view(rowvals(S), nzrange(S, j)), i)
        first(nzrange(S, j)) + r - 1
    end
    slot = [slotof(c.i[e], c.j[e]) for e in eachindex(c.i)]
    return SparseBlochWorkspace{D}(S, slot, c.pattern_id)
end

function sparse_bloch_hamiltonian!(ws::SparseBlochWorkspace{D}, model::HamiltonianModel{D},
                                   k_frac::AbstractVector{<:Real}) where {D}
    length(k_frac) == D || throw(DimensionMismatch(
        "k_frac has length $(length(k_frac)), expected $D."))
    k = SVector{D,Float64}(k_frac)
    _check_k_periodic(model.cell.lattice, k)
    c = model.couplings
    ws.source == c.pattern_id || throw(ArgumentError(
        "Sparsity-pattern mismatch: this model's coupling list was not produced from the " *
        "workspace's model by `reweight`/`change_gauge`. Slot indices are pattern-specific " *
        "-- an entry-count coincidence would assemble garbage silently, so the pattern " *
        "fingerprint is checked, not the size. Build a fresh `sparse_bloch_workspace(model)`."))
    nz = nonzeros(ws.S)
    fill!(nz, zero(ComplexF64))

    bT = SMatrix{D,D,Float64}(transpose(model.cell.lattice.b))
    for κ in eachindex(c.kernels)
        isempty(c.groups[κ]) && continue
        _accumulate_sparse!(nz, ws.slot, c.kernels[κ], c, c.groups[κ], k, bT, model.delta)
    end
    if model.gauge === :periodic
        Ws = _gauge_phases(model.cell, k)
        rows = rowvals(ws.S)
        @inbounds for j in 1:size(ws.S, 2), idx in nzrange(ws.S, j)
            nz[idx] *= Ws[rows[idx]] * conj(Ws[j])
        end
    end
    return ws.S
end

@inline function _accumulate_sparse!(nz::Vector{ComplexF64}, slot::Vector{Int}, ::ConstantKernel,
                                     c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                                     ::SMatrix{D,D,Float64},
                                     δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        nz[slot[e]] += c.w[e] * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

@inline function _accumulate_sparse!(nz::Vector{ComplexF64}, slot::Vector{Int},
                                     kern::CallbackKernel{BlochSumEnvelope},
                                     c::CouplingList{2}, r::UnitRange{Int}, k::SVector{2,Float64},
                                     bT::SMatrix{2,2,Float64},
                                     δ::Vector{SVector{2,Float64}})::Nothing
    env = kern.f
    tx, ty = _bloch_scratch(env)
    @inbounds for e in r
        kq = k + c.q[e]
        nz[slot[e]] += c.w[e] * _bloch_value(env, bT * kq, tx, ty) * bloch_phase(kq, δ[e])
    end
    return nothing
end

@inline function _accumulate_sparse!(nz::Vector{ComplexF64}, slot::Vector{Int}, kern,
                                     c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                                     bT::SMatrix{D,D,Float64},
                                     δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        p = bT * (k + c.q[e])
        nz[slot[e]] += c.w[e] * _kvalue(kern, p) * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

function bz_map(f, model::HamiltonianModel{D}, ks::AbstractVector; matrix::Symbol=:dense,
                kwargs...) where {D}
    nk = length(ks)
    nk == 0 && throw(ArgumentError("`ks` is empty; there is nothing to map over."))
    first_val = bloch_solver(model; matrix=matrix)(ks[1]; kwargs...) |> r -> f(ks[1], r)
    out = Vector{typeof(first_val)}(undef, nk)
    out[1] = first_val
    nk == 1 && return out
    threaded_chunks(nk - 1) do chunk
        solve = bloch_solver(model; matrix=matrix)     # ONE buffer per chunk
        for c in chunk
            i = c + 1
            out[i] = f(ks[i], solve(ks[i]; kwargs...))
        end
    end
    return out
end

function bloch_solver(model::HamiltonianModel{D}; matrix::Symbol=:dense) where {D}
    matrix in (:dense, :sparse) || throw(ArgumentError(
        "`matrix` must be `:dense` or `:sparse`; got `$matrix`."))
    if matrix === :sparse
        ws = sparse_bloch_workspace(model)
        return function (k; sigma=nothing, kwargs...)
            S = sparse_bloch_hamiltonian!(ws, model, k)
            meth = isnothing(sigma) ? :lanczos : :shift_invert
            r = diagonalize(S; method=meth, sigma=sigma, kwargs...)
            return (values=r.values, vectors=r.vectors, gauge=model.gauge)
        end
    end
    H = zeros(ComplexF64, n_states(model.cell), n_states(model.cell))
    return function (k; kwargs...)
        bloch_hamiltonian!(H, model, k)
        r = diagonalize(H; overwrite=true, kwargs...)
        return (values=r.values, vectors=r.vectors, gauge=model.gauge)
    end
end

struct BravaisType
    name::Symbol
    dimension::Int
    family::String
    centring::String
    params::Vector{Symbol}
    points_per_conventional::Int
    description::String
end

function _bravais_vectors(name::Symbol, p::NamedTuple)
    # ---------------- 1D ----------------
    name === :lp && return [[p.a]]

    # ---------------- 2D ----------------
    name === :mp && return [[p.a, 0.0], [p.b * cos(p.gamma), p.b * sin(p.gamma)]]
    name === :op && return [[p.a, 0.0], [0.0, p.b]]
    name === :oc && return [[p.a / 2, -p.b / 2], [p.a / 2, p.b / 2]]
    name === :tp && return [[p.a, 0.0], [0.0, p.a]]
    name === :hp && return [[p.a, 0.0], [-p.a / 2, p.a * sqrt(3) / 2]]

    # ---------------- 3D: cubic ----------------
    name === :cP && return [[p.a, 0.0, 0.0], [0.0, p.a, 0.0], [0.0, 0.0, p.a]]
    name === :cF && return [[0.0, p.a / 2, p.a / 2], [p.a / 2, 0.0, p.a / 2], [p.a / 2, p.a / 2, 0.0]]
    name === :cI && return [[-p.a / 2, p.a / 2, p.a / 2], [p.a / 2, -p.a / 2, p.a / 2],
                            [p.a / 2, p.a / 2, -p.a / 2]]

    # ---------------- 3D: tetragonal ----------------
    name === :tP && return [[p.a, 0.0, 0.0], [0.0, p.a, 0.0], [0.0, 0.0, p.c]]
    name === :tI && return [[-p.a / 2, p.a / 2, p.c / 2], [p.a / 2, -p.a / 2, p.c / 2],
                            [p.a / 2, p.a / 2, -p.c / 2]]

    # ---------------- 3D: orthorhombic ----------------
    name === :oP && return [[p.a, 0.0, 0.0], [0.0, p.b, 0.0], [0.0, 0.0, p.c]]
    name === :oS && return [[p.a / 2, -p.b / 2, 0.0], [p.a / 2, p.b / 2, 0.0], [0.0, 0.0, p.c]]
    name === :oI && return [[-p.a / 2, p.b / 2, p.c / 2], [p.a / 2, -p.b / 2, p.c / 2],
                            [p.a / 2, p.b / 2, -p.c / 2]]
    name === :oF && return [[0.0, p.b / 2, p.c / 2], [p.a / 2, 0.0, p.c / 2], [p.a / 2, p.b / 2, 0.0]]

    # ---------------- 3D: hexagonal / rhombohedral ----------------
    name === :hP && return [[p.a, 0.0, 0.0], [-p.a / 2, p.a * sqrt(3) / 2, 0.0], [0.0, 0.0, p.c]]
    name === :hR && return [[ p.a / 2, -p.a / (2 * sqrt(3)), p.c / 3],
                            [ 0.0,      p.a / sqrt(3),       p.c / 3],
                            [-p.a / 2, -p.a / (2 * sqrt(3)), p.c / 3]]

    # ---------------- 3D: monoclinic (unique axis b) ----------------
    name === :mP && return [[p.a, 0.0, 0.0], [0.0, p.b, 0.0],
                            [p.c * cos(p.beta), 0.0, p.c * sin(p.beta)]]
    name === :mS && return [[p.a / 2, p.b / 2, 0.0], [-p.a / 2, p.b / 2, 0.0],
                            [p.c * cos(p.beta), 0.0, p.c * sin(p.beta)]]

    # ---------------- 3D: triclinic ----------------
    if name === :aP
        ca, cb, cg, sg = cos(p.alpha), cos(p.beta), cos(p.gamma), sin(p.gamma)
        nu2 = 1 - ca^2 - cb^2 - cg^2 + 2 * ca * cb * cg
        nu2 > 0 || throw(DomainError((p.alpha, p.beta, p.gamma),
            "The given angles do not describe a valid triclinic cell " *
            "(1 − cos²α − cos²β − cos²γ + 2cosα cosβ cosγ = $nu2 ≤ 0)."
        ))
        return [[p.a, 0.0, 0.0],
                [p.b * cg, p.b * sg, 0.0],
                [p.c * cb, p.c * (ca - cb * cg) / sg, p.c * sqrt(nu2) / sg]]
    end

    throw(ArgumentError("No primitive-vector rule for Bravais type `$name`."))
end

const BRAVAIS_TYPES = Dict{Symbol,BravaisType}(
    # ---------------- 1D ----------------
    :lp => BravaisType(:lp, 1, "linear", "primitive", [:a], 1, "linear chain"),

    # ---------------- 2D ----------------
    :mp => BravaisType(:mp, 2, "monoclinic",   "primitive", [:a, :b, :gamma], 1, "oblique"),
    :op => BravaisType(:op, 2, "orthorhombic", "primitive", [:a, :b],         1, "rectangular"),
    :oc => BravaisType(:oc, 2, "orthorhombic", "centred",   [:a, :b],         2, "centred rectangular (rhombic)"),
    :tp => BravaisType(:tp, 2, "tetragonal",   "primitive", [:a],             1, "square"),
    :hp => BravaisType(:hp, 2, "hexagonal",    "primitive", [:a],             1, "triangular (2D hexagonal net)"),

    # ---------------- 3D ----------------
    :aP => BravaisType(:aP, 3, "triclinic",    "primitive",     [:a, :b, :c, :alpha, :beta, :gamma], 1, "triclinic"),
    :mP => BravaisType(:mP, 3, "monoclinic",   "primitive",     [:a, :b, :c, :beta], 1, "monoclinic, unique axis b"),
    :mS => BravaisType(:mS, 3, "monoclinic",   "base-centred",  [:a, :b, :c, :beta], 2, "base-centred monoclinic (C), unique axis b"),
    :oP => BravaisType(:oP, 3, "orthorhombic", "primitive",     [:a, :b, :c], 1, "orthorhombic"),
    :oS => BravaisType(:oS, 3, "orthorhombic", "base-centred",  [:a, :b, :c], 2, "base-centred orthorhombic (C)"),
    :oI => BravaisType(:oI, 3, "orthorhombic", "body-centred",  [:a, :b, :c], 2, "body-centred orthorhombic"),
    :oF => BravaisType(:oF, 3, "orthorhombic", "face-centred",  [:a, :b, :c], 4, "face-centred orthorhombic"),
    :tP => BravaisType(:tP, 3, "tetragonal",   "primitive",     [:a, :c], 1, "tetragonal"),
    :tI => BravaisType(:tI, 3, "tetragonal",   "body-centred",  [:a, :c], 2, "body-centred tetragonal"),
    :hR => BravaisType(:hR, 3, "trigonal",     "rhombohedral",  [:a, :c], 3, "rhombohedral, hexagonal setting"),
    :hP => BravaisType(:hP, 3, "hexagonal",    "primitive",     [:a, :c], 1, "hexagonal"),
    :cP => BravaisType(:cP, 3, "cubic",        "primitive",     [:a], 1, "simple cubic"),
    :cI => BravaisType(:cI, 3, "cubic",        "body-centred",  [:a], 2, "body-centred cubic (bcc)"),
    :cF => BravaisType(:cF, 3, "cubic",        "face-centred",  [:a], 4, "face-centred cubic (fcc)"),
)

const BRAVAIS_ALIASES = Dict{Symbol,Symbol}(
    # 1D
    :linear => :lp, :chain => :lp,
    # 2D
    :oblique => :mp,
    :rectangular => :op,
    :centered_rectangular => :oc, :centred_rectangular => :oc, :rhombic => :oc,
    :square => :tp,
    :triangular => :hp, :hexagonal2d => :hp,
    # 3D
    :triclinic => :aP,
    :monoclinic => :mP, :monoclinic_base_centered => :mS,
    :mC => :mS, :mI => :mS,
    :orthorhombic => :oP,
    :orthorhombic_base_centered => :oS, :oC => :oS,
    :orthorhombic_body_centered => :oI,
    :orthorhombic_face_centered => :oF,
    :tetragonal => :tP, :tetragonal_body_centered => :tI,
    :rhombohedral => :hR, :trigonal => :hR,
    :hexagonal => :hP,
    :cubic => :cP, :simple_cubic => :cP, :sc => :cP,
    :bcc => :cI, :body_centered_cubic => :cI,
    :fcc => :cF, :face_centered_cubic => :cF,
)

const NOT_BRAVAIS = Dict{Symbol,String}(
    :honeycomb  => "the triangular lattice (`:triangular`) with a two-site basis at [1/3,2/3] and [2/3,1/3]",
    :graphene   => "the triangular lattice (`:triangular`) with a two-site basis at [1/3,2/3] and [2/3,1/3]",
    :kagome     => "the triangular lattice (`:triangular`) with a three-site basis",
    :diamond    => "the face-centred cubic lattice (`:fcc`) with a two-site basis at [0,0,0] and [1/4,1/4,1/4]",
    :zincblende => "the face-centred cubic lattice (`:fcc`) with a two-site basis at [0,0,0] and [1/4,1/4,1/4]",
    :rocksalt   => "the face-centred cubic lattice (`:fcc`) with a two-site basis at [0,0,0] and [1/2,1/2,1/2]",
    :nacl       => "the face-centred cubic lattice (`:fcc`) with a two-site basis at [0,0,0] and [1/2,1/2,1/2]",
    :perovskite => "the simple cubic lattice (`:cubic`) with a five-site basis",
    :wurtzite   => "the hexagonal lattice (`:hexagonal`) with a four-site basis",
)

function bravais_canonical_name(name::Symbol)::Symbol
    haskey(BRAVAIS_TYPES, name) && return name
    haskey(BRAVAIS_ALIASES, name) && return BRAVAIS_ALIASES[name]

    if haskey(NOT_BRAVAIS, name)
        throw(ArgumentError(
            "`$name` is not a Bravais lattice: it is $(NOT_BRAVAIS[name]). " *
            "Build the Bravais lattice with `bravais_lattice`, then attach the basis states to a " *
            "`UnitCell`."
        ))
    end
    lowered = lowercase(String(name))
    near = sort([k for k in keys(BRAVAIS_TYPES) if lowercase(String(k)) == lowered]; by=String)
    if !isempty(near)
        suggestion = length(near) == 1 ? "Did you mean `$(near[1])`?" :
            "Did you mean one of `" * join(near, "`, `") * "`?"
        throw(ArgumentError(
            "Unknown Bravais lattice `$name`. " * suggestion * " " *
            "Pearson symbols are case sensitive: `:hp` is the 2D hexagonal net and `:hP` the 3D " *
            "hexagonal lattice. The spelled-out aliases (`:triangular`, `:hexagonal`) avoid this."
        ))
    end

    throw(ArgumentError(
        "Unknown Bravais lattice `$name`. Known types: $(sort(collect(keys(BRAVAIS_TYPES)))). " *
        "Aliases: $(sort(collect(keys(BRAVAIS_ALIASES))))."
    ))
end

function bravais_lattice(name::Symbol;
                         a::Union{Nothing,Real}=nothing,
                         b::Union{Nothing,Real}=nothing,
                         c::Union{Nothing,Real}=nothing,
                         alpha::Union{Nothing,Real}=nothing,
                         beta::Union{Nothing,Real}=nothing,
                         gamma::Union{Nothing,Real}=nothing,
                         degrees::Bool=true)
    canonical = bravais_canonical_name(name)
    spec = BRAVAIS_TYPES[canonical]

    supplied = Dict{Symbol,Any}()
    for (key, val) in ((:a, a), (:b, b), (:c, c), (:alpha, alpha), (:beta, beta), (:gamma, gamma))
        isnothing(val) || (supplied[key] = val)
    end

    missing_params = [k for k in spec.params if !haskey(supplied, k)]
    isempty(missing_params) || throw(ArgumentError(
        "`$canonical` ($(spec.description)) requires $(spec.params); missing $(missing_params)."
    ))
    extra = [k for k in keys(supplied) if !(k in spec.params)]
    isempty(extra) || throw(ArgumentError(
        "`$canonical` ($(spec.description)) uses only $(spec.params), but $(sort(extra)) " *
        "were also given. Either drop them or name a lattice type that uses them."
    ))
    vals = Dict{Symbol,Float64}()
    for k in spec.params
        v = Float64(supplied[k])
        if k in (:a, :b, :c)
            v > 0 || throw(DomainError(v, "Lattice parameter `$k` must be positive."))
            vals[k] = v
        else
            r = degrees ? deg2rad(v) : v
            (0 < r < π) || throw(DomainError(supplied[k],
                "Angle `$k` must lie strictly between 0 and 180 degrees."))
            vals[k] = r
        end
    end

    p = NamedTuple{Tuple(spec.params)}(Tuple(vals[k] for k in spec.params))
    return BravaisLattice(_bravais_vectors(canonical, p))
end

function Base.show(io::IO, ::MIME"text/plain", t::BravaisType)
    println(io, "BravaisType `", t.name, "` -- ", t.description)
    println(io, "  dimension : ", t.dimension)
    println(io, "  family    : ", t.family, " (", t.centring, ")")
    println(io, "  parameters: ", t.params)
    print(io,   "  primitive cell = 1/", t.points_per_conventional, " of the conventional cell")
end

function _assert_fully_periodic(lat::BravaisLattice{D}) where {D}
    P = nperiodic(lat)
    P == D && return nothing
    P == 0 && throw(ArgumentError(
        "This lattice has no periodic direction (a Γ-only finite cluster): there is no " *
        "Brillouin zone, hence no Bravais class to identify. All $D of its rows are " *
        "orthonormal padding axes, so any lattice constant read off them would be a property " *
        "of the coordinate frame and not of the cluster. The cluster's whole spectrum lives at " *
        "the single momentum k = $(Tuple(zeros(D))); ask for it with `bands(model, zeros($D))` " *
        "rather than along a k-path."))
    throw(ArgumentError(
        "This lattice is periodic along only $P of its $D directions; a partially periodic " *
        "lattice has no $(D)D Bravais class, because rows $(P + 1):$D are orthonormal padding " *
        "axes rather than translations. Its periodic block DOES classify, and there are two " *
        "ways to ask. For the class itself: `identify_bravais(periodic_block(lat))`, which " *
        "returns the $(P)D type and a $(P)x$(P) map valid for rows 1:$P of this lattice. For " *
        "k-points, nothing is needed: `high_symmetry_points(lat)` / `high_symmetry_path(lat)` " *
        "reduce to the $(P)-periodic block automatically and return k-vectors in the basis of " *
        "this lattice, with the aperiodic components pinned to 0."))
end

function identify_bravais(lat::BravaisLattice{1}; rtol::Float64=BRAVAIS_RTOL)
    _assert_fully_periodic(lat)
    return (name=:lp, params=(a=norm(primitive_vector(lat, 1)),),
            M=SMatrix{1,1,Int}(1))
end

function identify_bravais(lat::BravaisLattice{2}; rtol::Float64=BRAVAIS_RTOL)
    _assert_fully_periodic(lat)
    a1, a2, Uu = _lagrange_reduce(lat)
    l1, l2 = norm(a1), norm(a2)
    cosg = clamp(dot(a1, a2) / (l1 * l2), 0.0, 1.0)   # reduced + sign-normalized: γ ∈ [60°, 90°]

    equal = abs(l1 - l2) <= rtol * l2
    right = cosg <= rtol                              # γ = 90°
    hexa  = abs(cosg - 0.5) <= rtol                   # γ = 60°
    centred = abs(dot(a1, a2) - dot(a1, a1) / 2) <= rtol * dot(a1, a1)
    name, params = if equal && hexa
        :hp, (a=l1,)
    elseif equal && right
        :tp, (a=l1,)
    elseif right
        :op, (a=l1, b=l2)
    elseif equal
        :oc, (a=norm(a2 - a1), b=norm(a1 + a2))
    elseif centred
        :oc, NamedTuple{(:a, :b)}(minmax(l1, norm(2 * a2 - a1)))
    else
        :mp, (a=l1, b=l2, gamma=rad2deg(acos(cosg)))
    end
    canon = bravais_lattice(name; params...)
    c1, c2, Uc = _lagrange_reduce(canon)
    Gu = _gram_of_rows(_rows_to_matrix((a1, a2)))
    Gc = _gram_of_rows(_rows_to_matrix((c1, c2)))
    tol = rtol * max(dot(a2, a2), 1.0) * 10
    W = _metric_match(Gu, Gc, tol)
    isnothing(W) && throw(ErrorException(
        "identify_bravais classified the lattice as `$name` $params but could not match the " *
        "reduced metrics; this is an internal inconsistency, not a usage error."))
    M = round.(Int, inv(SMatrix{2,2,Float64}(Uu)) * W * Uc)
    return (name=name, params=params, M=SMatrix{2,2,Int}(M))
end
