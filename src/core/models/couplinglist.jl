##################################################################
###### export only the following functions from this module ######
##################################################################

export CouplingBuffer, CouplingList
export push_coupling!, kernel_id!, finalize_couplings, n_couplings, is_hermitian
export begin_term!, adopt_provenance!
export is_canonical_coupling, each_canonical, n_canonical, adopted_buffer




#################################################################
###### The (i, j, ΔR) sign and index convention             #####
#################################################################

"""
The coupling convention, fixed once for the whole package.

A coupling `(i, j, ΔR, q, w, κ)` contributes

    H[i, j] += w * f_κ( bᵀ(k + q) ) * exp(2πi (k + q) ⋅ δ)

with `δ = ΔR + τ_j - τ_i` in the `:atomic` gauge (the default) and `δ = ΔR` in `:periodic`,
`q = (G_i + G_j)/2 - (K_i + K_j)/2 - k0_term` the entry's momentum offset, and `f_κ` its kernel
(`f ≡ 1` for a constant coupling -- a bond). The electron hops **from** basis state `j` in the
cell displaced by `ΔR`, **to** basis state `i` in the home cell: `i` is the row (bra), `j` the
column (ket).

Hermiticity reads, for every kernel alike,

    w_ji(-ΔR, q) = conj(w_ij(ΔR, q))

because `q` is symmetric under `i ↔ j`, `δ → -δ` under the swap, and every kernel scalar is
REAL -- the one invariant the whole design rests on.

Every existing tight-binding model has `q = 0` on its bonds and every continuum model has
`δ = 0` on its k-entries, so `(k+q)·δ` reduces to the historical `k·δ` exactly there; the `q` in
the phase is what makes folded combs (`fold`) and per-state momentum origins work.

This is the single most bug-prone convention in the package. It is stated here, tested directly,
and every emitter is expected to honour it.
"""
const COUPLING_CONVENTION_DOC = nothing

"""
    _endpoint_q(Gi, Gj, k0i, k0j) -> SVector{D,Float64}

The endpoint-derived momentum offset of one coupling, `(G_i + G_j)/2 − (K_i + K_j)/2` -- the
midpoint convention stated above, in ONE place. The term emitter, the Wannier90 importer and
the legacy HDF5 reader all derive `q` through this function (the emitter subtracts its
per-term `k0` on top), so the convention cannot drift between them.
"""
@inline _endpoint_q(Gi::SVector{D,Int}, Gj::SVector{D,Int},
                    k0i::SVector{D,Float64}, k0j::SVector{D,Float64}) where {D} =
    (SVector{D,Float64}(Gi) + SVector{D,Float64}(Gj)) / 2 - (k0i + k0j) / 2

"""
    ConstantKernel

The kernel of a k-independent coupling -- a bond. `kvalue ≡ 1`, even parity. Always kernel id 1
of every `CouplingBuffer`/`CouplingList`, so the constant group is *literally* the bond loop.
"""
struct ConstantKernel end

"""
    is_canonical_coupling(i, j, dR) -> Bool

Whether `(i, j, dR)` is the canonical representative of the Hermitian pair it forms with
`(j, i, -dR)`.

The first non-zero component of `dR` must be positive, else `i <= j`. Exactly one of the two
orderings satisfies this, except for the self-conjugate case `i == j, dR == 0`, which is its own
partner (and canonical). `dR` is rational (integer for ordinary orbitals, fractional after a
`fold`), and the rule is exact either way.

This predicate is the stored-once contract made consumable: a compiled [`CouplingList`](@ref)
stores BOTH directions of every pair explicitly, so a transform that maps entries one by one
must map only the canonical half and let the finalize step regenerate the partners --
[`each_canonical`](@ref) packages exactly that filter, warning included.
"""
@inline function is_canonical_coupling(i::Int, j::Int, dR::SVector{D,Rational{Int}})::Bool where {D}
    for m in 1:D
        dR[m] > 0 && return true
        dR[m] < 0 && return false
    end
    return i <= j
end

"""
    _is_self_conjugate(i, j, dR) -> Bool

Whether a coupling is its own Hermitian partner, i.e. an on-site entry `i == j`, `ΔR == 0`.
Such an entry must carry a real amplitude.
"""
@inline _is_self_conjugate(i::Int, j::Int, dR::SVector{D,Rational{Int}}) where {D} =
    (i == j) && iszero(dR)




#################################################################
###### Mutable COO buffer that terms emit into              #####
#################################################################

"""
    CouplingBuffer{D}()

A growable, type-concrete COO buffer that Hamiltonian terms push couplings into during
compilation. The unified successor of the former bond buffer and per-term k-pattern buffers:
one buffer holds every kernel's entries, tagged by kernel id.

Fields:
-------
- `i`, `j`: target (bra) and source (ket) basis-state indices
- `dR`: the translation, rational (integer for ordinary orbitals, fractional after a `fold`)
- `q`: the entry's momentum offset (see the convention above)
- `w`: the k-independent prefactor
- `kernel`: index into `kernels`; `1` is always `ConstantKernel()`
- `kernels`: the kernel table
- `term`: index into `term_names` recording which term emitted the entry (`0` = untagged,
  e.g. an imported model); the provenance that makes [`reweight`](@ref) possible
- `term_names`, `term_amps`: the term registry -- name and the term's scalar amplitude at
  compile time (`NaN` when the amplitude is not a scalar, e.g. a callback)
- `current_term`: the id `push_coupling!` records; set by [`begin_term!`](@ref)
"""
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

"""
    begin_term!(buf, name, amp) -> Int

Register a term in the buffer's provenance table and make it the current one: every
subsequent `push_coupling!` is tagged with its id. `amp` is the term's scalar amplitude
(`NaN` when it has none -- a callback amplitude); [`reweight`](@ref) reads it back.
"""
function begin_term!(buf::CouplingBuffer, name::AbstractString, amp::ComplexF64)::Int
    push!(buf.term_names, String(name))
    push!(buf.term_amps, amp)
    buf.current_term = length(buf.term_names)
    return buf.current_term
end

"""
    adopt_provenance!(buf, cl) -> Nothing

Copy a source list's term registry (names + compile-time scalar amplitudes) into a fresh
buffer, so a transform that maps entries one by one can carry each entry's term tag through:
pass the source tag (the `term` field of [`each_canonical`](@ref)'s tuples) as the `tid`
argument of [`push_coupling!`](@ref). This is what keeps [`reweight`](@ref) working on the
transformed model.

Adoption must come FIRST -- a buffer that already registered terms is refused, because the
adopted ids must stay literally equal to the source ids. Terms the transform adds itself are
registered afterwards with [`begin_term!`](@ref) and get the ids after the adopted block.
"""
function adopt_provenance!(buf::CouplingBuffer, cl)::Nothing
    isempty(buf.term_names) || throw(ArgumentError(
        "The buffer already carries a term registry; adopt provenance before registering terms."))
    append!(buf.term_names, cl.term_names)
    append!(buf.term_amps, cl.term_amps)
    return nothing
end

"""
    push_coupling!(buf, i, j, dR, q, w, kid[, tid]) -> Nothing

Append one coupling. Either direction may be pushed; `finalize_couplings` canonicalizes.
Integer `dR` is accepted and promoted. Without an explicit term id `tid`, the entry is
tagged with the buffer's current term (see [`begin_term!`](@ref)).
"""
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

"""
    kernel_id!(buf, kern) -> Int

The id of `kern` in the buffer's kernel table, registering it if new. Built-in kernels compare
structurally (two `dirac` terms with the same velocity share one kernel). A `CallbackKernel`
whose payload is an `AbstractEnvelope` also compares structurally -- same payload type, equal
payloads, equal parity -- so two terms built from equal `GaussianEnvelope(σ)`s share one
kernel group; a raw-closure `CallbackKernel` compares by identity only -- two equal-in-spirit
closures are two kernels, which is correct but non-canonical.
"""
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




#################################################################
###### The compiled, Hermitian coupling list                #####
#################################################################

"""
    CouplingList{D}

The compiled Hamiltonian: ONE flat, sorted, coalesced, explicitly Hermitian list of couplings,
covering bonds and k-dependent entries alike -- a bond is the constant-kernel special case.

This is the canonical internal form. Every term lowers into it once, and `bloch_hamiltonian!`
is one branchless pass per kernel group. It is also what `supercell`, `fold`, `nambu_doubling`,
the symmetry layer and HDF5 serialization all operate on: one representation, one code path
each, where the previous split (bond list + per-term k-patterns) needed two.

Both directions are stored explicitly rather than one half being implied, for the same three
reasons as before: the per-k loop needs no branch, the serialized form is self-contained, and
downstream transforms never reason about which half was kept.

Fields:
-------
- `n::Int`: number of basis states (the matrix dimension)
- `i`, `j`, `dR`, `q`, `w`, `kernel`: the couplings, sorted by `(kernel, j, i, dR)` --
  column-fast, so consecutive writes into the column-major dense `H` (and the CSC sparse slots)
  are local
- `kernels`: the kernel table; `kernels[1]` is `ConstantKernel()`
- `groups`: `groups[κ]` is the contiguous entry range of kernel `κ` (possibly empty)

`q` is a **stored per-entry datum**: it is endpoint-derived at first compile
(`(G_i+G_j)/2 - (K_i+K_j)/2 - k0_term`) but NOT re-derivable afterwards -- `nambu_doubling`'s
hole entries carry `-q` under unflipped state labels. Transforms operate on the stored column.

Reading the list from user code (gauge verification, plaquette-flux checks, model diffing) is
supported and simple -- the fields above ARE the contract. `model.couplings` holds it, both
directions of every Hermitian pair are present, and the physical displacement of entry `e` is
`δ = dR[e] + τ_j - τ_i` with `τ = state_positions(model.cell)`:

    cl = model.couplings
    for e in 1:n_couplings(cl)
        cl.kernel[e] == 1 || continue                       # constant (bond) entries only
        i, j, dR, t = cl.i[e], cl.j[e], cl.dR[e], cl.w[e]   # t = ⟨i,0|H|j,dR⟩
    end

Sparsity-pattern identity:
--------------------------
`pattern_id` fingerprints exactly the columns the sparse slot map depends on --
`hash((n, i, j, dR, kernel))`; the values `w` and `q` (and the kernel *objects*) affect
values, not slots. Lists that share it assemble correctly through one
`sparse_bloch_workspace`: `reweight` and `change_gauge` preserve the columns, so a sweep
reuses one CSC pattern for the whole phase diagram. It is derived, never stored on disk --
a reloaded model legitimately needs a fresh workspace.
"""
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
    ### provenance: which term each entry came from (0 = untagged, e.g. imported models),
    ### plus the term registry -- name and scalar compile-time amplitude (NaN if not scalar).
    ### The SIGN of the tag records the entry's orientation relative to how the term emitted
    ### it: negative means the stored amplitude is the CONJUGATE of the emitted one (the entry
    ### was flipped to the canonical direction at finalize, or it is a Hermitian partner, or a
    ### BdG hole entry). `reweight` scales positive-tagged entries by f and negative-tagged
    ### ones by conj(f) -- without the sign, a complex factor would silently mis-phase every
    ### bond written in the non-canonical direction. The assembly loops never read these
    ### columns.
    term::Vector{Int}
    term_names::Vector{String}
    term_amps::Vector{ComplexF64}
    ### the sparsity-pattern fingerprint (see the docstring); defaults to the deterministic
    ### hash of the slot-determining columns, so two lists with identical columns -- a
    ### `reweight`, a re-read of the same file -- agree on it without sharing an object
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

"""
    n_couplings(c::CouplingList) -> Int

Number of stored couplings, counting both directions and every kernel.
"""
@inline n_couplings(c::CouplingList)::Int = length(c.w)

Base.length(c::CouplingList) = length(c.w)

"""
    has_kdependence(c::CouplingList) -> Bool

Whether any non-constant kernel carries entries -- i.e. whether `H(k)` has any k-dependence
beyond the Bloch phases.
"""
has_kdependence(c::CouplingList)::Bool =
    any(κ -> !(c.kernels[κ] isa ConstantKernel) && !isempty(c.groups[κ]),
        eachindex(c.kernels))

"""
    finalize_couplings(buf::CouplingBuffer{D}, nstates; atol=AMPLITUDE_ATOL, prune=false)
        -> CouplingList{D}

Compile a buffer of emitted couplings into a Hermitian `CouplingList` -- the ONE finalize,
where the previous design ran the same pipeline twice (`finalize_bonds` / `_finalize_kterm`,
whose docstring already called them "deliberately the same").

Pipeline:
---------
1. **Canonicalize** every entry to the canonical direction, conjugating the amplitude when
   flipped (`q` is `i↔j`-symmetric and rides along unchanged).
2. **Sort and coalesce** duplicates by summing. The key is `(kernel, i, j, ΔR, q)`: the `q` in
   the key is NOT optional -- two terms with the same kernel but different momentum origins
   (valley-resolved `dirac` at ±K on the same states) hit the same `(i, j, ΔR)` and must not
   merge.
3. **Check reality** of self-conjugate entries, after coalescing.
4. **Materialise the Hermitian partner** of every non-self-conjugate entry: `(j, i, -ΔR, q,
   conj(w), κ)` -- valid for every kernel because the kernel scalar is real.
5. **Sort** by `(kernel, j, i, ΔR)` -- column-fast for the writes -- and record the kernel
   groups.
"""
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
        ### the one funnel every amplitude passes through -- including callback-computed ones,
        ### which no constructor can validate. A NaN here would otherwise propagate into H(k)
        ### and come out as NaN bands with no pointer back to the coupling that caused it.
        isfinite(w) || throw(DomainError(w,
            "Coupling ($i, $j, $(Tuple(dR))) carries a non-finite amplitude $w. If the term " *
            "uses a callback amplitude, it returned NaN or Inf for this bond."
        ))
        if is_canonical_coupling(i, j, dR)
            ci[e] = i; cj[e] = j; cR[e] = dR; cw[e] = w
            ct[e] = buf.term[e]
        else
            ci[e] = j; cj[e] = i; cR[e] = -dR; cw[e] = conj(w)
            ### the flip conjugates the stored amplitude, so the orientation sign of the term
            ### tag flips with it -- this is what keeps complex `reweight` factors exact
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
            ### the partner carries the conjugated amplitude, hence the flipped orientation sign
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

"""
    _kernel_groups(kernel_ids_sorted, nkernels) -> Vector{UnitRange{Int}}

The contiguous entry range of each kernel in a kernel-major-sorted id column.
"""
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

"""
    each_canonical(cl::CouplingList{D}) -> iterator

Iterate the canonical halves of a compiled list, as NamedTuples

    (; e, i, j, dR, q, w, kernel, term)

with `e` the entry's index into `cl` and the rest the entry's columns verbatim (`kernel` is the
id into `cl.kernels`, `term` the orientation-signed provenance tag). This is the read side of
the transform recipe (see the `HamiltonianModel(cell, buf)` constructor): map each canonical
entry onto the new cell, `push_coupling!` it, and let the finalize inside the constructor
re-materialize every Hermitian partner.

!!! warning "Push ONLY these entries -- never both directions"
    The source list stores BOTH directions of every Hermitian pair explicitly. Mapping both
    would make the finalize step coalesce each coupling with its own image and **double every
    amplitude** -- the single easiest way to write a silently wrong transform. Push only what
    this iterator yields; the partners are regenerated at finalize, and nothing may rely on
    both directions being present in the buffer.
"""
function each_canonical(cl::CouplingList{D}) where {D}
    return ((; e, i=cl.i[e], j=cl.j[e], dR=cl.dR[e], q=cl.q[e], w=cl.w[e],
             kernel=cl.kernel[e], term=cl.term[e])
            for e in eachindex(cl.i) if is_canonical_coupling(cl.i[e], cl.j[e], cl.dR[e]))
end

"""
    n_canonical(cl::CouplingList) -> Int

Number of canonical halves in the list -- exactly the entries [`each_canonical`](@ref) yields,
i.e. the pushes a transform that maps entries one-to-one will make. The sizing input of
[`adopted_buffer`](@ref)'s capacity hint.
"""
n_canonical(cl::CouplingList)::Int =
    count(e -> is_canonical_coupling(cl.i[e], cl.j[e], cl.dR[e]), eachindex(cl.i))

"""
    adopted_buffer(cl::CouplingList{D}; per_canonical=1) -> (CouplingBuffer{D}, Vector{Int})

The standard transform preamble in ONE call: a fresh buffer with `cl`'s term registry adopted,
its kernel table re-registered (`kids[κ]` is the buffer id of source kernel `κ` -- the id to
pass to `push_coupling!` for an entry with `kernel == κ`), and capacity reserved for
`n_canonical(cl) * per_canonical` pushes (`per_canonical` = how many entries the transform
emits per canonical source entry: `|det M|` for a supercell, `L` for an added axis, ...).

This replaces the four-line preamble every shipped transform used to repeat:

    buf = CouplingBuffer{D}()
    adopt_provenance!(buf, cl)
    kids = [kernel_id!(buf, kern) for kern in cl.kernels]
"""
function adopted_buffer(cl::CouplingList{D};
                        per_canonical::Integer=1)::Tuple{CouplingBuffer{D},Vector{Int}} where {D}
    buf = CouplingBuffer{D}()
    adopt_provenance!(buf, cl)
    kids = [kernel_id!(buf, kern) for kern in cl.kernels]
    sizehint!(buf, n_canonical(cl) * Int(per_canonical))
    return buf, kids
end

"""
    is_hermitian(c::CouplingList{D}; atol=AMPLITUDE_ATOL) -> Bool

Check that the list satisfies `w_ji(-ΔR, q, κ) = conj(w_ij(ΔR, q, κ))` for every coupling.

`finalize_couplings` guarantees this by construction; this is the independent check used in
tests, on HDF5 input, and after any transform that rearranges a list.
"""
function is_hermitian(c::CouplingList{D}; atol::Float64=AMPLITUDE_ATOL)::Bool where {D}
    ### the term tag is deliberately NOT in the key: Hermiticity is a statement about H(k),
    ### which sums over terms -- and imported/legacy lists may be untagged anyway
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




#################################################################
###### Display                                              #####
#################################################################

### fractional dR entries print as rationals, integer ones as plain integers
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
        ### the trailing ' marks a stored amplitude that is the CONJUGATE of the emitted one
        t == 0 || print(io, "  term=", c.term_names[abs(t)], t > 0 ? "" : "'")
    end
    m > nshow && print(io, "\n  … ", m - nshow, " more")
    return nothing
end

Base.show(io::IO, c::CouplingList{D}) where {D} =
    print(io, "CouplingList{", D, "}(n=", c.n, ", couplings=", n_couplings(c),
          ", kernels=", length(c.kernels), ")")
