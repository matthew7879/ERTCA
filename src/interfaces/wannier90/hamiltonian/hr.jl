##################################################################
###### export only the following functions from this module ######
##################################################################

export read_wannier90_hr, write_wannier90_hr

import Dates




#############################################################
###### Wannier90 seedname_hr.dat import                 #####
#############################################################

"""
    read_wannier90_hr(cell::UnitCell{D}, filename; gauge=:periodic, centres=nothing,
                      name="wannier90", atol=1e-6, prune=false) -> HamiltonianModel{D}

Read a Wannier90 `seedname_hr.dat` file into a `HamiltonianModel` on the given cell.

The cell provides everything the file does not carry: the lattice, the state positions and any
internal axes. `n_states(cell)` must equal the file's `num_wann`, **with the cell's basis in the
file's Wannier-function order** -- index `m` of the file is state `m` of the cell.

Args:
-----
- `cell`: the unit cell, one state per Wannier function, in the file's order
- `filename`: path to the `seedname_hr.dat` file
- `gauge`: `:periodic` (default -- the file's own gauge) or `:atomic` -- see the gauge
  contract below
- `centres`: the true Wannier centres, either a vector of **cartesian** positions (length
  `num_wann`, each of length `D`, in the units of the lattice vectors) or the path to a Wannier90
  `seedname_centres.xyz` file. Verified against the cell, and out-of-cell centres are wrap-shift
  compensated -- see below.
- `name`: the model's name
- `atol`: the file tolerance (see Notes)
- `prune`: drop couplings whose coalesced amplitude falls below `atol`

Returns:
--------
- `HamiltonianModel{D}`

The gauge contract:
-------------------
The file stores `H_mn(R) = ⟨m,0|H|n,R⟩`, which is basis-convention independent: the **same**
imported numbers are correct in either gauge, and the flag only selects the Bloch-sum phase
convention at assembly. Position-dependent observables (`berry_curvature`, `spectral_weight`)
always read the atomic assembly whatever the flag says (an O(1) flip) -- so what decides
whether they come out physical is not the flag but the STATE POSITIONS, which must be the true
Wannier centres:

- `gauge=:periodic` (the default) phases with `e^{2πi k·ΔR}` alone -- Wannier90's own Fourier
  gauge, so the import needs nothing the file does not carry: spectra and every
  eigenvalue-derived quantity are exact whatever the positions. Position-dependent observables
  still read the atomic view of the same model; when every state sits at the origin (a bare
  `hr.dat` import, centres never attached) that view coincides with the zero-position
  gauge, and those observables WARN instead of reporting it as embedding-aware physics.
  Attach the centres (`centres=`, or build the cell at them) to make them physical.
- `gauge=:atomic` phases every coupling with the state positions,
  `e^{2πi k·(ΔR + τ_n - τ_m)}` -- the physical convention **only if the `τ` are the true
  Wannier centres**. A cell of more than one state whose positions are all ≈ 0 (within the
  cell's `atol`) is therefore **refused** under an explicitly requested `:atomic`: those are
  almost never the real centres, and the assembly would silently equal the periodic convention
  while claiming to be atomic. Either build the cell with its states at the Wannier centres
  (or pass `centres=`, which declares and verifies them), or keep the default `:periodic`,
  which accepts the zero-position convention openly and leaves the diagnostic to the
  observable layer.

The `:atomic` sum over `R + c_n - c_m` is what the tight-binding literature calls
convention I; the file's own convention is `:periodic`.

Verified centres and the wrap shift:
------------------------------------
When `centres=` is given, each centre is converted with `real_to_fractional` and must match the
corresponding cell state's position **modulo integer lattice translations**, within the cell's
`atol`; the first mismatch raises, naming the state. A centre lying outside the home cell differs
from the stored (wrapped into `[0,1)`) position by an integer `W_n`, and the importer then
compensates every element's translation,

    R  ->  R + W_n - W_m        for element (m, n, R),

which is exactly what keeps the atomic-gauge phase pinned to the true centres:
`ΔR + τ_n - τ_m = R + c_n - c_m` whether or not the `c` were wrapped. (In the periodic gauge the
same shift is a pure `e^{2πi k·W}` relabelling of the eigenvectors; eigenvalues are unchanged.)

File format (Wannier90 user guide): a comment line; `num_wann`; `nrpts`; `nrpts` Wigner-Seitz
degeneracy integers (15 per line); then one line per matrix element,

    R1 R2 R3  m n  Re(H)  Im(H)

holding `H_mn(R) = ⟨m,0|H|n,R⟩` -- the hop *from* Wannier function `n` in cell `R` *to* `m` in the
home cell, exactly this package's bond convention `(i, j, ΔR) = (m, n, R)`. Each amplitude is
divided by its R-vector's degeneracy on read, so `H(k) = Σ_R H(R)/ndegen(R) · e^{ik·R}` comes out
as the package's `:periodic`-gauge Bloch sum.

Notes:
------
- The file stores **both** directions of every coupling. Mapping them all and letting
  `finalize_couplings` coalesce would double every element, so only the canonical half is kept --
  after checking that the two stored directions agree, which is the file-level Hermiticity check.
  A coupling whose reverse is absent (a hand-trimmed file) is kept and Hermitized by the compiler.
- `atol` covers the file's finite precision (Wannier90 prints 6 decimals): it bounds the allowed
  `|H_nm(-R) - conj(H_mn(R))|` deviation and the imaginary part tolerated on on-site energies. It
  is deliberately looser than the compiler's `AMPLITUDE_ATOL`. (Positions are compared with the
  cell's own `atol`, not this one.)
- For a `D < 3` cell the trailing components of every `R` -- and of every centre -- must vanish.
- The compiled coupling list IS the model: an imported Hamiltonian has no symbolic term source.

Arriving in 3D -- import, bands, a shareable file:
--------------------------------------------------

    lat = bravais_lattice(:fcc; a=5.43)
    m   = read_wannier90_hr(lat, "si_hr.dat"; centres="si_centres.xyz")
    plot_bands(m; bravais=:cF)                        # 3D: DECLARE the Bravais type
    h5_write("si.h5", m; note="Si valence manifold, wannier90 v3.1")
    h5_info("si.h5")                                  # what a colleague sees before loading

In 3D [`identify_bravais`](@ref) raises by design, so the `bravais=` declaration is THE route
to the standard k-path (validated against the lattice metric; see
[`standard_kpath_nodes`](@ref)). For sharing the import, `h5_write(...; note=)` stamps the
file's provenance and [`h5_info`](@ref) reads it back without loading the model.
"""
function read_wannier90_hr(cell::UnitCell{D}, filename::AbstractString;
                           gauge::Symbol=:periodic, name::AbstractString="wannier90",
                           atol::Float64=1e-6, prune::Bool=false,
                           centres::Union{Nothing,AbstractString,AbstractVector}=nothing
                           )::HamiltonianModel{D} where {D}
    D <= 3 || throw(ArgumentError("Wannier90 files are at most three-dimensional; the cell has D = $D."))
    ### validated up front rather than left to the model constructor at the very end: the
    ### zero-position guard below branches on the gauge, so a typo'd flag must not skip it
    gauge in GAUGE_MODES || throw(ArgumentError(
        "Unknown gauge `$gauge`; expected one of $GAUGE_MODES."
    ))

    ### resolve the declared centres against the cell BEFORE touching the file: W[n] is the
    ### integer translation by which centre n was wrapped into the home cell, zero when no
    ### centres are declared or all of them already lie inside
    W = isnothing(centres) ? nothing :
        _wannier_wrap_shifts(cell, _wannier_centres_cartesian(centres, Val(D), atol);
                             atol=atol)

    ### the atomic-gauge contract (see the docstring): more than one state, every position ≈ 0,
    ### and no explicit centre declaration means the τ's are almost certainly NOT the Wannier
    ### centres -- refuse the explicit :atomic request rather than let it claim physicality.
    ### (Under the default :periodic the import succeeds and the diagnostic lives at the
    ### observable layer: `berry_curvature`/`spectral_weight` warn on the same predicate.)
    if gauge === :atomic && isnothing(centres) && n_states(cell) > 1 &&
       all(b -> _same_site(b.r_frac, zero(SVector{D,Float64}), cell.atol), cell.basis)
        throw(ArgumentError(
            "Atomic-gauge Wannier90 import needs the true Wannier centres, but all " *
            "$(n_states(cell)) states of the cell sit at the origin. The :atomic gauge phases " *
            "couplings with e^{2πi k·(ΔR + τ_n - τ_m)}; with every τ = 0 it silently equals the " *
            ":periodic convention while claiming to be atomic, and Berry curvature, spectral " *
            "weight and every position-dependent quantity come out in the unphysical " *
            "zero-position convention. Either (a) build the cell with its states at the Wannier " *
            "centres, or pass `centres=` (a vector of cartesian positions, or the " *
            "seedname_centres.xyz path) to declare and verify them, or (b) keep the default " *
            "`gauge=:periodic`, which accepts the zero-position convention openly (spectra " *
            "are exact; position-dependent observables warn instead of claiming physicality)."
        ))
    end

    lines = readlines(filename)
    length(lines) >= 4 || throw(ArgumentError(
        "`$filename` is not a Wannier90 _hr.dat file: expected a comment line, num_wann, nrpts " *
        "and a degeneracy list."
    ))
    nw = parse(Int, strip(lines[2]))
    nrpts = parse(Int, strip(lines[3]))
    nw == n_states(cell) || throw(DimensionMismatch(
        "`$filename` holds num_wann = $nw Wannier functions but the cell declares " *
        "$(n_states(cell)) states. The cell must carry one state per Wannier function, in the " *
        "file's order."
    ))

    ### the degeneracy list: nrpts integers, 15 per line
    ndeg = Int[]
    row = 4
    while length(ndeg) < nrpts && row <= length(lines)
        append!(ndeg, parse.(Int, split(strip(lines[row]))))
        row += 1
    end
    length(ndeg) == nrpts || throw(ArgumentError(
        "`$filename` lists $(length(ndeg)) Wigner-Seitz degeneracies for nrpts = $nrpts."
    ))
    all(>(0), ndeg) || throw(ArgumentError("`$filename` carries a non-positive WS degeneracy."))

    ### parse the matrix elements; degeneracy index follows the order of first appearance of each
    ### R block, which is the order of the degeneracy list
    table = Dict{Tuple{Int,Int,NTuple{D,Int}},ComplexF64}()
    rindex = Dict{NTuple{3,Int},Int}()
    for ln in @view lines[row:end]
        s = strip(ln)
        isempty(s) && continue
        f = split(s)
        length(f) >= 7 || throw(ArgumentError("Malformed _hr.dat line in `$filename`: `$ln`"))
        R3 = (parse(Int, f[1]), parse(Int, f[2]), parse(Int, f[3]))
        m, n = parse(Int, f[4]), parse(Int, f[5])
        (1 <= m <= nw && 1 <= n <= nw) || throw(ArgumentError(
            "`$filename` addresses Wannier index outside 1:$nw: `$ln`"
        ))
        all(R3[d] == 0 for d in (D+1):3) || throw(ArgumentError(
            "`$filename` carries the translation $R3, which is non-zero outside the cell's " *
            "$D dimension(s)."
        ))
        irpt = get!(rindex, R3, length(rindex) + 1)
        irpt <= nrpts || throw(ArgumentError(
            "`$filename` uses more distinct R vectors than its declared nrpts = $nrpts."
        ))
        key = (m, n, ntuple(d -> R3[d], D))
        haskey(table, key) && throw(ArgumentError(
            "`$filename` repeats the matrix element (R=$R3, m=$m, n=$n)."
        ))
        t = ComplexF64(parse(Float64, f[6]), parse(Float64, f[7])) / ndeg[irpt]
        iszero(t) || (table[key] = t)
    end

    ### keep one direction per coupling, verifying the file's Hermiticity where both are present
    buf = CouplingBuffer{D}()
    sizehint!(buf, length(table) ÷ 2 + 1)
    tables = cell_tables(cell)
    ### an imported bond's momentum offset is endpoint-derived exactly like a compiled one --
    ### zero on the ordinary G ≡ 0 Wannier cells this reader targets
    qof(i, j) = _endpoint_q(tables.G[i], tables.G[j], tables.k0[i], tables.k0[j])
    for ((m, n, Rt), t) in table
        R = SVector{D,Int}(Rt)
        ### the Hermiticity check runs on the FILE's own coordinates -- the partner of the file
        ### element (m, n, R) is the file element (n, m, -R), wrap shifts or not
        partner = get(table, (n, m, ntuple(d -> -Rt[d], D)), nothing)
        if !isnothing(partner) && abs(partner - conj(t)) > atol
            throw(ArgumentError(
                "`$filename` is not Hermitian: H[$n,$m](-R) = $partner but conj(H[$m,$n](R)) = " *
                "$(conj(t)) at R = $Rt, deviation $(abs(partner - conj(t))) > atol = $atol."
            ))
        end
        ### wrap-shift compensation: the file's R addresses the Wannier function whose centre is
        ### c_n + R; our state n stores the wrapped τ_n = c_n - W_n, so the same function lives in
        ### cell R + W_n of our labelling -- and the bra shifts by W_m likewise. This is a
        ### bijection in R for fixed (m, n), so it cannot collide two file elements, and it maps
        ### the file's Hermitian pair onto a compensated pair (m,n,Rc)/(n,m,-Rc).
        Rc = isnothing(W) ? R : R + W[n] - W[m]
        if is_canonical_coupling(m, n, SVector{D,Rational{Int}}(Rc))
            push_coupling!(buf, m, n, Rc, qof(m, n), t, 1)
        elseif isnothing(partner)
            ### half-specified file: keep the coupling by flipping it into the canonical direction
            push_coupling!(buf, n, m, -Rc, qof(n, m), conj(t), 1)
        end
    end

    ### the public gate: finalize (re-materializing every Hermitian partner), derive the phase
    ### displacements and the k-periodicity from the cell
    return HamiltonianModel(cell, buf; gauge=gauge, name=name, atol=atol, prune=prune)
end


"""
    read_wannier90_hr(lattice::BravaisLattice{D}, filename; centres, schema=TRIVIAL_SCHEMA,
                      site_names=nothing, kwargs...) -> HamiltonianModel{D}

Build the cell **from** the Wannier centres, then read the file onto it -- the ergonomic import
path:

    read_wannier90_hr(lat, "si_hr.dat"; centres="si_centres.xyz")

One `:localized` site per Wannier function, at its centre converted to fractional coordinates
(wrapped into the home cell; an out-of-cell centre is wrap-shift compensated exactly as the cell
method documents, because the centres are passed through to it for verification). The basis order
is the centres' order, which is the file's Wannier-function order.

Args:
-----
- `lattice`: the Bravais lattice, whose cartesian units the centres are in
- `filename`: path to the `seedname_hr.dat` file
- `centres`: **required** -- a vector of cartesian centre positions (each of length `D`) or the
  path to a Wannier90 `seedname_centres.xyz` file
- `schema`: the internal label space, needed only when several Wannier functions share one
  centre (see Notes); defaults to the trivial schema
- `site_names`: one `Symbol` per **distinct** centre, in first-appearance order; defaults to
  `:w1, :w2, ...`
- remaining keywords (`gauge`, `name`, `atol`, `prune`) are passed through to the cell method

Returns:
--------
- `HamiltonianModel{D}`

Notes:
------
Wannier functions sharing one centre -- an atom-centred orbital manifold -- cannot each be their
own localized site (site identity is the position). Pass `schema=` declaring an internal axis
whose level count reaches the largest shared multiplicity: co-centred functions then become one
site, and the g-th function at a centre, in file order, is stored at the schema's g-th internal
state. With the default trivial schema, coincident centres are refused with this rule spelled
out. Centres are considered coincident within the lattice tolerance (`LATTICE_ATOL`); centres
closer than twice that without coinciding are rejected by the cell constructor's separation
check, like any other ambiguous site registry.
"""
function read_wannier90_hr(lattice::BravaisLattice{D}, filename::AbstractString;
                           centres::Union{Nothing,AbstractString,AbstractVector}=nothing,
                           schema::InternalSchema=TRIVIAL_SCHEMA,
                           site_names::Union{Nothing,AbstractVector{Symbol}}=nothing,
                           atol::Float64=1e-6, kwargs...)::HamiltonianModel{D} where {D}
    isnothing(centres) && throw(ArgumentError(
        "The BravaisLattice method builds the cell FROM the Wannier centres; pass `centres=` " *
        "(a vector of cartesian positions, or the seedname_centres.xyz path). To use a cell " *
        "you already built, call `read_wannier90_hr(cell, filename)` instead."
    ))
    cart = _wannier_centres_cartesian(centres, Val(D), atol)
    isempty(cart) && throw(ArgumentError("`centres` declares no Wannier centres."))

    ### cluster the centres into sites: one site per distinct centre, co-centred functions
    ### distinguished by consecutive internal states (file order), which the duplicate check of
    ### the cell constructor requires and the schema must be able to name
    reps = SVector{D,Float64}[]              # cluster representatives, canonicalized
    first_at = Int[]                         # the WF that opened each cluster, for diagnostics
    mult = Int[]                             # running multiplicity per cluster
    state_site = Vector{Int}(undef, length(cart))
    internal = Vector{Int}(undef, length(cart))
    for (n, c) in pairs(cart)
        fc = canonicalize_fracpos(real_to_fractional(c, lattice))
        s = findfirst(r -> _same_site(fc, r, LATTICE_ATOL), reps)
        if isnothing(s)
            push!(reps, fc); push!(first_at, n); push!(mult, 0)
            s = length(reps)
        end
        mult[s] += 1
        mult[s] <= n_internal(schema) || throw(ArgumentError(
            "Wannier centres $(first_at[s]) and $n coincide at $(Tuple(reps[s])) (fractional), " *
            "and the schema holds only $(n_internal(schema)) internal state(s). Co-centred " *
            "Wannier functions cannot each be their own localized site: pass `schema=` " *
            "declaring an internal axis with at least the shared multiplicity (the g-th " *
            "function at a centre, in file order, becomes the schema's g-th internal state), " *
            "or build the `UnitCell` yourself and call the cell method with `centres=`."
        ))
        state_site[n] = s
        internal[n] = mult[s]
    end

    names = if isnothing(site_names)
        [Symbol("w", s) for s in eachindex(reps)]
    else
        length(site_names) == length(reps) || throw(DimensionMismatch(
            "`site_names` holds $(length(site_names)) names for $(length(reps)) distinct " *
            "Wannier centres (one name per distinct centre, in first-appearance order)."
        ))
        collect(Symbol, site_names)
    end

    basis = [BasisState{D}(canonicalize_fracpos(real_to_fractional(cart[n], lattice)),
                           zero(SVector{D,Int}), internal[n], schema) for n in eachindex(cart)]
    cell = UnitCell{D}(lattice, schema, names, reps, basis;
                       site_kind=fill(:localized, length(reps)), state_site=state_site)

    ### delegate with the ORIGINAL cartesian centres: the cell just built holds them wrapped, so
    ### the cell method recovers each wrap shift W_n and compensates the imported translations
    return read_wannier90_hr(cell, filename; centres=cart, atol=atol, kwargs...)
end




#############################################################
###### Wannier90 seedname_hr.dat export                 #####
#############################################################

"""
    write_wannier90_hr(model::HamiltonianModel{D}, filename) -> filename

Write a tight-binding `HamiltonianModel` as a Wannier90 `seedname_hr.dat` file -- the exit to
tbmodels, PythTB, WannierTools and this package's own [`read_wannier90_hr`](@ref).

What is emitted, and the gauge:
-------------------------------
Each line holds `H_mn(R) = ⟨m,0|H|n,R⟩` -- exactly the model's stored constant-kernel
amplitude `w` under the package's bond convention `(i, j, ΔR) = (m, n, R)`. These numbers are
GAUGE-FREE: the model's `gauge` flag selects a Bloch-sum phase convention at assembly and
never touches the stored amplitudes, so an `:atomic` and a `:periodic` spelling of the same
physics write the identical file, and the flag decides nothing here. What the file does NOT
carry are the Wannier centres -- they are the cell's state positions, and Wannier90 ships them
in the separate `seedname_centres.xyz`. Hand them back to the reader (read onto the same cell,
or `read_wannier90_hr(lattice, path; centres=[fractional_to_real(τ, lattice) for τ in
state_positions(cell)])`) to recover the physical `:atomic` assembly; without them only
eigenvalue-derived quantities are exact (the reader's gauge contract documents why).

Layout (Wannier90 user guide, mirrored by [`read_wannier90_hr`](@ref)): a date comment line;
`num_wann`; `nrpts`; the Wigner-Seitz degeneracies -- all written as `1`, 15 per line, because
a compiled model has no supercell double-counting to undo -- then one line per matrix element,

    R1 R2 R3   m n   Re(H)  Im(H)

Every `(R, m, n)` element of every listed `R` block is written, zeros included (Wannier90's
own layout, which position-indexed parsers rely on), `m` fastest within a block. The compiled
coupling list stores BOTH directions of every Hermitian pair, so the emitted file is
explicitly Hermitian and the reader's pair check passes with zero deviation. Amplitudes are
printed as shortest round-trip decimals rather than at a fixed column width, so
`write_wannier90_hr` followed by `read_wannier90_hr` reproduces `H(k)` to floating-point
equality. For `D < 3` the trailing `R` components are written as 0.

Refused, by name, BEFORE the target file is touched (the format cannot hold these):
- k-dependent kernel entries (`linear`/`quadratic`/`polynomial`/envelopes): an _hr.dat holds
  constant hoppings only. [`discretize`](@ref) a k·p model onto a lattice first, or share the
  model via [`h5_write`](@ref), which round-trips every registered kernel.
- a model on plane-wave states: a Wannier function is localized by definition, and the
  reciprocal (`G`) / momentum-origin (`k0`) labels have no _hr.dat spelling.
- a folded model (fractional `ΔR` from [`fold`](@ref)): the file's `R` are integer lattice
  translations. Write the supercell spelling of the same physics instead.
- a nonzero per-entry momentum offset `q` -- same reason, on hand-assembled lists.
- `D > 3`: the format is at most three-dimensional.

A partially periodic model (`nperiodic(lattice) < D` -- a ribbon, a slab) writes fine: its
`ΔR` vanish along the padding directions by construction, so every emitted `R` is a genuine
translation. The file itself carries no periodicity declaration -- the cell handed back to
`read_wannier90_hr` restores it, and a foreign reader treating the padding directions as
periodic sees zero hopping there, which changes no spectrum.

# Example

    m = compile(cell, [hopping(-1.0; shell=1), onsite(0.3; sites=[:A])])
    write_wannier90_hr(m, "graphene_hr.dat")
    m2 = read_wannier90_hr(cell, "graphene_hr.dat")     # H(k) equal to m's, to 1e-12
"""
function write_wannier90_hr(model::HamiltonianModel{D},
                            filename::AbstractString)::String where {D}
    D <= 3 || throw(ArgumentError(
        "Wannier90 files are at most three-dimensional; the model has D = $D."))
    c = model.couplings
    cell = model.cell

    if has_kdependence(model)
        live = unique(String[_hr_kernel_name(c.kernels[κ])
                             for κ in 2:length(c.kernels) if !isempty(c.groups[κ])])
        throw(ArgumentError(
            "`write_wannier90_hr` writes constant hoppings t_ij(R) only, but this model " *
            "carries k-dependent kernel entries ($(join(live, ", "))): the _hr.dat format " *
            "has no k-dependence beyond the Bloch phases. `discretize` a k·p model onto a " *
            "lattice first, or share the model via `h5_write`, which round-trips every " *
            "registered kernel."))
    end
    for s in 1:n_sites(cell)
        cell.site_kind[s] === :plane_wave && throw(ArgumentError(
            "`write_wannier90_hr` cannot express site `$(cell.site_names[s])`: it is " *
            ":plane_wave, and a Wannier90 _hr.dat is a localized-orbital format with no " *
            "spelling for reciprocal (G) or momentum-origin (k0) labels. Share continuum " *
            "models via `h5_write`."))
    end
    is_folded(cell) && throw(ArgumentError(
        "`write_wannier90_hr` cannot express a model compiled on a `fold`ed cell: folding " *
        "stores fractional ΔR and reciprocal combs, and the _hr.dat format holds integer " *
        "lattice translations only. Write the supercell spelling of the same physics " *
        "instead."))
    ### backstops for hand-assembled lists that dodge `is_folded`: the format is exactly
    ### t_ij(R) over integer R with no momentum offsets, so anything else refuses by entry
    for e in c.groups[1]
        all(r -> denominator(r) == 1, c.dR[e]) || throw(ArgumentError(
            "`write_wannier90_hr` needs integer translations, but entry " *
            "(i=$(c.i[e]), j=$(c.j[e])) carries the fractional ΔR = $(Tuple(c.dR[e])); the " *
            "_hr.dat format has no fractional R (this is `fold`'s spelling -- write the " *
            "supercell model instead)."))
        iszero(c.q[e]) || throw(ArgumentError(
            "`write_wannier90_hr` cannot express entry (i=$(c.i[e]), j=$(c.j[e]), " *
            "ΔR=$(Tuple(c.dR[e]))): it carries the momentum offset q = $(Tuple(c.q[e])), " *
            "and an _hr.dat element is a bare t_ij(R). Momentum offsets come from basis " *
            "states with reciprocal labels (G) or momentum origins (k0); share such models " *
            "via `h5_write`."))
    end

    nw = c.n
    ### group the constant entries by integer R; both directions are stored, so both blocks
    ### of every Hermitian pair land in the file
    blocks = Dict{SVector{D,Int},Vector{Int}}()
    for e in c.groups[1]
        push!(get!(() -> Int[], blocks, SVector{D,Int}(numerator.(c.dR[e]))), e)
    end
    ### the home block R = 0 is always listed, entries or not (Wannier90 itself always lists
    ### it, and an onsite-free model would otherwise emit a file with no R = 0 line at all)
    get!(() -> Int[], blocks, zero(SVector{D,Int}))
    Rs = sort!(collect(keys(blocks)); by=Tuple)
    nrpts = length(Rs)

    v = Base.pkgversion(@__MODULE__)
    stamp = Dates.format(Dates.now(Dates.UTC), Dates.dateformat"yyyy-mm-dd\THH:MM:SS\Z")
    H_R = zeros(ComplexF64, nw, nw)
    open(filename, "w") do io
        println(io, "written by SingleParticleModels.jl v", isnothing(v) ? "?" : v,
                " on ", stamp, isempty(model.name) ? "" : " -- " * model.name)
        println(io, lpad(nw, 12))
        println(io, lpad(nrpts, 12))
        for r in 1:nrpts
            print(io, lpad(1, 5))               # WS degeneracy 1 for every R, 15 per line
            (r % 15 == 0 || r == nrpts) && println(io)
        end
        for R in Rs
            fill!(H_R, zero(ComplexF64))
            for e in blocks[R]
                ### ACCUMULATE: entries are NOT unique per (i, j, ΔR). `finalize_couplings`
                ### coalesces on (kernel, TERM, i, j, ΔR, q), so two different terms landing
                ### on the same bond -- a uniform onsite plus a sublattice mass, two hopping
                ### terms on one shell -- stay separate entries, and `bloch_hamiltonian` sums
                ### them at evaluation time. Assigning here would keep only whichever term
                ### sorted last and emit a plausible, Hermitian file holding a DIFFERENT
                ### Hamiltonian, which no reader could detect.
                H_R[c.i[e], c.j[e]] += c.w[e]
            end
            R3 = _pad_components(R, Val(3), 0)
            for n in 1:nw, m in 1:nw            # m fastest: Wannier90's own element order
                t = H_R[m, n]
                println(io, lpad(R3[1], 5), lpad(R3[2], 5), lpad(R3[3], 5),
                        lpad(m, 7), lpad(n, 7), lpad(real(t), 24), lpad(imag(t), 24))
            end
        end
    end
    return String(filename)
end

### the human name of a kernel for the k-dependence refusal: the on-disk tag where one
### exists, the payload type for an unregistered closure
_hr_kernel_name(kern) = something(kernel_tag(kern), string(typeof(kern)))




#############################################################
###### Wannier-centre resolution helpers                #####
#############################################################

"""
    _wannier_centres_cartesian(centres, ::Val{D}, atol) -> Vector{SVector{D,Float64}}

Normalize a `centres=` specification into cartesian `SVector`s.

A string is read as a Wannier90 `seedname_centres.xyz` file: line 1 the record count, line 2 a
comment, then one record per line, `tag  x  y  z`. The Wannier centres are the **leading** records
tagged `X`; the trailing records, tagged with real element symbols, are the atoms and are ignored.
For `D < 3` the trailing cartesian components of every centre must vanish (within `atol`, the
file tolerance).

A vector must hold one length-`D` real position per Wannier function.
"""
function _wannier_centres_cartesian(centres::AbstractString, ::Val{D},
                                    atol::Float64)::Vector{SVector{D,Float64}} where {D}
    lines = readlines(centres)
    length(lines) >= 3 || throw(ArgumentError(
        "`$centres` is not a Wannier90 _centres.xyz file: expected a record count, a comment " *
        "line and the centre records."
    ))
    total = tryparse(Int, strip(lines[1]))
    isnothing(total) && throw(ArgumentError(
        "`$centres` is not an .xyz file: its first line (`$(strip(lines[1]))`) is not the " *
        "record count."
    ))
    raw = SVector{3,Float64}[]
    nrec = 0
    past_centres = false
    for ln in @view lines[3:end]
        s = strip(ln)
        isempty(s) && continue
        f = split(s)
        length(f) >= 4 || throw(ArgumentError("Malformed record in `$centres`: `$ln`"))
        nrec += 1
        if f[1] == "X"
            ### the centres are the LEADING X-tagged records; an X after an atom record is not
            ### the Wannier90 layout and most likely a hand-edited file worth refusing
            past_centres && throw(ArgumentError(
                "`$centres` holds an `X` centre record after an atom record; Wannier90 writes " *
                "all centres first, so the leading-`X` rule cannot identify the centres here."
            ))
            push!(raw, SVector{3,Float64}(parse(Float64, f[2]), parse(Float64, f[3]),
                                          parse(Float64, f[4])))
        else
            past_centres = true
        end
    end
    nrec == total || throw(ArgumentError(
        "`$centres` declares $total records but holds $nrec."
    ))
    isempty(raw) && throw(ArgumentError(
        "`$centres` holds no `X`-tagged Wannier-centre records."
    ))
    for (n, c) in pairs(raw)
        all(abs(c[d]) <= atol for d in (D+1):3) || throw(ArgumentError(
            "Wannier centre $n in `$centres` is $(Tuple(c)), non-zero outside the cell's " *
            "$D dimension(s)."
        ))
    end
    return [SVector{D,Float64}(ntuple(d -> c[d], D)) for c in raw]
end

function _wannier_centres_cartesian(centres::AbstractVector, ::Val{D},
                                    ::Float64)::Vector{SVector{D,Float64}} where {D}
    out = Vector{SVector{D,Float64}}(undef, length(centres))
    for (n, c) in pairs(centres)
        c isa AbstractVector{<:Real} || throw(ArgumentError(
            "`centres[$n]` is $(repr(c)); each centre must be a real-valued cartesian position " *
            "of length $D (or pass the seedname_centres.xyz path instead)."
        ))
        length(c) == D || throw(DimensionMismatch(
            "`centres[$n]` has length $(length(c)); centres are cartesian positions of the " *
            "cell's dimension D = $D."
        ))
        out[n] = SVector{D,Float64}(c)
    end
    return out
end

"""
    _wannier_wrap_shifts(cell, cart) -> Vector{SVector{D,Int}}

Verify the declared cartesian Wannier centres against the cell's state positions and return the
wrap shifts `W`: `W[n]` is the integer lattice translation by which centre `n` was wrapped into
the home cell, `c_n(frac) = τ_n + W_n` within the cell's `atol`. The importer compensates every
element's translation by `W_n - W_m`, which pins the atomic-gauge phase to the true centres.
"""
function _wannier_wrap_shifts(cell::UnitCell{D},
                              cart::Vector{SVector{D,Float64}};
                              atol::Float64=1e-6)::Vector{SVector{D,Int}} where {D}
    ### a Wannier function is localized by definition; a :plane_wave state's position is
    ### gauge-only and never enters a phase, so "its Wannier centre" is not a meaningful claim
    for s in 1:n_sites(cell)
        cell.site_kind[s] === :plane_wave && throw(ArgumentError(
            "`centres=` declares Wannier centres, but site `$(cell.site_names[s])` is " *
            ":plane_wave -- its position is gauge-only and never enters a phase. Wannier90 " *
            "import needs a cell of :localized states."
        ))
    end
    length(cart) == n_states(cell) || throw(DimensionMismatch(
        "`centres` declares $(length(cart)) Wannier centres but the cell holds " *
        "$(n_states(cell)) states; one centre per state, in the file's Wannier-function order."
    ))
    W = Vector{SVector{D,Int}}(undef, length(cart))
    for n in eachindex(cart)
        cfrac = real_to_fractional(cart[n], cell.lattice)
        d = cfrac - cell.basis[n].r_frac
        Wn = round.(d)
        ### the FILE tolerance governs the match, not the cell's position grid: Wannier90
        ### prints centres with finite precision (~8 decimals), so a genuine _centres.xyz
        ### essentially never lands within the default cell atol of 1e-12 -- the documented
        ### workflow would refuse every real file
        tol = max(atol, cell.atol)
        all(abs.(d .- Wn) .<= tol) || throw(ArgumentError(
            "Wannier centre $n sits at $(Tuple(cfrac)) in fractional coordinates, but state $n " *
            "of the cell sits at $(Tuple(cell.basis[n].r_frac)): they differ by " *
            "$(Tuple(d .- Wn)) beyond an integer lattice translation (tolerance = " *
            "$(tol), the larger of the reader's `atol` and the cell's). The cell's states " *
            "must sit at the true Wannier centres, in the file's order -- rebuild the cell " *
            "at the centres, or use the `BravaisLattice` method, which builds it for you."
        ))
        W[n] = SVector{D,Int}(round.(Int, d))
    end
    return W
end
