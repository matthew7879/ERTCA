##################################################################
###### export only the following functions from this module ######
##################################################################

export embed_into_higher_dimension, periodic_block, leading_block

# The boundary between an AMBIENT dimension `D` and a PERIODIC RANK `P = nperiodic`, and the
# single owner of every crossing of it.
#
# `BravaisLattice{D}` has carried both numbers since format v3, and the distinction is real: a
# monolayer is 2-periodic in a 3-dimensional space, a ribbon 1-periodic in 2, a layer group's
# operations need the normal axis that its lattice does not have. What was missing is that
# nothing in `src/` used the crossing this file now owns. `embed_into_higher_dimension` was
# exported, documented and tested, and called by nothing outside the test suite -- while eight
# production paths each grew their own version:
#
#   interfaces/wannier90/deck.jl      pad to 3D with a PHYSICAL vacuum length
#   interfaces/wannier90/writers.jl   pad a grid shape to 3D with trailing 1s
#   interfaces/wannier90/stencil.jl   synthesize the padding axis's reciprocal shell
#   interfaces/wannier90/hamiltonian/hr.jl        zero-pad a translation to 3D
#   hamiltonians/ertca/registry_scan  slice a 2x2 block out of an axis-aligned 3D slab
#   hamiltonians/ertca/{registry_scan,diagnostics}  zero-pad a shift and a momentum
#   crystallography/high_symmetry_paths  drop to the periodic block, metric only
#   examples/ertca/loader.jl          rebuild a slab as FULLY periodic and rely on a later guard
#
# They disagreed in a way that mattered: the exported routine pads with UNIT axes, while the
# Wannier90 deck needs a physical vacuum (20 A by default) because that length sets the third
# reciprocal vector and hence the out-of-plane neighbour shell and its weight. The exported API
# was therefore unusable for the one production case that needed it, which is why nothing used
# it. That is a missing owner, not a dimensionality bug, and `padding=` is the fix.

"""
    embed_into_higher_dimension(lat::BravaisLattice{D}, D_new; periodic=false, padding=1.0,
                                kwargs...) -> BravaisLattice{D_new}

Embed a Bravais lattice into a higher-dimensional space: the original rows are preserved in the
first `D` coordinates and the added rows lie along the new axes.

By default the added directions are APERIODIC (`nperiodic` stays at the original lattice's
value): the result is a partially periodic lattice -- a 2D lattice embedded in 3D gains a real
`z` coordinate for positions without gaining a translation symmetry, which is what a layered
continuum cell or a slab needs. Pass `periodic=true` to declare the added axes as genuine
primitive vectors instead.

Args:
-----
- `lat`: the original Bravais lattice
- `D_new`: the new embedding dimension (≥ the current one)
- `periodic`: whether the added directions are translation symmetries (default `false`)
- `padding`: the LENGTH of each added row (default `1.0`). Only meaningful with
  `periodic=true`, and refused otherwise -- an aperiodic padding row is a coordinate axis, and
  the orthonormal-complement contract that keeps `cell_volume` equal to the true `P`-volume
  requires it to be a unit vector. With `periodic=true` it is a physical length: this is the
  vacuum thickness of a slab, and it SETS the new reciprocal vector `2π/padding`, hence the
  out-of-plane neighbour shell and its finite-difference weight. That is why
  [`wannier_deck`](@ref) takes `vacuum` as a parameter rather than a formatting detail.

Returns:
--------
- `BravaisLattice{D_new}` with `nperiodic = periodic ? D_new : nperiodic(lat)`

See [`periodic_block`](@ref) for the inverse.
"""
function embed_into_higher_dimension(lat::BravaisLattice{D}, D_new::Int;
                                     periodic::Bool=false, padding::Real=1.0,
                                     kwargs...) where {D}
    if D_new < D
        throw(ArgumentError("New dimension must be greater than or equal to the current dimension."))
    end
    nperiodic(lat) == D || throw(ArgumentError(
        "The lattice is already partially periodic (nperiodic = $(nperiodic(lat)) < D = $D); " *
        "embedding it again is not supported -- build the embedding in one step."))
    padding > 0 || throw(ArgumentError(
        "`padding` must be a positive length; got $padding."))
    (periodic || padding == 1.0) || throw(ArgumentError(
        "`embed_into_higher_dimension`: `padding = $padding` with `periodic = false`. An " *
        "APERIODIC added row is a coordinate axis, not a translation, and must be a unit " *
        "vector: that contract is what keeps `cell_volume` equal to the true " *
        "$(nperiodic(lat))-volume and transverse fractional coordinates equal to cartesian " *
        "ones. A physical padding LENGTH describes a vacuum layer, which is a periodic " *
        "direction -- pass `periodic=true` alongside it."))
    incD = D_new - D
    newvecs = hcat(zeros(incD, D), diagm(fill(Float64(padding), incD)))
    a_embedded = vcat(hcat(lat.a, zeros(D, incD)), newvecs)
    P = periodic ? D_new : D
    return BravaisLattice(a_embedded; force=true, nperiodic=P, kwargs...)
end

"""
    periodic_block(lat::BravaisLattice{D}) -> BravaisLattice{P}

The lattice generated by the PERIODIC rows of a partially periodic lattice, in its own
`P`-dimensional space. The inverse of [`embed_into_higher_dimension`](@ref), and the route every
crystallographic question about a slab or a ribbon should take.

A `P`-periodic frame in `D` dimensions has no `D`-dimensional Bravais class -- rows `P+1:D` are
orthonormal coordinate axes, so a lattice constant read off them is a property of the frame and
not of the material. Its periodic block DOES classify, and fractional coordinates transfer
verbatim: `high_symmetry_points(lat)` reduces through here and pads the returned k-vectors back
with zeros in the aperiodic components.

    periodic_block(embed_into_higher_dimension(bravais_lattice(:hp; a=2.46), 3))   # the :hp net

Notes:
------
Built by Cholesky factorization of the periodic block of the metric (`_basis_from_gram`),
so the metric is reproduced to machine precision. The earlier private version rebuilt a 2D block
from `(a, b, gamma)`, which round-trips the angle through `acos` and `cos`; it also handled only
`P ∈ {1, 2}`. Orientation is not preserved and cannot be -- a `P`-dimensional block has no
memory of how its span sat in `D`. What comes back is the block's CANONICAL frame, `a₁` along
the first axis, which is also the frame the moire archive stores its lattices in.

One consequence is worth stating because it looks like a bug: `periodic_block(embed(lat, D)) ==
lat` is **false** in general. `==` on a `BravaisLattice` is a unimodular BASIS-change test, not
an isometry test, and it does not consider a rigidly rotated lattice equal (`rotate(fcc) != fcc`
too). Everything that actually consumes this is metric-based and therefore unaffected:
`metric`, `cell_volume`, `identify_bravais`, and every tabulated high-symmetry coordinate agree
exactly. It happens to return `true` for a triangular lattice only because that template already
has `a₁` along the first axis.

The returned dimension is a runtime value, so this is deliberately not type-stable; it belongs
in setup code, not in a k-loop. A fully periodic lattice is returned unchanged.
"""
function periodic_block(lat::BravaisLattice{D}) where {D}
    P = nperiodic(lat)
    P == D && return lat
    P == 0 && throw(ArgumentError(
        "This lattice has no periodic direction (a Γ-only finite cluster): every one of its " *
        "$D rows is an orthonormal padding axis, so there is no periodic block to take, no " *
        "Brillouin zone, and no Bravais class. The cluster's whole spectrum lives at the " *
        "single momentum k = $(Tuple(zeros(D)))."))
    G = metric(lat)
    return BravaisLattice(_basis_from_gram(SMatrix{P,P,Float64}(G[1:P, 1:P])))
end

"""
    leading_block(lat::BravaisLattice{D}, P; atol=distance_atol(lat)) -> BravaisLattice{P}

The leading `P × P` block of a FULLY periodic lattice whose remaining axes are orthogonal to it.

This is the DFT-slab case, and it is NOT [`periodic_block`](@ref). There, rows `P+1:D` are
declared aperiodic and are dropped because they are not translations at all. Here every row is a
genuine translation -- a slab in vacuum is periodic along its stacking axis, and Wannier90 and
Quantum ESPRESSO both insist on that -- but the vacuum axis is DECOUPLED: no hopping crosses it,
so the physics is `P`-dimensional even though the cell is not.

That makes the two refusals different in kind. Dropping a declared-aperiodic axis is safe by
construction. Dropping a decoupled one is a claim about the HAMILTONIAN, which only the caller
can check (`registry_scan` does, by refusing any coupling whose out-of-plane separation exceeds
half the cell height). What this verifies is the weaker, geometric precondition that makes the
leading block well defined at all: rows `P+1:D` must be orthogonal to the leading rows in both
directions, so that the block is a sublattice and not a projection that distorts the metric.

    leading_block(read_qe_lattice("slab.xml"), 2)     # the in-plane net of an axis-aligned slab
"""
function leading_block(lat::BravaisLattice{D}, P::Int;
                       atol::Float64=distance_atol(lat)) where {D}
    (1 <= P <= D) || throw(ArgumentError("`leading_block`: `P` must lie in 1:$D; got $P."))
    P == D && return lat
    nperiodic(lat) == D || throw(ArgumentError(
        "`leading_block`: this lattice is periodic along only $(nperiodic(lat)) of its $D " *
        "directions, so its trailing axes are padding rather than decoupled translations. " *
        "That is `periodic_block`'s case, not this one."))
    A = lat.a
    for m in (P + 1):D, l in 1:P
        (abs(A[m, l]) <= atol && abs(A[l, m]) <= atol) || throw(ArgumentError(
            "`leading_block`: row $m is not orthogonal to the leading $P × $P block " *
            "(a[$m,$l] = $(A[m, l]), a[$l,$m] = $(A[l, m]), tolerance $atol). Taking the " *
            "leading block would change the metric rather than restrict it -- re-express the " *
            "cell so the first $P rows span the periodic subspace and the rest are normal to " *
            "it."))
    end
    return BravaisLattice(SMatrix{P,P,Float64}(A[1:P, 1:P]))
end

### Zero-extend (or `fill`-extend) a coordinate tuple from its own length to `Dnew`. This is the
### `_pad_shift` / `_pad_k` / `_deck_frac3` / `_deck_frac3_int` / `hr.jl` R-padding family,
### which had five spellings and two fill values -- 0 for a position, momentum or translation,
### 1 for a GRID SHAPE, where a zero would describe a mesh with no points. The default fill is
### the INTEGER zero so that integer data (a translation, a grid shape) stays integer and float
### data promotes -- `promote_type` does the right thing in both directions.
@inline function _pad_components(v, ::Val{Dnew}, fill=0) where {Dnew}
    P = length(v)
    P <= Dnew || throw(ArgumentError(
        "`_pad_components`: cannot pad a length-$P vector down to $Dnew components."))
    T = promote_type(eltype(v), typeof(fill))
    return SVector{Dnew,T}(ntuple(j -> j <= P ? T(v[j]) : T(fill), Val(Dnew)))
end

### The other direction: drop the trailing components, refusing any that is not (near) zero.
### A silent truncation here discards a periodic direction, which is the failure the h5 reader
### refuses by name rather than performing.
@inline function _project_components(v, ::Val{P}; atol::Float64=0.0) where {P}
    D = length(v)
    P <= D || throw(ArgumentError(
        "`_project_components`: cannot project a length-$D vector up to $P components."))
    for m in (P + 1):D
        abs(v[m]) <= atol || throw(ArgumentError(
            "`_project_components`: component $m of $v is $(v[m]), not zero; dropping it " *
            "would discard information rather than a padding axis."))
    end
    return SVector{P,eltype(v)}(ntuple(m -> v[m], Val(P)))
end
