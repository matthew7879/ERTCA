##################################################################
###### export only the following functions from this module ######
##################################################################

export bravais_lattice, bravais_types, bravais_info, bravais_canonical_name
export identify_bravais




#####################################################################
###### The Bravais lattice types                                #####
#####################################################################

"""
Named Bravais lattice types: 1 in 1D, 5 in 2D, 14 in 3D.

Naming:
-------
The canonical names are the **Pearson-style symbols** -- lower-case crystal family followed by the
centring letter (`cF`, `hP`, `oS`, ...). They are standard, unambiguous, and case-distinguishes 2D
from 3D (`hp` is the 2D hexagonal net, `hP` the 3D hexagonal lattice).

Because that case distinction is easy to get wrong, every type also has a spelled-out alias, and
those are chosen to be unambiguous across dimensions: the 2D hexagonal net is `:triangular` (its
usual name in condensed matter) while `:hexagonal` means the 3D lattice.

What is stored:
---------------
`bravais_lattice` returns **primitive** vectors, not conventional ones -- a Bravais lattice is a set
of points, and the primitive cell is what generates it. So `bravais_lattice(:cF; a=1)` gives the
face-centred cubic primitive cell of volume `a³/4`, not the cubic cell of volume `a³`. The
`points_per_conventional` field records the ratio, which is what makes the volumes checkable.

Centring conventions:
---------------------
- `S` (base-centred) uses **C-centring**: the centred face is the one spanned by `a` and `b`.
- `hR` is given in the **hexagonal setting** (obverse), parameterized by the hexagonal `a` and `c`
  rather than by the rhombohedral angle. Its primitive cell is one third of the hexagonal cell.
"""
struct BravaisType
    name::Symbol
    dimension::Int
    family::String
    centring::String
    params::Vector{Symbol}
    points_per_conventional::Int
    description::String
end

"""
    _bravais_vectors(name, p) -> Vector{Vector{Float64}}

The primitive vectors of a named Bravais lattice, as rows.

`p` is a NamedTuple of the validated parameters. Angles arrive already in radians.
"""
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

"""
Spelled-out aliases for the Pearson symbols.

Deliberately unambiguous across dimensions: `:triangular` is the 2D hexagonal net and `:hexagonal`
the 3D lattice, so no alias depends on the `hp` / `hP` case distinction.
"""
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
    ### ITA prints BOTH `mC` and `mI` for the one centred monoclinic Bravais type, and the
    ### Niggli character table uses both: twelve of its thirteen mS rows are printed mC and
    ### character 43 is printed mI. `mI` is not a fifteenth Bravais type.
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

"""
Structures that are commonly asked for by name but are **not** Bravais lattices -- they are a
Bravais lattice plus a multi-site basis.

Naming them explicitly turns a confusing "unknown lattice" error into an answer.
"""
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




#####################################################################
###### Construction                                             #####
#####################################################################

"""
    bravais_canonical_name(name) -> Symbol

Resolve an alias to its canonical Pearson symbol, raising an informative error otherwise.
"""
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

    ### a wrong-case Pearson symbol is the most likely typo, so check for it explicitly.
    ### Several canonical symbols can share a lowercase form (`:hp`/`:hP`, `:mp`/`:mP`, ...),
    ### so ALL case-insensitive matches are suggested -- `only()` here once crashed the
    ### suggestion path itself.
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

"""
    bravais_info(name) -> BravaisType

Look up the record for a Bravais lattice type, accepting either a Pearson symbol or an alias.
"""
bravais_info(name::Symbol)::BravaisType = BRAVAIS_TYPES[bravais_canonical_name(name)]

"""
    bravais_types(D) -> Vector{Symbol}

The canonical names of the Bravais lattice types in `D` dimensions: 1 in 1D, 5 in 2D, 14 in 3D.
"""
function bravais_types(D::Integer)::Vector{Symbol}
    D in (1, 2, 3) || throw(ArgumentError("Bravais lattice types are enumerated for D = 1, 2, 3; got $D."))
    return sort([k for (k, v) in BRAVAIS_TYPES if v.dimension == D])
end

"""
    bravais_lattice(name; a, b, c, alpha, beta, gamma, degrees=true) -> BravaisLattice

Construct a named Bravais lattice from its cell parameters.

Args:
-----
- `name`: a Pearson symbol (`:cF`, `:hP`, `:tp`, ...) or an alias (`:fcc`, `:hexagonal`, `:square`, ...)
- the cell parameters the type requires; angles in degrees unless `degrees=false`

Returns:
--------
- `BravaisLattice{D}` with **primitive** vectors.

Examples:
---------
    bravais_lattice(:square; a=1.0)
    bravais_lattice(:triangular; a=1.0)              # the 2D lattice underlying graphene
    bravais_lattice(:fcc; a=3.57)                    # primitive cell, volume a³/4
    bravais_lattice(:hP; a=2.46, c=6.7)
    bravais_lattice(:aP; a=1, b=2, c=3, alpha=80, beta=95, gamma=105)

Notes:
------
Supplying a parameter the type does not use is an error rather than being ignored -- asking for
`bravais_lattice(:cubic; a=1, c=2)` almost certainly means the wrong type was named, and silently
dropping `c` would hide that.
"""
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

    ### validate and convert: lengths positive, angles in (0, 180) and expressed in radians
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




#####################################################################
###### Identifying the Bravais type of a given lattice          #####
#####################################################################



### Every `identify_bravais` method reads its class off the METRIC of the lattice rows, and the
### rows `P+1:D` of a partially periodic lattice are an orthonormal complement -- coordinate axes
### chosen by the constructor, not translations of the lattice. Their lengths (all 1) and their
### angles (all right) are therefore not lattice data, and classifying on them fabricates a
### Bravais type for a lattice that has none. The check lives here, once, because it lived in the
### `BravaisLattice{2}` method alone for long enough that `identify_bravais` on a Γ-only 1D
### cluster returned `:lp` with `a = 1.0` -- the length of the padding axis.
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

"""
    identify_bravais(lat::BravaisLattice; rtol=BRAVAIS_RTOL) -> (name, params, M)

Identify which named Bravais type a lattice is, **whatever primitive basis and orientation it was
given in**, returning everything needed to use the tabulated crystallography on it:

- `name::Symbol`: the Pearson symbol (`:hp`, `:tp`, `:op`, `:oc`, `:mp` in 2D; the fourteen
  mixed-case ones in 3D)
- `params::NamedTuple`: the conventional cell parameters, ready for
  `bravais_lattice(name; params...)` or `high_symmetry_points(name; params...)`
- `M::SMatrix{D,D,Int}`: the unimodular map from the canonical basis to the given one,
  `lat.a = M * a_canonical` up to a rigid rotation OR reflection -- so a fractional momentum
  `k_c` tabulated in the canonical basis is `k = M * k_c` in the basis of `lat`. The improper
  case is unavoidable for the oblique class: its point group has no mirror, so a γ > 90° input
  relates to the reduced γ < 90° canonical cell only through one. Every metric quantity (and
  every tabulated point) is reflection-invariant, so nothing downstream depends on the sign.

Args:
-----
- `rtol`: relative tolerance for the length and angle coincidences that separate the types.
  Deliberately tight: a lattice that is square only to a percent IS oblique, and classifying it
  as square would attach symmetry labels its Brillouin zone does not have.

Notes:
------
The classification runs on the Lagrange-reduced cell, whose `(|a₁|, |a₂|, γ)` with
`60° ≤ γ ≤ 90°` is a lattice invariant: γ = 60° with equal lengths is triangular whether the
lattice was written with γ = 60°, γ = 120°, or any unimodular mixture -- a moiré coincidence
lattice, a supercell.

In 3D the same job runs on the NIGGLI-reduced cell, whose character is looked up in the 44-row
table of `external/crystallography/niggli_characters_3d.csv` (see
`crystallography/niggli_characters.jl`). Eleven of the fourteen types return in full; `mP`, `mS`
and `aP` classify reliably but are REFUSED BY NAME, because their Setyawan-Curtarolo setting is
not derived yet and a conventional cell is not unique for them -- returning the first valid one
would hand `high_symmetry_points` a cell it refuses, or one for which it silently returns points
from a neighbouring Brillouin zone. The refusal names the type it found and the `bravais=`
escapes that need no setting.
"""
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
    ### the SECOND reduced form a centred-rectangular lattice can take: unequal lengths with
    ### `a₁·a₂ = |a₁|²/2`, i.e. a₂ reaching exactly the midpoint above a₁ -- then `2a₂ - a₁` is
    ### perpendicular to `a₁` and the two span the conventional rectangle
    centred = abs(dot(a1, a2) - dot(a1, a1) / 2) <= rtol * dot(a1, a1)

    ### the decision tree, most symmetric first -- each case OWNS its parameter extraction
    name, params = if equal && hexa
        :hp, (a=l1,)
    elseif equal && right
        :tp, (a=l1,)
    elseif right
        :op, (a=l1, b=l2)
    elseif equal
        ### rhombic, symmetric reduced form: the conventional rectangle sides are the two
        ### diagonals; γ < 90° makes |a₁+a₂| the LONG one, and the `oc` tables presuppose a < b
        :oc, (a=norm(a2 - a1), b=norm(a1 + a2))
    elseif centred
        :oc, NamedTuple{(:a, :b)}(minmax(l1, norm(2 * a2 - a1)))
    else
        :mp, (a=l1, b=l2, gamma=rad2deg(acos(cosg)))
    end

    ### canonical construction, reduced the same way; the two reduced METRICS agree by
    ### construction, up to the finitely many reduced bases a symmetric lattice admits -- that
    ### residual freedom is resolved by an explicit search over small unimodular W
    canon = bravais_lattice(name; params...)
    c1, c2, Uc = _lagrange_reduce(canon)
    Gu = _gram_of_rows(_rows_to_matrix((a1, a2)))
    Gc = _gram_of_rows(_rows_to_matrix((c1, c2)))
    tol = rtol * max(dot(a2, a2), 1.0) * 10
    W = _metric_match(Gu, Gc, tol)
    isnothing(W) && throw(ErrorException(
        "identify_bravais classified the lattice as `$name` $params but could not match the " *
        "reduced metrics; this is an internal inconsistency, not a usage error."))

    ### A_red_u = Uu A_u, A_red_c = Uc A_c, A_red_u ≅ W A_red_c (rigid rotation)
    ### ⇒ A_u = Uu⁻¹ W Uc A_c. Inverses of unimodular integer matrices are integer.
    M = round.(Int, inv(SMatrix{2,2,Float64}(Uu)) * W * Uc)
    return (name=name, params=params, M=SMatrix{2,2,Int}(M))
end

### The 3D method lives in `crystallography/niggli_characters.jl`, with the 44-character
### table it needs. It is included after this file, so it can reach `bravais_lattice` and
### `bravais_canonical_name` without a forward reference.






#####################################################################
###### Declared Bravais types: recovery and validation          #####
#####################################################################

"""
    _conventional_map(name) -> Matrix{Int}

The rows of the CONVENTIONAL cell as integer combinations of the standard primitive rows of
`_bravais_vectors`: `conv = T * prim` (rows are vectors, matching the package's row convention).

Each map is read off the corresponding `_bravais_vectors` template, per centring:

- primitive types (`P`, and `hp`/`tp`/`op`/`mp`/`lp`): the primitive cell IS the conventional
  cell, `T = I`;
- face-centred (`cF`, `oF`), template `p₁ = (0,b,c)/2` etc.: `conv₁ = -p₁+p₂+p₃`, cyclically;
- body-centred (`cI`, `tI`, `oI`), template `p₁ = (-a,b,c)/2` etc.: `conv₁ = p₂+p₃`, cyclically;
- base-centred orthorhombic (`oS`, C-centring), template `p₁ = (a,-b,0)/2`, `p₂ = (a,b,0)/2`:
  `conv₁ = p₁+p₂`, `conv₂ = -p₁+p₂`, `conv₃ = p₃`; the 2D centred net `oc` is its in-plane block;
- base-centred monoclinic (`mS`, C-centring), template `p₁ = (a,b,0)/2`, `p₂ = (-a,b,0)/2`
  (NOTE the sign difference from `oS`): `conv₁ = p₁-p₂`, `conv₂ = p₁+p₂`, `conv₃ = p₃` -- the
  ORDER of the difference matters, because β is read off `conv₁ · conv₃`;
- rhombohedral in the hexagonal setting (`hR`, obverse): `conv₁ = p₁-p₃`, `conv₂ = -p₁+p₂`,
  `conv₃ = p₁+p₂+p₃` (the hexagonal `a₁`, `a₂` and the full `c` axis).

`|det T|` equals the type's `points_per_conventional` in every case -- the conventional cell
holds that many lattice points.
"""
function _conventional_map(name::Symbol)::Matrix{Int}
    # ---------------- centred types ----------------
    (name === :cF || name === :oF) && return [-1 1 1; 1 -1 1; 1 1 -1]
    (name === :cI || name === :tI || name === :oI) && return [0 1 1; 1 0 1; 1 1 0]
    name === :oS && return [1 1 0; -1 1 0; 0 0 1]
    name === :mS && return [1 -1 0; 1 1 0; 0 0 1]
    name === :hR && return [1 0 -1; -1 1 0; 1 1 1]
    name === :oc && return [1 1; -1 1]

    # ---------------- primitive types: conventional = primitive ----------------
    haskey(BRAVAIS_TYPES, name) || throw(ArgumentError(
        "No conventional-cell map for Bravais type `$name`."))
    D = BRAVAIS_TYPES[name].dimension
    return Matrix{Int}(LinearAlgebra.I, D, D)
end

"""
    _recover_bravais_parameters(lat, canonical) -> NamedTuple

Recover the CONVENTIONAL cell parameters of `lat` under the ASSUMPTION that it is the standard
primitive basis of the declared type `canonical` (up to a rigid rotation) -- the parameters
`bravais_lattice(canonical; ...)` and the symbol-route tables take, angles in degrees.

Works entirely on the Gram matrix: `G_conv = T G Tᵀ` with `T = _conventional_map(canonical)` is
rotation-invariant, so a rotated copy of a `bravais_lattice(...)` cell recovers the exact same
parameters. Only the parameters the type actually uses are returned (`:cF` gives `(a=...,)`);
redundant conventional lengths/angles are NOT averaged in -- if the assumption is wrong they
disagree, and `_declared_bravais`'s reconstruction check is what catches that, by name.
"""
function _recover_bravais_parameters(lat::BravaisLattice{D}, canonical::Symbol) where {D}
    spec = BRAVAIS_TYPES[canonical]
    T = _conventional_map(canonical)
    Gc = T * Matrix(metric(lat)) * transpose(T)
    len(i) = sqrt(Gc[i, i])
    ang(i, j) = rad2deg(acos(clamp(Gc[i, j] / (len(i) * len(j)), -1.0, 1.0)))

    full = Dict{Symbol,Float64}(:a => len(1))
    if D >= 2
        full[:b] = len(2)
        full[:gamma] = ang(1, 2)
    end
    if D == 3
        full[:c] = len(3)
        full[:alpha] = ang(2, 3)
        full[:beta] = ang(1, 3)
    end
    return NamedTuple{Tuple(spec.params)}(Tuple(full[k] for k in spec.params))
end

"""
    _declared_bravais(lat, bravais; rtol=BRAVAIS_RTOL) -> (name, params)

Validate a USER-DECLARED Bravais type against a lattice object and return the canonical name
with the recovered conventional parameters -- the bypass that gives 3D lattices (where
`identify_bravais` raises) their tabulated high-symmetry data.

Validation is by reconstruction: the conventional parameters are recovered from the lattice's
metric assuming the declaration is true (`_recover_bravais_parameters`), the reference cell
`bravais_lattice(name; params...)` is rebuilt from them, and the two metrics must agree to
`rtol` relative to the metric scale. A declaration that fails this test is refused BY NAME
rather than trusted: the tabulated fractional coordinates presuppose the standard primitive
basis, so using them on any other basis silently labels wrong k-points.

Accepted: the standard primitive basis of the declared type in any rigid ORIENTATION (the check
is metric-only). Refused: a wrong type, a re-ordered or unimodularly re-mixed primitive basis
(even though it generates the same lattice points), a dimension mismatch, and any partially
periodic lattice (those classify automatically by their periodic block; see `periodic_block`).
"""
function _declared_bravais(lat::BravaisLattice{D}, bravais::Symbol;
                           rtol::Float64=BRAVAIS_RTOL) where {D}
    canonical = bravais_canonical_name(bravais)
    spec = BRAVAIS_TYPES[canonical]
    nperiodic(lat) == D || throw(ArgumentError(
        "`bravais = :$bravais` was declared for a lattice that is periodic along only " *
        "$(nperiodic(lat)) of its $D directions. A partially periodic lattice classifies " *
        "automatically by its periodic block (a slab gets its in-plane path with the aperiodic " *
        "components pinned to 0) -- drop the `bravais` keyword."))
    spec.dimension == D || throw(ArgumentError(
        "The declared Bravais type `:$canonical` ($(spec.description)) is " *
        "$(spec.dimension)-dimensional, but this lattice is $D-dimensional. Declare one of " *
        "$(bravais_types(D))."))

    params = _recover_bravais_parameters(lat, canonical)
    ref = bravais_lattice(canonical; params...)
    G = Matrix(metric(lat))
    dev = maximum(abs, G .- Matrix(metric(ref)))
    scale = maximum(abs, G)
    dev <= rtol * scale || throw(ArgumentError(
        "The declared Bravais type `:$canonical` does not describe this lattice: rebuilding " *
        "`bravais_lattice(:$canonical; ...)` from the conventional parameters recovered under " *
        "that assumption ($(map(v -> round(v; sigdigits=6), params))) leaves a metric " *
        "deviation of $(round(dev; sigdigits=3)) on a metric scale of " *
        "$(round(scale; sigdigits=3)) (rtol = $rtol). Using the `:$canonical` tables anyway " *
        "would silently put every tabulated label on a wrong k-point. The declaration requires " *
        "the lattice to be the type's STANDARD primitive basis, exactly as " *
        "`bravais_lattice(:$canonical; ...)` builds it -- a rigidly rotated copy is fine, but " *
        "a re-ordered or unimodularly re-mixed basis is not, even when it generates the same " *
        "lattice points."))
    return (name=canonical, params=params)
end
