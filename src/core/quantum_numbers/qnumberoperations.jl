##################################################################
###### export only the following functions from this module ######
##################################################################

export AbstractQnumRule
export LevelOrder, DeclaredOrder, TwoLevelOrder, NamedOrder
export AnyQnum, ConserveQnum, DiagonalQnum, OffDiagonalQnum, MatrixQnum, TwoLevelMatrixQnum,
       DirectionalQnum, ProductQnumRule
export qnum_factor, rule_axes, validate_rule, qnum_matrix, compose, otimes, ⊗
export conserve, pauli_0, pauli_x, pauli_y, pauli_z, project, raise, lower, matrix_rule
export directional_rule




#######################################################
###### Quantum-number selection rules: interface ######
#######################################################

"""
    AbstractQnumRule

A selection rule acting on the quantum numbers of a pair of basis states.

Physical representation:
------------------------
Because `qnums` are part of the `BasisState` -- and therefore part of the basis *index* -- a matrix
in an internal space is not a matrix here, it is a rule. A spin operator `λ σ_z` does not produce a
2×2 block; it produces the scalar statement "couple ↑→↑ with `+λ` and ↓→↓ with `-λ`, couple nothing
else". Every matrix element in this package stays a scalar.

Interface:
----------
A rule implements THREE methods -- two required, one optional:

    qnum_factor(rule, schema, i, j) -> ComplexF64            # REQUIRED
    rule_axes(rule, schema) -> Tuple{Vararg{Symbol}}         # REQUIRED
    validate_rule(rule, schema) -> Nothing                   # optional

[`qnum_factor`](@ref) returns the multiplicative factor for a coupling from internal state `j`
to internal state `i`; `0` means *no coupling*. [`rule_axes`](@ref) names the axes the rule
speaks for -- every axis NOT returned is completed with a Kronecker delta by
[`qnum_matrix`](@ref) at compile, so a wrong answer silently changes physics, which is why its
fallback is a named refusal rather than a guess. [`validate_rule`](@ref) may add checks ahead of
evaluation; the default resolves every axis in `rule_axes` against the schema, so an undeclared
axis raises at compile with no extra code.

Collapsing "does it couple?" and "by how much?" into one number is deliberate: selection and
modulation are then the same predicate, and composing rules is plain multiplication with a
short-circuit on zero.

Axes a rule does not name:
--------------------------
`qnum_factor` speaks only for the axes in `rule_axes`; every axis not returned there is completed
with a Kronecker delta by [`qnum_matrix`](@ref) at compile time. So "no rule on an axis" means
*diagonal on that axis*: a bare hopping on a spinful cell is spin-diagonal, a bare `kinetic` on a
layered cell is layer-diagonal, and anything that changes a quantum number is always written by
hand (`raise`, `pauli_x`, ...). Rules therefore compose on the union of their named axes -- see
the warning on [`otimes`](@ref).

Hermiticity discipline:
-----------------------
A user rule inherits the package's stored-once contract: the compiler stores ONE direction of
every coupling and materializes the conjugate partner. On any pair whose partner it supplies,
the rule's reverse factor must therefore be the conjugate of its forward factor -- or exactly
zero (a one-sided ladder, whose partner is generated). A violating rule is refused at compile
by the emitters' conjugate-symmetry check ("its quantum-number rule is non-Hermitian on the
unordered pair ..."), never silently Hermitized.

Notes:
------
Rules are evaluated against the cell's [`InternalSchema`](@ref), which is authoritative: naming an
axis or a level the schema does not declare is an error, raised by [`qnum_matrix`](@ref) at compile
time. Terms do not call `qnum_factor` per bond -- `qnum_matrix` compiles the rule to a small dense
matrix once, and emission is then one array index. For a complete worked subtype to copy as a
template, see [`TwoLevelMatrixQnum`](@ref): it compiles verbatim, its `rule_axes` method is one
line, and its `validate_rule` shows what a stricter (level-checking) validation looks like.
"""
abstract type AbstractQnumRule end

"""
    qnum_factor(rule::AbstractQnumRule, qi, qj) -> ComplexF64

Return the multiplicative factor of `rule` for a coupling `j -> i`; `0` means no coupling.

Args:
-----
- `schema`: the internal label space both indices refer to
- `i`: interned internal state of the target (row / bra)
- `j`: interned internal state of the source (column / ket)

Returns:
--------
- `ComplexF64`: the factor; exactly `zero(ComplexF64)` when the rule rejects the pair.
"""
function qnum_factor end

"""
    _rule_axis(schema, key, rule) -> Int

Resolve the axis a rule acts on, refusing to proceed when the schema does not declare it.

Notes:
------
Raising here is the point. The previous implementation read quantum numbers out of a dictionary
with a default, so a mistyped axis name -- or a rule applied to a cell that simply has no such axis
-- resolved every state to the same absent value, matched nothing, and made the term emit **no
bonds at all**. That is a wrong band structure with no diagnostic. With the schema authoritative,
it is an error at compile time.
"""
function _rule_axis(s::InternalSchema, key::Symbol, rule::AbstractQnumRule)::Int
    a = axis_position(s, key)
    a == 0 && throw(ArgumentError(
        "$(nameof(typeof(rule))) acts on the internal axis `$key`, which this cell does not " *
        "declare (its axes are $(axis_names(s))). The rule would reject every pair and the term " *
        "would silently emit nothing."
    ))
    return a
end

"""
    _check_rule_levels(schema, a, wanted, rule) -> Nothing

Refuse a rule that names a level the axis does not declare.

Notes:
------
The same failure mode as `_rule_axis`, one level down: `pauli_z(:spin)` defaults to levels `±1`, so
applying it to an axis declared as `[:up, :down]` would match nothing and emit nothing.
"""
function _check_rule_levels(s::InternalSchema, a::Int, wanted, rule::AbstractQnumRule)::Nothing
    for v in wanted
        level_ordinal(s, a, v) == 0 && throw(ArgumentError(
            "$(nameof(typeof(rule))) on axis `$(s.axes[a])` names the level $(repr(v)), which the " *
            "axis does not declare (its levels are $(s.levels[a])). No pair would match, so the " *
            "term would silently emit nothing."
        ))
    end
    return nothing
end




#######################################################
###### Quantum-number selection rules: instances ######
#######################################################

"""
    AnyQnum()

The empty rule: it names no axis, so after the unnamed-axis completion of `qnum_matrix` it
compiles to the **identity** on the internal space. This is the default for every term: a term
with no rule is diagonal in every internal quantum number, and any coupling that changes one is
written explicitly.

Notes:
------
`qnum_factor` alone returns `1` for every pair -- it speaks only for the axes the rule names,
which for this rule is none. The identity semantics live in the completion, not here.
"""
struct AnyQnum <: AbstractQnumRule end

@inline qnum_factor(::AnyQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64 = one(ComplexF64)


"""
    ConserveQnum(keys)

Couple only states whose quantum numbers agree on every key in `keys`; factor `1` when they do.

Example:
--------
`conserve(:spin)` restricts a hopping to be spin-diagonal.
"""
struct ConserveQnum <: AbstractQnumRule
    keys::Vector{Symbol}

    function ConserveQnum(keys::Vector{Symbol})
        isempty(keys) && throw(ArgumentError("`ConserveQnum` requires at least one key; use `AnyQnum()` for no constraint."))
        return new(sort(unique(keys)))
    end
end

function qnum_factor(rule::ConserveQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    for k in rule.keys
        a = _rule_axis(s, k, rule)
        ordinal_at(s, i, a) == ordinal_at(s, j, a) || return zero(ComplexF64)
    end
    return one(ComplexF64)
end


"""
    DiagonalQnum(key, values)

A rule that is diagonal in `key`: couples only states with equal `key`, with a factor depending on
its value. Values absent from `values` reject the pair.

This is the `σ_z`-like family -- Zeeman terms, valley-selective potentials, sublattice masses.

Example:
--------
`pauli_z(:spin)` is `DiagonalQnum(:spin, Dict(1 => 1, -1 => -1))`.
"""
struct DiagonalQnum <: AbstractQnumRule
    key::Symbol
    values::Dict{QLevel,ComplexF64}

    function DiagonalQnum(key::Symbol, values::AbstractDict)
        isempty(values) && throw(ArgumentError("`DiagonalQnum` for key `$key` has no values; it would reject every pair."))
        return new(key, Dict{QLevel,ComplexF64}(_as_level(k) => ComplexF64(v) for (k, v) in values))
    end
end

function qnum_factor(rule::DiagonalQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    oi = ordinal_at(s, i, a)
    oi == ordinal_at(s, j, a) || return zero(ComplexF64)
    return get(rule.values, s.levels[a][oi], zero(ComplexF64))
end


"""
    OffDiagonalQnum(key, pairs)

A rule resolved on the ordered pair `(qi[key], qj[key])`. Pairs absent from `pairs` reject.

This is the `σ_x` / `σ_y` / `σ_±` family, and the layer-tunneling family: anything that changes a
quantum number rather than reading it.

Example:
--------
`pauli_x(:spin)` is `OffDiagonalQnum(:spin, Dict((1,-1) => 1, (-1,1) => 1))`.
"""
struct OffDiagonalQnum <: AbstractQnumRule
    key::Symbol
    pairs::Dict{Tuple{QLevel,QLevel},ComplexF64}

    function OffDiagonalQnum(key::Symbol, pairs::AbstractDict)
        isempty(pairs) && throw(ArgumentError("`OffDiagonalQnum` for key `$key` has no pairs; it would reject every pair."))
        return new(key, Dict{Tuple{QLevel,QLevel},ComplexF64}((_as_level(k[1]), _as_level(k[2])) => ComplexF64(v) for (k, v) in pairs))
    end
end

function qnum_factor(rule::OffDiagonalQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    key = (s.levels[a][ordinal_at(s, i, a)], s.levels[a][ordinal_at(s, j, a)])
    return get(rule.pairs, key, zero(ComplexF64))
end


"""
    MatrixQnum(key, M)

A rule whose factor on the ordered level pair `(a, b)` of axis `key` is `M[a, b]`, indexed by
the axis's **declared level order**. Build with [`matrix_rule`](@ref).

This is the k·p workhorse: a 4-band Luttinger-Kohn or 8-band Kane block is a handful of
`k_quadratic(...) ⊗ matrix_rule(:band, Γ_αβ)` terms instead of an unreadable pile of Pauli
strings. The compiler's existing stored-once check (`reverse ≠ conj(forward)` is refused on
unordered pairs) is the Hermiticity check for it -- a non-Hermitian `M` on an on-site term
raises exactly like a hand-built non-Hermitian Pauli string would.
"""
struct MatrixQnum <: AbstractQnumRule
    key::Symbol
    M::Matrix{ComplexF64}

    function MatrixQnum(key::Symbol, M::AbstractMatrix{<:Number})
        size(M, 1) == size(M, 2) || throw(DimensionMismatch(
            "`MatrixQnum` for axis `$key` needs a square matrix; got $(size(M))."))
        all(isfinite, M) || throw(DomainError(M,
            "`MatrixQnum` for axis `$key` contains a non-finite entry."))
        return new(key, Matrix{ComplexF64}(M))
    end
end

function qnum_factor(rule::MatrixQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    return rule.M[ordinal_at(s, i, a), ordinal_at(s, j, a)]
end


"""
    TwoLevelMatrixQnum(key, M, up, down)

A 2×2 matrix rule on a two-level axis whose row/column order is fixed by LEVEL VALUES rather
than by declaration order: row/column 1 is the `up` level, 2 the `down` level. With
`up === nothing` the ordering resolves against the schema at compile time: an axis whose two
levels are exactly `{+1, -1}` takes `up = +1` -- the `pauli_x/y/z` and `raise`/`lower`
convention, whatever the declaration order -- and any other two-level axis falls back to
declaration order (first level up).

This is the rule the k·p and moiré builders (`dirac_cone`, `moire_tunnelling`,
`angular_harmonic`) attach their Pauli blocks with: it makes the builder spelling and the
`pauli_*` spelling agree on ±1 axes instead of silently constructing the time-reversed valley
on a `:sub => [-1, +1]` schema, while still working on named axes like `:band => [:c, :v]`.
"""
struct TwoLevelMatrixQnum <: AbstractQnumRule
    key::Symbol
    M::Matrix{ComplexF64}
    up::Any                                 # nothing = resolve against the schema at compile
    down::Any

    function TwoLevelMatrixQnum(key::Symbol, M::AbstractMatrix{<:Number}, up, down)
        size(M) == (2, 2) || throw(DimensionMismatch(
            "`TwoLevelMatrixQnum` for axis `$key` needs a 2×2 matrix; got $(size(M))."))
        all(isfinite, M) || throw(DomainError(M,
            "`TwoLevelMatrixQnum` for axis `$key` contains a non-finite entry."))
        isnothing(up) == isnothing(down) || throw(ArgumentError(
            "Give both `up` and `down` levels of axis `$key`, or neither (auto-resolution)."))
        return new(key, Matrix{ComplexF64}(M), up, down)
    end
end

"""
    _two_level_updown(rule, s, a) -> (up_level, down_level)

The resolved (up, down) level pair of a `TwoLevelMatrixQnum` on axis `a` of schema `s`:
explicit levels win; otherwise `{+1, -1}` axes are value-keyed (`up = +1`) and any other
two-level axis keeps declaration order.
"""
function _two_level_updown(rule::TwoLevelMatrixQnum, s::InternalSchema, a::Int)
    isnothing(rule.up) || return (rule.up, rule.down)
    lv = s.levels[a]
    if (isequal(lv[1], 1) && isequal(lv[2], -1)) || (isequal(lv[1], -1) && isequal(lv[2], 1))
        return (1, -1)
    end
    return (lv[1], lv[2])
end

function qnum_factor(rule::TwoLevelMatrixQnum, s::InternalSchema, i::Int, j::Int)::ComplexF64
    a = _rule_axis(s, rule.key, rule)
    up, _ = _two_level_updown(rule, s, a)
    lv = s.levels[a]
    pos(o) = isequal(lv[o], up) ? 1 : 2
    return rule.M[pos(ordinal_at(s, i, a)), pos(ordinal_at(s, j, a))]
end


### ------------------------------------------------------------------------------------
### How `f`'s matrix maps onto the axis's levels -- three cases, three types
### ------------------------------------------------------------------------------------
###
### This was a `mode::Symbol` on `DirectionalQnum` until 2026-09-17, with `up`, `down` and
### `levels` beside it as three parallel fields of which only a subset was ever meaningful:
### `up`/`down` on `:two_level` alone, `levels` on `:named` alone and empty otherwise. That is
### a tagged union written by hand, in a language whose dispatch does tagged unions natively,
### and its cost was that every consumer re-read the tag to find out which of its own fields
### were real. Each case now owns exactly the data it needs, and each validates itself in its
### own constructor rather than under an `if` in `DirectionalQnum`'s.

"""
    LevelOrder

How the rows and columns of a directional rule's matrix map onto an internal axis's levels.
One of [`DeclaredOrder`](@ref), [`TwoLevelOrder`](@ref) or [`NamedOrder`](@ref).
"""
abstract type LevelOrder end

"""
    DeclaredOrder()

Declared-level order, rows = target (bra), columns = source (ket): exactly
[`matrix_rule`](@ref)'s convention. What [`directional_rule`](@ref) builds.
"""
struct DeclaredOrder <: LevelOrder end

"""
    TwoLevelOrder(up = nothing, down = nothing)

The value-keyed two-level resolution of [`TwoLevelMatrixQnum`](@ref): `up = +1` on a `{+1, -1}`
axis whatever the declaration order, with `up`/`down` overriding when given. Used by `rashba`,
so a directional rule agrees with `pauli_*` on ±1 axes.

`nothing` for both means "resolve by value at compile".
"""
struct TwoLevelOrder <: LevelOrder
    up::Any
    down::Any
end
TwoLevelOrder() = TwoLevelOrder(nothing, nothing)

"""
    NamedOrder(levels)

Rows and columns follow `levels` (e.g. `[:px, :py, :pz]`), and the matrix is permuted into the
axis's declared order at compile. Used by `sk_hopping`, so a `[:py, :px]` declaration gets the
same physics as `[:px, :py]`.

Every level the axis declares must appear in `levels` -- checked against the schema by
`validate_rule`, since only the schema knows what the axis declares. What CAN be checked here,
and is, is that the list is non-empty and repeats nothing.
"""
struct NamedOrder <: LevelOrder
    levels::Vector{QLevel}

    function NamedOrder(levels::Vector{QLevel})
        isempty(levels) && throw(ArgumentError(
            "`NamedOrder` needs the level order of the rule's matrix; got an empty list."))
        allunique(levels) || throw(ArgumentError(
            "`NamedOrder` level order $(levels) repeats a level; each row/column of the " *
            "rule's matrix must map to one distinct level."))
        return new(levels)
    end
end

"""
    DirectionalQnum(key, f, order::LevelOrder)

A **bond-direction-dependent** matrix rule on the internal axis `key`: the matrix is not fixed
per term but produced per site pair by `f(d_cart::SVector{D,Float64}) -> AbstractMatrix`, with
`d_cart` the cartesian displacement FROM the source state (site `sj`, cell `ΔR`) TO the target
state (site `si`, home cell) -- so reversing a bond flips `d`. Build with
[`directional_rule`](@ref); this page documents the type because builders (`rashba`,
`sk_hopping`) construct it directly with a non-default [`LevelOrder`](@ref).

Where it lowers:
----------------
Entirely at **compile time**: `emit_term!` splits a composed rule into its static part and its
directional factors, and each emitter obtains its matrix per site pair through
`_pair_rule_matrix` -- `f` is evaluated once per pair, composed with the static part BEFORE the
unnamed-axis completion (so the Hadamard semantics of `compose` stay correct), and lowered to
constants in the flat coupling list. The per-k cost is untouched.

`order` fixes how the rows/columns of `f`'s matrix map onto the axis's levels, and each case is
its own type carrying exactly its own data: [`DeclaredOrder`](@ref), [`TwoLevelOrder`](@ref)
(which holds `up`/`down`) or [`NamedOrder`](@ref) (which holds the label order).

Hermiticity contract, per bond:
-------------------------------
On any pair whose Hermitian partner is supplied by `finalize_couplings` (an on-site pair, or an
undirected selector such as a distance shell), the **full product** must satisfy
`amp·M(-d)[b, a] == conj(amp·M(d)[a, b])` -- equivalently `M(-d) == M(d)†` for a real
amplitude. A violating pair is refused at compile, never silently Hermitized. Rashba's
`f(d) = iλ(σ_x d̂_y - σ_y d̂_x)` is odd and anti-Hermitian, so it passes exactly.

What it refuses (at compile, with the term named):
- `OnSiteSelector`, `ΔR = 0` same-site pairs, and `DeltaGSelector` -- no bond direction there;
- a matrix of the wrong size for the axis, or with non-finite entries;
- the conjugate-symmetry violation above.

`qnum_factor` cannot be evaluated without a direction, so it raises by name; the rule is still
a full citizen of the rule contract through `rule_axes` and `validate_rule`.
"""
struct DirectionalQnum{L<:LevelOrder} <: AbstractQnumRule
    key::Symbol
    f::Function
    order::L
end

"""
    directional_rule(f, axis::Symbol) -> DirectionalQnum

A matrix-valued quantum-number rule whose matrix depends on the **bond direction**:
`f(d_cart::SVector{D,Float64}) -> AbstractMatrix`, with `d_cart` the cartesian displacement
from the source state to the target state of the bond and the matrix given in the axis's
**declared level order**, rows = target (bra), columns = source (ket) -- exactly
[`matrix_rule`](@ref)'s convention, made a function of `d`.

Evaluated **once per site pair at compile time**: the term lowers to constants in the flat
coupling list, the per-k loop is untouched, and `reweight` still works (couplings stay linear
in the scalar amplitude). This is the first-class spelling for Rashba spin-orbit coupling,
Slater-Koster direction cosines, and Kitaev-type bond anisotropies; the shipped
`rashba`/`sk_hopping` builders are built on it.

    # Rashba on the nearest-neighbour shell: iλ (σ × d̂)·ẑ, the i INSIDE f
    f(d) = (dh = d / norm(d); im * (dh[2] * [0 1; 1 0] - dh[1] * [0 -im; im 0]))
    hopping(λ; shell=1) ⊗ directional_rule(f, :spin)

Contract (checked per bond, at compile, refusals naming the term):
- the selector must give the pair a direction: `onsite`, same-site `ΔR = 0` pairs and
  `delta_g` are refused;
- on pairs whose Hermitian partner is supplied automatically, the full product must satisfy
  `amp·f(-d)[b, a] == conj(amp·f(d)[a, b])` (for a real amplitude: `f(-d) == f(d)†`; an odd
  anti-Hermitian `f`, or an even Hermitian one, passes) -- otherwise the term would be
  silently Hermitized, which the package refuses to do;
- `f` must return an `n_levels × n_levels` matrix with finite entries.

Compose with `⊗` as usual: `directional_rule(f, :spin) ⊗ project(:layer, 1)` restricts the
Rashba coupling to one layer, and every axis the composition does not name stays diagonal.
"""
directional_rule(f::Function, axis::Symbol) =
    DirectionalQnum(axis, f, DeclaredOrder())

qnum_factor(rule::DirectionalQnum, ::InternalSchema, ::Int, ::Int) = throw(ArgumentError(
    "`directional_rule` on axis `$(rule.key)` has no per-pair factor without a bond " *
    "direction: its matrix is `f(d_cart)`, resolved per site pair when the term compiles. " *
    "Compile the term (or call `qnum_matrix` on its static factors only) instead of " *
    "evaluating the rule directly."))

rule_axes(rule::DirectionalQnum, ::InternalSchema) = (rule.key,)

### one method per level order: the checks that used to sit under an `if` on the tag now sit
### on the type that owns the data they check, and `DeclaredOrder` needs none.
function validate_rule(rule::DirectionalQnum{DeclaredOrder}, s::InternalSchema)::Nothing
    _rule_axis(s, rule.key, rule)
    return nothing
end

function validate_rule(rule::DirectionalQnum{TwoLevelOrder}, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    o = rule.order
    n_levels(s, a) == 2 || throw(DimensionMismatch(
        "A two-level directional rule on axis `$(rule.key)` needs exactly two levels; " *
        "the axis declares $(n_levels(s, a)) ($(levels(s, a)))."))
    isnothing(o.up) && return nothing
    _check_rule_levels(s, a, (o.up, o.down), rule)
    isequal(o.up, o.down) && throw(ArgumentError(
        "`up` and `down` of axis `$(rule.key)` must be the two DISTINCT levels; both " *
        "are $(o.up)."))
    return nothing
end

function validate_rule(rule::DirectionalQnum{NamedOrder}, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    for v in s.levels[a]
        any(l -> isequal(l, v), rule.order.levels) || throw(ArgumentError(
            "Directional rule on axis `$(rule.key)`: the axis declares the level " *
            "$(repr(v)), which is not one of the rule's labels $(rule.order.levels). The " *
            "rule cannot silently leave a level uncoupled -- declare the axis with these " *
            "labels only, or build the full block with `directional_rule`."))
    end
    return nothing
end


"""
    ProductQnumRule(rules)

The tensor product of several rules: the factors multiply, and a zero from any factor short-circuits.

Build with `⊗` rather than by hand -- see `compose`.
"""
struct ProductQnumRule <: AbstractQnumRule
    rules::Vector{AbstractQnumRule}

    function ProductQnumRule(rules::Vector{<:AbstractQnumRule})
        isempty(rules) && throw(ArgumentError("`ProductQnumRule` requires at least one rule; use `AnyQnum()` for no constraint."))
        return new(collect(AbstractQnumRule, rules))
    end
end

function qnum_factor(rule::ProductQnumRule, s::InternalSchema, i::Int, j::Int)::ComplexF64
    acc = one(ComplexF64)
    for r in rule.rules
        f = qnum_factor(r, s, i, j)
        iszero(f) && return zero(ComplexF64)     # short-circuit: the product is already dead
        acc *= f
    end
    return acc
end




#################################
###### Composition of rules #####
#################################

"""
    compose(a::AbstractQnumRule, b::AbstractQnumRule) -> AbstractQnumRule

Combine two rules into their product, flattening nested products and absorbing `AnyQnum`.

Notes:
------
The `AnyQnum` identities matter in practice: every term starts with `AnyQnum()` as its rule, so
without them a term built by two `⊗` applications would carry a `ProductQnumRule` whose first
factor is a no-op evaluated on every candidate bond.
"""
compose(a::AbstractQnumRule, b::AbstractQnumRule) = ProductQnumRule(AbstractQnumRule[a, b])
compose(::AnyQnum, b::AbstractQnumRule) = b
compose(a::AbstractQnumRule, ::AnyQnum) = a
compose(::AnyQnum, ::AnyQnum) = AnyQnum()
compose(a::ProductQnumRule, b::AbstractQnumRule) = ProductQnumRule(vcat(a.rules, AbstractQnumRule[b]))
compose(a::AbstractQnumRule, b::ProductQnumRule) = ProductQnumRule(vcat(AbstractQnumRule[a], b.rules))
compose(a::ProductQnumRule, b::ProductQnumRule) = ProductQnumRule(vcat(a.rules, b.rules))
compose(a::ProductQnumRule, ::AnyQnum) = a
compose(::AnyQnum, b::ProductQnumRule) = b

"""
    otimes(a, b)
    a ⊗ b

Tensor product. Between two quantum-number rules this is `compose`; between a Hamiltonian term and
a rule it attaches the rule to the term (defined in `terms.jl`).

`⊗` is an alias parsed at `*` precedence, so terms read as physics:

    hopping(-t; shell=1) ⊗ conserve(:spin)
    onsite(B) ⊗ pauli_z(:spin)

!!! warning "This is a Hadamard product, not a Kronecker product"
    Every rule is already expressed on the *full* internal space, so composing them multiplies
    the `qnum_factor`s entrywise. `⊗` reads like `kron`, and the two differ by exactly one
    silently wrong answer. The rules do not each carry their own little space that needs
    tensoring up -- the schema already gave them a common one.

    One consequence of the unnamed-axis completion (see `qnum_matrix`): composition happens
    BEFORE the completion, on the union of the named axes. So

        qnum_matrix(a ⊗ b, schema) == qnum_matrix(a, schema) .* qnum_matrix(b, schema)

    holds only when `a` and `b` name the same axes. In general the compiled matrices cannot be
    multiplied after the fact: `raise(:layer)` compiles spin-diagonal and `pauli_x(:spin)`
    compiles layer-diagonal, so their entrywise product is zero everywhere -- while
    `raise(:layer) ⊗ pauli_x(:spin)` is the intended spin-flip interlayer coupling. Compose
    rules, not matrices.
"""
otimes(a::AbstractQnumRule, b::AbstractQnumRule) = compose(a, b)

const ⊗ = otimes




#########################################
###### Named helpers for the rules ######
#########################################

"""
    conserve(keys::Symbol...) -> ConserveQnum

Require the listed quantum numbers to be equal across the coupling.

Notes:
------
Since the unnamed-axis completion (see `qnum_matrix`), this is what every axis gets by default,
so a bare `conserve(:spin)` no longer changes what a term emits. It remains useful as
documentation-in-code -- stating a conservation the reader might otherwise wonder about -- and
inside compositions that must name an axis explicitly.
"""
conserve(keys::Symbol...) = ConserveQnum(collect(Symbol, keys))

"""
    matrix_rule(key::Symbol, M) -> MatrixQnum

The rule whose factor on the ordered level pair `(a, b)` of axis `key` is `M[a, b]`, indexed by
the axis's declared level order -- the compact spelling for multi-band k·p blocks:

    k_quadratic(1.0) ⊗ matrix_rule(:band, Γ)      # a Luttinger-Kohn block in one line

Which index is which, since for a complex element it decides between the element and its
conjugate: **rows index the TARGET level (the bra, the row of `H[i,j]`), columns the SOURCE
(the ket)** -- `M[a, b]` multiplies `|a⟩⟨b|`. So `γ₁ |A₂⟩⟨B₁|` on `layer ⊗ sub` axes is
`matrix_rule(:layer, [0 0; 1 0]) ⊗ matrix_rule(:sub, [0 1; 0 0])`.

`M` need not be Hermitian or complete: a one-sided ladder (`[0 0; 1 0]`, or a shift-by-one
nilpotent on an N-level axis) or a plain diagonal (`diagm(u)` for a per-level potential) is the
idiomatic spelling -- the stored-once compiler supplies the Hermitian partner. **That is a
stored-once contract, exactly like bonds: give ONE direction only.** Splitting a Hermitian
block into separate σ₊ and σ₋ TERMS (`matrix_rule(:band, [0 1; 0 0])` plus
`matrix_rule(:band, [0 0; 1 0])`) silently doubles it -- each term is Hermitized on its own,
so the pair compiles to `2σₓ`, not `σₓ`. Write the full Hermitian matrix in one rule, or one
triangle only. Hermiticity of `M` itself is NOT checked here (the rule alone cannot know which
entries land on stored-once pairs): a genuinely non-Hermitian combination surfaces downstream,
as the compiler's stored-once
conjugate-symmetry refusal on unordered pairs or as `finalize_couplings`' reality `DomainError`
on diagonal entries.
"""
matrix_rule(key::Symbol, M::AbstractMatrix{<:Number}) = MatrixQnum(key, M)

"""
    pauli_0(key; up=1, down=-1) -> DiagonalQnum

Identity in the two-value sector of `key`: couples `up→up` and `down→down`, both with factor `1`.

Notes:
------
This is *not* the same as `AnyQnum()`: it restricts the coupling to the `{up, down}` sector and
rejects any other value of `key`. On a two-level axis the two now compile to the same matrix
(the default is diagonal since the unnamed-axis completion), but on an axis with more levels
`pauli_0` rejects everything outside its sector.
"""
pauli_0(key::Symbol; up::QLevel=1, down::QLevel=-1) =
    DiagonalQnum(key, Dict(up => one(ComplexF64), down => one(ComplexF64)))

"""
    pauli_x(key; up=1, down=-1) -> OffDiagonalQnum

σₓ in the two-value sector of `key`: flips the quantum number with factor `1`.
"""
pauli_x(key::Symbol; up::QLevel=1, down::QLevel=-1) =
    OffDiagonalQnum(key, Dict((up, down) => one(ComplexF64), (down, up) => one(ComplexF64)))

"""
    pauli_y(key; up=1, down=-1) -> OffDiagonalQnum

σ_y in the two-value sector of `key`.

Notes:
------
With `(i,j)` ordered as `(row, column)`, σ_y = [0 -i; i 0], so the `up←down` element is `-i` and
the `down←up` element is `+i`.
"""
pauli_y(key::Symbol; up::QLevel=1, down::QLevel=-1) =
    OffDiagonalQnum(key, Dict((up, down) => -im, (down, up) => im))

"""
    pauli_z(key; up=1, down=-1) -> DiagonalQnum

σ_z in the two-value sector of `key`: `+1` on `up`, `-1` on `down`.
"""
pauli_z(key::Symbol; up::QLevel=1, down::QLevel=-1) =
    DiagonalQnum(key, Dict(up => one(ComplexF64), down => -one(ComplexF64)))

"""
    project(key, value) -> DiagonalQnum

Project onto a single value of `key`: couples `value→value` with factor `1` and rejects everything
else. Useful for layer- or valley-resolved on-site terms.
"""
project(key::Symbol, value::QLevel) = DiagonalQnum(key, Dict(value => one(ComplexF64)))

"""
    raise(key; from=-1, to=1) -> OffDiagonalQnum
    lower(key; from=1, to=-1) -> OffDiagonalQnum

Ladder operators on `key`: couple `from -> to` only, with factor `1`.

`raise(:layer)` is the natural way to write one direction of an interlayer tunneling term; the
Hermitian conjugate is supplied by the compiler, so only one direction should be given.
"""
raise(key::Symbol; from::QLevel=-1, to::QLevel=1) =
    OffDiagonalQnum(key, Dict((to, from) => one(ComplexF64)))

lower(key::Symbol; from::QLevel=1, to::QLevel=-1) =
    OffDiagonalQnum(key, Dict((to, from) => one(ComplexF64)))
@doc (@doc raise) lower




#######################################################
###### Compiling a rule to a matrix                ####
#######################################################

"""
    qnum_matrix(rule, schema) -> Matrix{ComplexF64}

Compile `rule` into its `n_internal × n_internal` matrix: `qnum_factor` on the axes the rule
names, times a Kronecker delta on every axis it does not.

Purpose:
--------
Called **once per term, at compile time**. The bond-emission loop then reduces a selection rule to
one array index instead of two dictionary lookups plus a dynamic dispatch. Since `n_internal` is
small by construction -- 1 trivial, 2 spinful, 4 for a moire bilayer -- the matrix is a few hundred
bytes.

The unnamed-axis completion:
----------------------------
This is where "no rule on an axis = diagonal on that axis" is enforced, for every term, in one
place. Without it the default `AnyQnum` compiled to the all-ones matrix, so a bare `kinetic` on a
two-layer plane-wave cell coupled the layers with full strength -- eigenvalues `(0, 2E)` per `G`
instead of `(E, E)`, silently. The delta lives here rather than in `qnum_factor` so that partial
rules are completed too: `raise(:layer)` on a layer ⊗ spin cell is spin-diagonal without the
caller having to remember `⊗ conserve(:spin)`.

Notes:
------
This is also where a rule is validated against the cell it will act on: naming an axis or a level
the schema does not declare raises here rather than silently producing an empty term.
"""
function qnum_matrix(rule::AbstractQnumRule, s::InternalSchema)::Matrix{ComplexF64}
    validate_rule(rule, s)
    n = n_internal(s)
    touched = _touched_axes!(falses(n_axes(s)), rule, s)
    unnamed = findall(!, touched)
    M = Matrix{ComplexF64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        M[i, j] = _diagonal_on(s, i, j, unnamed) ? qnum_factor(rule, s, i, j) :
                                                   zero(ComplexF64)
    end
    return M
end

"""
    _diagonal_on(schema, i, j, axes) -> Bool

Whether the internal states `i` and `j` agree on every axis in `axes` -- the Kronecker delta of
the unnamed-axis completion.
"""
@inline function _diagonal_on(s::InternalSchema, i::Int, j::Int, axes)::Bool
    for a in axes
        ordinal_at(s, i, a) == ordinal_at(s, j, a) || return false
    end
    return true
end

"""
    rule_axes(rule, schema) -> Tuple{Vararg{Symbol}}

The internal axes the rule NAMES -- the axes its `qnum_factor` speaks for. REQUIRED for every
[`AbstractQnumRule`](@ref) subtype.

Every axis not returned is completed with a Kronecker delta by [`qnum_matrix`](@ref) at compile
("no rule on an axis = diagonal on that axis"), so a wrong answer here silently changes
physics: an axis the rule handles but does not return gets a delta it must not have, and an
axis returned but ignored by `qnum_factor` loses its delta. That is why the fallback below is a
named refusal, never a guess.
"""
rule_axes(rule::AbstractQnumRule, ::InternalSchema) = throw(ArgumentError(
    "Define `rule_axes(::$(typeof(rule)), ::InternalSchema)` returning the tuple of internal " *
    "axes the rule names: every axis the rule does not name is completed with a Kronecker " *
    "delta at compile, so the compiler must not guess."))

rule_axes(::AnyQnum, ::InternalSchema) = ()
rule_axes(rule::ConserveQnum, ::InternalSchema) = Tuple(rule.keys)
rule_axes(rule::DiagonalQnum, ::InternalSchema) = (rule.key,)
rule_axes(rule::OffDiagonalQnum, ::InternalSchema) = (rule.key,)
rule_axes(rule::MatrixQnum, ::InternalSchema) = (rule.key,)
rule_axes(rule::TwoLevelMatrixQnum, ::InternalSchema) = (rule.key,)
### a product names the union of its children's axes
rule_axes(rule::ProductQnumRule, s::InternalSchema) =
    Tuple(unique(Symbol[k for r in rule.rules for k in rule_axes(r, s)]))

"""
    _touched_axes!(mask, rule, schema) -> mask

Mark, in `mask`, every axis position the rule names -- one generic pass over
[`rule_axes`](@ref). Axes left unmarked receive the Kronecker delta completion in `qnum_matrix`.
"""
function _touched_axes!(mask::BitVector, rule::AbstractQnumRule, s::InternalSchema)
    for k in rule_axes(rule, s)
        mask[_rule_axis(s, k, rule)] = true
    end
    return mask
end

"""
    validate_rule(rule, schema) -> Nothing

Check, ahead of evaluation, that a rule is meaningful on the schema. Called by
[`qnum_matrix`](@ref) once per term at compile.

OPTIONAL for a user subtype: the generic default resolves every axis in [`rule_axes`](@ref)
against the schema, refusing an undeclared axis by name. The built-in rules add stricter
checks under the same name -- declared levels, matrix sizes matching the level count.
"""
function validate_rule(rule::AbstractQnumRule, s::InternalSchema)::Nothing
    for k in rule_axes(rule, s)
        _rule_axis(s, k, rule)
    end
    return nothing
end

function validate_rule(rule::DiagonalQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    return _check_rule_levels(s, a, keys(rule.values), rule)
end

function validate_rule(rule::OffDiagonalQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    return _check_rule_levels(s, a, Iterators.flatten(keys(rule.pairs)), rule)
end

function validate_rule(rule::MatrixQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    nl = n_levels(s, a)
    size(rule.M, 1) == nl || throw(DimensionMismatch(
        "`matrix_rule(:$(rule.key), M)` carries a $(size(rule.M, 1))×$(size(rule.M, 2)) matrix, " *
        "but the axis declares $nl levels ($(levels(s, a)))."))
    return nothing
end

function validate_rule(rule::TwoLevelMatrixQnum, s::InternalSchema)::Nothing
    a = _rule_axis(s, rule.key, rule)
    n_levels(s, a) == 2 || throw(DimensionMismatch(
        "A two-level Pauli block on axis `$(rule.key)` needs exactly two levels; the axis " *
        "declares $(n_levels(s, a)) ($(levels(s, a)))."))
    isnothing(rule.up) && return nothing
    _check_rule_levels(s, a, (rule.up, rule.down), rule)
    isequal(rule.up, rule.down) && throw(ArgumentError(
        "`up` and `down` of axis `$(rule.key)` must be the two DISTINCT levels; both are " *
        "$(rule.up)."))
    return nothing
end

function validate_rule(rule::ProductQnumRule, s::InternalSchema)::Nothing
    for r in rule.rules
        validate_rule(r, s)
    end
    return nothing
end
