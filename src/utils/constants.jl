export ROUND_SLACK_DECADES,
       LATTICE_ROUND_DIGITS, LATTICE_ATOL,
       INT_ROUND_DIGITS, INT_ROUND_ATOL,
       AMPLITUDE_ATOL, HERMITICITY_ATOL, LINDEP_RTOL, CHERN_ATOL, PHASE_PIN_ATOL,
       SHIFT_INVERT_RESID_RTOL,
       DISPLAY_DIGITS,
       MAX_INTERNAL, AUTO_SITE_PREFIX,
       DEFAULT_GAUGE, GAUGE_MODES, H5_FORMAT_VERSION, H5_V_NPERIODIC,
       H5_V_KPERIODICITY,
       SHELL_RTOL, SHELL_MAX_GROW, SHELL_DEGENERACY_FACTOR,
       GREEDY_U_ATOL, BRAVAIS_RTOL




#####################################################################
###### Rounding granularities and their comparison tolerances   #####
#####################################################################

"""
    ROUND_SLACK_DECADES::Int = 2

Decades of slack between a rounding granularity and its comparison tolerance.

Every quantity this package canonicalizes has two numbers attached: the number of decimal
digits it is rounded to, and the tolerance at which two such values are then compared. Those
two must not be chosen independently -- a comparison tolerance at or below the rounding step is
useless, because two values that round to the same representable grid point can still differ by
a full step. So there is exactly one free parameter, and both tolerances are derived from their
own granularity:

    atol = 10^(-digits + ROUND_SLACK_DECADES)

Two decades of slack means "equal up to a hundred rounding steps", which absorbs the few ulps a
matrix product or a change of basis accumulates without ever admitting a physically distinct
value -- lattice positions differ by ~1e-2 at the smallest, twelve decades above the tolerance.
"""
const ROUND_SLACK_DECADES::Int = 2

"""
    LATTICE_ROUND_DIGITS::Int = 14

Digits `r_frac` and any fractional coordinate are rounded to (after wrapping into [0,1)).
The paired comparison tolerance is [`LATTICE_ATOL`](@ref).
"""
const LATTICE_ROUND_DIGITS::Int = 14

"""
    LATTICE_ATOL::Float64

Comparison tolerance for fractional coordinates, derived as
`10^(-LATTICE_ROUND_DIGITS + ROUND_SLACK_DECADES)` -- see [`ROUND_SLACK_DECADES`](@ref) for why
the tolerance and the rounding granularity are coupled.
"""
const LATTICE_ATOL::Float64 = exp10(-LATTICE_ROUND_DIGITS + ROUND_SLACK_DECADES)

"""
    INT_ROUND_DIGITS::Int = 13

Digits used when snapping values that must land on integers: Miller indices, supercell
matrices, and anything transported through a change of basis that is only meaningful if it
comes back integral. The paired tolerance is [`INT_ROUND_ATOL`](@ref).

The slack matters more here than for positions: a Miller index of magnitude ~30 pushed through
two matrix products accumulates error well above one rounding step, so a tolerance equal to the
step would reject labels that are integers in every sense that counts.
"""
const INT_ROUND_DIGITS::Int = 13

"""
    INT_ROUND_ATOL::Float64

Comparison tolerance for must-be-integer values, derived as
`10^(-INT_ROUND_DIGITS + ROUND_SLACK_DECADES)` -- see [`INT_ROUND_DIGITS`](@ref).
"""
const INT_ROUND_ATOL::Float64 = exp10(-INT_ROUND_DIGITS + ROUND_SLACK_DECADES)

"""
    GREEDY_U_ATOL::Float64 = 1e-6

How far `A_red * A⁻¹` may miss the integers before `_greedy_reduce` refuses to round it.

Deliberately much looser than [`INT_ROUND_ATOL`](@ref), and for the opposite reason: this is not
a snapping grid, it is a TRUST test on a float inverse. `A_red * A⁻¹` is mathematically an
integer matrix, so any residual at all is error, and its size measures how much of the
transformation the inverse has lost. The question is not "is this close enough to an integer to
snap" but "is the integer it snaps to still the right one".

Calibrated by measurement rather than chosen, on rhombohedral, triclinic and face-centred cells
across five decades of conditioning, 400 scrambled bases per bin. At `1e-6` the certificate
admits ZERO wrong transformations in every bin, while refusing 5/400 recoverable ones at
`cond(A) ≈ 4e3` and 140/400 at `≈ 3e6`. At `1e-8` it admits zero as well but refuses 38/400 and
292/400 in the same bins -- strictly worse for no gain in safety. The asymmetry is deliberate: a
refusal is actionable (reduce the basis in stages), a wrong `U` is not detectable downstream.
"""
const GREEDY_U_ATOL::Float64 = 1e-6




#####################################################################
###### Tolerances that are not tied to a rounding grid          #####
#####################################################################

"""
    AMPLITUDE_ATOL::Float64 = 1e-12

Tolerance on hopping amplitudes and Hamiltonian matrix elements -- dimensionless quantities of
order one that are never rounded to a grid, so the tolerance is chosen directly rather than
derived from a granularity. Used for pruning negligible couplings (the `atol` default of
[`compile`](@ref)) and for the stored-list Hermiticity check.
"""
const AMPLITUDE_ATOL::Float64 = 1e-12

"""
    HERMITICITY_ATOL::Float64 = 1e-8

Tolerance for the runtime check `max |H(k) - H(k)'|` before diagonalization (the `atol` default
of [`bands`](@ref)). Looser than [`AMPLITUDE_ATOL`](@ref) on purpose: the deviation it inspects
is the *accumulated float noise* of the whole H(k) assembly (thousands of cispi products summed
in arbitrary order), not a single amplitude. It guards against structural bugs -- a
mis-Hermitized term -- which produce O(1) violations, so anything comfortably above float noise
and far below physics works.
"""
const HERMITICITY_ATOL::Float64 = 1e-8

"""
    LINDEP_RTOL::Float64 = 1e-10

Relative tolerance for the linear-independence test on a set of lattice vectors,
`|det A| / prod(norm(aᵢ)) > LINDEP_RTOL`. Relative because the quantity is scale-free by
construction; loose because it is a conditioning guard, not an equality test.
"""
const LINDEP_RTOL::Float64 = 1e-10

"""
    CHERN_ATOL::Float64 = 1e-2

Tolerance at which the Fukui-Hatsugai lattice flux of [`chern_number`](@ref) must sit on an
integer before that integer is returned. Deliberately loose: the discretized flux converges to
the exact integer exponentially in the grid size, so a genuine Chern number lands within ~1e-10
of an integer on any adequate grid, while an inadequate grid (or a gapless subspace) misses by
O(1). A deviation between those regimes means the grid cannot be trusted, and `chern_number`
refuses rather than rounds -- silently rounding a non-quantized flux is how wrong topology gets
published.
"""
const CHERN_ATOL::Float64 = 1e-2

"""
    PHASE_PIN_ATOL::Float64 = 1e-10

Amplitude a component must clear before it may serve as the per-band phase reference of a
stored eigenvector (`_pin_phase!`, in `spectra/diagonalize_over_bz.jl`). An eigenvector is
defined only up to a phase and every LAPACK build picks a different one, which is invisible in
observables but not across a FILE: a Wannier90 deck is written from these vectors and the
`_u.mat` the binary returns refers to the gauge it was handed. `diagonalize_over_bz` therefore
rotates each column so that its first component above this amplitude is real and positive,
making the stored grid a function of the Hamiltonian rather than of the BLAS. The vectors are
normalized, so `1e-10` is far below anything carrying weight -- the threshold exists only to
skip numerically-zero components, whose `arg` would be noise.
"""
const PHASE_PIN_ATOL::Float64 = 1e-10

"""
    SHIFT_INVERT_RESID_RTOL::Float64 = 1e-4

Relative bar on the back-transformed residual `‖Hv − λv‖ / max(1, |λ|)` of a shift-invert
eigenpair, above which `diagonalize` refuses the result. `lu` does not reliably raise when
`sigma` lands on an eigenvalue -- it factorizes on ~1e-16 pivots, the inverted map is then
dominated by that null space, and the solver converges in θ-space to pairs that are garbage in
the original problem. Those show O(1) RELATIVE residuals; a healthy solve at a shift merely
close to the spectrum shows ~1e-8. The bar sits between, with orders of margin on both sides,
and is deliberately relative: an absolute one tuned to a particular solver's convergence starts
refusing healthy results the moment the solver changes.
"""
const SHIFT_INVERT_RESID_RTOL::Float64 = 1e-4

"""
    DISPLAY_DIGITS::Int = 6

Digits used when rounding for *display only*. Never compare against this.
"""
const DISPLAY_DIGITS::Int = 6




#####################################################################
###### Internal (quantum-number) space                          #####
#####################################################################

"""
    MAX_INTERNAL::Int = 256

Largest `n_internal` an [`InternalSchema`](@ref) may declare. Every quantum-number rule
compiles to an `n_internal × n_internal` matrix of `ComplexF64`, so the cost is quadratic: 256
caps a rule at 1 MB. Anything approaching this bound means a large index has been put in the
schema that does not belong there -- plane waves belong in `G`, positions in the site registry.
"""
const MAX_INTERNAL::Int = 256

"""
    AUTO_SITE_PREFIX::String = "site"

Prefix used to name sites that the caller did not name: `:site1`, `:site2`, ...
"""
const AUTO_SITE_PREFIX::String = "site"




#####################################################################
###### Bloch gauge                                              #####
#####################################################################

"""
    GAUGE_MODES = (:atomic, :periodic)

The two Bloch-phase conventions a model can be evaluated in:

- `:atomic`   -> `H_ij(k) = Σ t_ij(ΔR) exp(2πi k⋅(ΔR + τ_j - τ_i))`
- `:periodic` -> `H_ij(k) = Σ t_ij(ΔR) exp(2πi k⋅ΔR)`

Models are always *stored* in `:atomic` (the physical convention); `:periodic` is produced at
the assembly exit by [`gauge_matrix`](@ref), and [`change_gauge`](@ref) is an O(1) flag flip.
"""
const GAUGE_MODES = (:atomic, :periodic)

"""
    DEFAULT_GAUGE::Symbol = :atomic

The gauge newly compiled models carry -- see [`GAUGE_MODES`](@ref).
"""
const DEFAULT_GAUGE::Symbol = :atomic

"""
    H5_FORMAT_VERSION::Int = 4

The HIGHEST on-disk format version this package can READ. A file stamped above it is refused
rather than guessed at.

**This is a ceiling, not a stamp.** Do not write it: each writer stamps the LOWEST version that
describes what it actually put in the file, so a file an older reader can read correctly keeps
round-tripping with that reader. The per-feature versions are named below, and conflating them
with this ceiling is a real bug that already happened once -- the lattice writer stamped
`H5_FORMAT_VERSION` for a partially periodic lattice, so raising the ceiling to 4 silently
started claiming v4 semantics for ribbons that only need v3.

The versions: files WITHOUT the dataset are legacy v1 (bonds + per-term kterm groups, no
k0/site-kind columns), read through a compatibility path; **v2** is the unified coupling-list
layout; **v3** is [`H5_V_NPERIODIC`](@ref); **v4** is [`H5_V_KPERIODICITY`](@ref). Optional
datasets an older reader can ignore WITHOUT being wrong (term provenance, `/producer`) do not
bump anything. The shared rule: a version goes up when an older reader would silently MISREAD
the file, never when it would merely ignore part of it.
"""
const H5_FORMAT_VERSION::Int = 4

"""
    H5_V_NPERIODIC::Int = 3

The version that added `/lattice/nperiodic`. Stamped only on files that actually carry a
partially periodic lattice, because a v2 reader would misread a ribbon as fully periodic.
"""
const H5_V_NPERIODIC::Int = 3

"""
    H5_V_KPERIODICITY::Int = 4

The version that marks `/sites/kperiodicity` as NOT re-derivable from the rest of the file.

Stamped only on that case -- in practice a folded plane-wave cell, where the `|det M|` coset
labels read as a truncation window and a k·p frame would come back as a truncated one with
every zone-requiring refusal laundered. A reader older than this version ignores the dataset
and derives, which is right whenever the derivation agrees, so a file whose class IS derivable
keeps writing 2 or 3.
"""
const H5_V_KPERIODICITY::Int = 4




#####################################################################
###### Crystallography                                          #####
#####################################################################

"""
    BRAVAIS_RTOL::Float64 = 1e-8

Relative tolerance for CLASSIFYING a lattice: the single source of truth for "do these two
lengths match" and "is this angle 90 degrees" across `identify_bravais`, `lattice_point_group`,
`high_symmetry_points`/`high_symmetry_path`/`bz_variant` and `standard_kpath_nodes`.

It was a bare `1e-8` repeated at every one of those entry points. That is the shape of the
tolerance-mismatch bug fixed in `harmonic_potential`: two numbers that must agree, spelled
separately, so that moving one moves the classification and the k-points apart.

RELATIVE is the whole point, and it is why this package does not copy spglib's single Cartesian
`symprec`: measured on spglib 2.7.0, one tetragonal shape with `c/a - 1 = 2e-6` classifies as
CUBIC at `a = 0.05` and TETRAGONAL at `a = 5`. A classification must not depend on the units the
lattice was written in. Where a metric rather than a length is compared, scale by
`_metric_scale(G) = det(G)^(1/D)`, which is unimodular-invariant and carries the units of `G`'s
own entries -- NOT by `maximum(abs, G)`, which is a property of the basis you were handed.

Distinct from [`SHELL_RTOL`](@ref), which happens to share the value but separates physically
distinct neighbour shells, and from the Krivy-Gruber reduction's own internal branch tolerance:
conflating a classification tolerance with a reduction tolerance is how the reduction fails to
converge.
"""
const BRAVAIS_RTOL::Float64 = 1e-8




#####################################################################
###### Neighbour shells                                         #####
#####################################################################

"""
    SHELL_RTOL::Float64 = 1e-8

Relative tolerance used when clustering pair distances into shells. Far looser than the
position tolerances, and deliberately so: it separates physically distinct shells (which differ
by percent-level amounts), not numerically distinct values.
"""
const SHELL_RTOL::Float64 = 1e-8

"""
    SHELL_MAX_GROW::Int = 8

Maximum number of cutoff doublings when growing a neighbour table to reach a shell count.
"""
const SHELL_MAX_GROW::Int = 8

"""
    SHELL_DEGENERACY_FACTOR::Float64 = 10.0

Warn when two adjacent shells are separated by less than this multiple of [`SHELL_RTOL`](@ref).
"""
const SHELL_DEGENERACY_FACTOR::Float64 = 10.0




#####################################################################
###### Data directories                                         #####
#####################################################################

# Root of the validated crystallography tables (space groups, high-symmetry points and paths).
# Defined once, here, so every reader in `crystallography/` resolves its CSVs against the same
# anchor. Deliberately not exported: an internal path, not user API. The `../..` climbs from
# `src/utils/` to the package root.
const _CRYSTAL_DATA_DIR::String = normpath(joinpath(@__DIR__, "..", "..", "external", "crystallography"))

# Root of the moiré continuum-model database (`moiredb`, v2.1): eleven gzipped JSONL entity tables
# plus `build.json`. Same rule as the crystallography anchor above -- one definition, resolved
# against the package root -- and likewise not exported. The reader is `interfaces/moiredb/moiredb.jl`.
const _MOIREDB_DIR::String = normpath(joinpath(@__DIR__, "..", "..", "external", "moiredb-v2", "db"))
