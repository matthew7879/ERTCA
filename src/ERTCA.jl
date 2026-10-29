"""Numerical routines retained for Figures 1–10 of the ERTCA paper."""
module ERTCA
using LinearAlgebra, SparseArrays, StaticArrays
using KrylovKit: eigsolve, Block, BlockLanczos
export AMPLITUDE_ATOL, AUTO_SITE_PREFIX, AbstractBondSelector, AbstractBondWeight, AbstractEnvelope, AbstractHamiltonianTerm
export AbstractQnumRule, AnyQnum, BRAVAIS_RTOL, BasisState, BlochSumEnvelope, BondTerm
export BravaisLattice, CellTables, Chirality, ChiralityPhase, CompileContext, ConserveQnum
export CouplingBuffer, CouplingList, DEFAULT_GAUGE, DISPLAY_DIGITS, DeclaredOrder, DeltaGSelector
export DiagonalQnum, DirectionalQnum, ExactlyPeriodic, ExplicitBondSelector, ExponentialDecay, GAUGE_MODES
export GaussianEnvelope, HERMITICITY_ATOL, HamiltonianModel, HoppingTerm, INT_ROUND_ATOL, INT_ROUND_DIGITS
export InternalSchema, KDependentTerm, KEnvelopeTerm, KLinearTerm, KPeriodicity, KPolynomialTerm
export KQuadraticTerm, LATTICE_ATOL, LATTICE_ROUND_DIGITS, LINDEP_RTOL, LevelOrder, MAX_INTERNAL
export MatrixQnum, NamedOrder, NeighborTable, NoZone, OffDiagonalQnum, OnSiteSelector
export ProductQnumRule, QLevel, ROUND_SLACK_DECADES, RadiusSelector, RegistryHarmonics, RegistryScan
export SHELL_DEGENERACY_FACTOR, SHELL_MAX_GROW, SHELL_RTOL, SHIFT_INVERT_RESID_RTOL, ShellSelector, TRIVIAL_SCHEMA
export TabulatedEnvelope, TranslationSelector, TruncatedWindow, TwistedBilayer, TwoLevelMatrixQnum, TwoLevelOrder
export UnitCell, UnitWeight, adopted_buffer, axis_names, axis_position, bands
export bloch_hamiltonian, bloch_solver, bond_weight, bravais_canonical_name, bravais_lattice, bz_corners
export bz_map, bz_volume, canonicalize_fracpos, canonicalize_int, canonicalize_pos, cell_tables
export cell_volume, change_gauge, compile, compose, delta_g, diagonalize
export display_name, distance_atol, each_canonical, embed_into_higher_dimension, ertca_model, finalize_couplings
export fold_to_bz, fractional_to_real, fractional_to_reciprocal, g_shells, g_star, gauge
export harmonic_potential, has_kdependence, identify_bravais, internal_index, is_canonical_coupling, is_folded
export is_hermitian, is_symmetric_weight, is_trivial, k_envelope, k_quadratic, kernel_parity
export kpath_line, kperiodicity, kvalue, lattice_parameters, layer_lattice, leading_block
export length_scale, level_at, level_ordinal, levels, lower, matrix_rule
export max_radius, max_shell, metric, moire_tunnelling, monolayer_bz_gvectors, n_axes
export n_canonical, n_couplings, n_internal, n_levels, n_monolayer_cells, n_orbitals
export n_shells, n_sites, n_stackings, n_states, ndim, neighbor_table
export neighbor_table_by_shell, nperiodic, ordinal_at, otimes, plane_wave_cell, primitive_recip_vector
export primitive_recip_vectors, primitive_vector, project, qnum_factor, qnum_matrix, raise
export read_wannier90_hr, read_wannier90_spn, read_wannier90_wout, real_to_fractional, recip_matrix, reciprocal
export reciprocal_to_fractional, registry_harmonics, registry_scan, round_atol, rule_axes, same_basis
export shell_entries, site_kinds, site_names, site_positions, sparse_bloch_workspace, spectrum_along_line
export standard_kpath_nodes, state_g_vectors, state_internals, state_origins, state_positions, states_on_site_with_G
export states_with_G, supercell_matrix, term_name, term_names, threaded_chunks, twist_angle
export twisted_bilayer, validate_internal, validate_rule, validate_weight, wrap01, ⊗

export adopt_provenance!, bands!, begin_term!, bloch_hamiltonian!, emit_term!, kernel_id!, push_coupling!, sparse_bloch_hamiltonian!

include("numerics.jl")
include("geometry.jl")
include("hamiltonian.jl")
include("wannier.jl")
include("construction.jl")
include("spectra.jl")
end
