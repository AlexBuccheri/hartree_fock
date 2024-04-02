"""Hartree Fock implementation following https://pycrawfordprogproj.readthedocs.io/en/latest/Project_03/Project_03.html
and  https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303

All inputs found here: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303/input/h2o/STO-3G

Separate the function implementations from the main code - want to test this as a plugin system

To run from root: python src/hartree_fock/main.py
"""
import dataclasses
from pathlib import Path

import numpy as np
import scipy
from numpy import floating
from numpy._typing import _64Bit
from scipy.spatial import distance_matrix

from hartree_fock.basis_attrs import BasisType
from hartree_fock.gaussian_basis import GaussianBasis
from hartree_fock.gridding import CubicGrid3D

# Reference core Hamiltonian
ref_hcore = np.array([[-32.5773954, -7.5788328, -0., -0.0144738, 0., -1.2401023, -1.2401023],
                      [-7.5788328, -9.2009433, -0., -0.1768902, 0., -2.9067098, -2.9067098],
                      [0., 0., -7.4588193, 0., 0., -1.6751501, 1.6751501],
                      [-0.0144738, -0.1768902, 0., -7.4153118, 0., -1.3568683, -1.3568683],
                      [0., 0., 0., 0., -7.3471449, 0., 0.],
                      [-1.2401023, -2.9067098, -1.6751501, -1.3568683, 0., -4.5401711, -1.0711459],
                      [-1.2401023, -2.9067098, 1.6751501, -1.3568683, 0., -1.0711459, -4.5401711]])


@dataclasses.dataclass
class H2O:
    # Copied from what looks like .xyz format, but LEAVE as angstrom as this gives an E_ion_ion
    # that is consistent with the reference value
    positions = np.array([[0.000000000000, -0.143225816552, 0.000000000000],
                          [1.638036840407, 1.136548822547, 0.000000000000],
                          [-1.638036840407, 1.136548822547, 0.000000000000]])
    species = ['o', 'h', 'h']
    atomic_number = [8.0, 1.0, 1.0]


def energy_nuclear_repulsion_with_loops(z: list, positions: np.ndarray) -> floating[_64Bit]:
    """ Nuclear-nuclear repulsion energy
    Less efficient implementation
    :return:
    """
    energy = 0
    r = positions
    n_atoms = len(z)
    for a in range(n_atoms):
        for b in range(a + 1, n_atoms):
            energy += (z[a] * z[b]) / np.linalg.norm(r[a, :] - r[b, :])
    return energy


def energy_nuclear_repulsion(z: list, positions: np.ndarray) -> floating[_64Bit]:
    """Nuclear-nuclear repulsion energy
    :return:
    """
    z_ab = np.outer(z, z)
    r_ab = distance_matrix(positions, positions)
    # Ignore divide by zero warning - we won't touch those elements
    with np.errstate(divide='ignore'):
        e_mat = z_ab / r_ab
    indices = np.triu_indices(e_mat.shape[0], k=1)
    energy = np.sum(e_mat[indices])
    return energy


def parse_one_electron_integrals(file):
    """ Parse overlap, kinetic or nuclear-attraction integrals.

    Preference would be numpy but only the lower triangle is stored
    so parse like a peasant.
    :return: Full matrix of one-electron integrals
    """
    with open(file, mode='r') as fid:
        lines = fid.readlines()

    n_ao = int(lines[-1].split()[0])
    matrix = np.empty(shape=(n_ao, n_ao))

    # Lines correspond to lower triangle only
    for line in lines:
        i, j, element = line.split()
        # Convert indexing to 0-indexing
        i, j = int(i) - 1, int(j) - 1
        matrix[i, j] = float(element)
        matrix[j, i] = float(element)

    return matrix


def compute_core_hamiltonian(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Add expression

    :param T:
    :param V:
    :return:
    """
    h_core = T + V
    assert np.allclose(h_core, ref_hcore)
    return h_core


def two_electron_symmetry_mapping(i, j, k, l, include_self=True):
    """ Generate symmetrically-equivalent indices for 2-electron integral (ij|kl)

    Consider if I can generate this algebraically, rather than hard-coding
    the definitions
    :return:
    """
    self_indices = []
    if include_self:
        self_indices = [[i, j, k, l]]

    mapping = self_indices + [[j, i, k, l], [i, j, l, k], [j, i, l, k]]
    # For periodic calculations, these permutations would correspond to complex conjugates
    conjugate_mappings = [[k, l, i, j], [l, k, i, j], [k, l, j, i], [l, k, j, i]]

    # TODO Need to look into this step. Not very efficient to transpose
    indices = np.array(mapping + conjugate_mappings)
    return tuple(indices.T)


def test_two_electron_integrals(file, two_e_integral):
    """ Check integral symmetry

    TODO Move me to a unit test. Prevents using the debugger on this file
    :return:
    """
    # Parse file so I can get the indices of the defined integrals
    # Explicitly compare to all 7 permutations
    with open(file, mode='r') as fid:
        lines = fid.readlines()

    # Expect True for every element
    for line in lines:
        i, j, k, l, integral = line.split()
        # Convert to 0-indexing
        i, j, k, l = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1

        equivalent_values = [two_e_integral[i, j, l, k],
                             two_e_integral[j, i, k, l],
                             two_e_integral[j, i, l, k],
                             two_e_integral[k, l, i, j],
                             two_e_integral[l, k, i, j],
                             two_e_integral[k, l, j, i],
                             two_e_integral[l, k, j, i]
                             ]
        all_close = np.allclose(equivalent_values, float(integral))
        print(i, j, k, l, all_close)


def parse_two_electron_integrals(file, store_all=True) -> np.ndarray:
    """

    Only the permutationally unique integrals are provided in the file
    however the working examples require us to store all elements.

    Regardless of whether all are stored, all elements are allocated
    :return:
    """
    with open(file, mode='r') as fid:
        lines = fid.readlines()

    n_ao = int(lines[-1].split()[0])
    # Must be initialised with zeros because even with assigning all symmtrically
    # equivalent elements, many will still not be touched
    matrix = np.zeros(shape=(n_ao, n_ao, n_ao, n_ao))

    if store_all:
        assign_permutations = two_electron_symmetry_mapping
    else:
        assign_permutations = lambda i, j, k, l: tuple(np.array([i, j, k, l]))

    for line in lines:
        i, j, k, l, integral = line.split()
        # Convert to 0-indexing
        i, j, k, l = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1
        mappings = assign_permutations(i, j, k, l)
        matrix[mappings] = float(integral)

    if np.isnan(matrix).any():
        raise ValueError('two_electron_integrals contains NaN/s')

    return matrix


def parse_two_electron_integrals_explicit_assignment(file) -> np.ndarray:
    """

    :return:
    """
    with open(file, mode='r') as fid:
        lines = fid.readlines()

    n_ao = int(lines[-1].split()[0])
    # Must be initialised with zeros because even with assigning all symmtrically
    # equivalent elements, many will still not be touched
    matrix = np.zeros(shape=(n_ao, n_ao, n_ao, n_ao))

    for line in lines:
        i, j, k, l, integral = line.split()
        # Convert to 0-indexing
        i, j, k, l = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1
        print(float(integral))
        matrix[i, j, k, l] = float(integral)
        matrix[i, j, l, k] = float(integral)
        matrix[j, i, k, l] = float(integral)
        matrix[j, i, l, k] = float(integral)

        matrix[k, l, i, j] = float(integral)
        matrix[l, k, i, j] = float(integral)
        matrix[k, l, j, i] = float(integral)
        matrix[l, k, j, i] = float(integral)

    if np.isnan(matrix).any():
        raise ValueError('two_electron_integrals contains NaN/s')

    return matrix


def compute_orthogonalisation_matrix(overlap: np.ndarray) -> np.ndarray:
    """
    ADD EXPRESSIONS
    :return:
    """
    eigenvalue, L = scipy.linalg.eigh(overlap)
    assert np.all(eigenvalue > 1.e-6), ("One or more eigenvalue is very close to zero, "
                                        "meaning S is semi-positive definite")
    # Diagonal matrix, so one operates element-wise on the vector,
    # then assigns it to the diagonal, rather than inverting
    inv_root_delta = np.diag(eigenvalue ** -0.5)
    # Doing L.T rather than np.conjugate(L.T), as L is real for finite systems with no external fields
    o_matrix = L @ (inv_root_delta @ L.T)
    return o_matrix


def build_density_matrix(eigenvectors: np.ndarray, n_occ: int):
    """

    :param eigenvectors:
    :param n_occ:
    :return:
    """
    # Contract over occupied molecular orbital state indices
    d_matrix = 2. * eigenvectors[0:n_occ, :].T @ eigenvectors[0:n_occ, :]
    assert d_matrix.shape == h0.shape

    # Enforce symmetry if required
    d_matrix_trans = np.copy(d_matrix.T)
    if not np.allclose(d_matrix - d_matrix_trans, 0.):
        print('Symmetrising the density matrix guess from H_core')
        d_matrix = 0.5 * (d_matrix + d_matrix_trans)

    return d_matrix


def density_matrix_guess_from_hcore(h0: np.ndarray, n_occ: int):
    """

    :param h0: Core Hamiltonian
    :param n_occ: Number of occupied states
    :return:
    """
    _, eigenvectors = scipy.linalg.eigh(h0)
    d_matrix = build_density_matrix(eigenvectors, n_occ)
    return d_matrix


def build_fock_coulomb(d_matrix, two_electon_integrals) -> np.ndarray:
    """

    :return:
    """
    n_ao = two_electon_integrals.shape[0]
    # Compute \sum_{kl} d_(kl) e_(ij)(kl) == D @ E^T, having recast to 1D and 2D, in the matrix expression
    f_coul_vector = d_matrix.reshape(n_ao * n_ao) @ np.transpose(
        two_electon_integrals.reshape(n_ao * n_ao, n_ao * n_ao))
    # Reshape to 2D
    f_coul = f_coul_vector.reshape(n_ao, n_ao)
    assert f_coul.shape == (n_ao, n_ao)
    return f_coul


def build_fock_exchange(d_matrix, two_electon_integrals) -> np.ndarray:
    """

    :return:
    """
    # The easy way. This will crunch for any realistic basis
    return np.einsum("ikjl, kl -> ij", two_electon_integrals, d_matrix)


def build_fock_matrix(h0, d_matrix, two_electon_integrals):
    """

    :param h0:  Core Hamiltonian
    :param d_matrix:  Density matrix
    :param two_electon_integrals: Two electron integrals, in unpacked form
    :return:
    """
    f_coul = build_fock_coulomb(d_matrix, two_electon_integrals)
    f_exc = build_fock_exchange(d_matrix, two_electon_integrals)
    return h0 + f_coul + f_exc


# TODO Move to reference
# # Put the molecule in a cubic box
# max_dim = np.max(positions)
# # Points per dimension
# n_points = int(2 * max_dim / spacing) + 1
# x = np.linspace(-max_dim, max_dim, n_points, endpoint=True)
# X, Y, Z = np.meshgrid(x, x, x)
# grid = np.stack((X, Y, Z), axis=-1)
# assert grid.shape == (n_points, n_points, n_points, 3)
# i_mid = int(0.5 * n_points)
# assert np.allclose(grid[i_mid, i_mid, i_mid, :], [0., 0., 0.]), "grid should be centred at [0,0,0] by construction"


def wavefunctions_on_realspace_grid(positions: np.ndarray, spacing: float, basis: GaussianBasis, mo_coeff: np.ndarray):
    """ Construct the Hartree-Fock Wave functions on a real-space grid.

    positions: Centre of mass should be at (0,0,0)
    :return:
    """
    # Grid should be supplied?
    # Put the molecule in a cubic box
    max_dim = np.max(positions)
    box_lengths = 2 * np.array([max_dim, max_dim, max_dim])
    spacings = np.array([spacing, spacing, spacing])
    grid = CubicGrid3D(box_lengths, spacings, zero_centred=True)
    grid_points = grid.all_points()

    # TODO Make this a func. Currently hard-coded for H2O sto-3g
    # AOs 0-4 belong to first oxy. AO 5 belongs to first H, and AO 6 belongs to second H.
    ao_index_to_atomic_site_index = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2}

    n_molecular_orbitals = mo_coeff.shape[1]
    n_total = grid.n_total()

    wave_functions = []
    for i_mo in range(n_molecular_orbitals):
        # Iterate over the basis, resolved in terms of components (== number of atomic orbitals)
        for ao_index, ao_function in enumerate(basis.functions):
            wave_function = np.zeros(shape=n_total)
            iatom = ao_index_to_atomic_site_index[ao_index]
            # Evaluate primitive GTOs of the AO basis function at all grid points
            gto_primitives = ao_function(grid_points - positions[iatom, :])
            assert gto_primitives.shape == (n_total, basis.n_primitives)
            wave_function += mo_coeff[ao_index, i_mo] * np.sum(gto_primitives, axis=1)
        wave_functions.append(wave_function)

    return wave_functions


if __name__ == "__main__":
    h2o = H2O
    e_ion_ion = energy_nuclear_repulsion(h2o.atomic_number, h2o.positions)
    assert np.isclose(e_ion_ion, 8.002367061810450)

    # Project root. Change this from being hard-coded
    root = Path("/Users/alexanderbuccheri/Codes/isdf_prototypes")

    # Parse one-electron integrals
    overlap = parse_one_electron_integrals(root / "data/hf_h2o/s.dat")
    ke_integrals = parse_one_electron_integrals(root / "data/hf_h2o/t.dat")
    nuc_attract_integrals = parse_one_electron_integrals(root / "data/hf_h2o/v.dat")

    # Core Hamiltonian
    h0 = compute_core_hamiltonian(ke_integrals, nuc_attract_integrals)

    # Parse and assign 2-electron integrals
    two_electon_integrals = parse_two_electron_integrals(root / "data/hf_h2o/eri.dat")

    o_matrix = compute_orthogonalisation_matrix(overlap)
    assert np.allclose(o_matrix, scipy.linalg.fractional_matrix_power(overlap, -0.5))

    # Guess at density matrix from diagonalising H0
    # H2O has 12 electrons, so in spin unpolarised, two electrons per state => 6 occupied states
    n_occ = 6
    d_matrix_guess = density_matrix_guess_from_hcore(h0, n_occ)

    f_coul = build_fock_coulomb(d_matrix_guess, two_electon_integrals)
    ref_f_coul = np.einsum("ijkl, kl -> ij", two_electon_integrals, d_matrix_guess)
    assert np.allclose(f_coul, ref_f_coul)

    f_exc = build_fock_exchange(d_matrix_guess, two_electon_integrals)
    fock_matrix = h0 + f_coul + f_exc

    eigenvalues, eigenvectors = scipy.linalg.eigh(fock_matrix, overlap)
    d_matrix = build_density_matrix(eigenvectors, n_occ)

    # Compute total energy
    energy_core = 0.5 * np.trace(np.dot(d_matrix.T, h0))
    energy_fock = 0.5 * np.trace(np.dot(d_matrix.T, fock_matrix))
    energy_total = energy_core + energy_fock + e_ion_ion

    print('Total energy from initial guess:', energy_total)

    sto_3g_basis = GaussianBasis([s.upper() for s in H2O.species], BasisType.STO_3G)
    wave_functions = wavefunctions_on_realspace_grid(h2o.positions, 0.4, sto_3g_basis, eigenvectors)

    # -------------------------
    # TODOs List
    # -------------------------
    # TODO. Now need to plot this => Grid should definitely be injected.
    # Visualisation options:
    # Opt 1. https://github.com/pyvista/pyvista/discussions/1904
    # Opt 2. http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.points3d
    # Opt 3. Cube format, then some visualiser

    # SCF loop
    #  Diagonalise fock matrix for MO coefficients

    # After SCF:
    #   Watch this: https://www.youtube.com/watch?v=eDAfpQIMde0 on overlap matrix construction
    #   - Should implement
    #   Check out the basis set exchange python program in the paper of Laura (?)

