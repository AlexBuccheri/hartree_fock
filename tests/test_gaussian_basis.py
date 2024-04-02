import numpy as np

from hartree_fock.gaussian_basis import norm_cartesian, GaussianBasis, cartesian_gaussian_primitive
from hartree_fock.sto_3g import O as Oxygen


def test_norm_cartesian():
    # For s-orbitals
    alpha = 1.0
    norm = norm_cartesian(0, 0, 0, alpha)
    assert np.allclose(norm, (2 * alpha / np.pi)**0.75)

    # Normalisation should be the same for all p-orbitals
    alpha = 1.0
    norm_px = norm_cartesian(1, 0, 0, alpha)
    norm_py = norm_cartesian(0, 1, 0, alpha)
    norm_pz = norm_cartesian(0, 0, 1, alpha)
    assert norm_px == norm_py == norm_pz

    numerator = (2 * alpha / np.pi)**0.75 * np.sqrt(4 * alpha)
    assert np.allclose(norm_px, numerator)


def test_gaussian_basis_class():
    from hartree_fock.basis_attrs import BasisType
    basis = GaussianBasis(['O', 'H', 'H'], BasisType.STO_3G)

    assert basis.n_primitives == 3, "STO-3G should use 3 primitives for all AOs"

    ref_components = [('O', '1s'), ('O', '2s'), ('O', '2px'), ('O', '2py'), ('O', '2pz'), ('H', '1s'), ('H', '1s')]
    assert basis.components == ref_components
    assert basis.n_ao == 7, "H2O in ST0-3G basis"
    assert len(ref_components) == basis.n_ao

    # Check oxygen 2py function agrees with what one expects
    basis_functions = basis.functions

    # Random position vector
    r_vector = np.array([1., 1., 1.])
    ao_oxygen_2py = basis_functions[3](r_vector)

    # Manually retrieve the data and construct the basis function
    sto_3g_oxygen = Oxygen()
    alpha = sto_3g_oxygen.get_exponents(2, 1)
    coeff = sto_3g_oxygen.get_coefficients(2, 1)
    ref_ao_oxygen_2py = cartesian_gaussian_primitive(alpha, coeff, r_vector, [0, 1, 0])

    assert np.allclose(ao_oxygen_2py, ref_ao_oxygen_2py)
