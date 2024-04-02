import numpy as np

from hartree_fock.sto_3g import H as Hydrogen, O as Oxygen


def test_sto3g():
    basis_h = Hydrogen()
    assert basis_h.n_max == 1, "PQN = 1"
    assert basis_h.l_max == 0, "s-orbital only"
    ref_coeffs = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
    assert np.allclose(basis_h.get_coefficients(1, 0), ref_coeffs)

    basis_oxy = Oxygen()
    assert basis_oxy.n_max == 2, "max(PQN) == 2"
    assert basis_oxy.l_max == 1, "s and p-orbitals"
    ref_2s_coeffs = [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]
    assert np.allclose(basis_oxy.get_coefficients(2, 0), ref_2s_coeffs)
