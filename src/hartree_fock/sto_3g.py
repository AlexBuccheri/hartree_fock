""" STO-3G Basis

I've only specified for H and O, but one would tabulate for all elements.
Note, hard-coding dataclasses does not scale well.

What would be better if is one parses JSON from the [Basis Set Exchange](https://www.basissetexchange.org)
and generates them dynamically with the type constructor

TODO Have a single array constructor per element, for exponent and contractions
"""
import dataclasses

import numpy as np

from hartree_fock.basis_attrs import GaussianBasisAttrs

# Number of primitives
n_primitives = 3


@dataclasses.dataclass
class H(GaussianBasisAttrs):
    name = 'hydrogen'
    # ['1s']  Replaced str with a tuple of the quantum numbers
    # (n, lx, ly, lz)
    components = [(1, 0, 0, 0)]
    n_max = 1
    l_max = 0
    exponent = np.zeros(shape=(n_primitives, n_max, l_max + 1))
    contractions = np.zeros(shape=(n_primitives, n_max, l_max + 1))
    # 1s
    exponent[:, 0, 0] = np.array([0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00])
    contractions[:, 0, 0] = np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00])


@dataclasses.dataclass
class O(GaussianBasisAttrs):
    name = 'oxygen'
    # ['1s', '2s', '2px', '2py', '2pz'] given in terms of (n, lx, ly, lz)
    components = [(1, 0, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 0, 1, 0), (2, 0, 0, 1)]
    n_max = 2
    l_max = 1
    exponent = np.zeros(shape=(n_primitives, n_max, l_max + 1))
    contractions = np.zeros(shape=(n_primitives, n_max, l_max + 1))
    # 1s
    exponent[:, 0, 0] = np.array([0.1307093214E+03, 0.2380886605E+02, 0.6443608313E+01])
    # 2s
    exponent[:, 1, 0] = np.array([0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00])
    # 2p == 2s
    exponent[:, 1, 1] = exponent[:, 1, 0]
    # 1s
    contractions[:, 0, 0] = np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00])
    # 2s
    contractions[:, 1, 0] = np.array([-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00])
    # 2p
    contractions[:, 1, 1] = np.array([0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00])
