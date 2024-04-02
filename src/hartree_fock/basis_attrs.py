""" Basis Attributes
"""
import enum
from typing import List

import numpy as np


class GaussianBasisAttrs:
    """ Base class of attributes associated with any Gaussian basis
    """
    name: str
    components: List[tuple]
    n_max: int
    l_max: int
    exponent: np.ndarray
    contractions: np.ndarray

    def get_exponents(self, n: int, l: int):
        """Get exponents of the primitives
        Allows use of correct indexing for (n,l)
        """
        if n > self.n_max:
            raise ValueError(f"n-value {n}, exceeds max of {self.n_max} for {self.name} basis")
        if l > self.l_max:
            raise ValueError(f"l-value {l}, exceeds max of {self.l_max} for {self.name} basis")
        return self.exponent[:, n - 1, l]

    def get_coefficients(self, n: int, l: int):
        """Get contraction coefficients of the primitives
        Allows use of correct indexing for (n,l)
        """
        if n > self.n_max:
            raise ValueError(f"n-value {n}, exceeds max of {self.n_max} for {self.name} basis")
        if l > self.l_max:
            raise ValueError(f"l-value {l}, exceeds max of {self.l_max} for {self.name} basis")
        return self.contractions[:, n - 1, l]

    def component_qns_to_strings(self):
        """ Given an AO component of the basis, convert its quantum numbers
        (n, l_x, l_y, l_z) to a string of the form 'nlm'
        """
        shell_strs = []
        # Not a great implementation, but not terrible
        l_str = np.empty(shape=(2, 2, 2), dtype=object)
        l_str[0, 0, 0] = 's'
        l_str[1, 0, 0] = 'px'
        l_str[0, 1, 0] = 'py'
        l_str[0, 0, 1] = 'pz'
        for component in self.components:
            n, l_x, l_y, l_z = component
            assert all(np.array([l_x, l_y, l_z]) <= 1), 'Not implemented for d or f orbitals'
            shell_strs.append(str(n) + l_str[l_x, l_y, l_z])
        return shell_strs


class BasisType(enum.Enum):
    """Implemented basis types

    NOTE, this may be in the wrong module
    """
    # Gaussian minimal basis
    STO_3G = enum.auto()
