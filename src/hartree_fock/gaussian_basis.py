""" Basis Module.
"""
import importlib
from typing import List, Callable

import numpy as np
from scipy.special import factorial2

from hartree_fock.basis_attrs import GaussianBasisAttrs, BasisType

# Map basis type enum to module in which the basis is defined
basis_module = {BasisType.STO_3G: 'hartree_fock.sto_3g'}


class GaussianBasis:

    def __init__(self, species: List[str], basis: BasisType):
        """

        :param species: List of species strings using chemical symbols, specified for all atoms in system
        :param basis: List of basis strings, specified for all atoms in system
        Or if single string, use the same basis for all atoms.
        """
        if not isinstance(basis, BasisType):
            raise NotImplementedError("GaussianBasis class must take the same basis for all atoms")

        if basis != BasisType.STO_3G:
            raise NotImplementedError("Only supported BasisType is STO_3G")

        # Dynamically load this module
        name = basis_module[basis]
        try:
            module_name = importlib.import_module(name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Module {name} was not found")

        # Number of primitives per basis function
        self.n_primitives = module_name.n_primitives

        # Store a list of dataclasses, with one per atom.
        # Note, may need to think about how this works if a basis function is not assigned to an atom
        basis_data: List[GaussianBasisAttrs] = [getattr(module_name, element)() for element in species]

        # TODO(Alex) Maybe these are just better refered to as AOs instead of components?
        # Component defined in terms of Cartesian quantum numbers
        self.components = self.set_components(species, basis_data)
        self.n_ao = np.sum([len(self.components)])

        # Basis functions
        self.functions: List[Callable] = self.basis_builder(basis_data)
        assert len(self.functions) == self.n_ao, "Number of basis_functions != expected number of AOs"

    @staticmethod
    def set_components(species, basis_data) -> List[tuple]:
        """ Create a list of all AO basis components, and their associated atomic species

        Note, I'm not sure if storing as [('O', '1s'), ('O', '2s'), ... is the most useful.
        :return: components: List of AO components
        """
        components = []
        for i in range(len(basis_data)):
            # Prepend element symbol to each
            subset_components = [(species[i], component) for component in basis_data[i].component_qns_to_strings()]
            components += subset_components
        return components

    def basis_builder(self, basis_data) -> List[Callable]:
        """ For each ao, store a basis_builder that takes a vector position and returns n_primitive Gaussians
        Note, this is resolved in terms of components, not shells.

        :return:
        """
        functions = []
        # Iterate over all atomic sites, and all AO components per site
        for atom_site in basis_data:
            for qns in atom_site.components:
                n, l_x, l_y, l_z = qns
                l = np.sum([l_x, l_y, l_z])
                exponent = atom_site.get_exponents(n, l)
                contractions = atom_site.get_coefficients(n, l)
                functions.append(self.cartesian_gaussian_primitive_builder(exponent, contractions, l_x, l_y, l_z))
        return functions

    @staticmethod
    def cartesian_gaussian_primitive_builder(alpha, coeff, l_x, l_y, l_z):
        """Given a set of exponents and coefficients for n_primitives,
        return a function that returns a sets of Cartesian GTOs, given a position vector.
        """
        def func(r):
            norm = norm_cartesian(l_x, l_y, l_z, alpha)
            x_y_z = r ** np.array([l_x, l_y, l_z])
            gaussian = np.exp(-alpha * np.linalg.norm(r))
            return norm * coeff * x_y_z * gaussian

        return func


def norm_cartesian(l_x: int, l_y: int, l_z: int, alpha: float):
    r""" Normalisation constant of a Cartesian Gaussian primitive function.

    This reference is from [IOData](https://iodata.readthedocs.io/en/latest/basis.html):

    .. math::

        N(l_x, l_y, l_z, \alpha) =
        \left( \frac{2 \alpha}{\pi} \right)^{3/4} \left(4 \alpha \right)^{\frac{l_x + l_y + l_z}{2}}
            \frac{1}{\sqrt{(2l_x - 1)!!(2l_y - 1)!!(2l_z - 1)!!}}

    :param l_x: Integer for x-component, >= 0
    :param l_y: Integer for x-component, >= 0
    :param l_z: Integer for x-component, >= 0
    :param alpha: Gaussian primitive exponent factor
    :return: norm: Normalisation factor
    """
    l_values = np.array([l_x, l_y, l_z])
    assert not any(l_values < 0), f"Cannot have negative lx, ly or lz values: {l_values}"
    numerator = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (0.5 * np.sum(l_values))
    # np.abs used for when l_i = 0, because scipy returns (-1)!! = 0, however we want (-1)!! = 1
    twol_minus_one = np.abs(2 * l_values - 1)
    denom = np.sqrt(np.prod(factorial2(twol_minus_one)))
    norm = numerator / denom
    return norm


def cartesian_gaussian_primitive(alpha, coeff, r_vector, l_values):
    """
    Definition from:
    https://chemistry.stackexchange.com/questions/41163/how-many-basis-functions-used-in-sto-3g-and-6-31g-for-the-water-molecule

    ADD LATEX

    :param alpha:
    :param coeff:
    :param r_vector:
    :param l_x:
    :param l_y:
    :param l_z:
    :return:
    """
    norm = norm_cartesian(*l_values, alpha)
    # element-wise power
    x_y_z = r_vector ** np.array(l_values)
    gaussian = np.exp(-alpha * np.linalg.norm(r_vector))
    gaussian *= norm * coeff * x_y_z
    return gaussian
