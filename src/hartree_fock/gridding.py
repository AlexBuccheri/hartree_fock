""" Grid Generation
"""
import numpy as np


class CubicGrid3D:
    def __init__(self, box: np.ndarray, spacing: np.ndarray, zero_centred=True):

        assert box.shape == (3, 3) or box.shape == (3,), f'Shape of box is {box.shape}'
        if box.shape == (3, 3):
            raise NotImplementedError("Non-orthogonal grids not implemented")
        self.box = box

        assert spacing.shape == (3,), "3D grid class must be supplied N grid points for each dimension"
        self.spacing = spacing

        self.n_points = self.spacing_to_points()

        self.origin = np.zeros(shape=3)
        if zero_centred:
            i_mid = (0.5 * self.n_points).astype(int)
            self.origin = self.indices_to_pos(*i_mid)

    def spacing_to_points(self) -> np.ndarray:
        return (self.box / self.spacing).astype(int) + 1

    def n_total(self):
        return np.prod(self.n_points)

    def indices_to_index(self, ix: int, iy: int, iz: int) -> int:
        nx, ny, nz = self.n_points
        return iz + (iy * nz) + (ix * ny * nz)

    def indices_to_pos(self, ix: int, iy: int, iz: int):
        return np.array([ix, iy, iz]) * self.spacing - self.origin

    def index_to_pos(self, ir: int):
        # Unroll ir to (ix, iy, iz)
        # Then call self.indices_to_pos(ix, iy, iz)
        raise NotImplementedError("See my fortran implementation and copy to 0-indexing")

    def all_points(self):
        """Return all grid points"""

        # Create coordinate arrays using broadcasting
        # TODO Check the inclusion of the end point
        # Values are generated within the half-open interval, [0, stop)
        # so end-point not included. Consistent with definition above (?)
        x = np.arange(self.n_points[0])
        y = np.arange(self.n_points[1])
        z = np.arange(self.n_points[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Reshape coordinate arrays into 1D arrays
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)

        grid = np.column_stack((X, Y, Z)) * self.spacing - self.origin

        return grid
