import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True)
class Interpolator(abc.ABC):
    nodes: npt.NDArray
    values: npt.NDArray

    def __post_init__(self):
        self.fit()

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        pass


class StandardLagrange(Interpolator):
    def fit(self):
        self.nodes = np.asarray(self.nodes).ravel()
        self.values = np.asarray(self.values).ravel()
        assert self.nodes.ndim == 1 and self.values.ndim == 1, (
            "nodes and values must be one-dimensional arrays"
        )
        assert self.nodes.shape[0] == self.values.shape[0], (
            "nodes and values must have the same length"
        )

    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        # Standard Lagrange interpolation.
        # ====================================================================
        x_arr = np.asarray(x)
        x_flat = x_arr.ravel()
        n_nodes = self.nodes.shape[0]

        num_diffs = x_flat[:, None, None] - self.nodes[None, None, :]
        num_diffs = np.tile(num_diffs, (1, n_nodes, 1))
        denom_diffs = self.nodes[None, :, None] - self.nodes[None, None, :]
        denom_diffs = np.tile(denom_diffs, (x_flat.shape[0], 1, 1))

        diag_i, diag_j = np.diag_indices(n_nodes)
        num_diffs[:, diag_i, diag_j] = 1
        denom_diffs[:, diag_i, diag_j] = 1

        lagrange_poly = np.prod(num_diffs / denom_diffs, axis=2)
        output_flat = np.sum(lagrange_poly * self.values, axis=1)
        # ====================================================================
        return output_flat.reshape(x_arr.shape)


class FirstBarycentricLagrange(Interpolator):
    weights: npt.NDArray

    def fit(self):
        self.nodes = np.asarray(self.nodes).ravel()
        self.values = np.asarray(self.values).ravel()
        assert self.nodes.ndim == 1 and self.values.ndim == 1, (
            "nodes and values must be one-dimensional arrays"
        )
        assert self.nodes.shape[0] == self.values.shape[0], (
            "nodes and values must have the same length"
        )

        diffs = self.nodes[:, None] - self.nodes[None, :]
        diffs[np.diag_indices_from(diffs)] = 1
        self.weights = 1 / np.prod(diffs, axis=1)

    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        # Barycentric Lagrange interpolation.
        # ====================================================================
        x_arr = np.asarray(x)
        x_flat = x_arr.ravel()
        nodes_mask, nodes_idxs = self._get_filter_mask(x_flat)
        output = np.zeros_like(x_flat, dtype=self.values.dtype)

        if np.any(nodes_mask):
            output[nodes_mask] = self.values[nodes_idxs[nodes_mask]]

        if not np.all(nodes_mask):
            x_clean = x_flat[~nodes_mask]
            diffs = x_clean[:, None] - self.nodes[None, :]
            L = np.prod(diffs, axis=1)
            sum_term = self.values[None, :] * self.weights[None, :] / diffs
            output[~nodes_mask] = L * np.sum(sum_term, axis=1)
        # ====================================================================

        return output.reshape(x_arr.shape)

    def _get_filter_mask(
        self, x: npt.NDArray, eps: float = 1e-10
    ) -> tuple[npt.NDArray, npt.NDArray]:
        # Get a mask of the evaluation points that are close to the nodes.
        diffs = np.abs(x[:, None] - self.nodes[None, :]) < eps
        mask = np.any(diffs, axis=1)
        idxs = np.argmax(diffs, axis=1)
        return mask, idxs
