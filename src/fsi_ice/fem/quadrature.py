"""Positive-weight quadrature on {(xi, eta): xi, eta >= 0, xi+eta <= 1}."""

import numpy as np


def triangle_quadrature(degree=5):
    """Return points and weights whose sum is the reference area 1/2."""
    if degree == 1:
        return np.array([[1/3, 1/3]]), np.array([0.5])
    if degree == 2:
        return np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]), np.full(3, 1/6)
    if degree == 5:
        a, b = 0.059715871789770, 0.470142064105115
        c, d = 0.797426985353087, 0.101286507323456
        points = np.array([[1/3, 1/3], [b, b], [a, b], [b, a],
                           [d, d], [c, d], [d, c]])
        weights = 0.5 * np.array([0.225, *([0.132394152788506] * 3),
                                 *([0.125939180544827] * 3)])
        return points, weights
    raise ValueError("Supported triangle quadrature degrees are 1, 2 and 5.")
