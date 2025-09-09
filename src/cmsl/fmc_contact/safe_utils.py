"""Ccontact simulation based on Fiber Monte Carlo.

Based on Richardson et al.'s work in "Fiber Monte Carlo":
- https://github.com/PrincetonLIPS/fibermc
- https://openreview.net/forum?id=sP1tCl2QBk

Adapted for JAX-FEM  and contact mechanics simulations.

"""

import jax.numpy as np

# @functools.partial(jax.jit, static_argnums=(1,))
def custom_norm(x: np.ndarray, dtype: type = np.float32) -> np.ndarray:

    is_zero: np.ndarray = np.allclose(x, 0.0)
    x: np.ndarray = np.where(is_zero, np.ones_like(x), x)
    return np.where(is_zero, 0.0, np.linalg.norm(x))


def divide00(
    numerator: np.ndarray, denominator: np.ndarray, dtype: type = np.float32
) -> np.ndarray:

    force_zero: np.ndarray = np.logical_and(numerator == 0, denominator == 0)
    quotient: np.ndarray = np.where(force_zero, dtype(0.0), numerator) / np.where(
        force_zero, dtype(1.0), denominator
    )
    return quotient


def zero_one_sign(arr: np.ndarray) -> np.ndarray:

    binary_arr: np.ndarray = 0.5 * (1.0 + np.sign(arr))
    return binary_arr