"""Ccontact simulation based on Fiber Monte Carlo.

Based on Richardson et al.'s work in "Fiber Monte Carlo":
- https://github.com/PrincetonLIPS/fibermc
- https://openreview.net/forum?id=sP1tCl2QBk

Adapted for JAX-FEM  and contact mechanics simulations.

"""


"""This module contains procedures for implicit differentiation setups in
conjunction with fiber sampling applications.
"""
import functools
from typing import TypeVar
import jax
import jax.numpy as np
from .safe_utils import divide00
from jaxopt.implicit_diff import custom_root
pytree: type = TypeVar("Pytree")

@functools.partial(jax.jit, static_argnums=(0, 2))
def bisect(f: callable, fiber: np.ndarray, num_iterations: int = 25) -> float:
    interpolant: callable = lambda x: fiber[0] + x * (fiber[1] - fiber[0])
    h: callable = lambda x: f(interpolant(x))
    # standardize so the 'left' endpoint has negative value
    endpoints: np.ndarray = jax.lax.cond(
        h(0.0) > 0.0,
        lambda _: np.array([1.0, 0]),
        lambda _: np.array([0.0, 1.0]),
        operand=None,
    )

    def body_fun(_, endpoints):
        left, right = endpoints
        midpoint: float = (left + right) / 2.0
        return jax.lax.cond(
            h(midpoint) < 0.0,
            lambda _: np.array([midpoint, right]),
            lambda _: np.array([left, midpoint]),
            operand=None,
        )

    endpoints = jax.lax.fori_loop(0, num_iterations, body_fun, endpoints)

    return endpoints[0]


def get_interpolant(alpha: np.ndarray, fiber: np.ndarray) -> np.ndarray:
    return fiber[0] + alpha * (fiber[1] - fiber[0])


def bind_optimality_condition(f: callable) -> np.ndarray:
    def optimality_condition(
        x: np.ndarray, params: dict, fiber: np.ndarray
    ) -> np.ndarray:
        z: np.ndarray = get_interpolant(x, fiber)
        constraint_value: np.ndarray = f(params, z)
        return constraint_value

    return optimality_condition


def bind_solver(f: callable) -> callable:
    @custom_root(bind_optimality_condition(f))
    def bisection_solver(
        x_init: np.ndarray, params: dict, fiber: np.ndarray
    ) -> np.ndarray:
        x: np.ndarray = bisect(functools.partial(f, params), fiber)
        return x

    return bisection_solver


def bisection_constraint(
    f: callable, x: np.ndarray, params: tuple, fiber: np.ndarray
) -> np.ndarray:

    z: np.ndarray = get_interpolant(x, fiber)
    field_constraint: float = np.squeeze(f(params, z))
    return field_constraint


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def bisection_solver(params, fiber: np.ndarray, f: callable) -> np.ndarray:
    fixed_point: np.ndarray = bisect(functools.partial(f, params), fiber)
    return fixed_point

