"""Ccontact simulation based on Fiber Monte Carlo.

Based on Richardson et al.'s work in "Fiber Monte Carlo":
- https://github.com/PrincetonLIPS/fibermc
- https://openreview.net/forum?id=sP1tCl2QBk

Adapted for JAX-FEM  and contact mechanics simulations.

"""
"""
This module contains procedures for defining the specific method to compute the total fiber integrals within the contact region, 
including determining whether the start and end points of fibers lie inside or outside the contact area, 
as well as calculating intersection points.
"""

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)
import functools
import chex
from .jax_utils import vectorize, jit_vectorize
from functools import partial


def _get_hull_orientation(hull: np.ndarray) -> np.ndarray:
    """Returns -1 if the provided hull is oriented clockwise and
    +1 otherwise.
    """
    chex.assert_shape(hull, (..., None, 2))
    return np.sign(np.cross(hull[1] - hull[0], hull[2] - hull[0]))


def _reduce_clip_params(clips: np.ndarray, axis: int = -2) -> np.ndarray:
    if axis == -1 or axis == clips.ndim - 1:
        raise ValueError(f"Do not reduce_clip_params over coordinate axis.")
    clips: np.ndarray = np.moveaxis(clips, axis, 0)
    lo, hi = np.moveaxis(clips, -1, 0)
    clip_params: np.ndarray = np.stack([lo.max(0), hi.min(0)], axis=-1)
    return clip_params


def _hull_to_walls(hull: np.ndarray) -> np.ndarray:
    """Computes the 'walls' (line-segments) associated with the given `hull`
    containing the vertices of the hull.
    """
    chex.assert_shape(hull, (..., None, 2))
    walls: np.ndarray = np.stack([hull, np.roll(hull, -1, axis=-2)], axis=-2)
    return walls


def _in01(x: np.ndarray, open: bool = True) -> np.ndarray:
    """Returns a boolean-valued np.ndarray representing whether elements
    of `x` lie in the interval (0, 1] or (0, 1); the latter if `open`=True.
    """
    return np.logical_and(0 < x, x < 1) if open else np.logical_and(0 <= x, x <= 1)


@partial(np.vectorize, signature="(n,2),()->(n,2)")
def _orient_hull(hull: np.ndarray, target_orientation: int):
    """Given a hull and a target orientation, ensures the hull
    is organized with that orientation; if it doesn't it reverses
    the hull's orientation; the correctly oriented hull is returned
    """
    chex.assert_shape(hull, (None, 2))

    # determine whether the hull ought to be reversed
    should_reverse = target_orientation != _get_hull_orientation(hull)
    reverse: callable = lambda hull: hull[::-1]
    dont_reverse: callable = lambda hull: hull
    oriented_hull: np.ndarray = jax.lax.cond(
        should_reverse, reverse, dont_reverse, operand=hull
    )
    return oriented_hull


@jit_vectorize(signature="(2,2),(2)->(2,2)")
def apply_fiber_clip(fiber: np.ndarray, clip_params: np.ndarray) -> np.ndarray:
    """Clips the provided fibers according to the clip parameters."""
    clipped_fibers: np.ndarray = interpolate(clip_params, fiber)
    return clipped_fibers


def orient_hull_ccw(hull: np.ndarray) -> np.ndarray:
    """Returns the same hull as provided but guaranteed to be
    oriented counter-clockwise.
    """
    oriented_hull: np.ndarray = _orient_hull(hull, +1)
    return oriented_hull


def orient_hull_cw(hull: np.ndarray) -> np.ndarray:
    """Returns the same hull as provided but guaranteed to be
    oriented clockwise.
    """
    oriented_hull: np.ndarray = _orient_hull(hull, -1)
    return oriented_hull


@vectorize(signature="(2,2),(2,2)->(2)")
def intersect_segments(
    first_segment: np.ndarray, second_segment: np.ndarray
) -> np.ndarray:
    """Computes the intersection between two line segments.

    Note: returns np.array([np.inf, np.inf]) if there is no intersection.
    """
    # setup the linear system
    A: np.ndarray = np.stack(
        [first_segment[1] - first_segment[0], second_segment[0] - second_segment[1]],
        axis=-1,
    )
    b: np.ndarray = second_segment[0] - first_segment[0]

    # determine if the coefficient matrix is singular
    abs_determinant: np.ndarray = np.abs(np.linalg.det(A))

    # specify both sides of the branch
    linear_solve: callable = partial(np.linalg.solve, A)
    is_no_solution: callable = lambda _: np.array([np.inf, np.inf])

    # compute the intersection
    is_solvable: bool = abs_determinant > 0
    intersection: np.ndarray = jax.lax.cond(
        is_solvable, linear_solve, is_no_solution, operand=b
    )
    return intersection


@jit_vectorize(signature="(2,2),(2,2)->(2),(2),()")
def clip_wrt_wall(fiber: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Clips a fiber with respect to an oriented line segment `wall`.
    """
    # closure taking a query point and returning which side of the wall it falls on
    which_side: callable = lambda query_point: np.sign(
        np.cross(wall[1] - wall[0], query_point - wall[0])
    )

    # which side of the wall each endpoint of the fiber falls on
    endpoint_sides: np.ndarray = np.stack([which_side(f) for f in fiber], axis=0)
    start_sides, end_sides = endpoint_sides

    # find the intersection (if it exists)
    intersection: np.ndarray = intersect_segments(fiber, wall)
    has_intersection: np.ndarray = _in01(intersection, open=True).all(-1)
    alpha, _ = np.moveaxis(intersection, -1, 0)

    intersection_select: np.ndarray = jax.lax.select(
        start_sides > end_sides,
        np.stack([0.0, alpha]),
        np.stack([alpha, 1.0]),
    )

    # determine the clip parameters
    clip_parameters: np.ndarray = jax.lax.select(
        has_intersection, intersection_select, np.array([0.0, 1.0])
    )
    return clip_parameters, endpoint_sides, has_intersection


@jit_vectorize(signature="(2,2),(n,2)->(2)")
def _clip_inside_convex_hull(fiber: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Clip fiber(s) to inside of a convex hull.
    """
    # ensure we have a valid hull with greater than 2 vertices
    num_vertices, _ = hull.shape
    assert (
        num_vertices > 2
    ), f"Hull must contain more than 2 vertices (got hull with {num_vertices} vertices)."

    # orient the hull and obtain its walls
    hull: np.ndarray = orient_hull_ccw(hull)
    walls = _hull_to_walls(hull)

    # clip the fibers with respect to the walls of the hull
    clips, endpoint_sides, has_intersections = clip_wrt_wall(fiber, walls)
    clip: np.ndarray = _reduce_clip_params(clips)

    # determine if there are any intersections
    any_intersections: np.ndarray = has_intersections.any(-1)

    is_fully_inside: np.ndarray = (endpoint_sides > 0).all((-1, -2))
    is_fully_outside: np.ndarray = np.logical_and(~is_fully_inside, ~any_intersections)

    # compute the clipping parameters
    clip_parameters: np.ndarray = jax.lax.select(
        is_fully_outside, np.zeros_like(clip), clip
    )
    return clip_parameters


def clip_inside_convex_hull(fibers: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Wraps src.utils.geometry_utils._clip_inside_convex_hull and
    applies the clipping parameters to the fibers.
    """
    clip_parameters = _clip_inside_convex_hull(fibers, hull)
    clipped_fibers: np.ndarray = apply_fiber_clip(fibers, clip_parameters)
    return clipped_fibers


@jit_vectorize(signature="(),(2,2)->(2)")
def interpolate(alpha: float, segment: np.ndarray) -> np.ndarray:
    """Interpolates `alpha` amount along `segment`."""
    x0, x1 = np.moveaxis(segment, -2, 0)
    interpolated: np.ndarray = (1 - alpha) * x0 + alpha * x1
    return interpolated


@jit_vectorize(signature="(2,2),(k,2)->(k,2,2)")
def apply_fiber_multiclip(fiber: np.ndarray, clip_params: np.ndarray) -> np.ndarray:
    return interpolate(clip_params, fiber)

