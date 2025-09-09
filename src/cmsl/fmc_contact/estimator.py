"""Ccontact simulation based on Fiber Monte Carlo.

Based on Richardson et al.'s work in "Fiber Monte Carlo":
- https://github.com/PrincetonLIPS/fibermc
- https://openreview.net/forum?id=sP1tCl2QBk

Adapted for JAX-FEM  and contact mechanics simulations.

"""

"""
This module contains procedures for defining the scalar field functions that represent contact boundaries 
and for computing fiber integrals located within the contact region.
"""

import jax
import jax.numpy as np
import functools
from .safe_utils import custom_norm, zero_one_sign
from .differentiation_solver import get_interpolant, bind_solver
from .geometry_utils import clip_inside_convex_hull
from .sdf import compute_contact_boundary_sdf1, compute_contact_boundary_sdf2

# create the hull sampling region
@jax.jit
def create_box_hull(box_params):
    """create the convex hull of the box"""
    x_min, x_max, y_min, y_max = box_params
    return np.array([
        [x_min, y_min],  # left bottom
        [x_max, y_min],  # right bottom
        [x_max, y_max],  # right top
        [x_min, y_max],  # left top
    ])

##create the contact field(SDF)
def create_contact_field1(problem, rigid_body_params, sol):
    """create the contact field(SDF) for elastic-rigid contact"""
    updated_elements = problem.mesh_functions['get_updated_elements'](sol)
    contact_elements = updated_elements[problem.contact_elements]
    get_coords = problem.mesh_functions['get_node_coords']
    @jax.jit
    def field(sol, point):  
        starts, ends = get_coords(sol, problem.boundary_edges)
        return compute_contact_boundary_sdf1(point, starts, ends, contact_elements, rigid_body_params)
    return field

def create_contact_field2(problem1, problem2, sol1, sol2):
    """create the contact field(SDF) for elastic-elastic contact"""
    updated_elements1 = problem1.mesh_functions['get_updated_elements'](sol1)
    updated_elements2 = problem2.mesh_functions['get_updated_elements'](sol2)
    contact_elements1 = updated_elements1[problem1.contact_elements]
    contact_elements2 = updated_elements2[problem2.contact_elements]
    get_coords1 = problem1.mesh_functions['get_node_coords']
    get_coords2 = problem2.mesh_functions['get_node_coords']
    @jax.jit
    def field(params, point):  
        sol1, sol2 = params
        starts1, ends1 = get_coords1(sol1, problem1.boundary_edges)
        starts2, ends2 = get_coords2(sol2, problem2.boundary_edges)
        return compute_contact_boundary_sdf2(point, starts1, ends1, contact_elements1, 
        starts2, ends2, contact_elements2)
    return field


def compute_box_intersection_length(fibers, box_params):

    hull = create_box_hull(box_params)
    clipped_fibers = clip_inside_convex_hull(fibers, hull)
    
    box_total_length = jax.vmap(custom_norm)(clipped_fibers[:, 1] - clipped_fibers[:, 0]).sum()
    
    return box_total_length

def estimate_field_length(
    field: callable,
    fibers: np.ndarray,
    params: tuple,
    fiber_length: float,
) -> np.ndarray:

    vector_field: callable = functools.partial(
        jax.vmap(field, in_axes=(None, 0)), params
    )
    solver_base: callable = bind_solver(field)
    solver: callable = jax.jit(
            jax.vmap(
                lambda fiber, params: get_interpolant(
                    solver_base(np.empty(0), params, fiber), fiber
                ),
                in_axes=(0, None),
            )
        )
    start_points, end_points = fibers[:, 0], fibers[:, 1]
    start_values, end_values = (
        vector_field(start_points).ravel(),
        vector_field(end_points).ravel(),
    )
    start_signs, end_signs = (
        zero_one_sign(start_values),
        zero_one_sign(end_values),
    )

    negative: float = 0.0
    positive: float = 1.0

    count_entire_fiber_cond: np.ndarray = np.logical_and(
        start_signs == negative, end_signs == negative
    )
    count_from_start_cond: np.ndarray = np.logical_and(
        start_signs == negative, end_signs == positive
    )
    count_from_end_cond: np.ndarray = np.logical_and(
        start_signs == positive, end_signs == negative
    )

    # count the entire fiber length
    total_length = np.zeros(1)
    total_length += np.sum(count_entire_fiber_cond) * fiber_length

    count_from_start = np.where(
        count_from_start_cond.reshape(-1, 1, 1),
        fibers,
        np.zeros_like(fibers)
    )
    start_intersections = solver(count_from_start, params)
    total_length += jax.vmap(custom_norm)(
        start_intersections - count_from_start[:, 0]
    ).sum()

    count_from_end = np.where(
        count_from_end_cond.reshape(-1, 1, 1),
        fibers,
        np.zeros_like(fibers)
    )
    end_intersections = solver(count_from_end, params)
    total_length += jax.vmap(custom_norm)(
        count_from_end[:, 1] - end_intersections
    ).sum()

    return total_length
