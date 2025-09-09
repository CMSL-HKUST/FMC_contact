
"""
This module contains procedures for defining sampling regions, performing random sampling, 
and utilizing fiber Monte Carlo methods to computate contact area and its derivatives for elastic-rigid bodies contact."""

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)
import time
import functools
from .estimator import compute_box_intersection_length, create_contact_field1, estimate_field_length


def get_elastic_body_boundary(updated_elements):
    """get the boundary of the elastic body"""
    contact_cell_index = 217
    contact_cell = updated_elements[contact_cell_index]
    y_coords = contact_cell[:, 1]
    x_coords = contact_cell[:, 0]
    y_max = np.max(y_coords)
    x_min = np.min(x_coords)
    return y_max, x_min

def define_sampling_box(rigid_body_params, updated_elements):
    """define the sampling box based on the known contact cell"""

    boundary_points = rigid_body_params
    rigid_x_max = np.max(boundary_points[:, 0])
    contact_y_max, contact_x_min = get_elastic_body_boundary(updated_elements)
    padding1 = 0.005
    padding2 = 0.015
    box_x_min = contact_x_min-padding1
    box_x_max = rigid_x_max
    box_y_min = -contact_y_max + padding2
    box_y_max = contact_y_max - padding2
    return box_x_min, box_x_max, box_y_min, box_y_max

def fibers_generator(rigid_body_params, updated_elements, n_fibers: int, fiber_length: float, key):
    """generate the fiber"""
    key = jax.random.PRNGKey(time.time_ns())
    fibers = generate_fibers(rigid_body_params, updated_elements, n_fibers, fiber_length, key)
    return fibers

@functools.partial(jax.jit, static_argnums=(2, 3))
def generate_fibers(rigid_body_params, updated_elements, n_fibers: int, fiber_length: float, key):
    """define the sampling box"""
    box_x_min, box_x_max, box_y_min, box_y_max = define_sampling_box(rigid_body_params, updated_elements)
    
    # enlarge the sampling box
    large_box_x_min = box_x_min - 2*fiber_length
    large_box_x_max = box_x_max + 2*fiber_length
    large_box_y_min = box_y_min - 2*fiber_length
    large_box_y_max = box_y_max + 2*fiber_length
    
    location_key, angular_key = jax.random.split(key)
    x_key, y_key = jax.random.split(location_key)
    
    starts = np.array((
        jax.random.uniform(
            x_key,
            shape=(n_fibers,),
            minval=large_box_x_min,
            maxval=large_box_x_max
        ),
        jax.random.uniform(
            y_key,
            shape=(n_fibers,),
            minval=large_box_y_min,
            maxval=large_box_y_max
        )
    )).T
    

    angles = jax.random.uniform(
        angular_key,
        shape=(n_fibers,),
        minval=-np.pi,
        maxval=np.pi
    )

    ends = starts + fiber_length * np.array([np.cos(angles), np.sin(angles)]).T
    

    fibers = np.stack([starts, ends], axis=1)  # (n_fibers, 2, 2)

    return {
            'fibers': fibers,
            'box_params': (box_x_min, box_x_max, box_y_min, box_y_max),
            'fiber_length': fiber_length
    }


def fiber_monte_carlo(problem, rigid_body_params, contact_problem, sol, fibers):
    """compute the contact information, using FiberMC, compute the contact area and the intersecting fiber information, and the derivative"""
    # get the fiber information
    fiber = fibers['fibers']
    box_params = fibers['box_params']
    fiber_length = fibers['fiber_length']
    # compute the area of the sampling box
    area_box = float((box_params[1] - box_params[0]) * (box_params[3] - box_params[2]))
    total_box_length = compute_box_intersection_length(fiber, box_params)
    intersection_field = create_contact_field1(problem, rigid_body_params, sol)
    
    def objective(sol: np.ndarray, fiber: np.ndarray) -> np.ndarray: 
        """
        use the implicit function theorem to compute the gradient:
        ∂(Area)/∂u = ∂(Area)/∂L × ∂L/∂α × ∂α/∂u
        """
        interection_length = estimate_field_length(intersection_field, fiber, sol, fiber_length)
        iou: np.ndarray = ((interection_length / total_box_length)*area_box).squeeze()
        return iou

    gradient_fn: callable = jax.value_and_grad(objective)
    contact_area, gradient= gradient_fn(sol, fiber)
    contact_info = {
        'contact_area': float(contact_area),  
        'gradient': gradient, 
        'fiber': fibers['fibers'],
        'box_params': fibers['box_params']
    }
    # if there is no contact
    if contact_info['contact_area'] < contact_problem.contact_params['area_threshold']:
        return {
            'contact_area': 0.0,
            'gradient': np.zeros_like(gradient),  
            'has_contact': False,
            'fiber': fibers['fibers'],
            'box_params': fibers['box_params']
            }
    
    contact_info['has_contact'] = True
    return contact_info




























