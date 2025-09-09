"""
This module contains procedures for defining sampling regions, performing random sampling, 
and utilizing fiber Monte Carlo methods to computate contact area and its derivatives for elastic-elastic bodies contact."""

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)
import time
from .estimator import compute_box_intersection_length, create_contact_field2, estimate_field_length
import functools

def get_elastic_body_boundary(updated_elements1, updated_elements2):
    """get the boundary of the elastic body"""
    contact_cell_index1 = 29 #181    #34
    contact_cell_index2 = 69 #181    #34
    contact_cell_index3 = 99 #181    #34

    contact_cell1 = updated_elements2[contact_cell_index1]
    contact_cell2 = updated_elements1[contact_cell_index2]
    contact_cell3 = updated_elements1[contact_cell_index3]
    y_coords2 = contact_cell1[:, 1]
    x_coords1 = contact_cell2[:, 0]
    y_coords3 = contact_cell3[:, 1]
    x_max1 = np.max(x_coords1)
    y_max = np.max(y_coords2) 
    y_min = np.min(y_coords3) 
    return y_max, y_min, x_max1

def define_sampling_box(updated_elements1, updated_elements2):
    y_min1, y_min2, x_max1 = get_elastic_body_boundary(updated_elements1, updated_elements2)

    y_padding1 = y_min1 + 0.002
    y_padding2 = y_min2 - 0.003
    x_padding =  x_max1 + 0.005 

    box_x_min = -x_padding
    box_x_max =  x_padding
    box_y_min = np.minimum(y_padding1, y_padding2)  
    box_y_max = np.maximum(y_padding1, y_padding2)  
    return box_x_min, box_x_max, box_y_min, box_y_max



def fibers_generator(updated_elements1, updated_elements2, n_fibers: int, fiber_length: float, key=None):

    key = jax.random.PRNGKey(time.time_ns())
    fibers = generate_fibers(updated_elements1, updated_elements2, n_fibers, fiber_length, key)
    return fibers

@functools.partial(jax.jit, static_argnums=(2, 3))
def generate_fibers(updated_elements1, updated_elements2, n_fibers: int, fiber_length: float, key):

    box_x_min, box_x_max, box_y_min, box_y_max = define_sampling_box(updated_elements1, updated_elements2)

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

def fiber_monte_carlo(problem1, problem2, ContactProblem, sol1, sol2, fibers):
    fiber = fibers['fibers']
    box_params = fibers['box_params']
    fiber_length = fibers['fiber_length']

    area_box = float((box_params[1] - box_params[0]) * (box_params[3] - box_params[2]))
    total_box_length = compute_box_intersection_length(fiber, box_params)
    intersection_field = create_contact_field2(problem1, problem2, sol1, sol2)
    def objective(params: np.ndarray, fiber: np.ndarray) -> np.ndarray: 
        interection_length = estimate_field_length(intersection_field, fiber, params, fiber_length)
        iou: np.ndarray = ((interection_length / total_box_length)*area_box).squeeze()
        return iou
    gradient_fn: callable = jax.value_and_grad(objective)
    contact_area, gradient= gradient_fn((sol1, sol2), fiber)
    gradient1, gradient2 = gradient
    contact_info = {
        'contact_area': float(contact_area), 
        'gradient1': gradient1,  
        'gradient2': gradient2, 
        'fiber': fibers['fibers'],
        'box_params': fibers['box_params']
    }

    if contact_info['contact_area'] < ContactProblem.contact_params['area_threshold']:
        return {
            'contact_area': 0.0,
            'gradient1': np.zeros_like(gradient1),  
            'gradient2': np.zeros_like(gradient2), 
            'has_contact': False,
            'fiber': fibers['fibers'],
            'box_params': fibers['box_params']
            }
    

    contact_info['has_contact'] = True
    return contact_info