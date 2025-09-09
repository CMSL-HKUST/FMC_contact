"""
This module contains procedures for predefining potential contact elements for specific cases, 
where the selected range is designed to be larger than the actual contact area. This process can avoid contact detection.
"""
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)


### Predefine the Potential Contact Area for Elastic and Rigid Body Contact
def find_contact_elements(elements):

    elements_in_contact = jax.vmap(lambda element: 
        jax.vmap(lambda point: 
            is_point_in_contact_zone(point)
        )(element).any()
    )(elements)
    elements_in_contact = np.where(elements_in_contact)[0]
    return elements_in_contact

def is_point_in_contact_zone(point):
        return (
            (point[0] >= -0.0) & (point[0] <= 0.05) &  # x range
            (point[1] >  -0.08) & (point[1] < 0.08)  # y range
        )

### Predfine the Potential Contact Area for Elastic and Elastic Body Contact
SCALE_FACTOR = 0.1  
def find_contact_elements_body1(elements):

    elements_in_contact = jax.vmap(lambda element: 
        jax.vmap(lambda point: 
            is_point_in_contact_zone_body1(point)
        )(element).any()
    )(elements)
    elements_in_contact = np.where(elements_in_contact)[0]
    return elements_in_contact

def find_contact_elements_body2(elements):

    elements_in_contact = jax.vmap(lambda element: 
        jax.vmap(lambda point: 
            is_point_in_contact_zone_body2(point)
        )(element).any()
    )(elements)
    elements_in_contact = np.where(elements_in_contact)[0]
    return elements_in_contact


def is_point_in_contact_zone_body1(point, scale_factor=SCALE_FACTOR):
    """potential contact area for Body1"""
    return (
        (point[0] >= -0.5 * scale_factor) & (point[0] <= 0.5 * scale_factor) &  # x range
        (point[1] > -1.0 * scale_factor) & (point[1] <= -0.59 * scale_factor)  # y range
    )

def is_point_in_contact_zone_body2(point, scale_factor=SCALE_FACTOR):
    """potential contact area for Body2"""
    return (
        (point[0] >= -0.3 * scale_factor) & (point[0] <= 0.3 * scale_factor) &  # x range
        (point[1] > -1.3 * scale_factor) & (point[1] <= -1.1 * scale_factor)  # y range 
    )

def compute_contact_parameters(contact_info, problem):
    """get the parameters for the energy model"""
    contact_area = contact_info['contact_area']
    # get the stiffness parameter and power parameter
    K = problem.contact_params['K']
    n = problem.contact_params['n']
    # compute the dPsi_dV
    dPsi_dV = K * n * (contact_area ** (n-1))
    return {
        'dPsi_dV': dPsi_dV
    }

def compute_contact_force(contact_params, gradient, problem):
    """
    compute the contact force 
    """
    # compute the derivative
    dPsi_dV = contact_params['dPsi_dV']
    dV_dd = gradient
    #compute the contact force
    F_contact = -(dPsi_dV * dV_dd)
    return F_contact.reshape(problem.fes[0].num_total_nodes, problem.fes[0].vec)