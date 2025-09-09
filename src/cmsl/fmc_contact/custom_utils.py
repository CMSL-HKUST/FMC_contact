"""This module contains procedures for post-processing and visualization"""

import os
import jax.numpy as np
import numpy as onp
from jax_fem.utils import save_sol
import meshio

def save_sol1(problem, sol, vtk_path, contact_problem=None):
    """
    save the displacement field, force field, and the rigid body boundary in the VTU file
    """
    # call the stress calculation function
    stress_tensors = problem.compute_stress(sol)
    JxW = problem.fes[0].JxW  # (num_cells, num_quads)
    cell_volumes = np.sum(JxW, axis=1)  # the volume of each cell
    sigma_avg = np.sum(stress_tensors * JxW[:,:,None,None], axis=1) / cell_volumes[:,None,None]
    stress_xx = sigma_avg[:, 0, 0]
    stress_yy = sigma_avg[:, 1, 1] 
    stress_xy = sigma_avg[:, 0, 1]

    s_dev = sigma_avg - 1/3 * np.trace(sigma_avg, axis1=1, axis2=2)[:,None,None] * np.eye(2)[None,:,:]
    von_mises = np.sqrt(3/2 * np.sum(s_dev * s_dev, axis=(1,2)))

    # prepare the VTK data
    F_contact = problem.F_contact
    force_vector = np.hstack((F_contact, np.zeros((len(F_contact), 1))))
    sol_3d = onp.array(sol.reshape(-1, problem.fes[0].vec))
    sol_3d = np.hstack((sol_3d, np.zeros((sol_3d.shape[0], 1))))
    # 6. save the VTK
    save_sol(problem.fes[0], sol_3d, vtk_path,
             cell_infos=[('sigma_xx', onp.array(stress_xx, dtype=onp.float32)),
                        ('sigma_yy', onp.array(stress_yy, dtype=onp.float32)),
                        ('sigma_xy', onp.array(stress_xy, dtype=onp.float32)),
                        ('von_mises', onp.array(von_mises, dtype=onp.float32))],
             point_infos=[('contact_force', onp.array(force_vector))])
    
    # 7. save the sampling box
    if contact_problem is not None:

        contact_info = contact_problem.get_contact_info()
        if contact_info is not None and contact_info.get('has_contact', False):
            box_vtk_path = vtk_path.replace('.vtu', '_sampling_box.vtu')
            box_points, box_lines = generate_box_lines(contact_info['box_params'])
            mesh_box = meshio.Mesh(
                points=box_points,
                cells={'line': box_lines}
            ).write(box_vtk_path)

    # 8. save the rigid body boundary
    rigid_points = problem.rigid_body_params 
    # create the line of the rigid body
    rigid_cells = np.array([[i, (i+1)%len(rigid_points)] 
                           for i in range(len(rigid_points))], dtype=np.int32)
    # add the z coordinate
    rigid_points_3d = np.hstack((rigid_points, np.zeros((len(rigid_points), 1))))
    # save the rigid body VTU
    rigid_vtk_path = vtk_path.replace('.vtu', '_rigid_body.vtu')
    mesh = meshio.Mesh(
        points=rigid_points_3d,
        cells={'line': rigid_cells}
    )
    mesh.write(rigid_vtk_path)


def save_sol2(problem, sol, vtk_path, contact_problem=None):

    stress_tensors = problem.compute_stress(sol)
    JxW = problem.fes[0].JxW  # (num_cells, num_quads)
    cell_volumes = np.sum(JxW, axis=1)  # the volume of each cell
    sigma_avg = np.sum(stress_tensors * JxW[:,:,None,None], axis=1) / cell_volumes[:,None,None]
    sigma_avg = sigma_avg[:, :2, :2]
    stress_xx = sigma_avg[:, 0, 0]
    stress_yy = sigma_avg[:, 1, 1] 
    stress_xy = sigma_avg[:, 0, 1]
    s_dev = sigma_avg - 1/3 * np.trace(sigma_avg, axis1=1, axis2=2)[:,None,None] * np.eye(2)[None,:,:]
    von_mises = np.sqrt(3/2 * np.sum(s_dev * s_dev, axis=(1,2)))

    F_contact = problem.F_contact
    force_vector = np.hstack((F_contact, np.zeros((len(F_contact), 1))))
    sol_3d = onp.array(sol.reshape(-1, problem.fes[0].vec))
    sol_3d = np.hstack((sol_3d, np.zeros((sol_3d.shape[0], 1))))
    
    save_sol(problem.fes[0], sol_3d, vtk_path,
             cell_infos=[('sigma_xx', onp.array(stress_xx, dtype=onp.float32)),
                        ('sigma_yy', onp.array(stress_yy, dtype=onp.float32)),
                        ('sigma_xy', onp.array(stress_xy, dtype=onp.float32)),
                        ('von_mises', onp.array(von_mises, dtype=onp.float32))],
             point_infos=[('contact_force', onp.array(force_vector))])
    

    if contact_problem is not None:

        contact_info = contact_problem.get_contact_info()
        if contact_info is not None and contact_info.get('has_contact', False):
            box_vtk_path = vtk_path.replace('.vtu', '_sampling_box.vtu')
            box_points, box_lines = generate_box_lines(contact_info['box_params'])
            mesh_box = meshio.Mesh(
                points=box_points,
                cells={'line': box_lines}
            ).write(box_vtk_path)


def generate_box_lines(box_params):
    """generate the line of the sampling box"""
    x_min, x_max, y_min, y_max = box_params
    points = onp.array([
        [x_min, y_min],  # 0 - left bottom
        [x_max, y_min],  # 1 - right bottom
        [x_max, y_max],  # 2 - right top
        [x_min, y_max],  # 3 - left top
    ])
    
    # define the 4 edges
    lines = onp.array([
        [0, 1], [1, 2], [2, 3], [3, 0]
    ])
    
    return points, lines


