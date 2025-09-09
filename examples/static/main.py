import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)
# Some useful package
import os
import sys
import meshio
import time
# JAX-FEM packages
from jax_fem.problem import Problem
from jax_fem import logger
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh

from src.cmsl.fmc_contact.fmc1 import fiber_monte_carlo, fibers_generator
from src.cmsl.fmc_contact.contact_utils import find_contact_elements, compute_contact_parameters, compute_contact_force
from src.cmsl.fmc_contact.sdf import create_mesh_functions
from src.cmsl.fmc_contact.static_solver import contact_solver
from src.cmsl.fmc_contact.custom_utils import save_sol1

class Elasticity1(Problem):

    def custom_init(self, rigid_body_params):

        self.rigid_body_params = rigid_body_params
        self.F_contact = np.zeros((self.fes[0].num_total_nodes, self.fes[0].vec))
        # Initialize potential contact elements
        self.mesh_functions = create_mesh_functions(
            self.fes[0].points,
            self.fes[0].cells
        )
        self.contact_elements = find_contact_elements(jax.vmap(lambda cell: jax.numpy.take(self.fes[0].points, cell, axis=0))(self.fes[0].cells))
        self.boundary_edges = self.mesh_functions['find_boundary_edges'](self.contact_elements)
        # geometry information
        self.Lx = np.max(self.fes[0].points[:, 0])
        self.Ly = np.min(self.fes[0].points[:, 1])
        # material parameters
        self.material_params = {
            'E': 500,       
            'nu': 0.3,         
        }
        self.E = self.material_params['E']
        self.nu = self.material_params['nu']
        self.Dindex = np.where(np.isclose(self.fes[0].points[:, 0], self.Lx, atol=1e-5) & 
                                         np.isclose(self.fes[0].points[:, 1], self.Ly, atol=1e-5))[0][0]
    def get_tensor_map(self):
        lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def compute_stress(self, sol):
        """obtain the stress for post processing"""
        # get the stress mapping function
        stress_fn = self.get_tensor_map()  
        vmap_stress = jax.vmap(jax.vmap(stress_fn))  
        u_grads = self.fes[0].sol_to_grad(sol)  # (num_cells, num_quads, vec, dim)
        stresses = vmap_stress(u_grads)  # (num_cells, num_quads, vec, dim)
        return stresses


    def newton_update(self, sol_list):
        """compute the residual R(u(n+1)) - F(u(n))"""
        R_internal = super().newton_update(sol_list)[0]  # compute the internal force in the new configuration u(n+1)
        Res = R_internal - self.F_contact
        return Res

    def check_displacement_convergence(self, sol):
        # Get the displacement of the midpoint on the top surface
        disp = sol.reshape(self.fes[0].num_total_nodes, self.fes[0].vec)
        disp = disp[self.Dindex, 0]  # x direction displacement
        # check if it is close to the target displacement
        return np.abs(disp - (-0.2)) < 0.005


class ContactProblem:  
    def __init__(self, elasticity1):
        """initialize the contact problem"""
        self.elasticity1 = elasticity1
        self.contact_params = {
            'K': 280.0,                    
            'K_target': 520.0,            
            'K_current': 280.0,             
            'K_step': 1.01,        
            'n': 1.0,
            'area_threshold': 5e-5
        }
        self.previous_force = None

        self.fiber_params = {
            'n_fibers': 700000,
            'fiber_length': 0.0015
        }
        self.sampling_params = {
            'num_samples': 1,
        }
        self.solver_params = {
            'relaxation_factor': 0.1,       # displacement relaxation factor
            'force_relaxation_factor': 0.3  # force relaxation factor
        }

    def initialize_contact_detection(self, sol1, fibers):
        """initialize the contact detection and update the contact state"""
        logger.info("\n=== contact detection starts ===")
        # clean the old state
        self.contact_state = {
            'contact_detected': False,
            'contact_info': None,
            'contact_params': None,
            'sol1': sol1
        }
        # compute the contact information
        contact_info = fiber_monte_carlo(
            problem=self.elasticity1,    
            rigid_body_params=self.elasticity1.rigid_body_params,
            contact_problem=self,           
            sol=sol1,
            fibers=fibers
        )
        
        if contact_info['has_contact']:
            self.contact_state = {
                'contact_detected': True,
                'contact_info': contact_info,
                'contact_params': compute_contact_parameters(contact_info, self),
                'sol1': sol1
            }
        else:
            self.contact_state = {
                'contact_detected': False,
                'contact_info': None,
                'contact_params': None,
                'sol1': sol1
            }
        logger.info("=== contact detection completed ===\n")
        
        return contact_info['has_contact']

    def compute_force(self, sol1):
        """compute the contact force"""
        # dynamically adjust the number of samples
        if self.elasticity1.check_displacement_convergence(sol1):
            self.sampling_params['num_samples'] = 3
        n_samples = self.sampling_params['num_samples']
        updated_elements1 = self.elasticity1.mesh_functions['get_updated_elements'](sol1)
        force1_samples = []
        for i in range(n_samples):
            fibers = fibers_generator(
                rigid_body_params=self.elasticity1.rigid_body_params,
                updated_elements=updated_elements1,
                n_fibers=self.fiber_params['n_fibers'],
                fiber_length=self.fiber_params['fiber_length'],
                key=jax.random.PRNGKey(time.time_ns())
            )
            if self.initialize_contact_detection(sol1, fibers):
                # compute the contact force
                F1 = compute_contact_force(
                    contact_params=self.contact_state['contact_params'],
                    gradient=self.contact_state['contact_info']['gradient'],
                    problem=self.elasticity1
                )
            else:
                # when there is no contact, return zero force
                F1 = np.zeros((self.elasticity1.fes[0].num_total_nodes, self.elasticity1.fes[0].vec))
            force1_samples.append(F1)
        
        # compute the average of all samples
        F1 = np.mean(np.array(force1_samples), axis=0)
        return F1

    #
    def compute_relaxed_force(self, sol, use_relaxation=False):
        if self.contact_params['K_current'] < self.contact_params['K_target']:
            new_K = min(self.contact_params['K_current'] * self.contact_params['K_step'], 
                        self.contact_params['K_target'])
            self.contact_params['K_current'] = new_K  # update K_current
            self.contact_params['K'] = new_K         # update K
        # compute the force using the current K value
        F_new = self.compute_force(sol)
        # when the displacement is close to the target or K reaches 50% of the target value, apply force relaxation
        k_ratio = self.contact_params['K'] / self.contact_params['K_target']
        apply_force_relax = (use_relaxation or k_ratio > 0.8)
        
        # apply force relaxation (if needed)
        if apply_force_relax and self.previous_force is not None:
            force_diff = np.linalg.norm(F_new - self.previous_force)
            force_norm = np.linalg.norm(F_new)
            rel_error = force_diff / (force_norm + 1e-10)
            F_relaxed = (self.solver_params['force_relaxation_factor'] * F_new + 
                       (1 - self.solver_params['force_relaxation_factor']) * self.previous_force)
            if rel_error < 8.5e-2 and k_ratio > 0.9:

                self.previous_force = F_new
                return F_new, True
            else:

                self.previous_force = F_relaxed
                return F_relaxed, False
        else:
            self.previous_force = F_new
            return F_new, False
    
    def get_contact_info(self):
        return self.contact_state.get('contact_info', None) if hasattr(self, 'contact_state') else None


def setup_Elasticity1():
    # Define input and output directories
    input_dir = os.path.join("examples", "static", "input")
    output_dir = os.path.join("examples", "static", "output")
    os.makedirs(output_dir, exist_ok=True)

    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(input_dir, "square.msh"))

    meshio_mesh.points = meshio_mesh.points[:, :2]
    meshio_mesh.points[:, 1] -= 2.5
    meshio_mesh.write(os.path.join(output_dir, 'vtk/mesh.vtu'))
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    Lx = np.max(meshio_mesh.points[:, 0])
    
    # Define boundary locations.    
    def symmetry(point):
        return np.isclose(point[1], 0, atol=2e-3)
    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)
    
    def dirichlet_val_left_x(point):
        return 0.

    def get_dirichlet_left_x(disp):
        def val_fn(point):
            return disp
        return val_fn
    # define the loading step
    initial_disp = -0.20 # initial compression displacement
    # set the Dirichlet boundary conditions
    location_fns1 = [symmetry, right, right]     
    vecs = [1, 0, 1]                           
    value_fns = [dirichlet_val_left_x, get_dirichlet_left_x(initial_disp), dirichlet_val_left_x]  
    dirichlet_bc_info = [location_fns1, vecs, value_fns]
    return {
        'mesh': mesh,
        'dirichlet_bc_info': dirichlet_bc_info,
    }

def simulation():
    """Main simulation function for the contact problem"""
    # Define input and output directories
    input_dir = os.path.join("examples", "static", "input")
    output_dir = os.path.join("examples", "static", "output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/vtk", exist_ok=True)
    
    # read the boundary points for the rigid body
    def read_boundary_points(filename):
        boundary_points = []
        with open(filename, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                boundary_points.append(np.array([x, y]))
        return np.array(boundary_points)
    rigid_body_params = read_boundary_points(os.path.join(input_dir, 'square_points.txt'))
  

    elasticity1_config = setup_Elasticity1()
    elasticity1 = Elasticity1(  
        mesh=elasticity1_config['mesh'], 
        ele_type='QUAD4',
        vec=2, 
        dim=2,
        dirichlet_bc_info=elasticity1_config['dirichlet_bc_info'], 
        additional_info=(rigid_body_params,))

    contact_problem = ContactProblem(elasticity1)
    
    # initialize the displacement field
    sol = np.zeros(elasticity1.num_total_dofs_all_vars)
    
    # set the solver options
    solver_options = {
        'petsc_solver': {
         'ksp_type': 'tfqmr',
         'pc_type': 'lu',
         'pc_factor_mat_solver_type': 'mumps'
        },
        'initial_guess': [sol]
    }
    # solve the contact problem
    sol = contact_solver(
        elasticity1, contact_problem, sol, solver_options)
    
    # save the results
    vtk_path = f"{output_dir}/vtk/result.vtu"
    save_sol1(elasticity1, sol[0], vtk_path, contact_problem)

if __name__ == "__main__":
    simulation()   