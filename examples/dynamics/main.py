import jax
jax.config.update("jax_enable_x64", True)
#The main module for solving 2D flexible-rigid contact problems with JAX-FEM
# Some useful package
import os
import numpy as onp
import jax.numpy as np
import meshio
import time
from jax_fem import logger
# JAX-FEM packages
from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh
##
from src.cmsl.fmc_contact.fmc2 import fiber_monte_carlo, fibers_generator
from src.cmsl.fmc_contact.contact_utils import (find_contact_elements_body1, find_contact_elements_body2, 
                                                compute_contact_parameters, compute_contact_force)
from src.cmsl.fmc_contact.sdf import create_mesh_functions
from src.cmsl.fmc_contact.dynamic_solver import contact_solver 
from src.cmsl.fmc_contact.custom_utils import save_sol2


class ElasticityMass(Problem):
    """elastic body mass problem"""
    def custom_init(self, rho):
        """custom initialization"""
        self.fe = self.fes[0]
        self.rho = rho
    def get_mass_map(self):
        """define the weak form of the mass"""
        def mass_map(u, x):
            return self.rho * u
        return mass_map

class Elasticity1(Problem):
    """the first elastic body"""
    def custom_init(self):
        """initialize the properties of the first elastic body"""
        # 统一定义求解参数
        self.mesh_functions = create_mesh_functions(
            self.fes[0].points,
            self.fes[0].cells
        )
        self.contact_elements = find_contact_elements_body1(jax.vmap(lambda cell: jax.numpy.take(self.fes[0].points, cell, axis=0))(self.fes[0].cells))
        self.boundary_edges = self.mesh_functions['find_boundary_edges'](self.contact_elements)
        # 几何信息
        self.Lx = np.min(self.fes[0].points[:, 0])
        self.Ly = np.min(self.fes[0].points[:, 1])

        self.material_params = {
            'E': 73.2e6,       
            'nu': 0.4,        
            'rho': 1010.0      
        }
        self.E = self.material_params['E']
        self.nu = self.material_params['nu'] 
        self.rho = self.material_params['rho']
        self.Dindex = np.where(np.isclose(self.fes[0].points[:, 0], 0, atol=1e-5) & 
                                         np.isclose(self.fes[0].points[:, 1], self.Ly, atol=1e-5))[0][0]
        self.F_contact = np.zeros((self.fes[0].num_total_nodes, self.fes[0].vec))
    def get_tensor_map(self):
        """
        Neo-Hookean material
        """
        def psi(F_2d):
            F = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                          [F_2d[1, 0], F_2d[1, 1], 0.],
                          [0., 0., 1.]])
            E = self.E
            nu = self.nu
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)        
            Jinv = J**(-2. / 3.)        
            I1 = np.trace(F.T @ F)      
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
    def get_mass_problem(self):
        """get the mass matrix"""
        mass_problem = ElasticityMass(
            mesh=self.fes[0].mesh,
            vec=self.fes[0].vec,
            dim=self.dim,
            ele_type='QUAD4',
            additional_info=(self.rho,)
        )
        return mass_problem

    def compute_stress(self, sol):
        """plane strain Cauchy stress"""
        u_grads = self.fes[0].sol_to_grad(sol) 
        def psi_3d(F_3d):
            E = self.E
            nu = self.nu
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F_3d)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F_3d.T @ F_3d)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn_3d = jax.grad(psi_3d)  
        def compute_cauchy_stress(u_grad_2d):
            I_2d = np.eye(2)
            F_2d = u_grad_2d + I_2d
            F_3d = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                             [F_2d[1, 0], F_2d[1, 1], 0.],
                             [0., 0., 1.]])
            J = np.linalg.det(F_3d)
            P_3d = P_fn_3d(F_3d)
            cauchy_stress = (1.0 / J) * P_3d @ F_3d.T
            return cauchy_stress
        vmap_convert = jax.vmap(jax.vmap(compute_cauchy_stress))
        cauchy_stress = vmap_convert(u_grads)
        return cauchy_stress 

    def newton_update(self, sol_list):
        """compute the residual R(u(n+1)) - F(u(n))"""
        R_internal = super().newton_update(sol_list)[0]  # compute the internal force in the new configuration u(n+1)
        Res = R_internal - self.F_contact
        return Res
    
    def compute_residual(self, dofs):
        sol_list = self.unflatten_fn_sol_list(dofs)
        res = super().compute_residual(sol_list)[0]
        return res - self.F_contact


    def check_displacement_convergence(self, sol):

        disp = sol.reshape(self.fes[0].num_total_nodes, self.fes[0].vec)
        disp = disp[self.Dindex, 1]  
        return np.abs(disp - (-0.01)) < 2e-4

class Elasticity2(Problem):
    """the second elastic body"""
    def custom_init(self):
        """initialize the properties of the second elastic body"""
        # 统一定义求解参数
        self.mesh_functions = create_mesh_functions(
            self.fes[0].points,
            self.fes[0].cells
        )
        self.contact_elements = find_contact_elements_body2(jax.vmap(lambda cell: jax.numpy.take(self.fes[0].points, cell, axis=0))(self.fes[0].cells))
        self.boundary_edges = self.mesh_functions['find_boundary_edges'](self.contact_elements)
        self.Lx_max = np.max(self.fes[0].points[:, 0])
        self.Ly_min = np.min(self.fes[0].points[:, 1])
        self.Ly_max = np.max(self.fes[0].points[:, 1])
        self.material_params = {
            'E': 73.2e5,       
            'nu': 0.4,         
            'rho': 1010.0      
        }
        self.E = self.material_params['E']
        self.nu = self.material_params['nu'] 
        self.rho = self.material_params['rho']
        self.F_contact = np.zeros((self.fes[0].num_total_nodes, self.fes[0].vec))

    def get_tensor_map(self):

        def psi(F_2d):
            F = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                          [F_2d[1, 0], F_2d[1, 1], 0.],
                          [0., 0., 1.]])
            E = self.E
            nu = self.nu
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)        
            Jinv = J**(-2. / 3.)        
            I1 = np.trace(F.T @ F)      
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
    def get_mass_problem(self):

        mass_problem = ElasticityMass(
            mesh=self.fes[0].mesh,
            vec=self.fes[0].vec,
            dim=self.dim,
            ele_type='QUAD4',
            additional_info=(self.rho,)
        )
        return mass_problem
    
    def compute_stress(self, sol):
        u_grads = self.fes[0].sol_to_grad(sol) 
        def psi_3d(F_3d):
            E = self.E
            nu = self.nu
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F_3d)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F_3d.T @ F_3d)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn_3d = jax.grad(psi_3d)  
        def compute_cauchy_stress(u_grad_2d):
            I_2d = np.eye(2)
            F_2d = u_grad_2d + I_2d
            F_3d = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                             [F_2d[1, 0], F_2d[1, 1], 0.],
                             [0., 0., 1.]])
            J = np.linalg.det(F_3d)
            P_3d = P_fn_3d(F_3d)
            cauchy_stress = (1.0 / J) * P_3d @ F_3d.T
            return cauchy_stress
        vmap_convert = jax.vmap(jax.vmap(compute_cauchy_stress))
        cauchy_stress = vmap_convert(u_grads)
        return cauchy_stress 

    def newton_update(self, sol_list):
        """compute the residual R(u(n+1)) - F(u(n))"""
        R_internal = super().newton_update(sol_list)[0]  
        Res = R_internal - self.F_contact
        return Res

    def compute_residual(self, dofs):
        sol_list = self.unflatten_fn_sol_list(dofs)
        res = super().compute_residual(sol_list)[0]
        return res - self.F_contact

class ContactProblem:  
    def __init__(self, elasticity1, elasticity2):
      
        self.elasticity1 = elasticity1
        self.elasticity2 = elasticity2
        
        # 自动调用各自的初始化
        self.elasticity1.custom_init()
        self.elasticity2.custom_init()
        self.contact_params = {
            'K': 2.8e7,  
            'n': 1.0,                      
            'area_threshold': 1.0e-9      
        }
        self.previous_force1 = None
        self.previous_force2 = None
    
        self.fiber_params = {
            'n_fibers': 300000,
            'fiber_length': 0.001
        }
        self.sampling_params = {
            'num_samples': 1,
        }

    def initialize_contact_detection(self, sol1, sol2, fibers):
        """initialize the contact detection and update the contact state"""
        logger.info("\n=== contact detection starts ===")
        # clean the old state
        self.contact_state = {
            'contact_detected': False,
            'contact_info': None,
            'contact_params': None,
            'sol1': sol1,
            'sol2': sol2
        }

        contact_info = fiber_monte_carlo(
            problem1=self.elasticity1,     
            problem2=self.elasticity2,      
            ContactProblem=self,            
            sol1=sol1,
            sol2=sol2,
            fibers=fibers
        )
        
        if contact_info['has_contact']:
            self.contact_state = {
                'contact_detected': True,
                'contact_info': contact_info,
                'contact_params': compute_contact_parameters(contact_info, self),
                'sol1': sol1,
                'sol2': sol2
            }
        else:
            self.contact_state = {
                'contact_detected': False,
                'contact_info': None,
                'contact_params': None,
                'sol1': sol1,
                'sol2': sol2
            }
        logger.info("=== contact detection completed ===\n")
        return contact_info['has_contact']
    
    def compute_force(self, sol1, sol2):
 
        if self.elasticity1.check_displacement_convergence(sol1):
            self.sampling_params['num_samples'] = 1
        n_samples = self.sampling_params['num_samples']
        
        updated_elements1 = self.elasticity1.mesh_functions['get_updated_elements'](sol1)
        updated_elements2 = self.elasticity2.mesh_functions['get_updated_elements'](sol2)
        
        force1_samples = []
        force2_samples = []
        y_balance_errors = []

        for i in range(n_samples):
            logger.info(f"\n=== sampling {i+1}/{n_samples} ===")
            fibers = fibers_generator(
                updated_elements1=updated_elements1,
                updated_elements2=updated_elements2,
                n_fibers=self.fiber_params['n_fibers'],
                fiber_length=self.fiber_params['fiber_length'],
                key=jax.random.PRNGKey(time.time_ns())
            )
            
            if self.initialize_contact_detection(sol1, sol2, fibers):
               
                F1 = compute_contact_force(
                    contact_params=self.contact_state['contact_params'],
                    gradient=self.contact_state['contact_info']['gradient1'],
                    problem=self.elasticity1
                )
                F2 = compute_contact_force(
                    contact_params=self.contact_state['contact_params'],
                    gradient=self.contact_state['contact_info']['gradient2'], 
                    problem=self.elasticity2
                )
            else:
                F1 = np.zeros((self.elasticity1.fes[0].num_total_nodes, self.elasticity1.fes[0].vec))
                F2 = np.zeros((self.elasticity2.fes[0].num_total_nodes, self.elasticity2.fes[0].vec))
            

            total_fy1 = np.sum(F1[:, 1])
            total_fy2 = np.sum(F2[:, 1]) 
            y_balance_error = np.abs(total_fy1 + total_fy2)
            

            force1_samples.append(F1)
            force2_samples.append(F2)
            y_balance_errors.append(y_balance_error)
        # select the best sample for y-axis force balance
        idx = np.argmin(np.array(y_balance_errors))
        F1 = force1_samples[idx]
        F2 = force2_samples[idx]
        self.elasticity1.F_contact = F1
        self.elasticity2.F_contact = F2
        return F1, F2
    

    def get_contact_info(self):
        """get updated contact information"""
        return self.contact_state.get('contact_info', None) if hasattr(self, 'contact_state') else None
   

def setup_Elasticity1():
    """setup the configuration of the first elastic body"""

    # Define input and output directories
    input_dir = os.path.join("examples", "dynamics", "input")
    output_dir = os.path.join("examples", "dynamics", "output")
    os.makedirs(output_dir, exist_ok=True)

    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(input_dir, "star.msh"))
    meshio_mesh.points = meshio_mesh.points[:, :2]
    scale_factor = 0.1  # 缩小10倍
    meshio_mesh.points = meshio_mesh.points * scale_factor
    meshio_mesh.write(os.path.join(output_dir, 'vtk/mesh1.vtu'))
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    
    return {
        'mesh': mesh,
        'dirichlet_bc_info': None,
    }

def setup_Elasticity2():
    """setup the configuration of the second elastic body"""

    input_dir = os.path.join("examples", "dynamics", "input")
    output_dir = os.path.join("examples", "dynamics", "output")
    os.makedirs(output_dir, exist_ok=True)

    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(input_dir, "block.msh"))
    meshio_mesh.points = meshio_mesh.points[:, :2]
    meshio_mesh.points[:, 1] += 0.08 
    scale_factor = 0.1  
    meshio_mesh.points = meshio_mesh.points * scale_factor
    meshio_mesh.write(os.path.join(output_dir, 'vtk/mesh2.vtu'))
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    Ly = np.min(meshio_mesh.points[:, 1])
    def bottom(point):
        return np.isclose(point[1], Ly, atol=1e-5)
    def dirichlet_val_zero(point):
        return 0.0
    location_fns = [bottom, bottom] 
    vecs = [0, 1]                    
    value_fns = [
        dirichlet_val_zero, 
        dirichlet_val_zero
    ]
    dirichlet_bc_info = [location_fns, vecs, value_fns]
    return {
        'mesh': mesh,
        'dirichlet_bc_info': dirichlet_bc_info,
    }

def simulation():
    """Main simulation function for the contact problem"""

    input_dir = os.path.join("examples", "dynamics", "input")
    output_dir = os.path.join("examples", "dynamics", "output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/vtk", exist_ok=True)


    elasticity1_config = setup_Elasticity1()  
    elasticity2_config = setup_Elasticity2()  
    

    elasticity1 = Elasticity1(  
        mesh=elasticity1_config['mesh'], 
        ele_type='QUAD4',
        vec=2, 
        dim=2,
        dirichlet_bc_info=elasticity1_config['dirichlet_bc_info'], 
        additional_info=()
    )
    
    elasticity2 = Elasticity2( 
        mesh=elasticity2_config['mesh'],
        ele_type='QUAD4',
        vec=2,
        dim=2, 
        dirichlet_bc_info=elasticity2_config['dirichlet_bc_info'], 
        additional_info=()
    )
    

    contact_problem = ContactProblem(elasticity1, elasticity2)
    
    

    solver_options = {
        'petsc_solver': {
         'ksp_type': 'tfqmr',
         'pc_type': 'lu',
         'pc_factor_mat_solver_type': 'mumps'
        },
        'dt': 4e-5,
        'max_iter': 10000
    }
    
    sol1_final, sol2_final = contact_solver(
        elasticity1, elasticity2, contact_problem, 
        solver_options)
    
    vtk_path1 = f"{output_dir}/vtk/body1.vtu"
    vtk_path2 = f"{output_dir}/vtk/body2.vtu"
    
    save_sol2(elasticity1, sol1_final[0], vtk_path1, contact_problem)
    save_sol2(elasticity2, sol2_final[0], vtk_path2, contact_problem)

if __name__ == "__main__":
    simulation()