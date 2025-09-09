"""
This module contains the sovler for quasi-statics problem.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from jax_fem.solver import (
    linear_incremental_solver,
    apply_bc_vec,
    get_A
)
from jax_fem import logger

## static contact solver
class StaticContactSolver:
    """static contact solver"""
    def __init__(self, elasticity1, contact_problem, solver_params=None):
        self.elasticity1 = elasticity1
        self.contact_problem = contact_problem
        self.solver_params = solver_params or {}

    def solve(self, sol1_init):
        """solve the contact problem: static contact: R(u*) - F(u*) = 0"""
        logger.debug(f"Calling the static contact solver")
        # set the convergence parameters
        fixed_point_tol1 = self.solver_params.get('fixed_point_tol1', 7e-3)   #7e-3
        fixed_point_max_iter = self.solver_params.get('fixed_point_max_iter', 100)
        relaxation_factor = self.contact_problem.solver_params['relaxation_factor']
        
        # fixed point iteration
        sol1 = sol1_init.copy()
        fixed_point_iter = 0
        while fixed_point_iter < fixed_point_max_iter:
            u1_n = sol1.copy()
            use_relaxation = self.elasticity1.check_displacement_convergence(u1_n)
            F1,_ = self.contact_problem.compute_relaxed_force(u1_n, use_relaxation)
            self.elasticity1.F_contact = F1
            
            # solve the Newton iteration
            logger.info("Solving the Newton iteration for the elastic body 1...")
            sol1_new = self._newton_solver(self.elasticity1, sol1, self.solver_params)
            sol1_final = sol1_new.copy()
            
            sol1_increment = np.linalg.norm(sol1_new - u1_n)
            sol1_norm = np.linalg.norm(sol1_new)
            rel_increment1 = sol1_increment / (sol1_norm + 1e-10)
            
            # check the convergence
            if rel_increment1 < fixed_point_tol1:
                logger.debug(f"Fixed point iteration converged in {fixed_point_iter + 1} steps")
                break
                
            # apply the relaxation factor
            sol1 = relaxation_factor * sol1_new + (1 - relaxation_factor) * u1_n
            fixed_point_iter += 1
            logger.debug(f"Update u(n) for next fixed point iteration, fixed_point_iter = {fixed_point_iter}")
        
        if fixed_point_iter == fixed_point_max_iter:
            logger.warning(f"Fixed point iteration did not converge in {fixed_point_max_iter} steps")
        
        # check the validity of the solution
        assert np.all(np.isfinite(sol1_final)), f"sol1_final contains NaN, stop the program!"
        sol1_list = self.elasticity1.unflatten_fn_sol_list(sol1_final)
        return sol1_list

    def _newton_solver(self, problem, sol_init, solver_options):
        """Newton solver"""
        dofs = sol_init.copy()
        newton_rel_tol = solver_options.get('rel_tol', 1e-8)
        newton_tol = solver_options.get('tol', 1e-6)
        newton_max_iter = solver_options.get('max_iter', 50)
        
        res_vec, A_fn = self._newton_update_helper(dofs, problem)
        res_val = np.linalg.norm(res_vec)
        res_val_initial = res_val
        rel_res_val = res_val / res_val_initial
        
        newton_iter = 0
        while ((rel_res_val > newton_rel_tol) and (res_val > newton_tol) and 
               (newton_iter < newton_max_iter)):
            dofs = linear_incremental_solver(problem, res_vec, A_fn, dofs, solver_options)
            res_vec, A_fn = self._newton_update_helper(dofs, problem)
            res_val = np.linalg.norm(res_vec)
            rel_res_val = res_val / res_val_initial
            logger.debug(f"Newton iteration {newton_iter + 1}: l_2 res = {res_val}, relative l_2 res = {rel_res_val}")
            newton_iter += 1
        
        if newton_iter == newton_max_iter:
            logger.warning(f"Newton iteration did not converge in {newton_max_iter} steps")
        else:
            logger.debug(f"Newton iteration converged in {newton_iter} steps")
        
        return dofs

    def _newton_update_helper(self, dofs, problem):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)  
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A_fn = get_A(problem)
        return res_vec, A_fn


def contact_solver(elasticity1, contact_problem, sol1_init, solver_params=None):
    solver = StaticContactSolver(elasticity1, contact_problem, solver_params)
    return solver.solve(sol1_init)



## dynamic contact solver

