
"""
This module contains the sovler for dynamics problem.
"""

import jax
import jax.numpy as np
import numpy as onp
import scipy.sparse
from jax_fem import logger
import time


class DynamicContactSolver():
    """dynamic contact solver"""
    def __init__(self, elasticity1, elasticity2, contact_problem, solver_params=None):
        self.elasticity1 = elasticity1
        self.elasticity2 = elasticity2
        self.contact_problem = contact_problem
        # initialize the solver parameters
        self.solver_params = solver_params or {}
        self._setup_two_body_system()

    def _setup_two_body_system(self):
        """set the mass matrix and initial state of the contact bodies"""
        # 1. set the mass matrix
        self.M1, self.M1_inv = self._setup_mass_matrix(self.elasticity1)
        self.M2, self.M2_inv = self._setup_mass_matrix(self.elasticity2)

        num_nodes1 = self.elasticity1.fes[0].num_total_nodes
        num_nodes2 = self.elasticity2.fes[0].num_total_nodes
        vec_dim = self.elasticity1.fes[0].vec  # usually is 2 (2D problem)
        # state variables: [displacement, velocity, acceleration] - (N, 6) for 2D
        self.state1 = np.zeros((num_nodes1, 3 * vec_dim))  
        self.state2 = np.zeros((num_nodes2, 3 * vec_dim))

        # 2. set the initial velocity
        v0_x = self.solver_params.get('initial_velocity_x', 0)
        v0_y = self.solver_params.get('initial_velocity_y', -50)
        self.state1 = self.state1.at[:, vec_dim].set(v0_x)      
        self.state1 = self.state1.at[:, vec_dim+1].set(v0_y)
        # 5. set the adaptive time step parameter
        self.current_dt = self.solver_params.get('dt', 1e-5)
        self.contact_detected = False    # contact state flag
        logger.info(f"initial time step: {self.current_dt:.2e}s")

    def _setup_mass_matrix(self, elasticity):
        """set the lump mass matrix"""
        problem_mass = elasticity.get_mass_problem()
        dofs = np.zeros(problem_mass.num_total_dofs_all_vars)
        sol_list = problem_mass.unflatten_fn_sol_list(dofs)
        problem_mass.newton_update(sol_list)
        A_sp_scipy = scipy.sparse.csr_array(
            (onp.array(problem_mass.V), 
             (problem_mass.I, problem_mass.J)),
            shape=(problem_mass.num_total_dofs_all_vars, 
                  problem_mass.num_total_dofs_all_vars)
        )
        M_flat = A_sp_scipy.sum(axis=1) 
        vec_dim = elasticity.fes[0].vec
        num_nodes = elasticity.fes[0].num_total_nodes
        M = M_flat.reshape(num_nodes, vec_dim)     
        M_inv = 1.0 / M                          
        return M, M_inv

    def solve(self):
        """solve the contact problem: dynamic contact"""
        start = time.time()
        iter_count = 0
        self.dynamics_mode = False
        self.max_iter = self.solver_params.get('max_iter', 10000)
        logger.info("calling dynamic solver...")
        self.current_dt = self._check_timestep_stability()

        current_time = 0.0 
        while iter_count < self.max_iter:
            vec_dim = self.elasticity1.fes[0].vec
            if not self.dynamics_mode:
                self._free_motion()  
                if self._check_contact_state():
                    self.dynamics_mode = True
                    self.state1, self.state2 = self._update_state(self.state1, self.state2, self.current_dt)
            else:
                # dynamic mode
                self.state1, self.state2 = self._two_body_leap_frog_step(self.state1, self.state2, self.current_dt)
            current_time += self.current_dt*1000
            
            iter_count += 1

            if iter_count % 1 == 0:
                self._log_step_info(iter_count)
            
        logger.info(f"dynamic solver finished, total steps: {iter_count}")
        
        # 返回最终位移
        u1_final = [self.state1[:, :vec_dim]]
        u2_final = [self.state2[:, :vec_dim]]
        end = time.time()
        solve_time = end - start
        logger.info(f"Solve took {solve_time} [s]")
        return u1_final, u2_final
    
    def _free_motion(self):
        """free motion"""
        logger.info("free motion")
        vec_dim = self.elasticity1.fes[0].vec
        dt = self.current_dt
 
        pos1 = self.state1[:, :vec_dim]
        vel1 = self.state1[:, vec_dim:2*vec_dim]
        pos1_new = pos1 + dt * vel1
        vel1_new = vel1 + dt * np.array([0.0, 0.0])  
        acc1_new = np.zeros_like(pos1) + np.array([0.0, 0.0])
        self.state1 = np.hstack((pos1_new, vel1_new, acc1_new))

    def _check_contact_state(self):
        """contact detection - for mode switch"""
        vec_dim = self.elasticity1.fes[0].vec
        u1 = self.state1[:, :vec_dim].flatten()
        u2 = self.state2[:, :vec_dim].flatten()
        _, _ = self.contact_problem.compute_force(u1, u2)
        contact_info = self.contact_problem.get_contact_info()
        return contact_info is not None and contact_info.get('has_contact', False)

    def _update_state(self, state1, state2, dt):
        """update the state using the contact force"""
        vec_dim = self.elasticity1.fes[0].vec
        # current state (_free_motion result)
        pos1 = state1[:, :vec_dim]           
        vel1 = state1[:, vec_dim:2*vec_dim]  
        pos2 = state2[:, :vec_dim]           
        vel2 = state2[:, vec_dim:2*vec_dim]  
        vel1_half = vel1  
        vel2_half = vel2
        u1_current = pos1.flatten()
        u2_current = pos2.flatten()
        F_contact1, F_contact2 = self.contact_problem.compute_force(u1_current, u2_current)
        self.elasticity1.F_contact = F_contact1 
        self.elasticity2.F_contact = F_contact2
        residual1_current = self.elasticity1.compute_residual(u1_current)
        residual2_current = self.elasticity2.compute_residual(u2_current)
        acc1_new = self.M1_inv * (-residual1_current)
        acc2_new = self.M2_inv * (-residual2_current)
        vel1_new = vel1_half + 0.5 * dt * acc1_new
        vel2_new = vel2_half + 0.5 * dt * acc2_new
        new_state1 = np.hstack((pos1, vel1_new, acc1_new))
        new_state2 = np.hstack((pos2, vel2_new, acc2_new))
        return new_state1, new_state2


    def _two_body_leap_frog_step(self, state1, state2, dt):
        """
       kick-drift-kick
        """
        vec_dim = self.elasticity1.fes[0].vec
        

        pos1 = state1[:, :vec_dim]
        vel1 = state1[:, vec_dim:2*vec_dim]
        acc1_old = state1[:, 2*vec_dim:]
        pos2 = state2[:, :vec_dim]
        vel2 = state2[:, vec_dim:2*vec_dim]
        acc2_old = state2[:, 2*vec_dim:]
        
        # Step 1: KICK1 
        vel1_half = vel1 + 0.5 * dt * acc1_old
        vel2_half = vel2 + 0.5 * dt * acc2_old
        
        # Step 2: DRIFT 
        pos1_new = pos1 + dt * vel1_half
        pos2_new = pos2 + dt * vel2_half

        u1_new = pos1_new.flatten()
        u2_new = pos2_new.flatten()
        
        # compute the contact force
        F_contact1, F_contact2 = self.contact_problem.compute_force(u1_new, u2_new)
        self.elasticity1.F_contact = F_contact1
        self.elasticity2.F_contact = F_contact2
        # Step 4: compute the updated acceleration
        residual1_new = self.elasticity1.compute_residual(u1_new)
        residual2_new = self.elasticity2.compute_residual(u2_new)
        acc1_new = self.M1_inv * (-residual1_new)
        acc2_new = self.M2_inv * (-residual2_new)
        
        # Step 5: KICK2
        vel1_new = vel1_half + 0.5 * dt * acc1_new
        vel2_new = vel2_half + 0.5 * dt * acc2_new
        
        # update
        new_state1 = np.hstack((pos1_new, vel1_new, acc1_new))
        new_state2 = np.hstack((pos2_new, vel2_new, acc2_new))
        
        return new_state1, new_state2

    def _compute_kinetic_energy(self):

        vec_dim = self.elasticity1.fes[0].vec
        v1 = self.state1[:, vec_dim:2*vec_dim]  
        v2 = self.state2[:, vec_dim:2*vec_dim]
   
        KE1 = np.sum(0.5 * self.M1 * v1 * v1)
        KE2 = np.sum(0.5 * self.M2 * v2 * v2)
        return KE1 + KE2

    def _compute_elastic_energy(self):

        vec_dim = self.elasticity1.fes[0].vec

        u1 = self.state1[:, :vec_dim]  
        u2 = self.state2[:, :vec_dim] 
        
        PE1 = self._compute_energy(self.elasticity1, u1)
        PE2 = self._compute_energy(self.elasticity2, u2)
        total_PE = PE1 + PE2 
        return total_PE

    def _compute_energy(self, elasticity, sol):
        u_grads = elasticity.fes[0].sol_to_grad(sol) 

        def get_psi_function():
            def psi(F_2d):
                F = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                              [F_2d[1, 0], F_2d[1, 1], 0.],
                              [0., 0., 1.]])
                E = elasticity.E
                nu = elasticity.nu
                mu = E / (2. * (1. + nu))
                kappa = E / (3. * (1. - 2. * nu))
                J = np.linalg.det(F)
                Jinv = J**(-2. / 3.)
                I1 = np.trace(F.T @ F)
                energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
                return energy
            return psi
        
        psi_fn = get_psi_function()
        def compute_energy_density(u_grad_2d):
            I_2d = np.eye(elasticity.dim)
            F_2d = u_grad_2d + I_2d
            return psi_fn(F_2d)
        
        vmap_psi = jax.vmap(jax.vmap(compute_energy_density))
        energy_densities = vmap_psi(u_grads)
        return np.sum(energy_densities * elasticity.fes[0].JxW)

    def _compute_contact_energy(self):

        contact_info = self.contact_problem.get_contact_info()
        if not contact_info:
            return 0.0
        contact_area = contact_info.get('contact_area', 0.0)
        K = self.contact_problem.contact_params['K']
        n = self.contact_problem.contact_params['n']
        return K * (contact_area ** n)

    def _get_system_status(self):
        """get information of system states """
        vec_dim = self.elasticity1.fes[0].vec
        
        # Body1 状态
        u1 = self.state1[:, :vec_dim]
        v1 = self.state1[:, vec_dim:2*vec_dim]
        a1 = self.state1[:, 2*vec_dim:]
        
        body1_status = {
            'max_u_x': np.max(np.abs(u1[:, 0])),
            'max_u_y': np.max(np.abs(u1[:, 1])),
            'max_v_x': np.max(np.abs(v1[:, 0])),
            'max_v_y': np.max(v1[:, 1]),
            'max_a_x': np.max(np.abs(a1[:, 0])),
            'max_a_y': np.max(np.abs(a1[:, 1]))
        }
        
        # Body2 状态
        u2 = self.state2[:, :vec_dim]
        v2 = self.state2[:, vec_dim:2*vec_dim]
        a2 = self.state2[:, 2*vec_dim:]
        
        body2_status = {
            'max_u_x': np.max(np.abs(u2[:, 0])),
            'max_u_y': np.max(np.abs(u2[:, 1])),
            'max_v_x': np.max(np.abs(v2[:, 0])),
            'max_v_y': np.max(v2[:, 1]),
            'max_a_x': np.max(np.abs(a2[:, 0])),
            'max_a_y': np.max(np.abs(a2[:, 1]))
        }
        
        # 能量状态
        KE = self._compute_kinetic_energy()
        PE_energy = self._compute_elastic_energy()
        PE_contact = self._compute_contact_energy()
        
        energy_status = {
            'kinetic': KE,
            'elastic': PE_energy,
            'contact': PE_contact,
            'total': KE + PE_energy + PE_contact
        }
        
        return body1_status, body2_status, energy_status

    def _log_step_info(self, step):
        """Log step information"""
        body1, body2, energy = self._get_system_status()
        logger.info(f"dynamic step {step}: dt={self.current_dt:.2e}s, mode={'contact' if self.dynamics_mode else 'free'}")
        logger.info(f"Body1 - u_x={body1['max_u_x']:.6f}, u_y={body1['max_u_y']:.6f}, "
                    f"v_x={body1['max_v_x']:.6f}, v_y={body1['max_v_y']:.6f}, "
                    f"a_x={body1['max_a_x']:.6f}, a_y={body1['max_a_y']:.6f}")
        logger.info(f"Body2 - u_x={body2['max_u_x']:.6f}, u_y={body2['max_u_y']:.6f}, "
                    f"v_x={body2['max_v_x']:.6f}, v_y={body2['max_v_y']:.6f}, "
                    f"a_x={body2['max_a_x']:.6f}, a_y={body2['max_a_y']:.6f}")
        logger.info(f"Energy - KE={energy['kinetic']:.6f}, PE_elastic={energy['elastic']:.6f}, "
                   f"PE_contact={energy['contact']:.6f}, Total_E={energy['total']:.6f}")
 
    def _compute_critical_timestep(self):
        """compute the critical time step satisfying CFL condition"""

        E1 = self.elasticity1.E  # Young's modulus1
        nu1 = self.elasticity1.nu  # Poisson's ratio1
        rho1 = self.elasticity1.rho  # density1
        E2 = self.elasticity2.E  # Young's modulus2
        nu2 = self.elasticity2.nu  # Poisson's ratio2
        rho2 = self.elasticity2.rho  # density2
        # 2. compute the wave speed
        # c = sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
        wave_speed1 = np.sqrt((E1 * (1 - nu1)) / ((1 + nu1) * (1 - 2*nu1) * rho1))
        wave_speed2 = np.sqrt((E2 * (1 - nu2)) / ((1 + nu2) * (1 - 2*nu2) * rho2))
        # 3. get the minimum cell size
        dx_min1 = 0.006
        dx_min2 = 0.004
        # 4. compute the critical time step
        dt_c1 = dx_min1 / wave_speed1
        dt_c2 = dx_min2 / wave_speed2
        # 5. get the minimum value of the two elastic bodies
        dt_critical = min(dt_c1, dt_c2)
        logger.info(f"critical timestep: dt_c1={dt_c1:.2e}, dt_c2={dt_c2:.2e}")
        return dt_critical
    
    def _check_timestep_stability(self):
        dt_critical = self._compute_critical_timestep()
        self.current_dt = 0.6 * dt_critical   
        logger.info(f"safe timestep: {self.current_dt:.2e}s")
        return self.current_dt

    def _compute_contact_force(self):
        """compute the contact force component"""

        if hasattr(self.elasticity1, 'F_contact') and hasattr(self.elasticity2, 'F_contact'):
            F_contact1 = self.elasticity1.F_contact  
            F_contact2 = self.elasticity2.F_contact  
            # 计算每个物体的x、y方向合力
            F1_x = np.sum(F_contact1[:, 0])  
            F1_y = np.sum(F_contact1[:, 1])  
            F2_x = np.sum(F_contact2[:, 0])  
            F2_y = np.sum(F_contact2[:, 1]) 
            
        else:
            F1_x = 0.0
            F1_y = 0.0
            F2_x = 0.0
            F2_y = 0.0
        # 获取接触面积
        contact_info = self.contact_problem.get_contact_info()
        if contact_info:
            contact_area = contact_info.get('contact_area', 0.0)
        else:
            contact_area = 0.0
        
        return F1_x, F1_y, F2_x, F2_y, contact_area


def contact_solver(elasticity1, elasticity2, contact_problem, solver_params=None):
    solver = DynamicContactSolver(elasticity1, elasticity2, contact_problem, solver_params)
    return solver.solve()
