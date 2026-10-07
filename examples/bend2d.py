from firedrake import *
from netgen.occ import *
from meshgenbend import create_geometry_bend
import os
from simpl import *

class Bend2D(NavierStokesMTW):

    def mesh(self):
        maxh = 0.1
        ngmesh, markers = create_geometry_bend(maxh)
        base = Mesh(ngmesh,distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)},)
        mh   = MeshHierarchy(base, 2)
        mesh = mh[-1]
        self.markers = markers
        return mesh

    def boundary_conditions(self, W):
        markers = self.markers
        BOTTOM_WALL_LEFT  = markers["bottom_wall_left"]
        BOTTOM_OUTLET     = markers["bottom_outlet"]
        BOTTOM_WALL_RIGHT = markers["bottom_wall_right"]
        RIGHT_WALL        = markers["right_wall"]
        TOP_WALL          = markers["top_wall"]
        LEFT_WALL_TOP     = markers["left_wall_top"]
        LEFT_INLET        = markers["left_inlet"]
        LEFT_WALL_BOTTOM  = markers["left_wall_bottom"]

        (x, y) = SpatialCoordinate(self.mesh)
        l = 1/5
        val1 = gbar * (1 - (2 * (y - 0.8) / l) ** 2)  # inlet parabolic profile
        bcs = [
            # u = 0 on all walls
            DirichletBC(W.sub(0), Constant((0.0,0.0)), [
                BOTTOM_WALL_LEFT,
                BOTTOM_WALL_RIGHT,
                RIGHT_WALL,
                TOP_WALL,
                LEFT_WALL_TOP,
                LEFT_WALL_BOTTOM,
            ]),
            DirichletBC(W.sub(0), as_vector([val1,0.0]), LEFT_INLET),
        ]
        return bcs

    def forward_sp(self):
        sp ={
            "snes_type": "newtonls",
            "mat_type": "aij",
            "snes_monitor": None,
            "ksp_type": "preonly",
            "snes_atol": 1e-6,
            # "ksp_monitor": None,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            # "snes_linesearch_type": "basic",
        }
        return sp

    def adj_sp(self):
        return self.forward_sp()

    def filter_sp(self):
        return self.forward_sp()

    def adj_filter_sp(self):
        return self.forward_sp()
    
    def _forward_sp(self):
        sp = {
            'mat_type': 'matfree',
            'snes_monitor': None,
            #'snes_converged_reason': None,
            'snes_max_it': 20,
            'snes_atol': 1e-6,
            'snes_rtol': 1e-8,
            'snes_stol': 1e-06,
            'ksp_type': 'fgmres',
            # 'ksp_converged_reason': None,
            'ksp_monitor_true_residual': None,
            'ksp_max_it': 300,
            'ksp_atol': 1e-7,
            'ksp_rtol': 1e-9,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_factorization_type': 'full',
            'pc_fieldsplit_0_fields': 1,
            'pc_fieldsplit_1_fields': 0,
            'fieldsplit_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'jacobi',
            'fieldsplit_1': {
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled': {
                    'pc_use_amat': False,
                    'pc_type': 'mg',
                    'pc_mg_type': 'full',
                    'mg_coarse_mat_type': 'aij',
                    'mg_coarse_pc_type': 'lu',
                    'mg_coarse_pc_factor_mat_solver_type': 'mumps',
                    'mg_coarse_mat_mumps_icntl_14': 1000,
                    'mg_levels': {
                        'ksp_convergence_test': 'skip',
                        'ksp_max_it': 5,
                        'ksp_type': 'gmres',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.ASMStarPC',
                    },
                },
            },
        }
        return sp

    def _adj_sp(self):
        sp = {
            'mat_type': 'matfree',
            'snes_monitor': None,
            #'snes_converged_reason': None,
            'snes_max_it': 20,
            'snes_atol': 1e-6,
            'snes_rtol': 1e-8,
            'snes_stol': 1e-06,
            'ksp_type': 'fgmres',
            #'ksp_converged_reason': None,
            'ksp_monitor_true_residual': None,
            'ksp_max_it': 300,
            'ksp_atol': 1e-07,
            'ksp_rtol': 1e-9,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_factorization_type': 'full',
            'pc_fieldsplit_0_fields': 1,
            'pc_fieldsplit_1_fields': 0,
            'fieldsplit_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'jacobi',
            'fieldsplit_1': {
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled': {
                'pc_use_amat': False,
                'pc_type': 'mg',
                'pc_mg_type': 'full',
                'mg_coarse_mat_type': 'aij',
                'mg_coarse_pc_type': 'lu',
                'mg_coarse_pc_factor_mat_solver_type': 'mumps',
                'mg_coarse_mat_mumps_icntl_14': 1000,
                'mg_levels': {
                    'ksp_convergence_test': 'skip',
                    'ksp_max_it': 5,
                    'ksp_type': 'gmres',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.ASMStarPC',
                },
                },
            },
        }
        return sp

    def _filter_sp(self):
        sp = {
            "ksp_type": "cg", # use conjugate gradients
            "ksp_monitor": None, # print info about iteration
            "ksp_rtol": 1.0e-10, # residual relative tolerance
            "pc_type": "mg", # use geometric multigrid
        }
        return sp

    def _adj_filter_sp(self):
        sp = {
            "ksp_type": "cg", # use conjugate gradients
            "ksp_monitor": None, # print info about iteration
            "ksp_rtol": 1.0e-10, # residual relative tolerance
            "pc_type": "mg", # use geometric multigrid
        }
        return sp

    def projection_sp(self):
        sp = {
            # "ksp_type": "cg",
            # "ksp_atol": 1e-6,
            # "ksp_rtol": 1e-6,
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        return sp

if __name__ == "__main__":
    Re            = Constant(1)  # Reynolds number
    gbar          = 1.0           # max inlet/outlet velocity
    dens          = Constant(1.0)  # density
    mu            = dens * gbar / Re
    nu            = 1.0 / Re       # nondimensional viscosity used in the forward problem
    alphaunderbar = 2.5 * mu / (1 / 5**2)   # alpha_min in the original dimensional scaling
    alphabar      = 1e4 * alphaunderbar     # alpha_max in the original dimensional scaling
    target_volume = 1/4
    alpha_init    = 2.5 * mu / (0.1**2)
    r_min         = 0.04                  #filter radius
    gamma         = Constant(1e4)        # augmented lagrangian penalty-coefficient

    c1 = 1e-3
    q_values=(250,)
    iters_per_q=(50,)


    
    problem = Bend2D(Re, gamma, alphaunderbar, alphabar, r_min, mu)


    if problem.rank0:
        os.makedirs("output", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        problem.comm.barrier()

    # Re_v = [1,10, float(Re)]
    # problem.continuation_solve(Re_v, "output")
 
    problem.simpl(tol=1e-5,
                  target_volume=target_volume,
                  q_values=q_values,
                  iters_per_q = iters_per_q,
                  c1=c1,
                  simpl_type="A",
                  max_backtrack=10)