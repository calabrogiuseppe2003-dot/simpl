from firedrake import *
from netgen.occ import *
from meshgenbend import create_geometry_bend
from simpl import *

class Bend2D(NavierStokesMTW):

    def __init__(self, Re, gamma, alphaunderbar, alphabar, r_min, mu, dens, mesh, markers, level):
        self._mesh = mesh
        self.markers = markers
        self.level = level
        super().__init__(Re, gamma, alphaunderbar, alphabar, r_min, mu, dens)

    def mesh(self):
        return self._mesh

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
        if self.level > 1:
            return self.forward_sp_mg()
        else:
            return self.forward_sp_lu()

    def adj_sp(self):
        if self.level > 1:
            return self.adj_sp_mg()
        else:
            return self.forward_sp_lu()

    def filter_sp(self):
        if self.level > 1:
            return self.filter_sp_mg()
        else:
            return self.forward_sp_lu()

    def adj_filter_sp(self):
        if self.level > 1:
            return self.adj_filter_sp_mg()
        else:
            return self.forward_sp_lu()


    def forward_sp_lu(self):
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
    
    def forward_sp_mg(self):
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

    def adj_sp_mg(self):
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

    def filter_sp_mg(self):
        sp = {
            "ksp_type": "cg", # use conjugate gradients
            "ksp_monitor": None, # print info about iteration
            "ksp_rtol": 1.0e-6, # residual relative tolerance
            "pc_type": "mg", # use geometric multigrid
        }
        return sp

    def adj_filter_sp_mg(self):
        sp = {
            "ksp_type": "cg", # use conjugate gradients
            "ksp_monitor": None, # print info about iteration
            "ksp_rtol": 1.0e-6, # residual relative tolerance
            "pc_type": "mg", # use geometric multigrid
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
    q_values_per_level = ((250,), (250,), (250,))
    iters_per_level = ((50,), (50,), (50,))
    rtols = (1e-3, 1e-7, 1e-7)
    atols = (1e-4, 1e-2, 1e-2)

    # Build the bend and its boundary markers once for the whole sequence.
    ngmesh, markers = create_geometry_bend(0.02)
    base = Mesh(ngmesh,distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)},)
    hierarchy = MeshHierarchy(base, 2)

    previous_problem = None
    transfer_manager = None
    for level, mesh in enumerate(hierarchy):
        problem = Bend2D(Re, gamma, alphaunderbar, alphabar, r_min, mu, dens, mesh, markers, level)
        w = problem.setup_parameters[1]
        rho_k = problem.setup_parameters[2]
        forward_solver = problem.setup_parameters[7]
        rho_initial = None
        if previous_problem is None:
            transfer_manager = forward_solver._ctx.transfer_manager
        else:
            forward_solver.set_transfer_manager(transfer_manager)
            w_coarse = previous_problem.setup_parameters[1]
            rho_coarse = previous_problem.setup_parameters[2]
            transfer_manager.prolong(rho_coarse, rho_k)
            # Prolong velocity and pressure
            for wc_i, wf_i in zip(w_coarse.subfunctions, w.subfunctions):
                transfer_manager.prolong(wc_i, wf_i)
            rho_initial = rho_k

        info_r(f"Grid level {level}: {mesh.num_cells()} local cells")

        problem.simpl(
            rtol=rtols[level],
            atol=atols[level],
            target_volume=target_volume,
            q_values=q_values_per_level[level],
            iters_per_q=iters_per_level[level],
            c1=c1,
            simpl_type="A",
            max_backtrack=10,
            output_dir=f"output/bend2d/level_{level}",
            save_iterates=True,
            rho_initial=rho_initial,
        )
        previous_problem = problem
