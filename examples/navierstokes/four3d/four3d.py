from firedrake import *
from firedrake import DistributedMeshOverlapType
from boxdouble import create_geometry_3d
from simpl import *


class Four3D(NavierStokesMTW):

    def mesh(self):
        maxh = 0.1
        delta = 1.5
        ngmesh, markers = create_geometry_3d(delta, maxh)
        base = Mesh(ngmesh,distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)},)
        mh = MeshHierarchy(base, 2)
        self.markers = markers
        return mh[-1]

    def boundary_conditions(self, W):
        markers = self.markers
        WALLS = [
            markers["wall_xmin"],
            markers["wall_xmax"],
            markers["wall_ymin"],
            markers["wall_ymax"],
            markers["wall_zmin"],
            markers["wall_zmax"],
        ]
        INLET1 = markers["inlet1"]
        INLET2 = markers["inlet2"]
        INLET3 = markers["inlet3"]
        INLET4 = markers["inlet4"]
        OUTLET1 = markers["outlet1"]
        OUTLET2 = markers["outlet2"]
        OUTLET3 = markers["outlet3"]
        OUTLET4 = markers["outlet4"]

        (x, y, z) = SpatialCoordinate(self.mesh)
        l = 1/6
        r_pipe = l / 2
        rho1_sq = (y - 0.3)**2 + (z - 0.3)**2
        val1 = gbar * (1 - rho1_sq / r_pipe**2)
        rho2_sq = (y - 0.3)**2 + (z - 0.7)**2
        val2 = gbar * (1 - rho2_sq / r_pipe**2)
        rho3_sq = (y - 0.7)**2 + (z - 0.3)**2
        val3 = gbar * (1 - rho3_sq / r_pipe**2)
        rho4_sq = (y - 0.7)**2 + (z - 0.7)**2
        val4 = gbar * (1 - rho4_sq / r_pipe**2)

        bcs = [
            DirichletBC(W.sub(0), as_vector([val1, 0, 0]), [INLET1, OUTLET1]),
            DirichletBC(W.sub(0), as_vector([val2, 0, 0]), [INLET2, OUTLET2]),
            DirichletBC(W.sub(0), as_vector([val3, 0, 0]), [INLET3, OUTLET3]),
            DirichletBC(W.sub(0), as_vector([val4, 0, 0]), [INLET4, OUTLET4]),
            # Use inlet markers alone above for natural outlet conditions.
            DirichletBC(W.sub(0), Constant((0, 0, 0)), WALLS),
        ]
        return bcs

    def forward_sp(self):
        return self.sp_mg()

    def adj_sp(self):
        return self.sp_mg()

    def filter_sp(self):
        return self.filter_sp_mg()

    def adj_filter_sp(self):
        return self.filter_sp_mg()

    def sp_lu(self):
        sp ={
            "mat_type": "aij",
            "snes_monitor": None,
            "ksp_type": "preonly",
            "snes_atol": 1e-6,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        return sp

    def forward_sp_mg(self):
        sp = {
            'mat_type': 'matfree',
            'snes_monitor': None,
            'snes_max_it': 20,
            'snes_atol': 5e-6,
            'snes_rtol': 1e-12,
            'snes_stol': 1e-6,
            'snes_monitor': None,
            'ksp_monitor': None,
            'ksp_type': 'fgmres',
            'ksp_monitor_true_residual': None,
            'ksp_max_it': 300,
            'ksp_atol': 1e-7,
            'ksp_rtol': 1e-10,
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
                        'ksp_type': 'fgmres',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.ASMStarPC',
                    },
                },
            },
        }
        return sp

    def filter_sp_mg(self):
        sp = {
            'ksp_type': 'cg',
            'ksp_rtol': 1e-10,
            'pc_type': 'mg',
            'ksp_monitor': None,
        }
        return sp

if __name__ == "__main__":
    Re            = Constant(2000)
    gbar          = 1.0
    dens          = Constant(1.0)
    mu            = dens * gbar / Re
    nu            = 1.0 / Re
    alphaunderbar = 0
    alphabar      = 2.5 * mu / (0.01**2)
    volfrac       = 0.0873
    delta         = 1.5
    target_volume = volfrac * delta
    alpha_init    = 2.5 * mu / (0.1**2)
    r_min         = 0.04
    gamma         = Constant(1e4)

    q_0 = 0.01*((alphabar - alpha_init) - volfrac * (alphabar - alphaunderbar)) / (volfrac * (alpha_init - alphaunderbar))
    q_vec =q_0 * np.array([1,0.5,0.1])
    c1 = 1e-3
    iters_per_q = (5,5,50)
    rtol = 3e-5
    atol = 1e-15

    problem = Four3D(Re, gamma, alphaunderbar, alphabar, r_min, mu, dens)
    Re_v = [1, 10, 100] + list(range(200, int(float(Re)) + 1, 400))
    problem.continuation_solve(Re_v, q_0, target_volume, "output/", save_iterates=True)

    rho_opt, J_filtered, _ = problem.simpl(
        rtol=rtol,
        atol=atol,
        target_volume=target_volume,
        q_values=q_vec,
        iters_per_q=iters_per_q,
        c1=c1,
        simpl_type="A",
        max_backtrack=10,
        descent_tol=1e-8,
        output_dir="output/",
        save_iterates=True,
    )

    info_b(f"\nFinal volume: {float(assemble(rho_opt * dx)):.6f}  (target {target_volume:.6f})")
    info_b(f"Final objective filtered: {J_filtered:.6e}")
