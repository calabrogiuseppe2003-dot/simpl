from firedrake import *
from create_geometry import create_geometry
from simpl import *


class Double2D(NavierStokesMTW):

    def mesh(self):
        maxh = 0.008*4
        delta = 1.5
        ngmesh, markers = create_geometry(delta, maxh)
        base = Mesh(ngmesh,distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)},)
        mh = MeshHierarchy(base, 1)
        self.markers = markers
        return mh[-1]

    def boundary_conditions(self, W):
        markers = self.markers
        y = SpatialCoordinate(self.mesh)[1]
        width = 1.0 / 6.0
        inlet1 = gbar * (1.0 - (2.0 * (y - 0.25) / width)**2)
        inlet2 = gbar * (1.0 - (2.0 * (y - 0.75) / width)**2)
        walls = [
            "bottom", "top", "left_wall_top", "left_wall_bottom",
            "left_wall_middle", "right_wall_bottom", "right_wall_top",
            "right_wall_middle",
        ]
        return [
            DirichletBC(W.sub(0), Constant((0.0, 0.0)), [markers[name] for name in walls]),
            DirichletBC(W.sub(0), as_vector((inlet1, 0.0)), markers["left_inlet1"]),
            DirichletBC(W.sub(0), as_vector((inlet2, 0.0)), markers["left_inlet2"]),
            DirichletBC(W.sub(0), as_vector([inlet1, 0.0]), markers["right_outlet1"]),
            DirichletBC(W.sub(0), as_vector([inlet2, 0.0]), markers["right_outlet2"]),
        ]

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

    def sp_mg(self):
        sp = {
            'mat_type': 'matfree',
            'snes_max_it': 20,
            'snes_atol': 1e-6,
            'snes_rtol': 1e-7,
            'snes_stol': 1e-6,
            'snes_monitor': None,
            'ksp_monitor_true_residual': None,
            'ksp_type': 'fgmres',
            'ksp_max_it': 300,
            'ksp_atol': 1e-7,
            'ksp_rtol': 1e-8,
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
            'ksp_atol': 1e-6,
            'ksp_rtol': 1e-7,
            'pc_type': 'mg',
            'ksp_monitor': None,
        }
        return sp


if __name__ == "__main__":
    Re            = Constant(1000)
    gbar          = 1.0
    dens          = Constant(1.0)
    mu            = dens * gbar / Re
    nu            = 1.0 / Re
    alphaunderbar = 2.5 * mu / (100**2)
    alphabar      = 2.5 * mu / (0.01**2)
    delta         = 1.5
    volfrac       = 1.0/3.0
    target_volume = volfrac * delta
    alpha_init    = 2.5 * mu / (0.1**2)
    r_min         = 0.04
    gamma         = Constant(1e4)

    q_0 = 200
    q_vec = q_0 * np.array([1,0.5,0.1])
    c1 = 1e-4
    iters_per_q = (5,5,50)
    rtol = 1e-5
    atol = 1e-4

    problem = Double2D(Re, gamma, alphaunderbar, alphabar, r_min, mu, dens)
    Re_v = [1, 10, 50] + list(range(100, int(float(Re)) + 1, 300))
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
        descent_tol=1e-9,
        output_dir="output/",
        save_iterates=True,
    )

    info_b(f"\nFinal volume: {float(assemble(rho_opt * dx)):.6f}  (target {target_volume:.6f})")
    info_b(f"Final objective filtered: {J_filtered:.6e}")
