
from firedrake import *
from create_geometry import create_geometry
from simpl import *



class DoublePipe(NavierStokesMeanOutletFlow, NavierStokesP1):

    def mesh(self):
        maxh = 0.008*2
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
            'snes_atol': 1e-5,
            'snes_rtol': 1e-6,
            'snes_stol': 1e-6,
            'snes_monitor': None,
            'ksp_monitor_true_residual': None,
            'ksp_type': 'fgmres',
            'ksp_max_it': 100,
            'ksp_atol': 1e-6,
            'ksp_rtol': 1e-6,
 
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
                'pc_type': 'jacobi',
            },
        }
        return sp

    def filter_sp_mg(self):
        sp = {
            'ksp_type': 'cg',
            'ksp_atol': 1e-5,
            'ksp_rtol': 1e-6,
            'pc_type': 'mg',
            'ksp_monitor': None,
        }
        return sp

if __name__ == "__main__":
    Re            = Constant(1)
    gbar          = 1.0
    dens          = Constant(1.0)
    mu            = dens * gbar / Re
    alphaunderbar = 2.5 * mu / (100**2)
    alphabar      = 2.5 * mu / (0.01**2)
    delta         = 1.5
    volfrac       = 1.0 / 3.0
    target_volume = volfrac * delta
    r_min         = 0.04
    gamma         = Constant(0)

    q_0 = 200
    q_vec = q_0 * np.array([1.0, 0.5, 0.1])
    c1 = 1e-4
    iters_per_q = (5, 5, 50,)
    rtol = 1e-5
    atol = 1e-4

    beta = 1e3
    beta_growth = 1.1

    problem = DoublePipe(
        Re, gamma, alphaunderbar, alphabar, r_min, mu, dens,
        inlet_markers=("left_inlet1", "left_inlet2"),
        outlet_markers=("right_outlet1", "right_outlet2"),
        outlet_fraction_bounds=((0.48, 0.52),),
        beta=beta,
        beta_growth=beta_growth,
    )
    rho_opt, J_augmented, _ = problem.simpl(
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
    info_b(f"Final physical objective: {problem.base_objective_value:.6e}")
    info_b(f"Final augmented objective: {J_augmented:.6e}")
    for i, fraction in enumerate(problem.fractions, start=1):
        info_b(f"Final outlet {i} fraction: {fraction:.4%}")
