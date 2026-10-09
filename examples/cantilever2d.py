from firedrake import *
from simpl import *

class Cantilever2D(Compliance):
    length = 2.0
    height = 1.0

    def __init__(self, mu, lmbda, E_min, *, mesh):
        self._mesh = mesh
        super().__init__(mu, lmbda, E_min)

    def mesh(self):
        return self._mesh

    def boundary_conditions(self, W):
        # RectangleMesh marker 1 is the left edge, marker 2 is the right edge.
        return [DirichletBC(W, Constant((0.0, 0.0)), 1)]

    def load_form(self, v):
        _, y = SpatialCoordinate(self.mesh)
        # A unit downward force distributed over the middle of the right edge.
        loaded = abs(y - self.height / 2.0) <=  0.05
        traction = as_vector((0.0, conditional(loaded, -1.0, 0.0)))
        return inner(traction, v) * ds(2, domain=self.mesh, degree=2)

    def forward_sp(self):
        return {
            "snes_atol": 1e-6,
            "snes_rtol": 1e-7,
            "snes_monitor": None,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    def filter_sp(self):
        return self.forward_sp()

    def adj_filter_sp(self):
        return self.forward_sp()


if __name__ == "__main__":
    # Build one hierarchy so consecutive problems share parent/child meshes.
    base = RectangleMesh(160, 80, Cantilever2D.length, Cantilever2D.height)
    hierarchy = MeshHierarchy(base, 2)
    mu = Constant(10)
    lmbda = Constant(50)
    E_min = Constant(1e-5)
    # Absolute material volume; the cantilever's domain volume is 2.
    target_volume = 0.4
    q_values_per_level = ((0.01,), (0.005,), (0.005,))
    iters_per_level = ((100,), (100,), (100,))
    rtols = (1e-4, 1e-4, 1e-4)
    atols = (1e-4,1e-4,1e-4)

    previous_problem = None
    for level, mesh in enumerate(hierarchy):
        problem = Cantilever2D(mu, lmbda, E_min, mesh=mesh)
        rho_k = problem.setup_parameters[2]
        if previous_problem is not None:
            rho_coarse = previous_problem.setup_parameters[2]
            prolong(rho_coarse, rho_k)
        else:
            rho_k.assign(target_volume/2)

        info_r(f"Grid level {level}: {mesh.num_cells()} local cells")
        problem.simpl(
            rtol=rtols[level],
            atol=atols[level],
            target_volume=target_volume,
            q_values=q_values_per_level[level],
            iters_per_q=iters_per_level[level],
            c1=1e-8,
            simpl_type="A",
            max_backtrack=20,
            descent_tol=1e-8,
            output_dir=f"output/cantilever2d/level_{level}",
            save_iterates=True,
            rho_initial=rho_k,
        )
        previous_problem = problem
