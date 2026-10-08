from firedrake import *
from simpl import *

class Cantilever2D(Compliance):
    length = 2.0
    height = 1.0

    def mesh(self):
        base = RectangleMesh(40, 20, self.length, self.height)
        self.mesh_hierarchy = MeshHierarchy(base, 3)
        return self.mesh_hierarchy[-1]

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
    problem = Cantilever2D(
        mu=Constant(10),
        lmbda=Constant(50),
        E_min=Constant(1e-5),
    )
    target_volume = 0.4

    problem.simpl(
        tol=1e-6,
        target_volume=target_volume,
        q_values=(0.01,0.005),
        iters_per_q=(20,100),
        c1=1e-8,
        simpl_type="A",
        max_backtrack=20,
        descent_tol=1e-8,
        output_dir="output/cantilever2d",
        save_iterates=True,
    )