from firedrake import *
from .logging import *
from .simpl import *

class _SNES:
    def getLinearSolveIterations(self):
        return 0

class Adj_Solver:
    def __init__(self, lam, w):
        self.snes = _SNES()
        self.lam = lam
        self.w = w

    def solve(self):
        self.lam.assign(-self.w)

class Compliance(SiMPL):

    def __init__(self, mu, lmbda, E_min, r_min, penalty=3.0):
        self.mesh = self.mesh()
        self.comm  = self.mesh.comm
        self.rank0 = (self.comm.rank == 0)   
        self.mu = mu
        self.lmbda = lmbda
        self.E_min = E_min
        self.r_min = r_min
        self.q = Constant(penalty)
        self.setup_parameters = self.setup(self.mesh)

    def primal_function_space(self, mesh):
        return VectorFunctionSpace(mesh, "CG", 1)

    def simp(self, rho):
        """Young's modulus evaluated at the filtered density."""
        return self.E_min + (Constant(1) - self.E_min) * rho**self.q

    def stress(self, u, rho):
        k = self.simp(rho)
        return k * (2.0 * self.mu * sym(grad(u)) + self.lmbda * div(u) * Identity(self.mesh.topological_dimension))

    def load_form(self, v):
        """Return load work, e.g. inner(body_force, v)*dx + inner(traction, v)*ds."""
        raise NotImplementedError

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        return inner(self.stress(w, rho_k_filtered), sym(grad(y_test)) )* dx - self.load_form(y_test)

    def construct_Jobj(self, w, rho_k_filtered):
        return self.load_form(w)

    def construct_primal_solvers(self, Res, w, lam, y_test, rho_k_filtered, bcs):
        a = derivative(Res, w)
        L = self.load_form(y_test)
        problem = LinearVariationalProblem(a, L, w, bcs=bcs)
        solver = LinearVariationalSolver(problem, solver_parameters=self.forward_sp())
        adj_solver = Adj_Solver(lam, w)
        return solver, adj_solver, self.construct_Jobj(w, rho_k_filtered)

    def initialize_save_solutions(self, w, rho_k_filtered, rho_k, output_dir):
        control_vtk = VTKFile(f"{output_dir}/control_iterations.pvd")
        rhofilt_vtk = VTKFile(f"{output_dir}/rho_filtered_iterations.pvd")
        displacement_vtk = VTKFile(f"{output_dir}/displacement_iterations.pvd")
        w.rename("Displacement")
        self.control_vtk = control_vtk
        self.rhofilt_vtk = rhofilt_vtk
        self.displacement_vtk = displacement_vtk

    def save_solutions(self, w, rho_k_filtered, rho_k):
        self.control_vtk.write(rho_k)
        self.rhofilt_vtk.write(rho_k_filtered)
        self.displacement_vtk.write(w)
