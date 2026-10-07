from firedrake import *
from firedrake.adjoint import *
from .logging import *
from .simpl import *

class NavierStokes(SiMPL):
    def __init__(self, Re, gamma, alphaunderbar, alphabar, r_min, mu, dens):
        self.mesh = self.mesh()
        self.comm  = self.mesh.comm
        self.rank0 = (self.comm.rank == 0)
        self.Re = Re
        self.gamma = gamma
        self.alphaunderbar = alphaunderbar
        self.alphabar = alphabar
        self.r_min = r_min
        self.mu = mu
        self.dens = dens
        self.q = Constant(1)
        self.setup_parameters = self.setup(self.mesh)

    def alpha_perm(self, rho):
        """Inverse permeability as a function of rho."""
        return self.alphaunderbar + (self.alphabar - self.alphaunderbar) * (1 - rho) / (1 + self.q * rho)
    
    def construct_primal_solvers(self, Res, w, lam, y_test, rho_k_filtered, bcs):
        (u, p) = split(w)
        (v, q) = split(y_test)
        gamma = self.gamma

        J = derivative(Res, w)  #jacobian

        if self.forward_sp()["pc_type"] == "fieldsplit":
            Fp = Res - inner(p/gamma, q)*dx  + inner(div(u)*gamma, div(v))*dx  # preconditioned residual 
        else:
            Fp = Res + inner(div(u)*gamma, div(v))*dx
        Jp = derivative(Fp, w)  # preconditioned jacobian
        

        problem = NonlinearVariationalProblem(Res, w, J=J , Jp=Jp, bcs=bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=self.forward_sp(), pre_apply_bcs=False)

        Jobj = self.construct_Jobj(w, rho_k_filtered)

        adjA = adjoint(derivative(Res, w))
        rhs = -derivative(Jobj, w)
        if self.forward_sp()["pc_type"] == self.adj_sp()["pc_type"]:
            JpT = adjoint(Jp)
        else:
            error("Currently require same pc for adjoint and forward solves.")
        bcs_hom = homogenize(bcs)
        adj_prob = LinearVariationalProblem(adjA, rhs, lam, aP=JpT, bcs=bcs_hom)
        adj_solver = LinearVariationalSolver(adj_prob, solver_parameters=self.adj_sp(), pre_apply_bcs=True)
        return solver, adj_solver, Jobj

    def  construct_Jobj(self, w, rho_k_filtered):
        (u, _) = split(w)
        Jobj = 0.5 * (
            2.0*self.mu * inner(sym(grad(u)), sym(grad(u)))
            + self.alpha_perm(rho_k_filtered) * inner(u, u)
        ) * dx
        return Jobj

    def continuation_solve(self, Re_v, output_dir, save_file=False):
        w = self.setup_parameters[1]
        Re = self.Re
        forward_solver = self.setup_parameters[7]

        info_g(f"Using continuing strategy to reach Re={float(Re)}.")
        Re_final = int(float(Re))
        if Re_v[-1] != Re_final:
            Re_v.append(Re_final)

        u_curr, _ = w.subfunctions

        for Re_ in Re_v:
            info_g(f"Current Re={Re_}")
            Re.assign(Re_)
            info_b("Starting forward solve.")
            forward_solver.solve()
        if save_file:
            VTKFile(f"{output_dir}/Velocity_{float(Re)}.pvd").write(u_curr)


# MTW discretization for Navier-Stokes
class NavierStokesMTW(NavierStokes):
    def primal_function_space(self, mesh):
        U_h = FunctionSpace(mesh, "MTW", 1) # function space velocity
        P_h = FunctionSpace(mesh, "DG", 0)  # function space pressure
        W = MixedFunctionSpace([U_h, P_h])
        return W

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        n = FacetNormal(mesh)
        (u, p) = split(w)
        (v, q) = split(y_test)
        uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u   #flux of u across internal facets to stabilise the advection term

        Res = (
            2/self.Re * inner(sym(grad(u)), sym(grad(v)))*dx
                    - inner(u ,div(outer(v,u)))*dx
                    + inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS
                    - inner(p, div(v))*dx
                    - inner(q, div(u))*dx
                    + self.alpha_perm(rho_k_filtered) * inner(u,v) * dx
            )
        def c_bc(u, v, bid, g):
            if g is None:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
            else:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
            return dot(v, uflux_ext)*ds(bid)
        exterior_markers = set(mesh.exterior_facets.unique_markers)

        for bc in bcs:
            if "DG" in str(bc._function_space):
                continue
            g = bc.function_arg
            bid = bc.sub_domain
            if isinstance(bid, Iterable):
                for marker in bid:
                    exterior_markers.remove(marker)
            else:
                exterior_markers.remove(bid)
            Res += c_bc(u, v, bid, g) 

        for bid in exterior_markers:
            Res += c_bc(u, v, bid, None)
        return Res

# Taylor-Hood
class NavierStokesTH(NavierStokes):
    def primal_function_space(self, mesh):
        U_h = VectorFunctionSpace(mesh, "CG", 2) # function space velocity
        P_h = FunctionSpace(mesh, "CG", 1)  # function space pressure
        W = MixedFunctionSpace([U_h, P_h])
        return W

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        (u, p) = split(w)
        (v, q) = split(y_test)

        Res = (
            2/self.Re * inner(sym(grad(u)), sym(grad(v)))*dx
                    - inner(u ,div(outer(v,u)))*dx
                    - inner(p, div(v))*dx
                    - inner(q, div(u))*dx
                    + self.alpha_perm(rho_k_filtered) * inner(u,v) * dx
            )
        return Res
    
# P1-stabilised
class NavierStokesP1(NavierStokes):
    def primal_function_space(self, mesh):
        U_h = VectorFunctionSpace(mesh, "CG", 1) # function space velocity
        P_h = FunctionSpace(mesh, "CG", 1)  # function space pressure
        W = MixedFunctionSpace([U_h, P_h])
        return W

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        (u, p) = split(w)
        (v, q) = split(y_test)

        Res = (
            2/self.Re * inner(sym(grad(u)), sym(grad(v)))*dx
                    - inner(u ,div(outer(v,u)))*dx
                    + self.alpha_perm(rho_k_filtered) * inner(u,v) * dx
                    - inner(p, div(v))*dx
                    - inner(q, div(u))*dx  
            )
        
        alpha = self.alpha_perm(rho_k_filtered)
        dens = self.dens
        mu = self.mu
        
        # SUPG and PSPG terms from Appendix B in https://arxiv.org/pdf/2207.13695
        R_m = dens * dot(grad(u), u) + grad(p) + alpha * u    
        h = CellDiameter(mesh)

        tau = 1.0/sqrt(4.0*inner(u, u)/h**2 + (12.0 * mu / (dens * h**2))**2 + (alpha / dens)**2)

        F_supg = tau* inner(dot(grad(u), v),R_m)* dx(degree=6)
        F_pspg = (tau / dens**2)* inner(grad(q),R_m)* dx(degree=6)


        return Res + F_supg - F_pspg