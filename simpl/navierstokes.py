from firedrake import *
from firedrake.adjoint import *
from .logging import *
from .simpl import *

class NavierStokesMTW(SiMPL):
    def primal_function_space(self, mesh):
        U_h = FunctionSpace(mesh, "MTW", 1) # function space velocity
        P_h = FunctionSpace(mesh, "DG", 0)  # function space pressure
        W = MixedFunctionSpace([U_h, P_h])
        return W

    def boundary_conditions(self, W):
        raise NotImplementedError
    
    def alpha_perm(self, rho):
        """Inverse permeability as a function of rho."""
        return self.alphaunderbar + (self.alphabar - self.alphaunderbar) * (1 - rho) / (1 + self.q * rho)

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        n = FacetNormal(mesh)
        (u, p) = split(w)
        (v, q_test) = split(y_test)
        uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u   #flux of u across internal facets to stabilise the advection term

        Res = (
            2/self.Re * inner(sym(grad(u)), sym(grad(v)))*dx#(degree=8)
                    - inner(u ,div(outer(v,u)))*dx#(degree=10)
                    + inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS#(degree=8)
                    - inner(p, div(v))*dx#(degree=8)
                    - inner(q_test, div(u))*dx#(degree=8)
                    + self.alpha_perm(rho_k_filtered) * inner(u,v) * dx#(degree=8)
            )
        def c_bc(u, v, bid, g):
            if g is None:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
            else:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
            return dot(v, uflux_ext)*ds(bid)#,degree=10)
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

    def construct_primal_solvers(self, Res, w, lam, y_test, rho_k_filtered, bcs):
        (u, p) = split(w)
        (v, q_test) = split(y_test)
        gamma = self.gamma

        J = derivative(Res, w)  #jacobian

        if self.forward_sp()["pc_type"] == "fieldsplit":
            Fp = Res - inner(p/gamma, q_test)*dx  + inner(div(u)*gamma, div(v))*dx  # preconditioned residual 
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
        w = self.setup_parameters[2]
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