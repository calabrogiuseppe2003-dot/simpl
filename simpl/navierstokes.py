from firedrake import *
from firedrake.adjoint import *
from .logging import *
from .simpl import *

class NavierStokes(SiMPL):
    def __init__(self, Re, gamma, alphaunderbar, alphabar, r_min, mu, dens):
        self.Re = Re
        self.gamma = gamma
        self.alphaunderbar = alphaunderbar
        self.alphabar = alphabar
        self.mu = mu
        self.dens = dens
        super().__init__(r_min)

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
            Fp = Res #+ inner(div(u)*gamma, div(v))*dx
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


        tm = self.transfer_manager()
        if tm is not None:
            solver.set_transfer_manager(tm)
            adj_solver.set_transfer_manager(tm)
        return solver, adj_solver, Jobj

    def  construct_Jobj(self, w, rho_k_filtered):
        (u, _) = split(w)
        Jobj = 0.5 * (
            2.0*self.mu * inner(sym(grad(u)), sym(grad(u)))
            + self.alpha_perm(rho_k_filtered) * inner(u, u)
        ) * dx
        return Jobj

    def continuation_solve(self, Re_v, q_0, target_volume, output_dir="output", save_iterates=False):
        domain_volume = float(assemble(Constant(1.0) * dx(domain=self.mesh)))
        self.setup_parameters[2].assign(target_volume/domain_volume)
        self.setup_parameters[8].solve()
        self.q.assign(float(q_0))
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
        if save_iterates:
            VTKFile(f"{output_dir}/Velocity_{float(Re)}.pvd").write(u_curr)

    def initialize_save_solutions(self, w, rho_k_filtered, rho_k, output_dir):
        control_vtk = VTKFile(f"{output_dir}/control_iterations.pvd")
        rhofilt_vtk = VTKFile(f"{output_dir}/rho_filtered_iterations.pvd")
        velocities_vtk = VTKFile(f"{output_dir}/velocity_iterations.pvd")
        control_vtk.write(rho_k)
        (u, _) = w.subfunctions
        u.rename("Velocity")
        self.control_vtk = control_vtk
        self.rhofilt_vtk = rhofilt_vtk
        self.velocities_vtk = velocities_vtk

    def save_solutions(self, w, rho_k_filtered, rho_k):
        (u, p) = w.subfunctions
        self.control_vtk.write(rho_k)
        self.rhofilt_vtk.write(rho_k_filtered)
        self.velocities_vtk.write(u)




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
                    - inner(u ,div(outer(v,u)))*dx(metadata={"max_quadrature_degree": 6})
                    + inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS(metadata={"max_quadrature_degree": 6})
                    - inner(p, div(v))*dx
                    - inner(q, div(u))*dx
                    + self.alpha_perm(rho_k_filtered) * inner(u,v) * dx
            )
        def c_bc(u, v, bid, g):
            if g is None:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
            else:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
            return dot(v, uflux_ext)*ds(bid, metadata={"max_quadrature_degree": 6})
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
                    + inner(dot(grad(u), u) , v)*dx(metadata={"max_quadrature_degree": 6})
                    # - inner(u ,div(outer(v,u)))*dx(metadata={"max_quadrature_degree": 6})
                    - inner(p, div(v))*dx
                    - inner(q, div(u))*dx
                    + self.alpha_perm(rho_k_filtered) * inner(u,v) * dx
            )
        return Res

# Alfeld-split SV
class NavierStokesSV(NavierStokesTH):
    def primal_function_space(self, mesh):
        U_h = VectorFunctionSpace(mesh, "CG", 3, variant="alfeld") # function space velocity
        P_h = FunctionSpace(mesh, "DG", 2, variant="alfeld")  # function space pressure
        W = MixedFunctionSpace([U_h, P_h])
        return W
    def transfer_manager(self):
        return CoarsePatchTransferManager()

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
                    # - inner(u ,div(outer(v,u)))*dx(metadata={"max_quadrature_degree": 6})
                    + inner(dot(grad(u), u) , v)*dx(metadata={"max_quadrature_degree": 6})
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

        tau = 1e0/sqrt(4.0*inner(u, u)/h**2 + (12.0 * mu / (dens * h**2))**2 + (alpha / dens)**2)

        # We scale our equations a bit differently
        tau = tau / dens

        F_supg = tau* inner(dot(grad(u), v),R_m)* dx(metadata={"max_quadrature_degree": 6})
        F_pspg = (tau / dens)* inner(grad(q),R_m)* dx(metadata={"max_quadrature_degree": 6})
        return Res + F_supg - F_pspg


# P1-stabilised + Upwind Flux
class NavierStokesP1_Flux(NavierStokesP1):

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        (u, p) = split(w)
        (v, q) = split(y_test)
        
        n = FacetNormal(mesh)
        uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u

        Res = (
            2/self.Re * inner(sym(grad(u)), sym(grad(v)))*dx
                    - inner(u ,div(outer(v,u)))*dx(metadata={"max_quadrature_degree": 6})
                    # + inner(dot(grad(u), u) , v)*dx(metadata={"max_quadrature_degree": 6})
                    + inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS(metadata={"max_quadrature_degree": 6})
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

        tau = 1e0/sqrt(4.0*inner(u, u)/h**2 + (12.0 * mu / (dens * h**2))**2 + (alpha / dens)**2)

        # We scale our equations a bit differently
        tau = tau / dens

        F_supg = tau* inner(dot(grad(u), v),R_m)* dx(metadata={"max_quadrature_degree": 6})
        F_pspg = (tau / dens)* inner(grad(q),R_m)* dx(metadata={"max_quadrature_degree": 6})


        def c_bc(u, v, bid, g):
            if g is None:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
            else:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
            return dot(v, uflux_ext)*ds(bid, metadata={"max_quadrature_degree": 6})
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

        return Res + F_supg - F_pspg


class NavierStokesMeanOutletFlow(NavierStokes):
    """Navier-Stokes mixin with augmented-Lagrangian outlet-flow bands.

    Compose this class before a discretization class, for example
    ``class Problem(NavierStokesMeanOutletFlow, NavierStokesP1)``. Marker
    entries may be mesh marker numbers or keys in ``self.markers``. Supply
    one ``(lower, upper)`` fraction band for each of the first N-1 outlets;
    conservation determines the final outlet fraction.
    """

    def __init__(self, *args, inlet_markers, outlet_markers,
                 outlet_fraction_bounds, beta=1e4, beta_growth=2.0):
        """Store flow constraints before SiMPL sets up the mesh and solvers."""
        self.inlet_markers = inlet_markers
        self.outlet_markers = outlet_markers
        self.outlet_fraction_bounds = outlet_fraction_bounds
        self.beta = beta
        self.beta_growth = beta_growth
        self.multipliers = [0.0 for _ in outlet_fraction_bounds]
        self.flux_coefficients = [Constant(0.0) for _ in outlet_fraction_bounds]
        self.inlet_flow = None
        super().__init__(*args)

    def boundary_id(self, marker):
        """Resolve marker names after the concrete problem has created its mesh."""
        if isinstance(marker, str):
            return self.markers[marker]
        return marker

    def construct_Jobj(self, w, rho_k_filtered):
        """Called during primal and adjoint setup; build the flux forms once."""
        self.base_objective = super().construct_Jobj(w, rho_k_filtered)
        u, _ = split(w)
        n = FacetNormal(self.mesh)
        inlet_ids = [self.boundary_id(marker) for marker in self.inlet_markers]
        outlet_ids = [self.boundary_id(marker) for marker in self.outlet_markers]

        self.inlet_flow_form = -dot(u, n) * ds(inlet_ids[0])
        for marker in inlet_ids[1:]:
            self.inlet_flow_form = self.inlet_flow_form - dot(u, n) * ds(marker)
        self.outlet_flow_forms = [dot(u, n) * ds(marker) for marker in outlet_ids]

        # The mutable coefficients provide dJ_AL/dQ_i to the adjoint RHS.
        J = self.base_objective
        for coefficient, flow_form in zip(self.flux_coefficients, self.outlet_flow_forms):
            J += coefficient * flow_form
        return J

    def augmented_objective(self):
        """Evaluate J_AL and refresh its outlet derivatives for the next adjoint."""
        value = self.base_objective_value
        self.fractions = [flow / self.inlet_flow for flow in self.outlet_flows]
        self.constraint_values = []
        for i, (lower, upper) in enumerate(self.outlet_fraction_bounds):
            midpoint = (lower + upper) / 2.0
            halfwidth = (upper - lower) / 2.0
            deviation = self.fractions[i] - midpoint
            g = deviation**2 - halfwidth**2
            active = max(0.0, self.multipliers[i] + self.beta * g)
            value += (active**2 - self.multipliers[i]**2) / (2.0 * self.beta)
            self.flux_coefficients[i].assign(
                2.0 * active * deviation / self.inlet_flow
            )
            self.constraint_values.append(g)
        return value

    def build_functional(self, filter_solver, forward_solver, Jobj):
        """Called for the initial design and line-search trials; solve the PDE and evaluate J_AL."""
        self.base_objective_value = float(super().build_functional(
            filter_solver, forward_solver, self.base_objective
        ))
        if self.inlet_flow is None:
            self.inlet_flow = float(assemble(self.inlet_flow_form))
        self.outlet_flows = [float(assemble(form)) for form in self.outlet_flow_forms]
        return self.augmented_objective()

    def on_step_accepted(self, objective_value):
        """Called only after a line-search step is accepted; update multipliers and penalty."""
        self.multipliers = [
            max(0.0, multiplier + self.beta * g)
            for multiplier, g in zip(self.multipliers, self.constraint_values)
        ]
        if any(g > 0.0 for g in self.constraint_values):
            self.beta *= self.beta_growth
        value = self.augmented_objective()
        fractions = ", ".join(
            f"outlet {i + 1}={fraction:.4%}"
            for i, fraction in enumerate(self.fractions)
        )
        lambdas = ", ".join(f"{lam_i:.3e}" for lam_i in self.multipliers)
        info_g(f"    Outlet fractions: {fractions}  lambda=[{lambdas}]  beta={self.beta:.3e}")
        return value

    def additional_convergence_check(self):
        """Called by SiMPL's stopping check; require every specified band to hold."""
        return all(g <= 0.0 for g in self.constraint_values)
