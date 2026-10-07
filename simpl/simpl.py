from firedrake import *
from firedrake.adjoint import *
from .logging import *
import numpy as np
from firedrake.petsc import PETSc
import os
import contextlib
from collections.abc import Iterable
import csv

class SiMPL:
    def __init__(self, r_min):
        self.mesh = self.mesh()
        self.comm  = self.mesh.comm
        self.rank0 = (self.comm.rank == 0)
        self.r_min = r_min
        self.q = Constant(1)
        self.setup_parameters = self.setup(self.mesh)

    def setup(self, mesh):
        A, F = self.density_function_spaces(mesh)
        W = self.primal_function_space(mesh)

        rho_k = Function(A)  # density function
        rho_k_filtered = Function(F)  # filtered density
        rho_k_filtered.rename("Filtered Density")
        w = Function(W)  # create function to hold the solution (u,p)

        y_test = TestFunction(W)
        lam = Function(W)  # primal adjoint
        lam2 = Function(F) # filter adjoint
        g_k = Function(A)

        projection = self.construct_projection_solver(A, lam2, g_k)

        bcs = self.boundary_conditions(W)
        Res = self.forward_form(mesh, rho_k_filtered, w, y_test, bcs)
        forward_solver, adj_solver, Jobj = self.construct_primal_solvers(Res, w, lam, y_test, rho_k_filtered, bcs)


        # dJ/d(rho_filtered) (built once, symbolic)
        dFdrhof = derivative(Res, rho_k_filtered)
        dJdrhof = derivative(Jobj, rho_k_filtered) + action(adjoint(dFdrhof), lam)

        filter_solver, filter_adj_solver = self.construct_filter_solvers(F, rho_k, rho_k_filtered, lam2, dJdrhof)


        info_g(f"Primal DoFs: {W.dim()}, Density DoFs: {A.dim()}")

        return A, w, rho_k, rho_k_filtered, g_k, Jobj, projection, forward_solver, filter_solver, adj_solver, filter_adj_solver

    def density_function_spaces(self, mesh):
        A    = FunctionSpace(mesh, "DG", 0)  # function space for rho
        F    = FunctionSpace(mesh, "CG", 1)  # function space for filtered rho
        return A, F

    def primal_function_space(self, mesh):
        raise NotImplementedError

    def boundary_conditions(self, W):
        raise NotImplementedError

    def forward_sp(self):
        raise NotImplementedError

    def adj_sp(self):
        raise NotImplementedError

    def filter_sp(self):
        raise NotImplementedError

    def adj_filter_sp(self):
        raise NotImplementedError

    def projection_sp(self):
        return {"ksp_type": "preonly", "pc_type": "jacobi"}

    def construct_projection_solver(self, A, lam2, g_k):
        a_test = TestFunction(A)
        b_trial = TrialFunction(A)
        a = inner(a_test, b_trial)*dx
        L = inner(lam2, a_test)*dx
        problem = LinearVariationalProblem(a, L, g_k)
        solver = LinearVariationalSolver(problem, solver_parameters = self.projection_sp())
        return solver

    def forward_form(self, mesh, rho_k_filtered, w, y_test, bcs):
        raise NotImplementedError

    def construct_primal_solvers(self, Res, w, lam, y_test, rho_k_filtered, bcs):
        raise NotImplementedError

    def construct_filter_solvers(self, F, rho_k, rho_k_filtered, lam2, dJdrhof):
        v_test = TestFunction(F)
        rho_trial = TrialFunction(F)
        a = self.r_min**2 * inner(grad(rho_trial), grad(v_test)) * dx + rho_trial * v_test * dx
        L = rho_k * v_test * dx#(degree=10)
        problem = LinearVariationalProblem(a, L, rho_k_filtered)
        solver = LinearVariationalSolver(problem, solver_parameters = self.filter_sp())
        # Filter adjoint problem (built once)
        filter_adj_prob = LinearVariationalProblem(a, dJdrhof, lam2)
        filter_adj_solver = LinearVariationalSolver(filter_adj_prob, solver_parameters=self.adj_filter_sp())
        return solver, filter_adj_solver

    def  construct_Jobj(self, w, rho_k_filtered):
        raise NotImplementedError

    def illinois(self, f, a, b, tol=1e-8, maxiter=100):
        fa = f(a)
        fb = f(b)

        if abs(fa) < tol:
            return a
        if abs(fb) < tol:
            return b
        if fa * fb > 0:
            raise ValueError("Opposite sign needed for a and b.")

        for _ in range(maxiter):
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            if abs(fc) < tol:
                return c
            if fa * fc < 0:
                b = c
                fb = fc
                fa *= 0.5
            else:
                a = c
                fa = fc
                fb *= 0.5

        return c

    def sigma(self, psi):
        return 1.0 / (1.0 + exp(-psi))

    def sigma_inv(self, rho):
        return ln(rho / (1.0 - rho))

    def divergence(self, div_scratch,  rho, q_ref):
        """KL-divergence-like term for type-B line search."""
        div_scratch.interpolate(rho * ln(rho / q_ref) + (1 - rho) * ln((1 - rho) / (1 - q_ref)))
        return assemble(div_scratch * dx)

    # ------------------------------------------------------------------
    # KKT estimator
    # ------------------------------------------------------------------
    def kkt_error(self, alpha_c, tmp_var, psi_new, psi_old, alpha_step, rho_old):
        eps = 1e-8
        alpha_c.assign(alpha_step)
        tmp_var.interpolate((psi_new - psi_old) / alpha_c)  # approximate Lagrange multiplier

        tmp_var.interpolate(max_value(-rho_old * tmp_var, (1 - rho_old) * tmp_var))

        return assemble(tmp_var * dx)

    # ------------------------------------------------------------------
    # Generalised Barzilai-Borwein step size
    # ------------------------------------------------------------------
    def alpha_gbb(self, tmp_var, psi_k, psi_prev, rho_k, rho_old, g_k, g_prev, alpha_prev):     
        tmp_var.interpolate((psi_k - psi_prev) * (rho_k - rho_old))
        num = assemble(tmp_var * dx)
        tmp_var.interpolate((g_k - g_prev) * (rho_k - rho_old))
        den = abs(assemble(tmp_var * dx))

        if num <= 0.0:
            raise ValueError("Negative numerator in GBB step size calculation.")

        return np.sqrt((num / den) * alpha_prev)

    # ------------------------------------------------------------------
    # Volume projection
    # ------------------------------------------------------------------
    def find_mu(self, alpha_c, psi_half, psi_k, g_k, vol_check_f, target_volume, mu_c, alpha_step):
        alpha_c.assign(alpha_step)
        psi_half.interpolate(psi_k - alpha_c * g_k)

        vol_check_f.interpolate(self.sigma(psi_half))
        if assemble(vol_check_f * dx) <= target_volume:  # inactive constraint
            return 0.0

        vol = self.sigma(psi_half - alpha_c * mu_c) * dx
        def residual(mu_val):
            mu_c.assign(mu_val)
            return assemble(vol) - target_volume

        with g_k.dat.vec_ro as gvec:
            _, min_val = gvec.min()
        b = float(-min_val)

        if b <= 0.0:
            info_r("Warning: non-positive b in find_mu. Returning 0.0.")
            return 0.0

        return self.illinois(residual, 0.0, b)

    def build_functional(self, filter_solver, forward_solver, Jobj):
        info_b("Starting filter solve.")
        filter_solver.solve()
        info_b("Starting forward solve.")
        forward_solver.solve()
        return assemble(Jobj)


    def compute_derivative(self, adj_solver, filter_adj_solver):
        """Solve the two adjoint systems (already built/factorised once
        above) and return the reduced gradient as an assembled cofunction."""
        info_b("Starting adjoint solve.")
        adj_solver.solve()
        info_b("Starting adjoint filter solve.")
        filter_adj_solver.solve()

    def initialise_save_solutions(self, rho_k, rho_k_filtered, w):
        raise NotImplementedError

    def save_solutions(self, rho_k, rho_k_filtered, w):
        raise NotImplementedError

    def simpl(
        self,
        tol,
        target_volume,
        q_values=(0.01, 0.1),
        iters_per_q=(18, 30),
        c1=1e-3,
        simpl_type="A",
        max_backtrack=50,
        descent_tol=None,
        output_dir="output",
        alpha_initial=None,
        save_iterates=False,
    ):
        if len(q_values) != len(iters_per_q):
            raise ValueError("q_values and iters_per_q must have the same length.")
        if simpl_type not in ("A", "B"):
            raise ValueError("simpl_type must be 'A' or 'B'.")

        if descent_tol is None:
            descent_tol = tol

        (A, w, rho_k, rho_k_filtered, g_k, Jobj, 
         projection, forward_solver, filter_solver, adj_solver, 
         filter_adj_solver) = self.setup_parameters
        # ------------------------------------------------------------------
        # Preallocate all working Functions BEFORE the closures that use them
        # ------------------------------------------------------------------
        psi_k = Function(A, name="psi_k")
        psi_prev = Function(A, name="psi_prev")
        g_prev = Function(A, name="g_prev")
        psi_half = Function(A, name="psi_half")
        psi_new = Function(A, name="psi_new")
        rho_old = Function(A, name="rho_old")

        tmp_var = Function(A)

        difference = Function(A, name="Difference")
        mu_c = Constant(0.0)

        alpha_c = Constant(1.0)
        mu_val_c = Constant(0.0)



        # ------------------------------------------------------------------
        # Initialise
        # ------------------------------------------------------------------
        domain_volume = float(assemble(Constant(1.0) * dx(domain=self.mesh)))
        rho_k.assign(target_volume/domain_volume)
        psi_k.interpolate(self.sigma_inv(rho_k))

        if self.rank0:
            os.makedirs(output_dir, exist_ok=True)
        self.comm.barrier()

        if save_iterates:
            self.initialize_save_solutions(w, rho_k_filtered, rho_k, output_dir)

        log_path = f"{output_dir}/iterations.csv"
        stage_summary_path = f"{output_dir}/stage_summary.csv"

        projection_its_history = []           # CG its, mass-matrix inversion (projection representative)
        filter_fwd_its_history = []      # CG its, forward filter solve
        filter_adj_its_history = []      # CG its, adjoint filter solve
        adj_ns_its_history = []          # FGMRES its, adjoint Navier-Stokes solve
        newton_its_history = []          # Newton its, forward Navier-Stokes solve
        krylov_per_newton_history = []   # average FGMRES its per Newton step, forward  solve

        log_ctx = open(log_path, "w", newline="") if self.rank0 else contextlib.nullcontext()
        stage_ctx = open(stage_summary_path, "w", newline="") if self.rank0 else contextlib.nullcontext()
        with log_ctx as log_file, stage_ctx as stage_file:
            writer = csv.writer(log_file) if self.rank0 else None
            stage_writer = csv.writer(stage_file) if self.rank0 else None
            if self.rank0:
                writer.writerow([
                    "stage", "iter", "kkt", "descent", "J", "alpha", "volume", "backtracking",
                    "newton_its", "krylov_its", "krylov_per_newton",
                    "projection_its", "filter_fwd_its", "filter_adj_its", "adj_ns_its",
                ])
                stage_writer.writerow([
                    "stage", "q_value", "n_iterations_requested", "n_iterations_run", "converged",
                    "final_kkt", "final_descent",
                ])

            # ------------------------------------------------------------------
            # Outer loop over q-continuation stages
            # ------------------------------------------------------------------
            for stage, (q_value, niter) in enumerate(zip(q_values, iters_per_q), start=1):
                info_g(f"\n{'=' * 60}")
                info_g(f"Stage {stage}: q = {float(q_value)}, max_iter = {niter}")
                info_g(f"{'=' * 60}")

                self.q.assign(q_value)

                alpha_prev = None
                converged = False
                g_initialised = False
                n_bt = 0
                k = -1  # in case niter == 0, so n_iterations_run below is well defined
                kkt = float("nan")
                descent_val = float("nan")

                for k in range(niter):
                    if k == 0:
                        J_current = float(self.build_functional(filter_solver, forward_solver, Jobj))

                    # ---- Gradient ------------------------------------------
                    # Update lam2 to contain new descent direction
                    self.compute_derivative(adj_solver, filter_adj_solver)

                    adj_ns_its = adj_solver.snes.getLinearSolveIterations()      # FGMRES its, adjoint forward
                    filter_adj_its = filter_adj_solver.snes.getLinearSolveIterations()  # CG its, adjoint filter

                    info_b("Starting projection solve.")
                    # Project descent direction lam2 into FEM space of g_k
                    projection.solve()

                    projection_its = projection.snes.getLinearSolveIterations()

                    with g_k.dat.vec_ro as gvec:
                        gnorm = gvec.norm(PETSc.NormType.NORM_INFINITY)

                    if gnorm < 1e-14:
                        info_r(f"  k={k}: zero gradient — stopping.")
                        converged = True
                        break

                    # ---- Step size -----------------------------------------
                    if not g_initialised:
                        if alpha_initial is None:
                            alpha_step = 1.0 / gnorm
                        else:
                            alpha_step = self.alpha_gbb(
                                tmp_var, psi_k, psi_prev, rho_k,
                                rho_old, g_k, g_prev, alpha_initial,
                            )
                    else:
                        alpha_step = self.alpha_gbb(
                            tmp_var, psi_k, psi_prev, rho_k,
                            rho_old, g_k, g_prev, alpha_prev,
                        )

                    rho_old.assign(rho_k)

                    # ---- Armijo backtracking --------------------------------
                    if k > 0 and n_bt == 0:
                        alpha_step *= 1.5
                    n_bt = 0
                    while True:
                        mu_val = self.find_mu(alpha_c, psi_half, psi_k, g_k, tmp_var, target_volume, mu_c, alpha_step)
                        mu_val_c.assign(mu_val)
                        alpha_c.assign(alpha_step)

                        psi_new.interpolate(psi_half - alpha_c * mu_val_c)
                        rho_k.interpolate(self.sigma(psi_new))

                        J_new = float(self.build_functional(filter_solver, forward_solver, Jobj))
                        difference.interpolate(rho_k - rho_old)
                        descent = assemble(inner(g_k, difference)*dx)

                        if descent > 0:
                            break

                        if simpl_type == "A":
                            armijo_rhs = J_current + c1 * descent
                        else:
                            armijo_rhs = J_current + descent + (1.0 / alpha_step) * self.divergence(tmp_var, rho_k, rho_old)

                        if J_new <= armijo_rhs:
                            break

                        alpha_step *= 0.5
                        n_bt += 1
                        info_r(f"Step size does not satisfy Armijo condition. Backtracking iteration {n_bt}.")
                        if n_bt > max_backtrack:
                            break

                    if descent > 0:
                        info_r(f"  k={k}: non-descent direction (descent={descent:.3e})")
                        break
                    if n_bt > max_backtrack:
                        info_r(f"  k={k}: maximum backtracking iterations reached.")
                        break
                    if alpha_step < 1e-8:
                        info_r(f"  k={k}: step size too small during backtracking.")
                        break

                    nonlinear_its = forward_solver.snes.getIterationNumber()
                    linear_its = forward_solver.snes.getLinearSolveIterations()
                    krylov_per_newton = (linear_its / nonlinear_its) if nonlinear_its > 0 else 0.0

                    filter_fwd_its = filter_solver.snes.getLinearSolveIterations()

                    # ---- KKT residual --------------------------------------
                    kkt = self.kkt_error(alpha_c, tmp_var, psi_new, psi_k, alpha_step, rho_k)
                    if k == 0 and stage == 1:
                        kkt0 = max(abs(kkt), 1e-16)
                        kkt_rel = 1.0
                    else:
                        kkt_rel = abs(kkt) / kkt0
                    descent_val = float(descent)
                    vol = assemble(rho_k * dx)
                    info_g(
                        f"  k={k:3d}  J={J_new:.6e}  "
                        f"KKT={kkt:.3e}  descent={descent_val:.3e}  vol={vol:.4f}  α={alpha_step:.3e}  bt={n_bt}  "
                        f"Newton={nonlinear_its}  Krylov={linear_its} ({krylov_per_newton:.1f}/Newton)  "
                        f"Projection={projection_its}  FiltFwd={filter_fwd_its}  FiltAdj={filter_adj_its}  Adj={adj_ns_its}"
                    )
                    if self.rank0:
                        writer.writerow([
                            stage,
                            k,
                            float(kkt),
                            float(descent_val),
                            float(J_new),
                            float(alpha_step),
                            float(vol),
                            n_bt,
                            int(nonlinear_its),
                            int(linear_its),
                            float(krylov_per_newton),
                            int(projection_its),
                            int(filter_fwd_its),
                            int(filter_adj_its),
                            int(adj_ns_its),
                        ])
                        log_file.flush()

                        projection_its_history.append(projection_its)
                        filter_fwd_its_history.append(filter_fwd_its)
                        filter_adj_its_history.append(filter_adj_its)
                        adj_ns_its_history.append(adj_ns_its)
                        newton_its_history.append(nonlinear_its)
                        krylov_per_newton_history.append(krylov_per_newton)

                    # ---- Update memory -------------------------------------
                    psi_prev.assign(psi_k)
                    g_prev.assign(g_k)
                    alpha_prev = alpha_step
                    g_initialised = True

                    # ---- Accept step ---------------------------------------
                    J_current = J_new
                    psi_k.assign(psi_new)

                    if save_iterates:
                        self.save_solutions(w, rho_k_filtered, rho_k)

                    # ---- Convergence check ---------------------------------
                    if kkt_rel <= tol or descent_val >= -descent_tol:
                        info_g(
                            f"  Stage {stage}: converged (KKT + descent) in {k + 1} iterations."
                        )
                        converged = True
                        break

                n_iterations_run = k + 1
                if self.rank0:
                    stage_writer.writerow([
                        stage,
                        float(q_value),
                        niter,
                        n_iterations_run,
                        bool(converged),
                        float(kkt),
                        float(descent_val),
                    ])
                    stage_file.flush()

                if not converged:
                    info_r(f"  Stage {stage}: maximum iterations reached without convergence.")
            if save_iterates:
                self.save_solutions(w, rho_k_filtered, rho_k)

        def _safe_mean(history):
            return float(np.mean(history)) if len(history) > 0 else float("nan")

        avg_projection = _safe_mean(projection_its_history)
        avg_filter_fwd = _safe_mean(filter_fwd_its_history)
        avg_filter_adj = _safe_mean(filter_adj_its_history)
        avg_adj_ns = _safe_mean(adj_ns_its_history)
        avg_newton_its = _safe_mean(newton_its_history)
        avg_krylov_per_newton = _safe_mean(krylov_per_newton_history)

        if self.rank0:
            info_g(f"\n{'=' * 60}")
            info_g("Average solver iterations over the whole optimization run:")
            info_g(f"  Projection / mass-matrix inversion .... ......... {avg_projection:.2f}")
            info_g(f"  Forward filter solve ..... ................. {avg_filter_fwd:.2f}")
            info_g(f"  Adjoint filter solve ....................... {avg_filter_adj:.2f}")
            info_g(f"  Adjoint solve ...................... ....... {avg_adj_ns:.2f}")
            info_g(f"  Forward Newton iterations .... {avg_newton_its:.2f}")
            info_g(f"  Forward FGMRES per Newton .... {avg_krylov_per_newton:.2f}")
            info_g(f"{'=' * 60}")

            summary_path = f"{output_dir}/solver_iterations_summary.csv"
            with open(summary_path, "w", newline="") as f:
                summary_writer = csv.writer(f)
                summary_writer.writerow(["quantity", "average_iterations", "n_samples"])
                summary_writer.writerow(["projection_mass_matrix_cg", avg_projection, len(projection_its_history)])
                summary_writer.writerow(["filter_forward_cg", avg_filter_fwd, len(filter_fwd_its_history)])
                summary_writer.writerow(["filter_adjoint_cg", avg_filter_adj, len(filter_adj_its_history)])
                summary_writer.writerow(["adjoint_ns_fgmres", avg_adj_ns, len(adj_ns_its_history)])
                summary_writer.writerow(["forward_ns_newton_its", avg_newton_its, len(newton_its_history)])
                summary_writer.writerow(["forward_ns_fgmres_per_newton", avg_krylov_per_newton, len(krylov_per_newton_history)])

        return rho_k, J_current, alpha_step