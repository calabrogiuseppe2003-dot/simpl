import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

from firedrake import *
from netgen.occ import *
from netgen.meshing import meshsize
from simpl import *


def create_geometry(maxh):
    box = Box((0, 0, 0), (2, 1, 1))
    for axis, side, name in (
        (X, "Min", "wall_xmin"), (X, "Max", "wall_xmax"),
        (Y, "Min", "wall_ymin"), (Y, "Max", "wall_ymax"),
        (Z, "Min", "wall_zmin"), (Z, "Max", "wall_zmax"),
    ):
        getattr(box.faces, side)(axis).name = name

    openings = [
        ("inlet", (0, 0.5, 0.5), (-1, 0, 0), (0, 1, 0), 1.0),
        ("outlet1", (2, 0.35, 0.62), (1, 0, 0), (0, 1, 0), 0.25),
        ("outlet2", (0.55, 0, 0.45), (0, -1, 0), (1, 0, 0), 0.25),
        ("outlet3", (1.35, 1, 0.62), (0, 1, 0), (1, 0, 0), 0.2),
        ("outlet4", (0.85, 0.4, 0), (0, 0, -1), (1, 0, 0), 0.1),
        ("outlet5", (1.55, 0.65, 1), (0, 0, 1), (1, 0, 0), 0.1),
        ("outlet6", (1.5, 0, 0.55), (0, -1, 0), (1, 0, 0), 0.1),
    ]
    faces = [box]
    inlet_radius = 0.2
    for name, center, normal, horizontal, area_fraction in openings:
        plane = WorkPlane(Axes(Pnt(*center), n=Dir(*normal), h=Dir(*horizontal)))
        face = plane.Circle(inlet_radius * area_fraction**0.5).Face()
        face.faces.name = name
        faces.append(face)

    ngmesh = OCCGeometry(Glue(faces), dim=3).GenerateMesh(meshsize.coarse,maxh=maxh)
    markers = {name: i + 1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1))}
    return ngmesh, markers


class Manifold3D(NavierStokesMeanOutletFlow, NavierStokesMTW):

    def __init__(self, Re, gamma, alphaunderbar, alphabar, r_min, mu, dens, mesh, markers, level):
        self._mesh = mesh
        self.markers = markers
        self.level = level
        fractions = (0.25, 0.25, 0.2, 0.1, 0.1)
        bounds = tuple((fraction - 0.05, fraction + 0.05) for fraction in fractions)
        super().__init__(
            Re, gamma, alphaunderbar, alphabar, r_min, mu, dens,
            inlet_markers=("inlet",),
            outlet_markers=tuple(f"outlet{i}" for i in range(1, 7)),
            outlet_fraction_bounds=bounds,
            beta=1e3,
            beta_growth=1.1,
        )

    def mesh(self):
        return self._mesh

    def boundary_conditions(self, W):
        _, y, z = SpatialCoordinate(self.mesh)
        inlet = gbar * (1 - ((y - 0.5)**2 + (z - 0.5)**2) / 0.3**2)
        walls = [self.markers[name] for name in (
            "wall_xmin", "wall_xmax", "wall_ymin", "wall_ymax", "wall_zmin", "wall_zmax"
        )]
        return [
            DirichletBC(W.sub(0), Constant((0, 0, 0)), walls),
            DirichletBC(W.sub(0), as_vector((inlet, 0, 0)), self.markers["inlet"]),
        ]

    def forward_sp(self):
        return self.sp_lu() if self.level == 0 else self.sp_mg()

    def adj_sp(self):
        return self.forward_sp()

    def filter_sp(self):
        # return self.sp_lu() if self.level == 0 else self.filter_sp_mg()
        return self.filter_sp_mg()

    def adj_filter_sp(self):
        return self.filter_sp()

    def sp_lu(self):
        sp = {
            "snes_monitor": None,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 1000,
        }
        return sp

    def sp_mg(self):
        sp = {
            "mat_type": "matfree",
            "snes_monitor": None,
            "snes_max_it": 20,
            "snes_atol": 1e-5,
            "snes_rtol": 1e-6,
            "ksp_type": "fgmres",
            "ksp_max_it": 300,
            "ksp_atol": 1e-6,
            "ksp_rtol": 1e-7,
            "ksp_monitor_true_residual": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_0_fields": 1,
            "pc_fieldsplit_1_fields": 0,
            "fieldsplit_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "jacobi",
            "fieldsplit_1": {
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {
                    "pc_use_amat": False,
                    "pc_type": "mg",
                    "mg_coarse_mat_type": "aij",
                    "mg_coarse_pc_type": "lu",
                    "mg_coarse_pc_factor_mat_solver_type": "mumps",
                    "mg_coarse_mat_mumps_icntl_14": 1000,
                    "mg_levels": {
                        "ksp_convergence_test": "skip",
                        "ksp_max_it": 5,
                        "ksp_type": "gmres",
                        "pc_type": "python",
                        "pc_python_type": "firedrake.ASMStarPC",
                    },
                },
            },
        }
        return sp

    def filter_sp_mg(self):
        sp = {
            "ksp_type": "cg", 
            "ksp_atol": 1e-7, 
            "ksp_rtol": 1e-8,
            "ksp_monitor": None, 
            "pc_type": "mg"}
        return sp


if __name__ == "__main__":
    Re_values = [1.0]
    Re = Constant(Re_values[-1])
    gbar = 1.0
    dens = Constant(1.0)
    mu = dens * gbar / Re
    alphaunderbar = 2.5 * mu / 100**2
    alphabar = 2.5 * mu / 0.01**2
    target_volume = 1.0 / 3.0
    r_min = 0.04
    gamma = Constant(1e4)

    ngmesh, markers = create_geometry(0.2)
    base = Mesh(ngmesh, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
    })
    hierarchy = MeshHierarchy(base, 2)

    # VTKFile("output/mesh.pvd").write(hierarchy[-1])

    timing_path = Path(__file__).resolve().parent / "output" / "manifold3d_gs.csv"
    if base.comm.rank == 0:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().astimezone().isoformat(timespec="seconds")
        with timing_path.open("a", newline="") as timing_file:
            if timing_file.tell() == 0:
                csv.writer(timing_file).writerow(("run_id", "level", "phase", "Re", "seconds"))
    base.comm.barrier()

    def save_timing(level, phase, seconds):
        if base.comm.rank == 0:
            with timing_path.open("a", newline="") as timing_file:
                csv.writer(timing_file).writerow((run_id, level, phase, float(Re), f"{seconds:.6f}"))

    previous_problem = None
    transfer_manager = None
    for level, mesh in enumerate(hierarchy):
        problem = Manifold3D(
            Re, gamma, alphaunderbar, alphabar, r_min, mu, dens,
            mesh, markers, level,
        )
        w = problem.setup_parameters[1]
        rho_k = problem.setup_parameters[2]
        forward_solver = problem.setup_parameters[7]
        rho_initial = None
        if previous_problem is None:
            transfer_manager = forward_solver._ctx.transfer_manager
            if len(Re_values) > 1:
                base.comm.barrier()
                continuation_start = perf_counter()
                problem.continuation_solve(Re_values, 200, target_volume)
                base.comm.barrier()
                save_timing(level, "re_continuation_total", perf_counter() - continuation_start)
        else:
            w_coarse = previous_problem.setup_parameters[1]
            rho_coarse = previous_problem.setup_parameters[2]
            transfer_manager.prolong(rho_coarse, rho_k)
            for wc_i, wf_i in zip(w_coarse.subfunctions, w.subfunctions):
                transfer_manager.prolong(wc_i, wf_i)
            rho_initial = rho_k

        info_r(f"Grid level {level}: {mesh.num_cells()} local cells")
        base.comm.barrier()
        start = perf_counter()
        problem.simpl(
            rtol=1e-5,
            atol=1e-4,
            target_volume=target_volume,
            q_values=(200, 100, 20) if level == 0 else (20,),
            iters_per_q=(5, 5, 50) if level == 0 else (50,),
            c1=1e-4,
            simpl_type="A",
            max_backtrack=10,
            output_dir=f"output/manifold3d/level_{level}",
            save_iterates=True,
            rho_initial=rho_initial,
        )
        base.comm.barrier()
        save_timing(level, "optimization", perf_counter() - start)
        previous_problem = problem

    for i, fraction in enumerate(problem.fractions, start=1):
        info_b(f"Final outlet {i} fraction: {fraction:.4%}")
