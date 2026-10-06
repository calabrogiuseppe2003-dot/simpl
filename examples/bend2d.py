from firedrake import *
from netgen.occ import *
from meshgenbend import create_geometry_bend
import os
from simpl import *

class Bend2D(SiMPL):

    def mesh(self):
        maxh = 0.2
        ngmesh, markers = create_geometry_bend(maxh)
        base = Mesh(ngmesh,distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)},)
        mh   = MeshHierarchy(base, 2)
        mesh = mh[-1]
        self.markers = markers
        return mesh

    def boundary_conditions(self, W):
        markers = self.markers
        BOTTOM_WALL_LEFT  = markers["bottom_wall_left"]
        BOTTOM_OUTLET     = markers["bottom_outlet"]
        BOTTOM_WALL_RIGHT = markers["bottom_wall_right"]
        RIGHT_WALL        = markers["right_wall"]
        TOP_WALL          = markers["top_wall"]
        LEFT_WALL_TOP     = markers["left_wall_top"]
        LEFT_INLET        = markers["left_inlet"]
        LEFT_WALL_BOTTOM  = markers["left_wall_bottom"]

        (x, y) = SpatialCoordinate(self.mesh)
        l = 1/5
        val1 = gbar * (1 - (2 * (y - 0.8) / l) ** 2)  # inlet parabolic profile
        bcs = [
            # u = 0 on all walls
            DirichletBC(W.sub(0), Constant((0.0,0.0)), [
                BOTTOM_WALL_LEFT,
                BOTTOM_WALL_RIGHT,
                RIGHT_WALL,
                TOP_WALL,
                LEFT_WALL_TOP,
                LEFT_WALL_BOTTOM,
            ]),
            DirichletBC(W.sub(0), as_vector([val1,0.0]), LEFT_INLET),
        ]
        return bcs

if __name__ == "__main__":
    Re            = Constant(1)  # Reynolds number
    gbar          = 1.0           # max inlet/outlet velocity
    dens          = Constant(1.0)  # density
    mu            = dens * gbar / Re
    nu            = 1.0 / Re       # nondimensional viscosity used in the forward problem
    alphaunderbar = 2.5 * mu / (1 / 5**2)   # alpha_min in the original dimensional scaling
    alphabar      = 1e4 * alphaunderbar     # alpha_max in the original dimensional scaling
    volfrac       = 1/4            # fluid volume fraction
    target_volume = volfrac
    alpha_init    = 2.5 * mu / (0.1**2)
    r_min         = 0.04                  #filter radius
    gamma         = Constant(1e4)        # augmented lagrangian penalty-coefficient

    c1 = 1e-3
    q_values=(250,)
    iters_per_q=(50,)


    
    problem = Bend2D(Re, gamma, alphaunderbar, alphabar, r_min, mu)


    if problem.rank0:
        os.makedirs("output", exist_ok=True)
        os.makedirs("output", exist_ok=True)

    problem.comm.barrier()

    # Re_v = [1,10, float(Re)]
    # problem.continuation_solve(Re_v, "output")
 
    problem.simpl(tol=1e-5,
                  target_volume=target_volume,
                  q_values=q_values,
                  iters_per_q = iters_per_q,
                  c1=c1,
                  simpl_type="A",
                  max_backtrack=10)