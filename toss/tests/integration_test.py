""" This test checks whether or not the integration is performed correctly """

import sys
sys.path.append("../..")

# Import relevant modules
import toss
from toss import equations_of_motion
from toss import mesh_utility
from toss import trajectory_tools


# Core packages
from dotmap import DotMap
from math import pi
import numpy as np
import pykep as pk

def test_integration():

    args = DotMap()

    # Setup body parameters
    args.body.density = 533                  # https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.mu = 665.666                   # Gravitational parameter for 67P/C-G
    args.body.declination = 64               # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.right_ascension = 69           # [degrees] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_period = 12.06*3600       # [seconds] https://sci.esa.int/web/rosetta/-/14615-comet-67p
    args.body.spin_velocity = (2*pi)/args.body.spin_period
    args.body.spin_axis = equations_of_motion.setup_spin_axis(args)

    # Setup specific integrator parameters:
    args.integrator.algorithm = 3
    args.integrator.dense_output = True
    args.integrator.rtol = 1e-12
    args.integrator.atol = 1e-12

    # Setup problem parameters
    args.problem.start_time = 0                     # Starting time [s]
    args.problem.final_time = 20*3600.0             # Final time [s]
    args.problem.initial_time_step = 600            # Initial time step size for integration [s]
    args.problem.radius_bounding_sphere = 4000      # Radius of spherical risk-zone for collision with celestial body [m]
    args.problem.event = 1                          # Event configuration (0 = no event, 1 = collision with body detection)
    args.problem.number_of_maneuvers = 0 

    args.mesh.body, args.mesh.vertices, args.mesh.faces, args.mesh.largest_body_protuberant = mesh_utility.create_mesh()


    # Initial position for integration (in cartesian coordinates):
    x = [-1.36986549e+03, -4.53113817e+03, -8.41816487e+03, -1.23505256e-01, -1.59791505e-01, 2.21471017e-01, 0, 0, 0, 0]
    x_cartesian = pk.ic2par(r=x[0:3], v=x[3:6], mu=args.body.mu)

    # Compute trajectory via numerical integration as in UDP.
    trajectory_info, _, _  = trajectory_tools.compute_trajectory(x_cartesian, args, equations_of_motion.compute_motion)

    # Final state from previous working results:
    final_state_historical = [3.07216681e+03, -2.45740917e+02, -9.03288997e+03, 2.48147088e-01, -2.18190890e-02, -2.68369809e-01]

    # New final state:
    final_state_new = trajectory_info[0:6,-1]

    assert all(np.isclose(final_state_historical,final_state_new,rtol=1e-5, atol=1e-5))