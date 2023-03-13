# General
import numpy as np
from typing import Union, Callable

# For Plotting
import pyvista as pv

# For orbit representation (reference frame)
import pykep as pk

# For working with the mesh
from toss.mesh_utility import is_outside

# For choosing numerical integration method
from toss.Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')


def compute_trajectory(x: np.ndarray, args, func: Callable) -> Union[np.ndarray, float, bool]:
    """compute_trajectory computes trajectory of satellite using numerical integation techniques 

    Args:
        x (np.ndarray): State vector.
        args (dotmap.DotMap):
            body: Parameters related to the celestial body:
                density (float): Mass density of celestial body.
                mu (float): Gravitational parameter for celestial body.
                declination (float): Declination angle of spin axis.
                right_ascension (float): Right ascension angle of spin axis.
                spin_velocity (float): Angular velocity of the body's rotation.
                spin_axis (np.ndarray): The axis around which the body rotates.
            integrator: Specific parameters related to the integrator:
                algorithm (int): Integer representing specific integrator algorithm.
                dense_output (bool): Dense output status of integrator.
                rtol (float): Relative error tolerance for integration.
                atol (float): Absolute error tolerance for integration.
            problem: Parameters related to the problem:
                start_time (int): Start time (in seconds) for the integration of trajectory.
                final_time (int): Final time (in seconds) for the integration of trajectory.
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                event (int): Event configuration (0 = no event, 1 = collision with body detection).
                number_of_maneuvers (int): Number of possible maneuvers.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
        func (Callable): A function handle for the state update equation required for integration.

    Returns:
        trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
        squared_altitudes (float): Sum of squared altitudes above origin for every position
        collision_penalty (bool): Penalty value given for the event of a collision with the celestial body.
    """
    # Convert osculating orbital elements to cartesian for integration
    r, v = pk.par2ic(E=x[0:6], mu=args.body.mu)
    x_cartesian = np.array(r+v)

    # Setup time intervals with/without maneuvers
    if args.problem.number_of_maneuvers == 0:
        time_list = [args.problem.start_time, args.problem.final_time]
    else:
        # Add state variables related to the impulsive maneuver in args
        args.state.time_of_maneuver = int(x[6])
        args.state.delta_v = np.array([x[7], x[8], x[9]])
        time_list = [args.problem.start_time, args.state.time_of_maneuver, args.problem.final_time]

    # Integrate trajectory for each subinterval
    trajectory_info = 0
    for i in range(0, len(time_list)-1):
        args.integrator.t0 = time_list[i]
        args.integrator.tf = time_list[i+1]

        if i > 0:
            # Start at end position of previous interval, now adding impulsive manuever
            x_cartesian = trajectory_info[0:6,-1]
            x_cartesian[3:6] += args.state.delta_v

        # Compute trajectory by numerical integration
        trajectory, trajectory_memory = integrate_trajectory(func, x_cartesian, args)

        if i == 0:
            trajectory_info = trajectory_memory

        else:
            trajectory_info = np.hstack((trajectory_info[:,0:-1] ,trajectory_memory))
        
        # Check for potential collisions
        event_triggers = np.empty((len(trajectory.events), 3), dtype=np.float64)
        k = 0
        for event in trajectory.events:
            event_triggers[k,:] = event.y[0:3]
            k += 1

        if i == 0: 
            points_inside_risk_zone = event_triggers
        else:
            points_inside_risk_zone = np.vstack((points_inside_risk_zone, event_triggers))


    # Compute average distance to target altitude
    squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2
    
    # Check for collisions
    collisions_avoided = point_is_outside_mesh(points_inside_risk_zone, args.mesh.vertices, args.mesh.faces)
    if all(collisions_avoided) == True:
        collision_detected = False
    else:
        collision_detected = True
    
    # Return trajectory and neccessary values for computing fitness in udp.
    return trajectory_info, squared_altitudes, collision_detected


def integrate_trajectory(func: Callable, x: np.ndarray, args):

    """ Integrates trajectory numerically using DeSolver library.

    Args:
        func (Callable): A function handle for the equ_rhs (state update equation) required for integration.
        x (np.ndarray): Initial position and velocity of satelite in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
            integrator: Specific parameters related to the integrator:
                algorithm (int): Integer representing specific integrator algorithm.
                dense_output (bool): Dense output status of integrator.
                rtol (float): Relative error tolerance for integration.
                atol (float): Absolute error tolerance for integration.
            problem: Parameters related to the problem:
                start_time (int): Start time (in seconds) for the integration of trajectory.
                final_time (int): Final time (in seconds) for the integration of trajectory.
                initial_time_step (float): Size of initial time step (in seconds) for integration of trajectory.
                radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                event (int): Event configuration (0 = no event, 1 = collision with body detection)
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
    Returns:
        trajectory (desolver.differential_system.OdeSystem): The integration object provided by desolver.
        trajectory_info (np.ndarray): Numpy array containing information on position and velocity at every time step (columnwise).
    """

    # Setup parameters
    dense_output = args.integrator.dense_output
    t0 = args.integrator.t0
    tf = args.integrator.tf
    dt = args.problem.initial_time_step
    rtol = args.integrator.rtol
    atol = args.integrator.atol
    numerical_integrator = IntegrationScheme(args.integrator.algorithm).name
    event = args.problem.event

    # Integrate trajectory
    initial_state = D.array(x)
    trajectory = de.OdeSystem(
        func, 
        y0 = initial_state, 
        dense_output = dense_output, 
        t = (t0, tf), 
        dt = dt, 
        rtol = rtol, 
        atol = atol,
        constants=dict(args = args))
    trajectory.method = str(numerical_integrator)

    if event==0:
        trajectory.integrate()

    elif event==1:
        point_is_inside_risk_zone.is_terminal = False
        trajectory.integrate(events=point_is_inside_risk_zone)

    # Add integration times to trajectory info
    trajectory_info = np.vstack((np.transpose(trajectory.y), trajectory.t))

    return trajectory, trajectory_info


def point_is_inside_risk_zone(t: float, state: np.ndarray, args) -> int:
    """ Checks for event: collision with the celestial body.

    Args:
        t (float): Current time step for integration.
        state (np.ndarray): Current state, i.e position and velocity
        args (dotmap.DotMap):
            problem:
                radius_bounding_sphere (float): Radius of bounding sphere around mesh. 

    Returns:
        (int): Returns 1 when the satellite enters the risk-zone, and 0 otherwise.
    """
    risk_zone_radius = args.problem.radius_bounding_sphere
    position = state[0:3]
    distance = risk_zone_radius**2 - position[0]**2 + position[1]**2 + position[2]**2
    if distance >= 0:
        return 0
    return 1


def point_is_outside_mesh(x: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> bool:
    """
    Uses is_outside to check if a set of positions (or current) x is is inside mesh.
    Returns boolean with corresponding results.

    Args:
        x (np.ndarray): Array containing current, or a set of, positions expressed in 3 dimensions.
        mesh_vertices (np.ndarray): Array containing all points on mesh.
        mesh_faces (np.ndarray): Array containing all triangles on the mesh.

    Returns:
        collision_boolean (bool): A one dimensional array with boolean values corresponding to each
                                position kept in x. Returns "False" if point is inside mesh, and 
                                "True" if point is outside mesh (that is, there no collision).
    """
    collision_boolean = is_outside(x, mesh_vertices, mesh_faces)
    return collision_boolean


def plot_trajectory(r_store: np.ndarray, mesh):
    """plot_trajectory plots the body mesh and satellite trajectory.

    Args:
        r_store (np.ndarray): Array containing values on position at each time step for the trajectory (columnwise).
        mesh (tetgen.pytetgen.TetGen): Tetgen mesh object of celestial body.
    """
    # Plotting mesh of asteroid/comet
    mesh_plot = pv.Plotter(window_size=[500, 500])
    mesh_plot.add_mesh(mesh.grid, show_edges=True)
    #mesh_plot.show_bounds() # minor_ticks=True, grid='front',location='outer',all_edges=True 

    # Plotting trajectory
    trajectory = np.transpose(r_store)
    for i in range(0,len(r_store[0])-1):
        traj = np.vstack((trajectory[i,:], trajectory[i+1,:]))
        mesh_plot.add_lines(traj, color="red", width=40)
                    
    # Plotting final position as a white dot
    trajectory_plot = pv.PolyData(np.transpose(r_store[:,-1]))
    mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')

    mesh_plot.add_axes(x_color='red', y_color='green', z_color='blue', xlabel='X', ylabel='Y', zlabel='Z', line_width=2, shaft_length = 10)
    
    mesh_plot.show(jupyter_backend = 'panel') 
