# General
import numpy as np
from typing import Union

# For Plotting
import pyvista as pv

# For working with the mesh
import mesh_utility

# For computing the next state
from Equations_of_motion import Equations_of_motion

# For choosing numerical integration method
from Integrator import IntegrationScheme

# D-solver (performs integration)
import desolver as de
import desolver.backend as D
D.set_float_fmt('float64')



# Class representing UDP
class udp_initial_condition:
    """ 
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for input variables, trajectory propagation and plotting of results. 
    """

    def __init__(self, body_density, target_altitude, final_time, start_time, time_step, lower_bounds, upper_bounds, algorithm, radius_bounding_sphere):
        """ Setup udp attributes.

        Args:
            body_density (_float_): Mass density of body of interest
            target_altitude (_float_): Target altitude for satellite trajectory. 
            final_time (_float_): Final time for integration.
            start_time (_float_): Start time for integration of trajectory (often zero)
            time_step (_float_): Step size for integration. 
            lower_bounds (_np.ndarray_): Lower bounds for domain of initial state.
            upper_bounds (_np.ndarray_): Upper bounds for domain of initial state. 
            algorithm (_int_): User defined algorithm of choice
            radius_bounding_sphere (_float_)_: Radius for the bounding sphere around mesh.
        """
        # Creating the mesh (TetGen)
        self.body_mesh, self.mesh_vertices, self.mesh_faces, largest_body_protuberant = mesh_utility.create_mesh()

        # Assertions:
        assert body_density > 0
        assert target_altitude > 0
        assert final_time > start_time
        assert time_step <= (final_time - start_time)
        assert lower_bounds.all() < upper_bounds.all()
        assert radius_bounding_sphere > largest_body_protuberant

        # Setup equations of motion class
        self.eq_of_motion = Equations_of_motion(self.mesh_vertices, self.mesh_faces, body_density)

        # Additional hyperparameters
        self.start_time = start_time
        self.final_time = final_time
        self.time_step = time_step
        self.target_altitude = target_altitude     
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.algorithm = algorithm
        self.radius_bounding_sphere = radius_bounding_sphere


        #Test
        self.k = 1

    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x (_np.ndarray_): State vector containing values for position and velocity of satelite in three dimensions. 

        Returns:
            fitness value (_float_): Difference between squared values of current and target altitude of satellite.
        """
        # Integrate trajectory
        _, squared_altitudes, collision_penalty = self.compute_trajectory(np.array(x))

        # Compute fitness value for the integrated trajectory
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_altitude)) + collision_penalty

        ### FOR TEST ###
        print(self.k)
        self.k += 1
        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            lower_bounds (_np.ndarray_): Lower boundary values for the initial state vector.
            upper_bounds (_np.ndarray_): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)

    def compute_trajectory(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """compute_trajectory computes trajectory of satellite using numerical integation techniques 

        Args:
            x (_np.ndarray_): State vector containing values for position and velocity of satelite in three dimensions.

        Returns:
            trajectory_info (_np.ndarray_): Numpy array containing information on position and velocity at every time step (columnwise).
            squared_altitudes (_float_): Sum of squared altitudes above origin for every position
            collision_penalty (_float_): Penalty value given for the event of a collision with the celestial body.
        """

        # Integrate trajectory
        initial_state = D.array(x)
        trajectory = de.OdeSystem(
            self.eq_of_motion.compute_motion, 
            y0 = initial_state, 
            dense_output = True, 
            t = (self.start_time, self.final_time), 
            dt = self.time_step, 
            rtol = 1e-12, 
            atol = 1e-12,
            constants=dict(risk_zone_radius = self.radius_bounding_sphere)) #, mesh_vertices = self.mesh_vertices, mesh_faces = self.mesh_faces
        trajectory.method = str(IntegrationScheme(self.algorithm).name)

        Check_inside_risk_zone.is_terminal = False
        trajectory.integrate(events=Check_inside_risk_zone)
        trajectory_info = np.vstack((np.transpose(trajectory.y), trajectory.t))

        # Compute average distance to target altitude
        squared_altitudes = trajectory_info[0,:]**2 + trajectory_info[1,:]**2 + trajectory_info[2,:]**2

        # Add collision penalty
        points_inisde_risk_zone = np.empty((len(trajectory.events), 3), dtype=np.float64)
        i = 0
        for j in trajectory.events:
            points_inisde_risk_zone[i,:] = j.y[0:3]
            i += 1
        
        collision_avoided = point_is_outside_mesh(points_inisde_risk_zone, self.mesh_vertices, self.mesh_faces)
        if all(collision_avoided) == True:
            collision_penalty = 0
        else:
            collision_penalty = 1e30
        
        # Return trajectory and neccessary values for fitness.
        return trajectory_info, squared_altitudes, collision_penalty



    def plot_trajectory(self, r_store: np.ndarray):
        """plot_trajectory plots the body mesh and satellite trajectory.

        Args:
            r_store (_np.ndarray_): Array containing values on position at each time step for the trajectory (columnwise).
        """

        # Plotting mesh of asteroid/comet
        mesh_plot = pv.Plotter(window_size=[500, 500])
        mesh_plot.add_mesh(self.body_mesh.grid, show_edges=True)
        mesh_plot.show_grid() #grid='front',location='outer',all_edges=True 

        # Plotting trajectory
        trajectory_plot = np.transpose(r_store)
        if (len(trajectory_plot[:,0]) % 2) != 0:
            trajectory_plot = trajectory_plot[0:-1,:,]
        mesh_plot.add_lines(trajectory_plot[:,0:3], color="red", width=40)        

        # Plotting final position as a white dot
        trajectory_plot = pv.PolyData(np.transpose(r_store[-1,0:3]))
        mesh_plot.add_mesh(trajectory_plot, color=[1.0, 1.0, 1.0], style='surface')
        
        mesh_plot.show(jupyter_backend = 'panel') 



def Check_inside_risk_zone(t: float, state: np.ndarray, risk_zone_radius: float) -> float: #, mesh_vertices: np.ndarray, mesh_faces: np.ndarray
    """ Checks for event: collision with the celestial body.

    Args:
        t (_float_): Current time step for integration.
        state (_np.ndarray_): Current state, i.e position and velocity
        risk_zone_radius (_float_): Radius of bounding sphere around mesh. 

    Returns:
        (_float_): Returns 1 when the satellite enters the risk-zone, and 0 otherwise.
    """
    position = state[0:3]
    distance = risk_zone_radius - D.norm(position)
    if distance >= 0:
        return 0
    return 1


def point_is_outside_mesh(x: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> bool:
    """
    Uses is_outside to check if a set of positions (or current) x is is inside mesh.
    Returns boolean with corresponding results.

    Args:
        x (_np.ndarray_): Array containing current, or a set of, positions expressed in 3 dimensions.

    Returns:
        collision_boolean (_bool): A one dimensional array with boolean values corresponding to each
                                position kept in x. Returns "False" if point is inside mesh, and 
                                "True" if point is outside mesh (that is, there no collision).
    """
    collision_boolean = mesh_utility.is_outside(x, mesh_vertices, mesh_faces)
    return collision_boolean

