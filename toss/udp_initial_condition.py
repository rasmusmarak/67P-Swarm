# General
import numpy as np
from typing import Union

# For computing trajectory
import trajectory_tools

# For computing the next state
import equations_of_motion

# Class representing UDP 
class udp_initial_condition:
    """ 
    Sets up the user defined problem (udp) for use with pygmo.
    The object holds attributes in terms of variables and constants that
    are used for trajectory propagation. 
    The methods of the class defines the objective function for the optimization problem,
    boundaries for the state variables and computation of the fitness value for a given intial state. 
    """

    def __init__(self, args, lower_bounds, upper_bounds):
        """ Setup udp attributes.

        Args:
            args (dotmap.Dotmap)
                body: Parameters related to the celestial body:
                    density (float): Mass density of celestial body.
                    mu (float): Gravitational parameter for celestial body.
                    declination (float): Declination angle of spin axis.
                    right_ascension (float): Right ascension angle of spin axis.
                    spin_period (float): Rotational period around spin axis of the body.
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
                    target_squared_altitude (float): Squared value of the satellite's orbital target altitude.
                    radius_bounding_sphere (float): Radius of the bounding sphere representing risk zone for collisions with celestial body.
                    event (int): Event configuration (0 = no event, 1 = collision with body detection).
                    number_of_maneuvers (int): Number of possible maneuvers.
                mesh:
                    vertices (np.ndarray): Array containing all points on mesh.
                    faces (np.ndarray): Array containing all triangles on the mesh.
            lower_bounds (np.ndarray): Lower bounds for domain of initial state.
            upper_bounds (np.ndarray): Upper bounds for domain of initial state. 
        """

        # Declerations
        self.target_sq_alt = args.problem.target_squared_altitude
        mu = args.body.mu
        ti = args.problem.start_time
        tf = args.problem.final_time
        dt = args.problem.initial_time_step
        r_sphere = args.problem.radius_bounding_sphere
        largest_protuberant = args.mesh.largest_body_protuberant
        density = args.body.density
        dec = args.body.declination
        ra = args.body.right_ascension
        period = args.body.spin_period

        # Assertions:
        assert self.target_sq_alt > 0
        assert all(np.greater(upper_bounds, lower_bounds))
        assert mu > 0
        assert tf > ti
        assert dt <= tf - ti
        assert r_sphere > largest_protuberant
        assert density > 0
        assert dec >= 0
        assert ra >= 0
        assert period >= 0


        # Additional hyperparameters
        self.args = args
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def fitness(self, x: np.ndarray) -> float:
        """ fitness evaluates the proximity of the satallite to target altitude.

        Args:
            x (np.ndarray): State vector. 

        Returns:
            fitness value (_float_): Difference between squared values of current and target altitude of satellite.
        """

        # Integrate trajectory
        _, squared_altitudes, collision_detected = trajectory_tools.compute_trajectory(x, self.args, equations_of_motion.compute_motion)

        # Define fitness penalty in the event of at least one collision along the trajectory
        if collision_detected == True:
            collision_penalty = 1e30
        else:
            collision_penalty = 0

        # Compute fitness value for the integrated trajectory
        fitness_value = np.mean(np.abs(squared_altitudes-self.target_sq_alt)) + collision_penalty

        return [fitness_value]


    def get_bounds(self) -> Union[np.ndarray, np.ndarray]:
        """get_bounds returns upper and lower bounds for the domain of the state vector.

        Returns:
            lower_bounds (np.ndarray): Lower boundary values for the initial state vector.
            upper_bounds (np.ndarray): Lower boundary values for the initial state vector.
        """
        return (self.lower_bounds, self.upper_bounds)