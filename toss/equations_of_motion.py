# General
import numpy as np
from math import pi, radians

# For computing acceleration and potential
import polyhedral_gravity as model

# For computing rotations of orbits
from pyquaternion import Quaternion


def setup_spin_axis(args):
    """ 
    Defines a spin axis given the declination and
    right ascension of the body's rotation.

    Args:
        args (dotmap.DotMap):
            body:
                declination (float): Declination angle of spin axis.
                right_ascension (float): Right ascension angle of spin axis.

    Returns:
        spin_axis (np.ndarray): The axis around which the body rotates.
    """

    # Setup spin axis of the body
    q_dec = Quaternion(axis=[1,0,0], angle=radians(args.body.declination)) # Rotate spin axis according to declination
    q_ra = Quaternion(axis=[0,0,1], angle=radians(args.body.right_ascension)) # Rotate spin axis accordining to right ascension
    q_axis = q_dec * q_ra  # Composite rotation of q1 then q2 expressed as standard multiplication
    spin_axis = q_axis.rotate([0,0,1])
    return spin_axis


def compute_acceleration(x: np.ndarray, args) -> np.ndarray:
    """ 
    Computes acceleration at a given point with respect to a mesh representing
    a celestial body of interest. This is done using the Polyhedral-Gravity-Model package.

    Args:
        x (np.ndarray): Vector containing information on current position expressed in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.

    Returns:
        (np.ndarray): The acceleration at the given point x with respect to the mesh (celestial body).
    """
    _, a, _ = model.evaluate(args.mesh.vertices, args.mesh.faces, args.body.density, x)
    return -np.array(a)

# Used by all RK-type algorithms
def compute_motion(t: float, x: np.ndarray, args) -> np.ndarray:
    """ State update equation for RK-type algorithms. 

    Args:
        t (float): Time value in seconds corresponding to current state
        x (np.ndarray): State vector containing position and velocity expressed in 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                density (float): Mass density of celestial body.
                spin_velocity (float): Angular velocity of the body's rotation.
                spin_axis (np.ndarray): The axis around which the body rotates.
            mesh:
                vertices (np.ndarray): Array containing all points on mesh.
                faces (np.ndarray): Array containing all triangles on the mesh.
            state: Parameters provided by the state vector
                time_of_maneuver (float): Time for adding impulsive maneuver [seconds].
                delta_v (np.ndarray): Array containing the cartesian componants of the impulsive maneuver.
    Returns:
        (np.ndarray): K vector used for computing state at the following time step.
    """
    rotated_position = rotate_point(t, x[0:3], args)
    a = compute_acceleration(rotated_position, args)

    kx = x[3:6]
    kv = a

    return np.concatenate((kx, kv))


def rotate_point(t: float, x: np.ndarray, args) -> np.ndarray:
    """ Rotates position x according to the analyzed body's real rotation.
        The rotation is made in the 3D cartesian inertial body frame.
    Args:
        t (float): Time value in seconds when position x occurs.
        x (np.ndarray): Position of satellite expressed in the 3D cartesian coordinates.
        args (dotmap.DotMap):
            body:
                spin_velocity (float): Angular velocity of the body's rotation.
                spin_axis (np.ndarray): The axis around which the body rotates.
    Returns:
        x_rotated (np.ndarray): Rotated position of satellite expressed in the 3D cartesian coordinates.
    """

    # Get Quaternion object for rotation around spin axis
    q_rot = Quaternion(axis=args.body.spin_axis, angle=(2*pi-args.body.spin_velocity*t))
    
    # Rotate satellite position using q_rot
    x_rotated = q_rot.rotate(x)

    return np.array(x_rotated)

