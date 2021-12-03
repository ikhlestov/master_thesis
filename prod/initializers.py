"""Initializers to populate initial state of the domain"""
import numpy as np

from const import (
    VECTORS_VELOCITIES_X,
    VECTORS_VELOCITIES_Y,
    VECTORS_WEIGHTS,
    N_VECTORS,
    U_MAX,
)


def poiseuille_profile(y_phys: np.ndarray, u_max: float, pipe_diameter: int):
    return 4 * u_max / (pipe_diameter ** 2) * (y_phys * pipe_diameter - y_phys * y_phys)


def init_poiseuille(length_x: int, length_y: int, u_max: int=U_MAX):
    rho = 1
    y, x = np.meshgrid(np.arange(length_y), np.arange(length_x))
    F = np.empty((length_x, length_y, N_VECTORS))
    ux = poiseuille_profile(
        y_phys=y,
        u_max=u_max,
        pipe_diameter=length_y,
    )
    uy = np.zeros((length_x, length_y))
    
    for idx in range(N_VECTORS):
        cu = 3 * (VECTORS_VELOCITIES_X[idx] * ux + VECTORS_VELOCITIES_Y[idx] * uy)
        res = rho * VECTORS_WEIGHTS[idx] * (1 + cu + 1/2 * cu ** 2 - 3/2*(ux**2 + uy **2))
        F[:, :, idx] = res
    F = np.rot90(F)
    return F


def calc_init_params(scale: int, Re, D: float, u_max: float=U_MAX):
    """
    Args:
        Re: Reynolds number
        D: normalized object size, in (0..1)
    """
    Re_calc = Re / scale
    u = u_max
    nu = u * D / Re_calc   # Kinematic viscosity
    tau = 3 * nu + 1/2  # Get relaxation parameters
    omega = 1 / tau
    return nu, tau, omega


# Shapes masks initializers
def get_square(obj_d, x_mesh, y_mesh, obj_center_x, obj_center_y, scale, y_shift):
    """x_mes, y_mesh = np.meshgrid(range(LENGTH_X), range(LENGTH_Y))"""
    norm_square_edge = obj_d
    square_edge = int(norm_square_edge * scale)
    object_mask = np.logical_and(
        abs(x_mesh - obj_center_x) - square_edge < 0,
        abs(y_mesh - obj_center_y + y_shift) - square_edge < 0
    )
    return object_mask


def get_cylinder(obj_d, x_mesh, y_mesh, obj_center_x, obj_center_y, scale, y_shift):
    norm_cylinder_radius = obj_d / 2
    cylinder_radius = int(norm_cylinder_radius * scale)
    # True within object boundaries
    object_mask = (x_mesh - obj_center_x) ** 2 + (y_mesh - obj_center_y + y_shift) ** 2 < (cylinder_radius) ** 2
    return object_mask


def get_plate(obj_d, x_mesh, y_mesh, obj_center_x, obj_center_y, scale, y_shift):
    plate_size = int(obj_d * scale)
    object_mask = np.logical_and(
            abs(x_mesh - obj_center_x) == 0,
            abs(y_mesh - obj_center_y + y_shift) - plate_size < 0
        )
    return object_mask
