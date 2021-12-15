import numpy as np
import tensorflow as tf


### Settings and parameters
LENGTH_X  = 200    # resolution x direction  # default 400
LENGTH_Y  = 50    # resolution y direction
RHO_0     = 1      # average density
TAU       = 0.6    # collision timescale (relaxation term)
N_STEPS   = 4000   # number of timesteps
U_MAX     = 0.1    # maximum velocity of Poiseuille inflow
INLET_IDX = 0
OUTLET_IDX = LENGTH_X - 1
PIPE_LENGTH = LENGTH_Y  # L
INLET_SL = np.s_[:, 0]
OUTLET_SL = np.s_[:, LENGTH_X - 1]

### Cylinder parameters
# X.shape: (100, 400) Y shape: (100, 400)
X, Y = np.meshgrid(range(LENGTH_X), range(LENGTH_Y))
# INFO: shape the same as all space, but only partially filled with cylinder
# cylinder shape: (100, 400)
CYLINDER_RADIUS = 4
# True within cylinder boundaries
CYLINDER_MASK = (X - LENGTH_X / 4) ** 2 + (Y - LENGTH_Y / 2) ** 2 < (LENGTH_Y // CYLINDER_RADIUS) ** 2

### Vectors params
# General params
LEFT_COL_NAMES = ["NW", "W", "SW"]
CENT_COL_NAMES = ["N", "C", "S"]
RIGHT_COL_NAMES = ["NE", "E", "SE"]
N_VECTORS = 9
VECTOR_INDEXES = np.arange(N_VECTORS)

VECTORS_VELOCITIES_X = np.array([
    [-1, 0, 1,],
    [-1, 0, 1,],
    [-1, 0, 1,],
]).reshape(-1)
VECTORS_VELOCITIES_Y = np.array([
     [1,  1,  1,],
     [0,  0,  0,],
    [-1, -1, -1,],
]).reshape(-1)
VECTORS_WEIGHTS = np.array([
    [1/36, 1/9, 1/36,],
    [1/9,  4/9, 1/9,],
    [1/36, 1/9, 1/36,],
]).reshape(-1)
LAT_LEFT_COL_SL = np.s_[:, [0, 3, 6]]
LAT_CENT_COL_SL = np.s_[:, [1, 4, 7]]
LAT_RIGHT_COL_SL= np.s_[:, [2, 5, 8]]
# 'NW' 'N' 'NE' 'W' 'C' 'E' 'SW' 'S' 'SE'
#   0   1    2   3   4   5    6   7    8
VECTORS_DIRECTIONS = np.array([
    ['NW', 'N', 'NE',],
    ['W',  'C',  'E',],
    ['SW', 'S', 'SE',],
]).reshape(-1)
CYL_BOUNCE_BACK_DIRECTIONS = [8, 7, 6, 5, 4, 3, 2, 1, 0]


### Initial conditions
def init_random_cos():
    # F.shape: (100, 400, 9)
    F = np.ones((LENGTH_Y, LENGTH_X, N_VECTORS)) #* RHO_0 / N_VECTORS
    np.random.seed(42)
    F += 0.01 * np.random.randn(LENGTH_Y, LENGTH_X, N_VECTORS)
    X, _ = np.meshgrid(range(LENGTH_X), range(LENGTH_Y))
    # F[0, 0] - 0.99 .. 3.45
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / LENGTH_X * 4))  # 1.6..2.4
    rho = np.sum(F, 2)
    for i in VECTOR_INDEXES:
        F[:, :, i] *= RHO_0 / rho
    # F[0, 0] - 8 .. 29
    return F

def poiseuille_profile(y_phys, u_max=U_MAX, pipe_length=PIPE_LENGTH):
    return 4 * u_max / (pipe_length ** 2) * (y_phys * pipe_length - y_phys * y_phys)

def init_poiseuille():
    rho = 1
    y, x = np.meshgrid(np.arange(LENGTH_Y), np.arange(LENGTH_X))
    F = np.empty((LENGTH_X, LENGTH_Y, N_VECTORS))
    y_phys = y - 0.5
    ux = poiseuille_profile(y_phys)
    uy = np.zeros((LENGTH_X, LENGTH_Y))
    
    for idx in range(9):
        # 300, 100
        cu = 3 * (VECTORS_VELOCITIES_X[idx] * ux + VECTORS_VELOCITIES_Y[idx] * uy)
        # 300, 100
        res = rho * VECTORS_WEIGHTS[idx] * (1 + cu + 1/2 * cu ** 2 - 3/2*(ux**2 + uy **2))
        F[:, :, idx] = res
    F = np.rot90(F)
    return F
