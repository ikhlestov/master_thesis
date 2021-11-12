import numpy as np
import tensorflow as tf


### Settings and parameters
LENGTH_X  = 400    # resolution x direction  # default 400
LENGTH_Y  = 100    # resolution y direction
RHO_0     = 1      # average density
TAU       = 0.6    # collision timescale (relaxation term)
# tau = 1.9739
# tau = 0.9
N_STEPS   = 4000   # number of timesteps
U_MAX     = 0.1    # maximum velocity of Poiseuille inflow
INLET_IDX = 0
OUTLET_IDX = LENGTH_X - 1
PIPE_LENGTH = LENGTH_Y - 2  # L
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
# pylab.imshow(CYLINDER_MASK, cmap='gray')

### Vectors params
# General params
LEFT_COL_NAMES = ["NW", "W", "SW"]
CENT_COL_NAMES = ["N", "C", "S"]
RIGHT_COL_NAMES = ["NE", "E", "SE"]
N_VECTORS = 9
VECTOR_INDEXES = np.arange(N_VECTORS)


# Old style lattices definitions
#                                      0    1     2    3    4     5     6    7     8
#                                      C    N     NE   E    SE    S    SW    W     NW
VECTORS_VELOCITIES_X_OLD = np.array([  0,   0,    1,   1,   1,    0,   -1,  -1,   -1,  ])
VECTORS_VELOCITIES_Y_OLD = np.array([  0,   1,    1,   0,  -1,   -1,   -1,   0,    1,  ])
VECTORS_WEIGHTS_OLD = np.array([      4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36,  ]) # sums to 1
# NOTE: this should be updated to the new style
LAT_LEFT_COL_SL_OLD = np.s_[:, [8, 7, 6]]
LAT_CENT_COL_SL_OLD = np.s_[:, [1, 0, 5]]
LAT_RIGHT_COL_SL_OLD= np.s_[:, [2, 3, 4]]
VECTORS_DIRECTIONS_OLD = np.array("C N NE E SE S SW W NW".split())
assert np.all(VECTORS_DIRECTIONS_OLD[LAT_LEFT_COL_SL_OLD[1]] == LEFT_COL_NAMES)
assert np.all(VECTORS_DIRECTIONS_OLD[LAT_CENT_COL_SL_OLD[1]] == CENT_COL_NAMES)
assert np.all(VECTORS_DIRECTIONS_OLD[LAT_RIGHT_COL_SL_OLD[1]] == RIGHT_COL_NAMES)
                                # C  S  SW W  NW  
CYL_BOUNCE_BACK_DIRECTIONS_OLD = [0, 5, 6, 7, 8, 1, 2, 3, 4]


# New style lattices definitions
VECTORS_VELOCITIES_X_NEW = np.array([
    [-1, 0, 1,],
    [-1, 0, 1,],
    [-1, 0, 1,],
]).reshape(-1)
VECTORS_VELOCITIES_Y_NEW = np.array([
     [1,  1,  1,],
     [0,  0,  0,],
    [-1, -1, -1,],
]).reshape(-1)
VECTORS_WEIGHTS_NEW = np.array([
    [1/36, 1/9, 1/36,],
    [1/9,  4/9, 1/9,],
    [1/36, 1/9, 1/36,],
]).reshape(-1)
LAT_LEFT_COL_SL_NEW = np.s_[:, [0, 3, 6]]
LAT_CENT_COL_SL_NEW = np.s_[:, [1, 4, 7]]
LAT_RIGHT_COL_SL_NEW= np.s_[:, [2, 5, 8]]
# 'NW' 'N' 'NE' 'W' 'C' 'E' 'SW' 'S' 'SE'
#   0   1    2   3   4   5    6   7    8
VECTORS_DIRECTIONS_NEW = np.array([
    ['NW', 'N', 'NE',],
    ['W',  'C',  'E',],
    ['SW', 'S', 'SE',],
]).reshape(-1)
assert np.all(VECTORS_DIRECTIONS_NEW[LAT_LEFT_COL_SL_NEW[1]] == LEFT_COL_NAMES)
assert np.all(VECTORS_DIRECTIONS_NEW[LAT_CENT_COL_SL_NEW[1]] == CENT_COL_NAMES)
assert np.all(VECTORS_DIRECTIONS_NEW[LAT_RIGHT_COL_SL_NEW[1]] == RIGHT_COL_NAMES)
CYL_BOUNCE_BACK_DIRECTIONS_NEW = [8, 7, 6, 5, 4, 3, 2, 1, 0]


# check that new and old style colums slices are the same
assert np.all(
    VECTORS_DIRECTIONS_NEW[LAT_LEFT_COL_SL_NEW[1]] ==
    VECTORS_DIRECTIONS_OLD[LAT_LEFT_COL_SL_OLD[1]]
)
assert np.all(
    VECTORS_DIRECTIONS_NEW[LAT_CENT_COL_SL_NEW[1]] ==
    VECTORS_DIRECTIONS_OLD[LAT_CENT_COL_SL_OLD[1]]
)
assert np.all(
    VECTORS_DIRECTIONS_NEW[LAT_RIGHT_COL_SL_NEW[1]] ==
    VECTORS_DIRECTIONS_OLD[LAT_RIGHT_COL_SL_OLD[1]]
)


# Helper call to convert old lattices to the new one
OLD_TO_NEW_INDEXES = [list(VECTORS_DIRECTIONS_OLD).index(item) for item in VECTORS_DIRECTIONS_NEW]
NEW_TO_OLD_INDEXES = [list(VECTORS_DIRECTIONS_NEW).index(item) for item in VECTORS_DIRECTIONS_OLD]


# check that old and new coordinates are the same
assert (VECTORS_DIRECTIONS_NEW == VECTORS_DIRECTIONS_OLD[OLD_TO_NEW_INDEXES]).all()
assert (VECTORS_DIRECTIONS_OLD == VECTORS_DIRECTIONS_NEW[NEW_TO_OLD_INDEXES]).all()
assert (VECTORS_DIRECTIONS_OLD == VECTORS_DIRECTIONS_OLD[OLD_TO_NEW_INDEXES][NEW_TO_OLD_INDEXES]).all()

assert (VECTORS_VELOCITIES_X_NEW[NEW_TO_OLD_INDEXES] == VECTORS_VELOCITIES_X_OLD).all()
assert (VECTORS_VELOCITIES_Y_NEW[NEW_TO_OLD_INDEXES] == VECTORS_VELOCITIES_Y_OLD).all()
assert (VECTORS_WEIGHTS_NEW[NEW_TO_OLD_INDEXES] == VECTORS_WEIGHTS_OLD).all()


style = "NEW"
VECTORS_VELOCITIES_X = eval(f"VECTORS_VELOCITIES_X_{style}")
VECTORS_VELOCITIES_Y = eval(f"VECTORS_VELOCITIES_Y_{style}")
VECTORS_WEIGHTS = eval(f"VECTORS_WEIGHTS_{style}")
LAT_LEFT_COL_SL = eval(f"LAT_LEFT_COL_SL_{style}")
LAT_CENT_COL_SL = eval(f"LAT_CENT_COL_SL_{style}")
LAT_RIGHT_COL_SL = eval(f"LAT_RIGHT_COL_SL_{style}")
VECTORS_DIRECTIONS = eval(f"VECTORS_DIRECTIONS_{style}")
CYL_BOUNCE_BACK_DIRECTIONS = eval(f"CYL_BOUNCE_BACK_DIRECTIONS_{style}")


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

def poiseuille_profile(y_phys):
    return 4 * U_MAX / (PIPE_LENGTH ** 2) * (y_phys * PIPE_LENGTH - y_phys * y_phys)

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
