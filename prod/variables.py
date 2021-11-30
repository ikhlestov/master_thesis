import numpy as np


NORM_LENGTH_X = 4
NORM_LENGTH_Y = 1
SCALE = 100
LENGTH_X  = NORM_LENGTH_X * SCALE
LENGTH_Y  = NORM_LENGTH_Y * SCALE

### Object parameters
# X.shape: (LENGTH_Y, LENGTH_X) Y shape: (LENGTH_Y, LENGTH_X)
X, Y = np.meshgrid(range(LENGTH_X), range(LENGTH_Y))
# INFO: shape the same as all space, but only partially filled with object
NORM_CYLINDER_RADIUS = 0.125
CYLINDER_RADIUS = int(NORM_CYLINDER_RADIUS * SCALE)
# True within object boundaries
OBJECT_MASK = (X - LENGTH_X / 4) ** 2 + (Y - LENGTH_Y / 2) ** 2 < (LENGTH_Y // CYLINDER_RADIUS) ** 2
