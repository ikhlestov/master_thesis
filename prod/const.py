"""List of totally non modifiable variables"""
import numpy as np
import tensorflow as tf


U_MAX = 0.1  # maximum velocity of Poiseuille inflow
RHO_0 = 1  # average density

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

# 'NW' 'N' 'NE' 'W' 'C' 'E' 'SW' 'S' 'SE'
#   0   1    2   3   4   5    6   7    8
VECTORS_DIRECTIONS = np.array([
    ['NW', 'N', 'NE',],
    ['W',  'C',  'E',],
    ['SW', 'S', 'SE',],
]).reshape(-1)
OBJ_BOUNCE_BACK_DIRECTIONS = [8, 7, 6, 5, 4, 3, 2, 1, 0]

dtype = tf.float32
