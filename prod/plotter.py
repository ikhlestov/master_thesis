import tensorflow as tf
import numpy as np

from const import (
    VECTORS_VELOCITIES_X,
    VECTORS_VELOCITIES_Y,
    dtype
)
from helpers import tf_to_numpy


def _copy_small_arr_to_large(arr, arr_large):
    arr_large[1:-1, 1:-1].assign(arr)
    arr_large[0, 1:-1].assign(arr[-1, :])
    arr_large[-1, 1:-1].assign(arr[0, :])
    arr_large[1:-1, 0].assign(arr[:, -1])
    arr_large[1:-1, -1].assign(arr[:, 0])
    arr_large[0, 0].assign(arr[-1, -1])
    arr_large[-1, -1].assign(arr[0, 0])
    arr_large[0, -1].assign(arr[-1, 0])
    arr_large[-1, 0].assign(arr[0, -1])
    return arr_large


@tf.function
def pre_plot_tf(F, ux_l_var, uy_l_var):
    # shape: (length_y, length_x)
    rho = tf.math.reduce_sum(F, axis=2)
    ux  = tf.math.reduce_sum(F * VECTORS_VELOCITIES_X, 2) / rho
    uy  = tf.math.reduce_sum(F * VECTORS_VELOCITIES_Y, 2) / rho
    u = tf.sqrt(ux ** 2 + uy ** 2)

    ux_l_var = _copy_small_arr_to_large(ux, ux_l_var)
    uy_l_var = _copy_small_arr_to_large(uy, uy_l_var)
    vorticity = (
        (ux_l_var[2:, 1:-1] - ux_l_var[0:-2, 1:-1]) -
        (uy_l_var[1:-1, 2:] - uy_l_var[1:-1, 0:-2])
    )
    return u, vorticity, rho


@tf_to_numpy
def pre_plot(F, ux_l=None, uy_l=None):
    if ux_l is None:
        ux_l = np.zeros((F.shape[0] + 2, F.shape[1] + 2))
        uy_l = np.zeros((F.shape[0] + 2, F.shape[1] + 2))
    # TODO: drop unnecessary variable creation
    F_var = tf.Variable(F, dtype=dtype)
    ux_l_var = tf.Variable(ux_l, dtype=dtype)
    uy_l_var = tf.Variable(uy_l, dtype=dtype)
    return pre_plot_tf(F_var, ux_l_var, uy_l_var)
