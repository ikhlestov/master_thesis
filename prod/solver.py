import tensorflow as tf
import numpy as np

from initializers import init_poiseuille
from const import (
    VECTORS_VELOCITIES_X,
    VECTORS_VELOCITIES_Y,
    VECTORS_WEIGHTS,
    VECTOR_INDEXES,
    dtype
)
from helpers import tf_to_numpy


def build_graph():
    dtype = tf.float32
    velocities_x_tf = tf.constant(VECTORS_VELOCITIES_X, dtype=dtype)
    velocities_y_tf = tf.constant(VECTORS_VELOCITIES_Y, dtype=dtype)
    weights_tf = tf.constant(VECTORS_WEIGHTS, dtype=dtype)
    
    def ones_init(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[0, 0, :, 0] = 1.0
        return tf.cast(kernel, dtype)

    sum_conv = tf.keras.layers.Conv2D(1, (1, 1), kernel_initializer=ones_init)

    def vel_x_init_many_to_one(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[0, 0, :, 0] = VECTORS_VELOCITIES_X
        return tf.cast(kernel, dtype)

    vel_x_conv = tf.keras.layers.Conv2D(1, (1, 1), kernel_initializer=vel_x_init_many_to_one)

    def vel_y_init_many_to_one(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[0, 0, :, 0] = VECTORS_VELOCITIES_Y
        return tf.cast(kernel, dtype)

    vel_y_conv = tf.keras.layers.Conv2D(1, (1, 1), kernel_initializer=vel_y_init_many_to_one)
    return velocities_x_tf, velocities_y_tf, weights_tf, sum_conv, vel_x_conv, vel_y_conv


TF_PARAMS = build_graph()
def calc_tf_eq_core(data_flat):
    velocities_x_tf, velocities_y_tf, weights_tf, sum_conv, vel_x_conv, vel_y_conv = TF_PARAMS
    batch = data_flat
    rho = sum_conv(batch)
    # rho = tf.cast(tf.expand_dims(tf.math.reduce_sum(batch, axis=3), axis=-1), tf.float32)
    ux_lattices = vel_x_conv(batch) / rho
    uy_lattices = vel_y_conv(batch) / rho
    ux_elements = tf.math.multiply(ux_lattices, velocities_x_tf)
    uy_elements = tf.math.multiply(uy_lattices, velocities_y_tf)
    before_weights = (
        1 + 3 * (ux_elements + uy_elements) +
        9 * (ux_elements + uy_elements) ** 2 / 2 - 
        3 * (ux_lattices ** 2 + uy_lattices ** 2) / 2
    )
    after_weights = tf.math.multiply(before_weights, weights_tf)
    F_eq = tf.math.multiply(rho, after_weights)
    return F_eq


def boundary_init(shape, dtype=None, partition_info=None):
    kernel = np.zeros(shape)
    kernel[0, 0, :, :] = tf.constant(np.array([
       [0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    )
    return tf.cast(kernel, dtype)

boundary_conv = tf.keras.layers.Conv2D(9, (1, 1), kernel_initializer=boundary_init)


@tf.function
def tf_step(F, F_var, wide_F_var, cyl_mask_np, tau):
    bndryF = tf.cast(
        tf.squeeze(boundary_conv(tf.expand_dims(F, axis=0))),
        dtype
    )
    F_eq = tf.cast(
        tf.squeeze(calc_tf_eq_core(tf.expand_dims(F, axis=0))),
        dtype
    )
    F = tf.cast(F, dtype)
    omega = 1 / tau
    F = (1 - omega) * F + omega * F_eq
    cyl_mask = tf.repeat(
        tf.expand_dims(
            # tf.constant(cyl_mask_np, dtype=np.uint8),
            cyl_mask_np,
            -1
        ), 9, axis=2
    )
    F = F * tf.cast(1 - cyl_mask, dtype) + bndryF * tf.cast(cyl_mask, dtype)

    wide_F_var[1:-1, 1:-1, :].assign(F)
    wide_F_var[0, 1:-1, :].assign(F[-1, :, :])
    wide_F_var[-1, 1:-1, :].assign(F[0, :, :])
    wide_F_var[1:-1, 0, :].assign(F[:, -1, :])
    wide_F_var[1:-1, -1, :].assign(F[:, 0, :])
    wide_F_var[0, 0, :].assign(F[-1, -1, :])
    wide_F_var[-1, -1, :].assign(F[0, 0, :])
    wide_F_var[0, -1, :].assign(F[-1, 0, :])
    wide_F_var[-1, 0, :].assign(F[0, -1, :])
    for i, cx, cy in zip(VECTOR_INDEXES, VECTORS_VELOCITIES_X, VECTORS_VELOCITIES_Y):
        F_var[:, :, i].assign(wide_F_var[:, :, i][1-cy:F.shape[0] + 1 - cy, 1-cx:F.shape[1] + 1 - cx])

    F = F_var
    return F


@tf_to_numpy
def calc(F, wide_F, obj_mask_np, tau):
    F_var = tf.Variable(F, dtype=dtype)
    wide_F_var = tf.Variable(wide_F, dtype=dtype)
    return tf_step(F, F_var, wide_F_var, tf.constant(obj_mask_np), tau)
