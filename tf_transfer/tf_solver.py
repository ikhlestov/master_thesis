try:
    from .constants import *
except ImportError:
    from constants import *

import tensorflow as tf

def tf_to_numpy(func):
    def inner(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, (list, tuple)):
            res = [item.numpy() for item in res]
        else:
            res = res.numpy()
        return res
    return inner  

CYL_MASKED_SHAPE = np.sum(CYLINDER_MASK)
indexes = []
# for idx in range(int(bndryF_reshaped.shape[0])):
for idx in range(CYL_MASKED_SHAPE * 9):
    init_idx = CYL_BOUNCE_BACK_DIRECTIONS[idx % 9]
    indexes.append(init_idx + (idx // 9) * 9)


### Tensorflow helper methods
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


tf_params = build_graph()
def calc_tf_eq_core(data_flat):
    velocities_x_tf, velocities_y_tf, weights_tf, sum_conv, vel_x_conv, vel_y_conv = tf_params
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
dtype = tf.float32

F = init_poiseuille()
bndryF = tf.Variable(F)
F_var = tf.Variable(F)
# To insert by indexes - https://www.tensorflow.org/api_docs/python/tf/scatter_nd
# Extract complex indexes - https://www.tensorflow.org/api_docs/python/tf/gather_nd
def calc_inner(F, F_res, wide_F_tf, cyl_mask_np, tau):
    # Set reflective boundaries
    # shape: (3405, 9)
    # F_r = tf.expand_dims(F, axis=0)
    
    # bndryF = tf.boolean_mask(F, CYLINDER_MASK)
    # # Action: to all cylinder coordinates assign such f_i values
    # bndryF_reshaped = tf.reshape(bndryF, [CYL_MASKED_SHAPE * 9])
    # bndryF_reshaped_mixed = tf.gather(bndryF_reshaped, indexes)
    # bndryF = tf.reshape(bndryF_reshaped_mixed, (CYL_MASKED_SHAPE, 9))
    
    
    # for_stack = []
    # for idx_old, idx_new in enumerate(CYL_BOUNCE_BACK_DIRECTIONS):
    #     for_stack.append(F[:, :, idx_new])
    #     bndryF[:, :, idx_old].assign(F[:, :, idx_new])
    
    # bndryF = tf.stack(for_stack, axis=2)
    
    bndryF = tf.cast(
        tf.squeeze(boundary_conv(tf.expand_dims(F, axis=0))),
        dtype
    )
    
    ### 1. Compute moments (for each latice)
    # rho = tf.math.reduce_sum(F, axis=2)
    # ux  = tf.math.reduce_sum(F * VECTORS_VELOCITIES_X, 2) / rho   # shape: (100, 400)
    # uy  = tf.math.reduce_sum(F * VECTORS_VELOCITIES_Y, 2) / rho   # shape: (100, 400)
    F_eq = tf.cast(
        tf.squeeze(calc_tf_eq_core(tf.expand_dims(F, axis=0))),
        dtype
    )
    F = tf.cast(F, dtype)
    bndryF = tf.cast(bndryF, dtype)
    # F = F - (F - F_eq) / TAU
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

    # TODO: we can work with wide F, and only slice in some places where it required
    # wide_F_tf = tf.constant(0, shape=(100 + 2, 400 + 2, 9))
    wide_F_tf[1:-1, 1:-1, :].assign(F)
    wide_F_tf[0, 1:-1, :].assign(F[-1, :, :])
    wide_F_tf[-1, 1:-1, :].assign(F[0, :, :])
    wide_F_tf[1:-1, 0, :].assign(F[:, -1, :])
    wide_F_tf[1:-1, -1, :].assign(F[:, 0, :])
    wide_F_tf[0, 0, :].assign(F[-1, -1, :])
    wide_F_tf[-1, -1, :].assign(F[0, 0, :])
    wide_F_tf[0, -1, :].assign(F[-1, 0, :])
    wide_F_tf[-1, 0, :].assign(F[0, -1, :])
    for i, cx, cy in zip(VECTOR_INDEXES, VECTORS_VELOCITIES_X, VECTORS_VELOCITIES_Y):
        F_res[:, :, i].assign(wide_F_tf[:, :, i][1-cy:F.shape[0] + 1 - cy, 1-cx:F.shape[1] + 1 - cx])

    F = F_res
    return F

wide_F = np.zeros((F.shape[0] + 2, F.shape[1] + 2, 9))
tf_step = tf.function(calc_inner)

@tf_to_numpy
def calc(F, wide_F=None, cyl_mask_np=None, tau=None):
    # return calc_inner(F)
    if wide_F is None:
        wide_F = np.zeros((F.shape[0] + 2, F.shape[1] + 2, 9))
        cyl_mask_np = CYLINDER_MASK.astype(np.uint8)
        tau = TAU
    F_res = tf.Variable(F, dtype=dtype)
    wide_F_tf = tf.Variable(wide_F, dtype=dtype)
    return tf_step(F, F_res, wide_F_tf, tf.constant(cyl_mask_np), tau)


def pre_plot_inner(F, ux_l_var, uy_l_var):
    rho = tf.math.reduce_sum(F, axis=2)
    ux  = tf.math.reduce_sum(F * VECTORS_VELOCITIES_X, 2) / rho   # shape: (100, 400)
    uy  = tf.math.reduce_sum(F * VECTORS_VELOCITIES_Y, 2) / rho   # shape: (100, 400)
    u = tf.sqrt(ux ** 2 + uy ** 2)

    # vorticity = (
    #     (tf.roll(ux, -1, axis=0) - tf.roll(ux, 1, axis=0)) - 
    #     (tf.roll(uy, -1, axis=1) - tf.roll(uy, 1, axis=1))
    # )

    ux_l_var = copy_small_arr_to_large(ux, ux_l_var)
    uy_l_var = copy_small_arr_to_large(uy, uy_l_var)
 
    vorticity = (
        (ux_l_var[2:, 1:-1] - ux_l_var[0:-2, 1:-1]) -
        (uy_l_var[1:-1, 2:] - uy_l_var[1:-1, 0:-2])
    )

    return u, vorticity, rho

pre_plot_tf_func = tf.function(pre_plot_inner)


def copy_small_arr_to_large(arr, arr_large):
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


def enlarge_array(arr):
    """Create enlarged array with shape height x width"""
    arr_large = np.zeros((arr.shape[0] + 2, arr.shape[1] + 2))
    arr_large = copy_small_arr_to_large(arr, arr_large)
    return arr_large


@tf_to_numpy
def pre_plot(F, ux_l=None, uy_l=None):
    if ux_l is None:
        ux_l = np.zeros((F.shape[0] + 2, F.shape[1] + 2))
        uy_l = np.zeros((F.shape[0] + 2, F.shape[1] + 2))
    F_tf = tf.Variable(F, dtype=dtype)
    ux_l_var = tf.Variable(ux_l, dtype=dtype)
    uy_l_var = tf.Variable(uy_l, dtype=dtype)
    return pre_plot_tf_func(F_tf, ux_l_var, uy_l_var)
