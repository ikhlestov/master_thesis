try:
    from .constants import *
except ImportError:
    from constants import *

def calc_ba_eq(F, rho, ux, uy):
    F_eq = np.zeros(F.shape)
    for i, cx, cy, w in zip(VECTOR_INDEXES, VECTORS_VELOCITIES_X, VECTORS_VELOCITIES_Y, VECTORS_WEIGHTS):
        # (100, 400)
        lat_u = cx*ux+cy*uy
        F_eq[:,:,i] = rho * w * (
            1
            + 3 * lat_u
            + 9 * lat_u ** 2 / 2
            - 3 * (ux ** 2 + uy ** 2) / 2 
        )
    # print_v("F_eq_ba:\n", F_eq[0][0])
    return F_eq

def calc(F, cylinder_mask=None, tau=None, u_max=None, pipe_length=None):
    if cylinder_mask is None:
        tau = TAU
        cylinder_mask = CYLINDER_MASK
        u_max = U_MAX
        pipe_length = PIPE_LENGTH

    # Set reflective boundaries
    bndryF = F[cylinder_mask, :]   # shape: (3405, 9)
    # Action: to all cylinder coordinates assign such f_i values
    bndryF = bndryF[:, CYL_BOUNCE_BACK_DIRECTIONS]

    ### 1. Compute moments (for each latice)
    rho = np.sum(F, 2)  # shape: (100, 400)
    ux  = np.sum(F * VECTORS_VELOCITIES_X, 2) / rho   # shape: (100, 400)
    uy  = np.sum(F * VECTORS_VELOCITIES_Y, 2) / rho   # shape: (100, 400)

    ### 1.1 Apply fixes for inlet / outlet
    # y_phys = np.arange(pipe_length)
    # ux[INLET_SL] = poiseuille_profile(y_phys, u_max=u_max, pipe_length=pipe_length)
    # uy[INLET_SL] = 0
    # rho[INLET_SL] = 1 / (1 - ux[INLET_SL]) * (
    #     F[INLET_SL][LAT_CENT_COL_SL].sum(axis=1) +
    #     2 * F[INLET_SL][LAT_LEFT_COL_SL].sum(axis=1))

    F_eq = calc_ba_eq(F, rho, ux, uy)
    # F = F - (F - F_eq) / tau
    omega = 1 / tau
    F = (1 - omega) * F + omega * F_eq


    # Apply boundary
    # F: 100, 400, 9
    # cylinder_mask: 100, 400
    # F[cylinder_mask]: 1941, 9
    # bndryF: 1941, 9
    F[cylinder_mask, :] = bndryF

    ### 4. Propagate to the neighbours
    for i, cx, cy in zip(VECTOR_INDEXES, VECTORS_VELOCITIES_X, VECTORS_VELOCITIES_Y):
        F[:, :, i] = np.roll(F[:, :, i], (cx, cy), axis=(1, 0))
    return F


def pre_plot(F):
    rho = np.sum(F, 2)
    ux  = np.sum(F * VECTORS_VELOCITIES_X, 2) / rho
    uy  = np.sum(F * VECTORS_VELOCITIES_Y, 2) / rho
    u = np.sqrt(ux ** 2 + uy ** 2)
    
    vorticity = (
        (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - 
        (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    )
    return u, vorticity
    