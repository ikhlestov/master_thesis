from constants import *

def calc_ba_eq(F, rho, ux, uy):
    F_eq = np.zeros(F.shape)
    for i, cx, cy, w in zip(VECTOR_INDEXES, VECTORS_VELOCITIES_X, VECTORS_VELOCITIES_Y, VECTORS_WEIGHTS):
        # (100, 400)
        F_eq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
    # print_v("F_eq_ba:\n", F_eq[0][0])
    return F_eq

def calc(F):

    # Set reflective boundaries
    bndryF = F[CYLINDER_MASK, :]   # shape: (3405, 9)
    # Action: to all cylinder coordinates assign such f_i values
    bndryF = bndryF[:, CYL_BOUNCE_BACK_DIRECTIONS]

    ### 1. Compute moments (for each latice)
    rho = np.sum(F, 2)  # shape: (100, 400)
    ux  = np.sum(F * VECTORS_VELOCITIES_X, 2) / rho   # shape: (100, 400)
    uy  = np.sum(F * VECTORS_VELOCITIES_Y, 2) / rho   # shape: (100, 400)
    F_eq = calc_ba_eq(F, rho, ux, uy)
    F = F - (F - F_eq) / TAU

    # Apply boundary
    # F: 100, 400, 9
    # CYLINDER_MASK: 100, 400
    # F[CYLINDER_MASK]: 1941, 9
    # bndryF: 1941, 9
    F[CYLINDER_MASK, :] = bndryF

    ### 4. Propagate to the neighbours
    for i, cx, cy in zip(VECTOR_INDEXES, VECTORS_VELOCITIES_X, VECTORS_VELOCITIES_Y):
        F[:, :, i] = np.roll(F[:, :, i], (cx, cy), axis=(1, 0))
    # for vis
    vorticity = (
        (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - 
        (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    )
    rho = np.sum(F, 2)
    ux  = np.sum(F * VECTORS_VELOCITIES_X, 2) / rho
    uy  = np.sum(F * VECTORS_VELOCITIES_Y, 2) / rho
    u = np.sqrt(ux ** 2 + uy ** 2)
    return F
