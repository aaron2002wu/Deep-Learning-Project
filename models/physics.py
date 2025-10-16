import numpy as np

def compute_fossen_accel(nu, tau, params):
    u, v, r = nu
    m, Iz = params["m"], params["Iz"]
    X_u_dot, Y_v_dot, N_r_dot = params["X_u_dot"], params["Y_v_dot"], params["N_r_dot"]
    Xu, Yv, Nr = params["Xu"], params["Yv"], params["Nr"]

    # Mass matrix
    M = np.array([
        [m - X_u_dot, 0, 0],
        [0, m - Y_v_dot, 0],
        [0, 0, Iz - N_r_dot]
    ])

    # Coriolis and damping
    C = np.array([
        [0, -m*r, 0],
        [m*r, 0, 0],
        [0, 0, 0]
    ])

    D = np.diag([-Xu, -Yv, -Nr])

    # Compute acceleration
    nu_dot = np.linalg.inv(M) @ (tau - (C + D) @ nu)
    return nu_dot
