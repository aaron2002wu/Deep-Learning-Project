# make_dataset.py


## Loads raw CSV logs.
## Synchronizes timestamps
## Computes u,v,r thrusts, and accelerations
## Computes dot_u dot_v dot_r

import numpy as np
import pandas as pd
from models.physics import compute_fossen_accel

def make_dataset(csv_in, csv_out, params):
    df = pd.read_csv(csv_in)

    # Compute time difference
    df["dt"] = df["time"].diff().fillna(0.1)

    # Finite differences for measured accelerations
    df["u_dot_meas"] = df["u"].diff() / df["dt"]
    df["v_dot_meas"] = df["v"].diff() / df["dt"]
    df["r_dot_meas"] = df["r"].diff() / df["dt"]

    # Compute model accelerations using Fossen 3-DOF
    u_model, v_model, r_model = [], [], []
    for _, row in df.iterrows():
        nu = np.array([row["u"], row["v"], row["r"]])
        tau = np.array([row["thrust_port"], row["thrust_starboard"], 0.0])
        accel = compute_fossen_accel(nu, tau, params)
        u_model.append(accel[0])
        v_model.append(accel[1])
        r_model.append(accel[2])

    df["u_dot_model"] = u_model
    df["v_dot_model"] = v_model
    df["r_dot_model"] = r_model

    # Compute residuals (labels for training)
    df["u_dot_res"] = df["u_dot_meas"] - df["u_dot_model"]
    df["v_dot_res"] = df["v_dot_meas"] - df["v_dot_model"]
    df["r_dot_res"] = df["r_dot_meas"] - df["r_dot_model"]

    df.to_csv(csv_out, index=False)
    print(f"✅ Saved processed dataset to {csv_out}")
