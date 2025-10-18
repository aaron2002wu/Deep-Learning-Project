import argparse, math, json
import numpy as np
import pandas as pd
from scipy.signal import butter as _butter, filtfilt as _filtfilt

def quat_to_yaw(w,x,y,z):
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def bw_filter(series, fs_hz, cutoff=1.2, order=4):
    nyq = 0.5*fs_hz
    Wn = min(max(cutoff/nyq, 1e-6), 0.999999)
    b,a = _butter(order, Wn, btype='low')
    return pd.Series(_filtfilt(b,a, series.to_numpy(), padlen=9), index=series.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cutoff", type=float, default=1.2)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--resample-hz", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # one column per field
    wide = (df.pivot_table(index="timestamp", columns="field",
                           values="value", aggfunc="last")
              .sort_index())

    # Make datetime index
    wide.index = pd.to_datetime(wide.index, unit='s')

    # RESAMPLE + INTERPOLATE to sync IMU and odom
    if args.resample_hz and args.resample_hz > 0:
        rule = f"{int(1000/args.resample_hz)}L"  # e.g., '100L' for 10 Hz
        wide = (wide
                .resample(rule).mean()
                .interpolate(method="time")  # time-based interpolation
                .ffill().bfill())            # guard edges
        fs = float(args.resample_hz)
    else:
        # fall back to median original rate
        dt_med = wide.index.to_series().diff().dt.total_seconds().median()
        fs = 1.0 / dt_med if (dt_med and dt_med > 0) else 10.0

    # Derive states and yaw from quaternion
    if {"odom.orientation.w","odom.orientation.x","odom.orientation.y","odom.orientation.z"}.issubset(wide.columns):
        q = wide[["odom.orientation.w","odom.orientation.x","odom.orientation.y","odom.orientation.z"]].to_numpy()
        wide["psi"] = np.array([quat_to_yaw(w,x,y,z) for (w,x,y,z) in q])
    else:
        wide["psi"] = np.nan

    # states
    wide["u"] = wide.get("odom.twist.linear.x")
    wide["v"] = wide.get("odom.twist.linear.y")
    wide["r"] = wide["odom.twist.angular.z"] if "odom.twist.angular.z" in wide.columns else wide.get("imu.angular_velocity.z")

    # Filter
    for c in ["u","v","r"]:
        if c in wide.columns:
            s = wide[c].interpolate().ffill().bfill()
            wide[c+"_filt"] = bw_filter(s, fs_hz=fs, cutoff=args.cutoff, order=args.order)

    # Accelerations with uniform dt
    dt = 1.0/fs
    for src_col, out_col in [("u_filt","du_dt"), ("v_filt","dv_dt"), ("r_filt","dr_dt")]:
        if src_col in wide.columns:
            arr = wide[src_col].to_numpy()
            acc = np.empty_like(arr, dtype=float)
            acc[1:-1] = (arr[2:] - arr[:-2]) / (2*dt)
            acc[0]     = (arr[1] - arr[0]) / dt
            acc[-1]    = (arr[-1] - arr[-2]) / dt
            wide[out_col] = acc

    # Trig features + normalization
    wide["cos_psi"] = np.cos(wide["psi"])
    wide["sin_psi"] = np.sin(wide["psi"])

    norm_cols = [c for c in ["u_filt","v_filt","r_filt","du_dt","dv_dt","dr_dt"] if c in wide.columns]
    stats = {}
    for c in norm_cols:
        mu = float(wide[c].mean())
        sd = float(wide[c].std(ddof=0)) or 1.0
        wide[c+"_norm"] = (wide[c] - mu)/sd
        stats[c] = {"mean": mu, "std": sd}

    # Export
    out = wide.reset_index().rename(columns={"index": "time", "timestamp": "time"})
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize(None)

    out.to_csv(args.output, index=False)
    with open(args.output + ".norm.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to {args.output} (N={len(out)}), fs≈{fs:.2f} Hz")

if __name__ == "__main__":
    main()
