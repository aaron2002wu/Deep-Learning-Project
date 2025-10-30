import math, json, re, ast
import numpy as np
import pandas as pd
from scipy.signal import butter as _butter, filtfilt as _filtfilt

# Math helpers
def quat_to_yaw(w, x, y, z):
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def bw_filter(series, fs_hz, cutoff=1.2, order=4):
    """Zero-phase lowpass on a 1D pandas Series."""
    nyq = 0.5*fs_hz
    Wn = min(max(cutoff/nyq, 1e-6), 0.999999)
    b, a = _butter(order, Wn, btype='low')
    return pd.Series(_filtfilt(b, a, series.to_numpy(), padlen=9), index=series.index)



# Parsing helpers
_BAD_SUFFIX = {"check_fields", "covariance", "covariance_type"}

def _ok_suffix(s):
    return (s not in _BAD_SUFFIX) and (not s.endswith("_covariance"))

def map_long_to_wide_col(topic: str, field: str):
    if "/mavros/local_position/odom" in topic:
        for prefix, out in [
            ("pose.pose.orientation.", "odom.orientation."),
            ("pose.pose.position."   , "odom.position."),
            ("twist.twist.linear."   , "odom.twist.linear."),
            ("twist.twist.angular."  , "odom.twist.angular."),
        ]:
            if field.startswith(prefix):
                comp = field.rsplit(".", 1)[-1]
                return f"{out}{comp}" if _ok_suffix(comp) else None
        return None
    if "/mavros/imu/data" in topic:
        for prefix, out in [
            ("angular_velocity.", "imu.angular_velocity."),
            ("linear_acceleration.", "imu.linear_acceleration."),
            ("orientation.", "imu.orientation."),
        ]:
            if field.startswith(prefix):
                comp = field.rsplit(".", 1)[-1]
                return f"{out}{comp}" if _ok_suffix(comp) else None
        return None
    if "/mavros/global_position/global" in topic:
        if field in ("latitude", "longitude", "altitude"):
            return f"global.{field}"
        return None
    if "/mavros/mission/waypoints" in topic:
        return "wp.current_seq" if field == "current_seq" else None
    
    if "cmd_thrust" in topic:
        return field
    
    return None

def coerce_numeric(val):
    """strings->floats and drop non numeric vals"""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def expand_cmd_thrust(df_long):
    """
    Expand /cmd_thrust rows with array('d', [port, center, starboard]) values
    into separate long-format rows for port and starboard thrusters.
    """
    mask = df_long["topic"].str.contains("cmd_thrust", na=False) & df_long["field"].eq("data")
    thrust_rows = df_long.loc[mask].copy()

    if thrust_rows.empty:
        print("[expand_cmd_thrust] No cmd_thrust rows found.")
        return pd.DataFrame(columns=df_long.columns)

    def parse_array(val):
        """Extract list of floats from array('d', [..]) or [..] string."""
        match = re.search(r"\[([^\]]+)\]", str(val))
        if not match:
            return [None, None, None]
        try:
            nums = [float(x.strip()) for x in match.group(1).split(",")]
            # pad or trim to exactly 3 values
            return (nums + [None] * 3)[:3]
        except Exception:
            return [None, None, None]

    thrust_rows["values"] = thrust_rows["value"].apply(parse_array)

    # Split into columns
    thrust_expanded = pd.DataFrame(
        thrust_rows["values"].tolist(),
        columns=["cmd_thrust.port", "cmd_thrust.center", "cmd_thrust.starboard"]
    )
    thrust_expanded["timestamp"] = thrust_rows["timestamp"].values

    # Drop center if always zero or None
    if "cmd_thrust.center" in thrust_expanded:
        if thrust_expanded["cmd_thrust.center"].fillna(0).abs().sum() == 0:
            thrust_expanded = thrust_expanded.drop(columns=["cmd_thrust.center"])

    # Melt into long-format compatible with main pipeline
    thrust_long = thrust_expanded.melt(
        id_vars=["timestamp"], var_name="field", value_name="value"
    )
    thrust_long["topic"] = "/bb04/experiment1/cmd_thrust"

    thrust_long = thrust_long[["timestamp", "topic", "field", "value"]]
    print(f"[expand_cmd_thrust] Expanded {len(thrust_long)} thrust samples.")
    return thrust_long


# Pipeline functions
def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Inject cmd_thrust expansion before mapping ---
    thrust_long = expand_cmd_thrust(df)
    df = pd.concat([df, thrust_long], ignore_index=True)

    mapped = df.assign(
        col=df.apply(lambda r: map_long_to_wide_col(r["topic"], r["field"]), axis=1),
        val=df["value"].map(coerce_numeric),
    )
    mapped = mapped.dropna(subset=["col", "val"])

    wide = (mapped
            .pivot_table(index="timestamp", columns="col", values="val", aggfunc="last")
            .sort_index())
    wide.index = pd.to_datetime(wide.index, unit='s')
    return wide

def resample_sync(wide: pd.DataFrame, resample_hz: float | None):
    """Returns (wide_resampled, fs_hz)"""
    if resample_hz and resample_hz > 0:
        rule = f"{int(1000/resample_hz)}ms"
        wide = (wide
                .resample(rule).mean()
                .interpolate(method="time")
                .ffill().bfill())
        fs = float(resample_hz)
    else:
        dt_med = wide.index.to_series().diff().dt.total_seconds().median()
        fs = 1.0/dt_med if (dt_med and dt_med > 0) else 10.0
    return wide, fs

def derive_states(wide: pd.DataFrame):
    """Compute psi, u, v, r"""
    need = {"odom.orientation.w","odom.orientation.x","odom.orientation.y","odom.orientation.z"}
    if need.issubset(wide.columns):
        q = wide[["odom.orientation.w","odom.orientation.x","odom.orientation.y","odom.orientation.z"]].to_numpy()
        wide["psi"] = np.array([quat_to_yaw(w,x,y,z) for (w,x,y,z) in q])
    else:
        wide["psi"] = np.nan

    wide["u"] = wide.get("odom.twist.linear.x")
    wide["v"] = wide.get("odom.twist.linear.y")
    wide["r"] = (wide["odom.twist.angular.z"]
                 if "odom.twist.angular.z" in wide.columns
                 else wide.get("imu.angular_velocity.z"))
    return wide

def filter_signals(wide: pd.DataFrame, fs: float, cutoff=1.2, order=4):
    for c in ["u","v","r"]:
        if c in wide.columns:
            s = wide[c].interpolate().ffill().bfill()
            wide[c+"_filt"] = bw_filter(s, fs_hz=fs, cutoff=cutoff, order=order)
    return wide

def finite_diff(wide: pd.DataFrame, fs: float):
    """Centered finite differences for filtered states"""
    dt = 1.0/fs
    for src_col, out_col in [("u_filt","du_dt"), ("v_filt","dv_dt"), ("r_filt","dr_dt")]:
        if src_col in wide.columns:
            arr = wide[src_col].to_numpy()
            acc = np.empty_like(arr, dtype=float)
            acc[1:-1] = (arr[2:] - arr[:-2]) / (2*dt)
            acc[0]    = (arr[1] - arr[0]) / dt
            acc[-1]   = (arr[-1] - arr[-2]) / dt
            wide[out_col] = acc
    return wide

def trig_and_norm(wide: pd.DataFrame):
    wide["cos_psi"] = np.cos(wide["psi"])
    wide["sin_psi"] = np.sin(wide["psi"])

    norm_cols = [c for c in ["u_filt","v_filt","r_filt","du_dt","dv_dt","dr_dt"] if c in wide.columns]
    stats = {}
    for c in norm_cols:
        mu = float(wide[c].mean())
        sd = float(wide[c].std(ddof=0)) or 1.0
        wide[c+"_norm"] = (wide[c] - mu)/sd
        stats[c] = {"mean": mu, "std": sd}
    return wide, stats

def finalize_export(wide: pd.DataFrame):
    out = wide.reset_index().rename(columns={"index": "time", "timestamp": "time"})
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize(None)
    return out

def save_outputs(df_out: pd.DataFrame, out_csv: str, norm_stats: dict):
    df_out.to_csv(out_csv, index=False)
    with open(out_csv + ".norm.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
