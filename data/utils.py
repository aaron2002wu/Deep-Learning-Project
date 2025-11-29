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
    
    # Handle converted thrust commands from RC/Out
    if "cmd_thrust" in topic:
        if field in ("cmd_thrust.port", "cmd_thrust.stbd"):
            return field
        return None
    
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


def expand_rc_out_to_thrust(df_long):
    """
    Convert /mavros/rc/out PWM values to thrust commands.
    Extracts channels 0 (port) and 2 (starboard) and converts PWM to thrust_cmd.
    """
    # Find all rc/out rows with 'channels' field
    mask = (df_long["topic"].str.contains("rc/out", na=False) & 
            df_long["field"].eq("channels"))
    rc_rows = df_long.loc[mask].copy()

    if rc_rows.empty:
        print("[expand_rc_out_to_thrust] No rc/out 'channels' rows found.")
        return pd.DataFrame(columns=df_long.columns)

    print(f"[expand_rc_out_to_thrust] Found {len(rc_rows)} RC/Out channel rows")
    
    def parse_channels_and_convert(val):
        """
        Parse channels array string like "array('H', [1500, 0, 1500, ...])"
        Extract port (ch0) and stbd (ch2) PWM values.
        Returns: (port_thrust, stbd_thrust)
        """
        try:
            val_str = str(val).strip()
            
            # Extract the list from "array('H', [...])" format
            # Find the content between [ and ]
            match = re.search(r'\[([^\]]+)\]', val_str)
            if not match:
                return None, None
            
            # Parse the comma-separated numbers
            numbers_str = match.group(1)
            channels = [int(x.strip()) for x in numbers_str.split(',')]
            
            if len(channels) < 3:
                return None, None
            
            # Channel 0 = port, Channel 2 = starboard
            port_pwm = channels[0]
            stbd_pwm = channels[2]
            
            # Skip if PWM values are 0 (invalid)
            if port_pwm == 0 or stbd_pwm == 0:
                return None, None
            
            # Convert PWM to thrust command
            def pwm_to_thrust(pwm):
                normalized = (pwm - 1500) / 500
                return (normalized * 70) - 0.5
            
            port_thrust = pwm_to_thrust(port_pwm)
            stbd_thrust = pwm_to_thrust(stbd_pwm)
            
            return port_thrust, stbd_thrust
            
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            return None, None
    
    # Parse all rows and extract thrust values
    thrust_data = rc_rows['value'].apply(parse_channels_and_convert)
    
    # Split into separate port and stbd columns
    port_values = [t[0] for t in thrust_data]
    stbd_values = [t[1] for t in thrust_data]
    
    # Create port thrust dataframe
    port_thrust = pd.DataFrame({
        'timestamp': rc_rows['timestamp'].values,
        'topic': '/bb04/mavros/cmd_thrust',
        'field': 'cmd_thrust.port',
        'value': port_values
    })
    
    # Create starboard thrust dataframe
    stbd_thrust = pd.DataFrame({
        'timestamp': rc_rows['timestamp'].values,
        'topic': '/bb04/mavros/cmd_thrust',
        'field': 'cmd_thrust.stbd',
        'value': stbd_values
    })
    
    # Combine and remove None values
    thrust_long = pd.concat([port_thrust, stbd_thrust], ignore_index=True)
    initial_len = len(thrust_long)
    thrust_long = thrust_long.dropna(subset=['value'])
    
    print(f"[expand_rc_out_to_thrust] Dropped {initial_len - len(thrust_long)} invalid values")
    
    if not thrust_long.empty:
        port_valid = port_thrust.dropna(subset=['value'])
        stbd_valid = stbd_thrust.dropna(subset=['value'])
        
        if not port_valid.empty:
            print(f"[expand_rc_out_to_thrust] Port thrust: n={len(port_valid)}, range={port_valid['value'].min():.2f} to {port_valid['value'].max():.2f} rad/s")
        if not stbd_valid.empty:
            print(f"[expand_rc_out_to_thrust] Stbd thrust: n={len(stbd_valid)}, range={stbd_valid['value'].min():.2f} to {stbd_valid['value'].max():.2f} rad/s")
        
        print(f"[expand_rc_out_to_thrust] Successfully converted {len(thrust_long)} thrust samples")
    else:
        print("[expand_rc_out_to_thrust] WARNING: No valid thrust values extracted")
    
    return thrust_long


def add_thrust_forces(wide: pd.DataFrame):
    """
    Convert thrust commands (rad/s) to forces (Newtons).
    
    Uses Gazebo thruster model from model.sdf:
    F = thrust_coefficient × fluid_density × propeller_diameter^4 × ω^2
    
    Parameters:
    - thrust_coefficient: -0.02 (port, CW), +0.02 (starboard, CCW) [kg·m]
    - fluid_density: 1025 kg/m³
    - propeller_diameter: 0.112 m
    """
    thrust_coef_port = -0.02  # kg·m (negative for CW rotation)
    thrust_coef_stbd = 0.02   # kg·m (positive for CCW rotation)
    fluid_density = 1025      # kg/m³
    prop_diameter = 0.112     # m
    
    if "cmd_thrust.port" in wide.columns:
        omega_port = wide["cmd_thrust.port"]
        # F = coef × density × diameter^4 × ω^2, preserving sign
        wide["force.port"] = (thrust_coef_port * fluid_density * 
                             (prop_diameter ** 4) * (omega_port.abs() ** 2) * 
                             np.sign(omega_port))
    
    if "cmd_thrust.stbd" in wide.columns:
        omega_stbd = wide["cmd_thrust.stbd"]
        wide["force.stbd"] = (thrust_coef_stbd * fluid_density * 
                             (prop_diameter ** 4) * (omega_stbd.abs() ** 2) * 
                             np.sign(omega_stbd))
    
    # Total force and yaw moment
    if "force.port" in wide.columns and "force.stbd" in wide.columns:
        wide["force.total"] = wide["force.port"] + wide["force.stbd"]
        
        # Thrusters are at ±0.218m from centerline (from xacro)
        # Moment arm for differential thrust
        thruster_separation = 0.436  # meters (2 × 0.218)
        wide["moment.yaw"] = (wide["force.stbd"] - wide["force.port"]) * thruster_separation / 2
    
    return wide


# Pipeline functions
def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format bag data to wide-format time series."""
    df = df.copy()

    # Convert RC/Out PWM to thrust commands
    thrust_long = expand_rc_out_to_thrust(df)
    if not thrust_long.empty:
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
    """Apply Butterworth lowpass filter to velocity signals."""
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
    """Add trigonometric functions and normalize states."""
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
    """Prepare final dataframe for export."""
    out = wide.reset_index().rename(columns={"index": "time", "timestamp": "time"})
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize(None)
    return out

def save_outputs(df_out: pd.DataFrame, out_csv: str, norm_stats: dict):
    """Save processed data and normalization statistics."""
    df_out.to_csv(out_csv, index=False)
    with open(out_csv + ".norm.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"[save_outputs] Saved {len(df_out)} rows to {out_csv}")
    print(f"[save_outputs] Saved normalization stats to {out_csv}.norm.json")