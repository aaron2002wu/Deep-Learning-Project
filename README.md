# Deep-Learning-Project
Fall 2025 CS 7643 DL project

## Data Preprocessing
### make_dataset.py
#### Args

| Flag | Description | Default |
|------|--------------|----------|
| `--input` | Path to the raw CSV log file (must contain `timestamp`, `topic`, `field`, `value` columns) | *Required* |
| `--output` | Path to save the processed dataset CSV | *Required* |
| `--cutoff` | Butterworth filter cutoff frequency (Hz) | 1.2 |
| `--order` | Butterworth filter order | 4 |
| `--resample-hz` | Target sampling frequency after interpolation (Hz) | 10 |

#### Input

A long-form CSV file containing ROS with the following columns:

- timestamp, topic, field, value


#### Output (wide, synchronized)

CSV contains one row per uniform timestamp with the exact header order shown:

```
time,
imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z,
odom.orientation.w, odom.orientation.x, odom.orientation.y, odom.orientation.z,
odom.position.x, odom.position.y, odom.position.z,
odom.twist.angular.x, odom.twist.angular.y, odom.twist.angular.z,
odom.twist.linear.x, odom.twist.linear.y, odom.twist.linear.z,
u, v, r, psi,
u_filt, v_filt, r_filt,
du_dt, dv_dt, dr_dt,
cos_psi, sin_psi
```

Notes:
- `u, v, r` are taken from `odom.twist.linear.x`, `odom.twist.linear.y`, and `odom.twist.angular.z` (with IMU fallback for `r` if needed)
- `psi` is computed from the quaternion orientation
- `u_filt, v_filt, r_filt` are zero-phase Butterworth filtered velocities
- `du_dt, dv_dt, dr_dt` are central-difference accelerations on the resampled grid
- If thrust signals are available later (e.g., `thrust_port`, `thrust_starboard`), they can be appended and used downstream



