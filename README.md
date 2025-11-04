# Deep-Learning-Project

Fall 2025 CS 7643 DL project

## Introduction

Modeling Unmanned Surface Vehicle (USV) dynamics is difficult due to nonlinear hydrodynamics and unmodeled disturbances. We propose a hybrid residual learning framework that augments Fossen’s 3-DOF model with a neural network trained on Blue Robotics BlueBoat data in Gazebo. Sequence models (LSTMs, TCNs) capture temporal effects like wake buildup, while physics-informed regularization enforces energy and damping consistency. This approach enhances robustness and generalization while maintaining physical interpretability for reliable autonomous navigation.

## Data Preprocessing
### data/main.py
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
global.altitude, global.latitude, global.longitude,
imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z,
imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z,
odom.orientation.w, odom.orientation.x, odom.orientation.y, odom.orientation.z,
odom.position.x, odom.position.y, odom.position.z,
odom.twist.angular.x, odom.twist.angular.y, odom.twist.angular.z,
odom.twist.linear.x, odom.twist.linear.y, odom.twist.linear.z,
wp.current_seq,
psi,                                    # yaw (from quaternion)
u, v, r,                                # raw body-frame velocities and yaw rate
u_filt, v_filt, r_filt,                 # filtered versions (Butterworth)
du_dt, dv_dt, dr_dt,                    # accelerations (central difference)
cos_psi, sin_psi,                       # trigonometric features
u_filt_norm, v_filt_norm, r_filt_norm,  # normalized velocity features
du_dt_norm, dv_dt_norm, dr_dt_norm      # normalized acceleration features
```

Notes:
- u, v, r are extracted from odometry twist signals
- psi (yaw) is computed from odom.orientation.* quaternion
- Filtered columns are smoothed using a zero-phase Butterworth filter
- Normalized columns use mean and std values recorded in <output>.norm.json
- The wp.current_seq field identifies the current waypoint index



