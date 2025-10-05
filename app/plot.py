import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load your data
data = pd.read_csv("dataset/data_P1_DEVICE_001_Jumping_20251005_020345.csv")

# Filter by participant and activity if needed
participant_id = 1
activity = "Jumping"

filtered = data[
    (data["Participant_ID"] == participant_id) & 
    (data["Activity_Type"] == activity)
].copy()

# Take first 200 samples for clarity
filtered = filtered.head(200)

print(f"Plotting {len(filtered)} samples for Participant {participant_id}, Activity: {activity}")

# Create a figure with multiple 3D subplots
fig = plt.figure(figsize=(16, 12))

# ============= MPU6050 SENSOR =============
# Accelerometer
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(filtered['mpu_ax'], filtered['mpu_ay'], filtered['mpu_az'], 
         color='blue', linewidth=1, alpha=0.7)
ax1.scatter(filtered['mpu_ax'].iloc[0], filtered['mpu_ay'].iloc[0], 
            filtered['mpu_az'].iloc[0], color='green', s=100, label='Start')
ax1.scatter(filtered['mpu_ax'].iloc[-1], filtered['mpu_ay'].iloc[-1], 
            filtered['mpu_az'].iloc[-1], color='red', s=100, label='End')
ax1.set_xlabel('Accel X (m/s²)')
ax1.set_ylabel('Accel Y (m/s²)')
ax1.set_zlabel('Accel Z (m/s²)')
ax1.set_title('MPU6050 Accelerometer')
ax1.legend()

# Gyroscope
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(filtered['mpu_gx'], filtered['mpu_gy'], filtered['mpu_gz'], 
         color='orange', linewidth=1, alpha=0.7)
ax2.scatter(filtered['mpu_gx'].iloc[0], filtered['mpu_gy'].iloc[0], 
            filtered['mpu_gz'].iloc[0], color='green', s=100, label='Start')
ax2.scatter(filtered['mpu_gx'].iloc[-1], filtered['mpu_gy'].iloc[-1], 
            filtered['mpu_gz'].iloc[-1], color='red', s=100, label='End')
ax2.set_xlabel('Gyro X (rad/s)')
ax2.set_ylabel('Gyro Y (rad/s)')
ax2.set_zlabel('Gyro Z (rad/s)')
ax2.set_title('MPU6050 Gyroscope')
ax2.legend()

# ============= ADXL345 SENSOR =============
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot(filtered['adxl_ax'], filtered['adxl_ay'], filtered['adxl_az'], 
         color='purple', linewidth=1, alpha=0.7)
ax3.scatter(filtered['adxl_ax'].iloc[0], filtered['adxl_ay'].iloc[0], 
            filtered['adxl_az'].iloc[0], color='green', s=100, label='Start')
ax3.scatter(filtered['adxl_ax'].iloc[-1], filtered['adxl_ay'].iloc[-1], 
            filtered['adxl_az'].iloc[-1], color='red', s=100, label='End')
ax3.set_xlabel('Accel X')
ax3.set_ylabel('Accel Y')
ax3.set_zlabel('Accel Z')
ax3.set_title('ADXL345 Accelerometer')
ax3.legend()

# ============= L3GD20 SENSOR =============
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.plot(filtered['l3gd_gx'], filtered['l3gd_gy'], filtered['l3gd_gz'], 
         color='cyan', linewidth=1, alpha=0.7)
ax4.scatter(filtered['l3gd_gx'].iloc[0], filtered['l3gd_gy'].iloc[0], 
            filtered['l3gd_gz'].iloc[0], color='green', s=100, label='Start')
ax4.scatter(filtered['l3gd_gx'].iloc[-1], filtered['l3gd_gy'].iloc[-1], 
            filtered['l3gd_gz'].iloc[-1], color='red', s=100, label='End')
ax4.set_xlabel('Gyro X')
ax4.set_ylabel('Gyro Y')
ax4.set_zlabel('Gyro Z')
ax4.set_title('L3GD20 Gyroscope')
ax4.legend()

# ============= LSM303 SENSOR =============
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.plot(filtered['lsm_ax'], filtered['lsm_ay'], filtered['lsm_az'], 
         color='magenta', linewidth=1, alpha=0.7)
ax5.scatter(filtered['lsm_ax'].iloc[0], filtered['lsm_ay'].iloc[0], 
            filtered['lsm_az'].iloc[0], color='green', s=100, label='Start')
ax5.scatter(filtered['lsm_ax'].iloc[-1], filtered['lsm_ay'].iloc[-1], 
            filtered['lsm_az'].iloc[-1], color='red', s=100, label='End')
ax5.set_xlabel('Accel X')
ax5.set_ylabel('Accel Y')
ax5.set_zlabel('Accel Z')
ax5.set_title('LSM303 Accelerometer')
ax5.legend()

# ============= JOYSTICK DATA =============
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
# Plot left joystick (using button state as Z)
ax6.plot(filtered['joy_lx'], filtered['joy_ly'], filtered['joy_lb'], 
         color='red', linewidth=1, alpha=0.7, label='Left Stick')
# Plot right joystick
ax6.plot(filtered['joy_rx'], filtered['joy_ry'], filtered['joy_rb'], 
         color='blue', linewidth=1, alpha=0.7, label='Right Stick')
ax6.set_xlabel('X Position')
ax6.set_ylabel('Y Position')
ax6.set_zlabel('Button State')
ax6.set_title('Joystick Input')
ax6.legend()

plt.suptitle(f'Sensor Data Visualization - {activity} (Participant {participant_id})', 
             fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

# ============= ADDITIONAL 2D TIME SERIES PLOTS =============
# MPU6050 Time Series
fig2, axes = plt.subplots(2, 1, figsize=(14, 8))

# Accelerometer over time
axes[0].plot(filtered.index, filtered['mpu_ax'], label='Accel X', alpha=0.7)
axes[0].plot(filtered.index, filtered['mpu_ay'], label='Accel Y', alpha=0.7)
axes[0].plot(filtered.index, filtered['mpu_az'], label='Accel Z', alpha=0.7)
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Acceleration (m/s²)')
axes[0].set_title('MPU6050 Accelerometer - Time Series')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gyroscope over time
axes[1].plot(filtered.index, filtered['mpu_gx'], label='Gyro X', alpha=0.7)
axes[1].plot(filtered.index, filtered['mpu_gy'], label='Gyro Y', alpha=0.7)
axes[1].plot(filtered.index, filtered['mpu_gz'], label='Gyro Z', alpha=0.7)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Angular Velocity (rad/s)')
axes[1].set_title('MPU6050 Gyroscope - Time Series')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============= JOYSTICK VISUALIZATION =============
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left Joystick 2D
axes[0].scatter(filtered['joy_lx'], filtered['joy_ly'], 
                c=filtered.index, cmap='viridis', s=30, alpha=0.6)
axes[0].axhline(y=2048, color='r', linestyle='--', alpha=0.3, label='Center Y')
axes[0].axvline(x=2048, color='r', linestyle='--', alpha=0.3, label='Center X')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
axes[0].set_title('Left Joystick Movement')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right Joystick 2D
axes[1].scatter(filtered['joy_rx'], filtered['joy_ry'], 
                c=filtered.index, cmap='plasma', s=30, alpha=0.6)
axes[1].axhline(y=2048, color='r', linestyle='--', alpha=0.3, label='Center Y')
axes[1].axvline(x=2048, color='r', linestyle='--', alpha=0.3, label='Center X')
axes[1].set_xlabel('X Position')
axes[1].set_ylabel('Y Position')
axes[1].set_title('Right Joystick Movement')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nData Summary:")
print(f"MPU6050 Accel range: X[{filtered['mpu_ax'].min():.2f}, {filtered['mpu_ax'].max():.2f}], "
      f"Y[{filtered['mpu_ay'].min():.2f}, {filtered['mpu_ay'].max():.2f}], "
      f"Z[{filtered['mpu_az'].min():.2f}, {filtered['mpu_az'].max():.2f}]")
print(f"Temperature range: [{filtered['mpu_temp'].min():.1f}, {filtered['mpu_temp'].max():.1f}]°C")