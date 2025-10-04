import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("ai_training/dataset/all_activity_data.csv")

# Sort by participant and timestamp
data = data.sort_values(by=["Participant_ID", "Timestamp"])

# Parameters
window_size = 20  # Number of samples in the window
threshold_accel = 15  # Threshold for acceleration change (jump detection)
threshold_gyro = 200  # Threshold for gyroscope change

def detect_jumps(data, participant_id, movement, window_size=20, threshold_accel=30, threshold_gyro=200):
    """
    Detect jumps using sliding window
    """
    # participant_data = data[data["Participant_ID"] == participant_id].copy()
    # participant_data = participant_data.sort_values(by="Timestamp")
    participant_data = data[(data["Participant_ID"] == participant_id) & (data["Activity_Type"] == movement)].copy()
    participant_data= participant_data.sort_values(by="Timestamp")  

    participant_data = participant_data.head(200)
    
    # Calculate magnitude of acceleration and gyroscope
    # participant_data['accel_mag'] = np.sqrt(
    #     participant_data['Ax']**2 + 
    #     participant_data['Ay']**2 + 
    #     (participant_data['Az']-9.81)**2
    # )
    
    participant_data['gyro_mag'] = np.sqrt(
        participant_data['Gz']**2
    )
    participant_data['accel_mag'] = (participant_data['Az']-9.81)**2
    
    jumps = []
    
    # Sliding window
    for i in range(len(participant_data) - window_size):
        window = participant_data.iloc[i:i+window_size]
        
        # Calculate changes in window
        accel_std = window['accel_mag'].std()
        accel_max_change = window['accel_mag'].max() - window['accel_mag'].min()
        gyro_max_change = window['gyro_mag'].max() - window['gyro_mag'].min()
        
        # Detect jump if changes exceed threshold
        if accel_max_change > threshold_accel or gyro_max_change > threshold_gyro:
            jumps.append({
                'index': i + window_size // 2,  # Middle of window
                'timestamp': window.iloc[window_size // 2]['Timestamp'],
                'accel_change': accel_max_change,
                'gyro_change': gyro_max_change,
                'activity': window.iloc[0]['Activity_Type']
            })
    
    return participant_data, jumps

# Detect jumps for participant 1
participant_id = 1
participant_data, jumps = detect_jumps(data, participant_id, "Jumping", 20, 30, 500)

print(f"\nDetected {len(jumps)} potential jumps/sudden movements:")
for jump in jumps[:10]:  # Show first 10
    print(f"  Index: {jump['index']}, Activity: {jump['activity']}, "
          f"Accel change: {jump['accel_change']:.2f}, Gyro change: {jump['gyro_change']:.2f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Plot acceleration magnitude
ax1.plot(participant_data.index, participant_data['accel_mag'], label='Acceleration Magnitude')
for jump in jumps:
    ax1.axvline(x=jump['index'], color='r', alpha=0.3, linestyle='--')
ax1.set_title(f'Acceleration Magnitude - Participant {participant_id}')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Acceleration Magnitude')
ax1.legend()
ax1.grid(True)

# Plot gyroscope magnitude
ax2.plot(participant_data.index, participant_data['gyro_mag'], label='Gyroscope Magnitude', color='orange')
for jump in jumps:
    ax2.axvline(x=jump['index'], color='r', alpha=0.3, linestyle='--', label='Jump' if jump == jumps[0] else '')
ax2.set_title(f'Gyroscope Magnitude - Participant {participant_id}')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Angular Velocity Magnitude')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
#plt.savefig('jump_detection.png')
plt.show()

print(f"\nVisualization saved as 'jump_detection.png'")