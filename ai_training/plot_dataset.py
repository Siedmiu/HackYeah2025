import pandas as pd
import matplotlib.pyplot as plt

# for basketball dataset
# data = pd.read_csv("ai_training/dataset/simulated_basketball_IMU_dataset.csv")
# #how data are saved: Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz,target,Player_ID
# 
# player_id = "P001"   
# player_data = data[data["Player_ID"] == player_id]

# #print(unique_targets := player_data["target"].unique()) # result -> ['set_shot' 'layup' 'jump_shot' 'tip_in']

# movement =  "jump_shot" 

# filtered = data[(data["Player_ID"] == player_id) & (data["target"] == movement)]

# print(f"Liczba próbek dla {player_id}, ruch: {movement} = {len(filtered)}")

# subset = filtered.head(200)
# print(f"Próbki dla {player_id}, ruch: {movement} (pierwsze 200):")
# filtered = subset

#same but for human movements datset
data = pd.read_csv("ai_training/dataset/all_activity_data.csv")
#how data are saved: Participant_ID,Activity_Type,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z,Timestamp
#first line: 1,Walking,-6.752,-1.462,6.634,50.458,-119.29,330.14,1727961275

player_id = 1  
player_data = data[data["Participant_ID"] == player_id]

print(unique_targets := player_data["Activity_Type"].unique()) # result -> ['set_shot' 'layup' 'jump_shot' 'tip_in']

movement =  "Jumping" 

filtered = data[(data["Participant_ID"] == player_id) & (data["Activity_Type"] == movement)]
filtered = filtered.sort_values(by="Timestamp")  

print(f"Liczba próbek dla {player_id}, ruch: {movement} = {len(filtered)}")

subset = filtered.head(200)
print(f"Próbki dla {player_id}, ruch: {movement} (pierwsze 200):")
filtered = subset
filtered = filtered.sort_values(by="Timestamp")  

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(subset["Ax"], subset["Ay"], subset["Az"], label="Trajectory in 3D")

ax.set_xlabel("Ax")
ax.set_ylabel("Ay")
ax.set_zlabel("Az")
ax.set_title(f"3D Accelerometer trajectory - {player_id}, {movement}")
ax.legend()

#accel
plt.figure(figsize=(12, 6))
plt.plot(filtered["Ax"], label="Ax")
plt.plot(filtered["Ay"], label="Ay")
plt.plot(filtered["Az"], label="Az")
plt.title(f"Accelerometer signals - {player_id}")
plt.xlabel("Sample index")
plt.ylabel("Acceleration")
plt.legend()
plt.show()

#gyro
plt.figure(figsize=(12, 6))
plt.plot(filtered["Gx"], label="Gx")
plt.plot(filtered["Gy"], label="Gy")
plt.plot(filtered["Gz"], label="Gz")
plt.title(f"Gyroscope signals - {player_id}")
plt.xlabel("Sample index")
plt.ylabel("Angular velocity")
plt.legend()
plt.show()

# #magneto
# plt.figure(figsize=(12, 6))
# plt.plot(filtered["Mx"], label="Mx")
# plt.plot(filtered["My"], label="My")
# plt.plot(filtered["Mz"], label="Mz")
# plt.title(f"Magnetometer signals - {player_id}")
# plt.xlabel("Sample index")
# plt.ylabel("Magnetic field")
# plt.legend()
# plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    player_data["Ax"], player_data["Ay"], player_data["Az"],
    c=pd.factorize(player_data["Activity_Type"])[0], cmap="tab10", s=20
)
ax.set_xlabel("Ax")
ax.set_ylabel("Ay")
ax.set_zlabel("Az")
ax.set_title(f"3D scatter of accelerometer data - {player_id}")
plt.legend(*scatter.legend_elements(), title="Gesture")
plt.show()
