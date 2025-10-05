# HackYeah 2025 — Open Controller
**Affordable, open-source motion controller for everyone!**

---

## Project Overview

**Open Controller** is a low-cost, open-source motion game controller designed to make gaming more **active, inclusive, and fun**.  
It uses **three IMUs** (Inertial Measurement Units) - one on each hand and one on the torso — to recognize gestures and map them to keyboard inputs.

Our goal is to let **anyone — kids, parents, or hobbyists — build their own motion controller** using accessible hardware and open software tools.  
You can build it, play, and stay active while gaming - without needing expensive VR systems or proprietary gear. You don't need to generate whole personal dataset - just add files with your specific gestures to our - literally plug and play.

https://youtube.com/shorts/PNGoJlM1TY0?si=B0IRf1yLYddQADC-

---

## How It Works

1. **Hardware Setup**
   - Three IMU sensors track motion of both hands and torso.
   - An additional accelerometer detects jumps.
   - All sensors are connected to a **single data pin**, allowing the setup to work even on compact boards like **ESP32-C3 mini**.
   - A joystick and tactile switches provide extra control options.

2. **Data & AI Model**
   - Our team and mentors recorded movement data to create a **gesture dataset**.
   - A **neural network** was trained to recognize gestures (e.g., arm swings, jumps, torso twists).
   - The trained model runs on your computer and classifies gestures in real-time from incoming IMU data.

3. **Desktop Application**
   - Built with **Python + Qt**, it visualizes sensor data and manages gesture recognition.
   - Recognized gestures can be **mapped to any keyboard key**, allowing compatibility with **any PC game**.

---

## Components

| Component | Function |
|------------|-----------|
| **ESP Dev Kit C** | Main microcontroller collecting and transmitting sensor data |
| **3× IMU sensors** | Track hand and torso motion |
| **Accelerometer** | Detects jump or rapid movement |
| **Joystick** | Optional in-game directional control |
| **Tactile Switches** | Extra buttons for gameplay or calibration |

---

## Tech Stack

- **Firmware:** PlatformIO (Arduino Framework)  
- **Desktop App:** Python + Qt  
- **ML Model:** TensorFlow / PyTorch for gesture recognition  
- **Communication:** Serial / Bluetooth  
- **Hardware:** ESP32 Dev Kit C, IMUs (MPU, L3GD, LSM, ADX series)

---

## Game Integration

Our demo setup integrates with **Minecraft** for bow-shooting and parkour activities - but thanks to the **custom key mapping**, the controller can be configured for **any game**.  
Simply assign gestures to keyboard actions (e.g. `Jump`, `Shoot`, `Run`) and start playing!

---

## Build It Yourself

Detailed step-by-step **hardware and software setup instructions** are available in the repository.  
You can build your controller:  
- solo,  
- with friends,  
- or even as a **STEM / school project**.  

Everyone can contribute and expand gesture sets for different games!

---

## hy It Matters

- **Affordable:** built from cheap, widely available components.  
- **Open-source:** free to modify, extend, and share.  
- **Educational:** teaches hardware, coding, and ML concepts.  
- **Active Gaming:** promotes movement while playing your favorite games.

---

## Team & Credits

Built with passion during **HackYeah 2025** by KN SafeIDEA team, a team of makers, gamers, and mentors who believe gaming should move you - literally.

---
