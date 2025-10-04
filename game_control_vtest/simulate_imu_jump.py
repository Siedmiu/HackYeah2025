#!/usr/bin/env python3

"""
Simulated IMU controller for Chrome Dino game using Selenium.
- Automatically opens Dino in Chrome.
- Simulates IMU jumps (pressing space) automatically.
- Replace the random jump simulation with real IMU logic.
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# === CONFIGURATION ===
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"  # Change to your chromedriver path
JUMP_INTERVAL = 0.5  # seconds between checks
SIMULATE_RANDOM_JUMPS = True  # True = simulate jumps randomly

# === SETUP CHROME OPTIONS ===
chrome_options = Options()
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--mute-audio")
chrome_options.add_argument("--start-maximized")

# === START CHROME DRIVER ===
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.get("chrome://dino")

time.sleep(2)  # wait for Dino to load

# Focus on the page
body = driver.find_element("tag name", "body")

print("=== Dino IMU Simulation Started ===")
print("Close Chrome or press Ctrl+C to stop.")

try:
    while True:
        # Simulate IMU jump detection
        jump_detected = False
        if SIMULATE_RANDOM_JUMPS:
            jump_detected = random.choice([True, False])  # random jump simulation

        # Trigger jump
        if jump_detected:
            print("Jump!")
            body.send_keys(Keys.SPACE)

        time.sleep(JUMP_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping simulation...")
    driver.quit()
