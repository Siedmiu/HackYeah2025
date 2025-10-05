from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import numpy as np
from collections import deque
from datetime import datetime

class HeuristicGestureWindow(QDialog):
    def __init__(self, parent=None, serial_reader=None):
        super().__init__(parent)
        self.serial_reader = serial_reader
        
        # Buffers for sensors (50 samples = ~0.5s at 100Hz)
        self.window_size = 50
        self.mpu_accel = deque(maxlen=self.window_size)
        self.mpu_gyro = deque(maxlen=self.window_size)
        self.adxl = deque(maxlen=self.window_size)
        self.l3gd = deque(maxlen=self.window_size)
        self.lsm = deque(maxlen=self.window_size)
        
        self.is_detecting = False
        self.last_detection_time = None
        self.cooldown_ms = 800  # 0.8s cooldown
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Heuristic Gesture Detector")
        self.setGeometry(200, 200, 700, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Real-Time Gesture Detection (Rule-Based)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Current detection
        self.gesture_label = QLabel("---")
        self.gesture_label.setFont(QFont("Arial", 48, QFont.Bold))
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet(
            "padding: 30px; background-color: #e3f2fd; "
            "border: 3px solid #2196F3; border-radius: 10px;"
        )
        layout.addWidget(self.gesture_label)
        
        # Confidence bars
        conf_layout = QVBoxLayout()
        self.confidence_bars = {}
        gestures = ['Jumping', 'Udarzenia', 'Udarzenia_lewa', 'Kucanie', 'Strzal', 'Syf']
        
        for gesture in gestures:
            row = QHBoxLayout()
            label = QLabel(f"{gesture}:")
            label.setFixedWidth(120)
            row.addWidget(label)
            
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            row.addWidget(bar)
            
            self.confidence_bars[gesture] = bar
            conf_layout.addLayout(row)
        
        layout.addLayout(conf_layout)
        
        # Buffer status
        self.buffer_label = QLabel("Buffer: 0/50")
        layout.addWidget(self.buffer_label)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        layout.addWidget(self.log_text)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("background-color: green; color: white; padding: 10px;")
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: red; color: white; padding: 10px;")
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Timer for detection
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect)
        
    def start_detection(self):
        if not self.serial_reader or not self.serial_reader.is_connected:
            self.add_log("Error: No sensor connection!")
            return
        
        self.is_detecting = True
        self.serial_reader.data_received.connect(self.on_sensor_data)
        self.timer.start(100)  # Check every 100ms
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_log("Detection started")
        
    def stop_detection(self):
        self.is_detecting = False
        self.timer.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.gesture_label.setText("---")
        self.add_log("Detection stopped")
        
    def on_sensor_data(self, data):
        if not self.is_detecting:
            return
        
        self.mpu_accel.append([data.get('mpu_ax', 0), data.get('mpu_ay', 0), data.get('mpu_az', 0)])
        self.mpu_gyro.append([data.get('mpu_gx', 0), data.get('mpu_gy', 0), data.get('mpu_gz', 0)])
        self.adxl.append([data.get('adxl_ax', 0), data.get('adxl_ay', 0), data.get('adxl_az', 0)])
        self.l3gd.append([data.get('l3gd_gx', 0), data.get('l3gd_gy', 0), data.get('l3gd_gz', 0)])
        self.lsm.append([data.get('lsm_ax', 0), data.get('lsm_ay', 0), data.get('lsm_az', 0)])
        
        self.buffer_label.setText(f"Buffer: {len(self.mpu_accel)}/{self.window_size}")
        
    def detect(self):
        if len(self.mpu_accel) < self.window_size:
            return
        
        # Cooldown check
        if self.last_detection_time:
            elapsed = (datetime.now() - self.last_detection_time).total_seconds() * 1000
            if elapsed < self.cooldown_ms:
                return
        
        # Convert to numpy
        mpu_a = np.array(self.mpu_accel)
        mpu_g = np.array(self.mpu_gyro)
        adxl = np.array(self.adxl)
        l3gd = np.array(self.l3gd)
        lsm = np.array(self.lsm)
        
        # Run all detectors
        confidences = {
            'Jumping': self._detect_jumping(mpu_a, mpu_g, adxl, l3gd, lsm),
            'Udarzenia': self._detect_udarzenia(mpu_a, mpu_g, adxl, l3gd, lsm),
            'Udarzenia_lewa': self._detect_udarzenia_lewa(mpu_a, mpu_g, adxl, l3gd, lsm),
            'Kucanie': self._detect_kucanie(mpu_a, mpu_g, adxl, l3gd, lsm),
            'Strzal': self._detect_strzal(mpu_a, mpu_g, adxl, l3gd, lsm),
            'Syf': self._detect_syf(mpu_a, mpu_g, adxl, l3gd, lsm)
        }
        
        # Update bars
        for gesture, conf in confidences.items():
            self.confidence_bars[gesture].setValue(int(conf * 100))
        
        # Get best
        best_gesture = max(confidences, key=confidences.get)
        best_conf = confidences[best_gesture]
        
        if best_conf > 0.55:
            self.gesture_label.setText(best_gesture)
            color = "#4CAF50" if best_conf > 0.7 else "#FF9800"
            self.gesture_label.setStyleSheet(
                f"padding: 30px; background-color: {color}20; "
                f"border: 3px solid {color}; border-radius: 10px; color: {color};"
            )
            self.last_detection_time = datetime.now()
            self.add_log(f"{best_gesture} ({best_conf:.0%})")
        
    def _detect_jumping(self, mpu_a, mpu_g, adxl, l3gd, lsm):
        conf = 0.0
        
        # ADXL Z-axis main indicator
        adxl_z = adxl[:, 2]
        max_z = np.max(adxl_z)
        min_z = np.min(adxl_z)
        range_z = max_z - min_z
        
        # Strong upward spike
        if max_z > 15:
            conf += 0.35
        
        # Downward component (landing)
        if min_z < -8:
            conf += 0.25
        
        # Large range (W-pattern)
        if range_z > 20:
            conf += 0.25
        
        # Low gyro (vertical jump, no rotation)
        gyro_mag = np.linalg.norm(mpu_g, axis=1)
        if np.max(gyro_mag) < 0.4:
            conf += 0.15
        
        return conf
    
    def _detect_udarzenia(self, mpu_a, mpu_g, adxl, l3gd, lsm):
        conf = 0.0
        
        # L3GD gyro - main indicator
        l3gd_mag = np.linalg.norm(l3gd, axis=1)
        max_gyro = np.max(l3gd_mag)
        
        if max_gyro > 0.5:
            conf += 0.4
        
        # LSM forward spike
        lsm_x = np.abs(lsm[:, 0])
        if np.max(lsm_x) > 5000:
            conf += 0.35
        
        # Quick peak (early in window)
        peak_idx = np.argmax(l3gd_mag)
        peak_pos = peak_idx / len(l3gd_mag)
        if 0.2 < peak_pos < 0.7:
            conf += 0.25
        
        return conf
    
    def _detect_udarzenia_lewa(self, mpu_a, mpu_g, adxl, l3gd, lsm):
        conf = 0.0
        
        # MPU Y-axis (left movement)
        mpu_y = mpu_a[:, 1]
        min_y = np.min(mpu_y)
        
        if min_y < -6:
            conf += 0.4
        
        # MPU gyro rotation
        gyro_z = mpu_g[:, 2]
        if np.max(np.abs(gyro_z)) > 0.3:
            conf += 0.3
        
        # Energy check
        energy = np.sum(mpu_a ** 2)
        if energy > 300:
            conf += 0.3
        
        return conf
    
    def _detect_kucanie(self, mpu_a, mpu_g, adxl, l3gd, lsm):
        conf = 0.0
        
        # MPU Z-axis V-pattern
        mpu_z = mpu_a[:, 2]
        min_idx = np.argmin(mpu_z)
        min_val = mpu_z[min_idx]
        
        # Minimum in center
        min_pos = min_idx / len(mpu_z)
        if 0.3 < min_pos < 0.7:
            conf += 0.3
        
        # Downward before min
        if min_idx > 5:
            before = np.mean(mpu_z[:min_idx])
            if before > min_val:
                conf += 0.25
        
        # Upward after min
        if min_idx < len(mpu_z) - 5:
            after = np.mean(mpu_z[min_idx:])
            if after > min_val:
                conf += 0.25
        
        # Some gyro activity (bending)
        gyro_energy = np.sum(mpu_g ** 2)
        if 50 < gyro_energy < 200:
            conf += 0.2
        
        return conf
    
    def _detect_strzal(self, mpu_a, mpu_g, adxl, l3gd, lsm):
        conf = 0.0
        
        # ADXL upward
        adxl_z = adxl[:, 2]
        if np.max(adxl_z) > 12:
            conf += 0.35
        
        # LSM upward range
        lsm_z = lsm[:, 2]
        lsm_range = np.max(lsm_z) - np.min(lsm_z)
        if lsm_range > 3000:
            conf += 0.35
        
        # Correlation between ADXL and LSM
        if len(adxl_z) == len(lsm_z):
            adxl_norm = (adxl_z - np.mean(adxl_z)) / (np.std(adxl_z) + 1e-6)
            lsm_norm = (lsm_z - np.mean(lsm_z)) / (np.std(lsm_z) + 1e-6)
            corr = np.corrcoef(adxl_norm, lsm_norm)[0, 1]
            if corr > 0.3:
                conf += 0.3
        
        return conf
    
    def _detect_syf(self, mpu_a, mpu_g, adxl, l3gd, lsm):
        conf = 0.0
        
        # Low energy
        mpu_energy = np.sum(mpu_a ** 2)
        adxl_energy = np.sum(adxl ** 2)
        
        if mpu_energy < 100 and adxl_energy < 50:
            conf += 0.5
        
        # Low variance
        if np.var(mpu_a) < 5:
            conf += 0.3
        
        # Small range
        if (np.max(adxl) - np.min(adxl)) < 5:
            conf += 0.2
        
        return conf
    
    def add_log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())