from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QPushButton, QTextEdit, 
                             QFileDialog, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
import csv
from datetime import datetime
import os

class DataGatheringWindow(QDialog):
    def __init__(self, parent=None, serial_reader=None):
        super().__init__(parent)
        self.serial_reader = serial_reader
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.data_buffer = []
        self.sample_count = 0
        self.is_connected = False 
        
        self.init_ui()

        if self.serial_reader:
            self.serial_reader.data_received.connect(self.record_data)

        
    def init_ui(self):
        self.setWindowTitle("Zbieranie danych - Data Gathering")
        self.setGeometry(150, 150, 500, 600)
        
        layout = QVBoxLayout()
        
        # Participant ID
        participant_layout = QHBoxLayout()
        participant_layout.addWidget(QLabel("Participant ID:"))
        self.participant_input = QSpinBox()
        self.participant_input.setMinimum(1)
        self.participant_input.setMaximum(9999)
        self.participant_input.setValue(1)
        participant_layout.addWidget(self.participant_input)
        layout.addLayout(participant_layout)
        
        # Device ID
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device ID:"))
        self.device_input = QLineEdit()
        self.device_input.setPlaceholderText("np. DEVICE_001")
        self.device_input.setText("DEVICE_001")
        device_layout.addWidget(self.device_input)
        layout.addLayout(device_layout)
        
        # Activity Type
        activity_layout = QHBoxLayout()
        activity_layout.addWidget(QLabel("Activity Type:"))
        self.activity_combo = QComboBox()
        self.activity_combo.addItems([
            "Walking",
            "Running",
            "Jumping",
            "Standing",
            "Sitting",
            "Climbing_Stairs",
            "Descending_Stairs",
            "Cycling",
            "Custom"
        ])
        activity_layout.addWidget(self.activity_combo)
        layout.addLayout(activity_layout)
        
        # Custom activity input (hidden by default)
        self.custom_activity_input = QLineEdit()
        self.custom_activity_input.setPlaceholderText("Wprowadź własną aktywność")
        self.custom_activity_input.setVisible(False)
        layout.addWidget(self.custom_activity_input)
        
        # Connect signal to show/hide custom input
        self.activity_combo.currentTextChanged.connect(self.on_activity_changed)
        
        # File path selection
        # file_layout = QHBoxLayout()
        # file_layout.addWidget(QLabel("Save to:"))
        # self.file_path_input = QLineEdit()
        # self.file_path_input.setPlaceholderText("Wybierz lokalizację pliku...")
        # self.file_path_input.setReadOnly(True)
        # file_layout.addWidget(self.file_path_input)
        
        # self.browse_button = QPushButton("Browse")
        # self.browse_button.clicked.connect(self.browse_file)
        # file_layout.addWidget(self.browse_button)
        # layout.addLayout(file_layout)
        
        file_path = self.get_file_path()

        # Display it in the QLineEdit (read-only)
        self.file_path_input = QLineEdit()
        self.file_path_input.setText(file_path)
        self.file_path_input.setReadOnly(True)

        # Add it to layout
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Save to:"))
        file_layout.addWidget(self.file_path_input)

        # No need for Browse button
        # layout.addLayout(file_layout)
        layout.addLayout(file_layout)

        # Sample count display
        self.sample_label = QLabel("Samples collected: 0")
        self.sample_label.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.sample_label)
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("Status będzie wyświetlany tutaj...")
        layout.addWidget(self.status_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)
        
        self.setLayout(layout)
    
    def get_file_path(self):
        participant_id = self.participant_input.value()
        device_id = self.device_input.text().strip()
        activity = self.activity_combo.currentText()
        if activity == "Custom":
            activity = self.custom_activity_input.text().strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset/data_P{participant_id}_{device_id}_{activity}_{timestamp}.csv"
        
    def on_activity_changed(self, text):
        """Show/hide custom activity input based on selection"""
        self.custom_activity_input.setVisible(text == "Custom")
        
    def browse_file(self):
        """Open file dialog to choose save location"""
        default_name = f"data_P{self.participant_input.value()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data File",
            default_name,
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.file_path_input.setText(file_path)

    def start_recording(self):
        """Start data recording"""
        # Validation
        if not self.file_path_input.text():
            self.add_status("Error: Please select a file location first!")
            return
            
        if not self.device_input.text().strip():
            self.add_status("Error: Device ID cannot be empty!")
            return
            
        activity = self.activity_combo.currentText()
        if activity == "Custom" and not self.custom_activity_input.text().strip():
            self.add_status("Error: Please enter custom activity name!")
            return
            
        if not self.serial_reader or not self.serial_reader.is_connected:
            self.add_status("Error: Serial device not connected!")
            return
        
        # Get activity name
        if activity == "Custom":
            activity = self.custom_activity_input.text().strip()
        
        try:
            # Generate fresh file path before recording
            file_path = self.get_file_path()
            self.file_path_input.setText(file_path)
            
            # Create dataset directory if it doesn't exist
            os.makedirs("dataset", exist_ok=True)
            
            self.csv_file = open(file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header matching Arduino output plus metadata
            self.csv_writer.writerow([
                'Participant_ID', 'Activity_Type', 'Timestamp',
                'mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz',
                'adxl_ax', 'adxl_ay', 'adxl_az',
                'l3gd_gx', 'l3gd_gy', 'l3gd_gz',
                'lsm_ax', 'lsm_ay', 'lsm_az',
                'joy_lx', 'joy_ly', 'joy_lb', 'joy_rx', 'joy_ry', 'joy_rb',
                'mpu_temp'
            ])
            
            self.is_recording = True
            self.sample_count = 0
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.participant_input.setEnabled(False)
            self.device_input.setEnabled(False)
            self.activity_combo.setEnabled(False)
            self.custom_activity_input.setEnabled(False)
            # Remove this line: self.browse_button.setEnabled(False)
            
            self.add_status(f"Recording started: {activity}")
            self.add_status(f"Participant: {self.participant_input.value()}, Device: {self.device_input.text()}")
            
        except Exception as e:
            self.add_status(f"Error opening file: {str(e)}")
            
    def stop_recording(self):
        """Stop data recording"""
        self.is_recording = False
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.participant_input.setEnabled(True)
        self.device_input.setEnabled(True)
        self.activity_combo.setEnabled(True)
        self.custom_activity_input.setEnabled(True)
        #self.browse_button.setEnabled(True)
        
        self.add_status(f"Recording stopped. Total samples: {self.sample_count}")
        
    def record_data(self, sensor_data):
        """Record incoming sensor data to CSV"""
        if not self.is_recording or not self.csv_writer:
            return
            
        try:
            activity = self.activity_combo.currentText()
            if activity == "Custom":
                activity = self.custom_activity_input.text().strip()
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Build row with all sensor data
            row = [
                self.participant_input.value(),  # Participant_ID
                activity,                         # Activity_Type
                #self.device_input.text().strip(), # Device_ID
                timestamp,                        # Timestamp
                # MPU6050 data
                sensor_data.get('mpu_ax', 0),
                sensor_data.get('mpu_ay', 0),
                sensor_data.get('mpu_az', 0),
                sensor_data.get('mpu_gx', 0),
                sensor_data.get('mpu_gy', 0),
                sensor_data.get('mpu_gz', 0),
                # ADXL345 data
                sensor_data.get('adxl_ax', 0),
                sensor_data.get('adxl_ay', 0),
                sensor_data.get('adxl_az', 0),
                # L3GD20 data
                sensor_data.get('l3gd_gx', 0),
                sensor_data.get('l3gd_gy', 0),
                sensor_data.get('l3gd_gz', 0),
                # LSM303 data
                sensor_data.get('lsm_ax', 0),
                sensor_data.get('lsm_ay', 0),
                sensor_data.get('lsm_az', 0),
                # Joystick data
                sensor_data.get('joy_lx', 0),
                sensor_data.get('joy_ly', 0),
                sensor_data.get('joy_lb', 0),
                sensor_data.get('joy_rx', 0),
                sensor_data.get('joy_ry', 0),
                sensor_data.get('joy_rb', 0),
                # Temperature
                sensor_data.get('mpu_temp', 0)
            ]
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # Ensure data is written immediately
            self.sample_count += 1
            self.sample_label.setText(f"Samples collected: {self.sample_count}")
            
        except Exception as e:
            self.add_status(f"Error writing data: {str(e)}")
            
    def add_status(self, message):
        """Add status message to text display"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.append(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.is_recording:
            self.stop_recording()
        event.accept()