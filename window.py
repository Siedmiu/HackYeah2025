# main.py

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        # Window settings
        self.setWindowTitle("Open controller")
        self.setGeometry(100, 100, 600, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add stretch to push content to center
        layout.addStretch()
        
        # Large label
        self.label = QLabel("Status: Ready")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 18px; padding: 20px;")
        layout.addWidget(self.label)
        
        # Buttons
        self.button_start = QPushButton("Start")
        self.button_start.clicked.connect(self.on_button_start_click)
        self.button_start.setFixedSize(200, 50)
        layout.addWidget(self.button_start, alignment=Qt.AlignCenter)
        
        self.button_hotkey = QPushButton("Przypisanie klawiszy")
        self.button_hotkey.clicked.connect(self.on_button_hotkey_click)
        self.button_hotkey.setFixedSize(200, 50)
        layout.addWidget(self.button_hotkey, alignment=Qt.AlignCenter)
        
        self.button_calibration = QPushButton("Kalibracja")
        self.button_calibration.clicked.connect(self.on_button_calibration_click)
        self.button_calibration.setFixedSize(200, 50)
        layout.addWidget(self.button_calibration, alignment=Qt.AlignCenter)
        
        self.button_close = QPushButton("Wyjdź")
        self.button_close.clicked.connect(self.on_button_close_click)
        self.button_close.setFixedSize(200, 50)
        layout.addWidget(self.button_close, alignment=Qt.AlignCenter)
        
        # Add stretch to push content to center
        layout.addStretch()

    # Funkcje obsługi przycisków (dodaj do klasy MainWindow):
    def on_button_start_click(self):
        self.label.setText("Button 1 clicked!")

    def on_button_hotkey_click(self):
        self.label.setText("Button 2 clicked!")

    def on_button_calibration_click(self):
        self.label.setText("Button 3 clicked!")

    def on_button_close_click(self):
        self.close()
    
    def on_button_click(self):
        """Handle button click event"""
        self.label.setText("Button clicked!")
    
    def update_label(self, text):
        """Update label text"""
        self.label.setText(text)
    
    def get_input(self):
        """Get user input from widgets"""
        pass
    
    def display_data(self, data):
        """Display data in GUI"""
        pass

    def update_sensor_data(self, data):
        """Update GUI with sensor data from Arduino"""
        status_text = f"IMU Data:\n"
        status_text += f"MPU Accel: ({data['mpu_ax']:.2f}, {data['mpu_ay']:.2f}, {data['mpu_az']:.2f}) m/s²\n"
        status_text += f"MPU Gyro: ({data['mpu_gx']:.2f}, {data['mpu_gy']:.2f}, {data['mpu_gz']:.2f}) rad/s\n"
        status_text += f"Temp: {data['mpu_temp']:.1f}°C\n\n"
        status_text += f"Joystick L: ({data['joy_lx']}, {data['joy_ly']}) Btn: {data['joy_lb']}\n"
        status_text += f"Joystick R: ({data['joy_rx']}, {data['joy_ry']}) Btn: {data['joy_rb']}"
        self.label.setText(status_text)