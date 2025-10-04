from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtCore import Qt
import serial.tools.list_ports
import parameters
from data_gathering import DataGatheringWindow
from imu_visualizer import IMUVisualizerWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_reader = None
        self.is_connected = False
        self.visualizer_window = None
        self.data_window = None
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

        # Connection status label
        self.connection_label = QLabel("Status: Offline")
        self.connection_label.setAlignment(Qt.AlignCenter)
        self.connection_label.setStyleSheet("font-size: 14px; padding: 10px; color: orange;")
        layout.addWidget(self.connection_label)

        # Port selection dropdown
        self.port_combo = QComboBox()
        self.port_combo.setFixedSize(200, 35)
        self.refresh_ports()
        layout.addWidget(self.port_combo, alignment=Qt.AlignCenter)

        # Connect/Disconnect button
        self.button_connect = QPushButton("Connect")
        self.button_connect.clicked.connect(self.on_button_connect_click)
        self.button_connect.setFixedSize(200, 50)
        layout.addWidget(self.button_connect, alignment=Qt.AlignCenter)

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

        self.button_hotkey = QPushButton("Zbierz dataset")
        self.button_hotkey.clicked.connect(self.on_button_data_click)
        self.button_hotkey.setFixedSize(200, 50)
        layout.addWidget(self.button_hotkey, alignment=Qt.AlignCenter)

        self.button_visualizer = QPushButton("Wizualizacja IMU (Live)")
        self.button_visualizer.clicked.connect(self.on_button_visualizer_click)
        self.button_visualizer.setFixedSize(200, 50)
        layout.addWidget(self.button_visualizer, alignment=Qt.AlignCenter)

        self.button_close = QPushButton("Wyjdź")
        self.button_close.clicked.connect(self.on_button_close_click)
        self.button_close.setFixedSize(200, 50)
        layout.addWidget(self.button_close, alignment=Qt.AlignCenter)

        # Add stretch to push content to center
        layout.addStretch()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.esc_press()
        super().keyPressEvent(event)

    # Funkcje obsługi przycisków
    def on_button_start_click(self):
        self.label.setText("Game mode activated\nPress [ESC] to cancel")
        parameters.game_state = "Game"

    def on_button_hotkey_click(self):
        self.label.setText("Button 2 clicked!")

    def on_button_calibration_click(self):
        self.label.setText("Button 3 clicked!")

    def on_button_close_click(self):
        self.close()
    
    def on_button_data_click(self):
        self.create_data_gathering_window()

    def on_button_visualizer_click(self):
        """Open IMU visualizer window"""
        if not self.is_connected:
            self.label.setText("Error: Please connect to a device first!")
            return

        if self.visualizer_window is None or not self.visualizer_window.isVisible():
            self.visualizer_window = IMUVisualizerWindow(self)
            self.visualizer_window.show()
            self.label.setText("IMU Visualizer opened")
        else:
            self.visualizer_window.raise_()
            self.visualizer_window.activateWindow()

    def create_data_gathering_window(self):
        #choose player id, movement type, device id
        #get data from terminal and save to csv
        #csv scheme: Participant_ID,Activity_Type,Device_ID,Ax,Ay,Az,Gx,Gy,Gz,Timestamp
        #or insert what you want to save in csv structure
        """Open data gathering window"""
        if not self.serial_reader:
            self.label.setText("Error: Serial reader not initialized!")
            return
            
        if not self.is_connected:
            self.label.setText("Error: Please connect to a device first!")
            return
        
        self.data_window = DataGatheringWindow(self, self.serial_reader)
        self.data_window.show()
        pass
    
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

    def esc_press(self):
        if parameters.game_state == "Game":
            parameters.game_state = "Main_menu"
            self.label.setText("Ready")

    def update_sensor_data(self, data):
        """Update GUI with sensor data from Arduino"""
        if parameters.game_state == "Game":
            status_text = f"IMU Data:\n"
            status_text += f"MPU Accel: ({data['mpu_ax']:.2f}, {data['mpu_ay']:.2f}, {data['mpu_az']:.2f}) m/s²\n"
            status_text += f"MPU Gyro: ({data['mpu_gx']:.2f}, {data['mpu_gy']:.2f}, {data['mpu_gz']:.2f}) rad/s\n\n"
            status_text += f"Joystick L: ({data['joy_lx']}, {data['joy_ly']}) Btn: {data['joy_lb']}\n"
            status_text += f"Joystick R: ({data['joy_rx']}, {data['joy_ry']}) Btn: {data['joy_rb']}"
            self.label.setText(status_text)
        if self.data_window and self.data_window.is_recording:
            self.data_window.record_data(data)

        # Update visualizer if open
        if self.visualizer_window and self.visualizer_window.isVisible():
            print(f"MainWindow sending data to visualizer: {list(data.keys())}")
            self.visualizer_window.update_data(data)

    def set_serial_reader(self, serial_reader):
        """Set the serial reader instance"""
        self.serial_reader = serial_reader

    def set_serial_reader(self, serial_reader):
        """Set the serial reader instance"""
        self.serial_reader = serial_reader

    def refresh_ports(self):
        """Refresh the list of available serial ports"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}", port.device)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("No ports available", None)

    def on_button_connect_click(self):
        """Handle connect/disconnect button click"""
        if self.is_connected:
            # Disconnect
            if self.serial_reader:
                self.serial_reader.stop()
            self.is_connected = False
            self.button_connect.setText("Connect")
            self.connection_label.setText("Status: Disconnected")
            self.connection_label.setStyleSheet("font-size: 14px; padding: 10px; color: orange;")
        else:
            # Refresh ports and connect
            self.refresh_ports()
            selected_port = self.port_combo.currentData()
            if selected_port and self.serial_reader:
                self.serial_reader.stop()  # Stop previous connection if any
                self.serial_reader.connect(selected_port)
                self.connection_label.setText(f"Status: Connecting to {selected_port}...")
                self.connection_label.setStyleSheet("font-size: 14px; padding: 10px; color: yellow;")

    def update_connection_status(self, connected, message):
        """Update connection status from serial reader"""
        self.is_connected = connected
        if connected:
            self.connection_label.setText(f"Status: {message}")
            self.connection_label.setStyleSheet("font-size: 14px; padding: 10px; color: green;")
            self.button_connect.setText("Disconnect")
        else:
            self.connection_label.setText(f"Status: {message}")
            self.connection_label.setStyleSheet("font-size: 14px; padding: 10px; color: red;")
            self.button_connect.setText("Connect")