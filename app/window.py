from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox, QLineEdit
from PyQt5.QtCore import Qt
import serial.tools.list_ports
import parameters
from data_gathering import DataGatheringWindow
from keybinding_dialog import KeyBindingDialog
from PyQt5.QtWidgets import QDialog
from joystick import Joystick


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_reader = None
        self.is_connected = False
        self.data_window = None  # Inicjalizacja okna data gathering
        self.joystick_controller = Joystick()
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

        # Pole tekstowe (nic nie robi)
        self.text_field = QLineEdit()
        self.text_field.setPlaceholderText("Wpisz coś tutaj...")
        self.text_field.setFixedSize(200, 35)
        self.text_field.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_field, alignment=Qt.AlignCenter)

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
        dialog = KeyBindingDialog(self)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:  # Jeśli kliknięto "Zapisz"
            bindings = dialog.get_bindings()
            self.label.setText(f"Zapisano przypisania!\n{len(bindings)} klawiszy skonfigurowanych")
        else:  # Jeśli kliknięto "Anuluj"
            self.label.setText("Anulowano przypisanie klawiszy")


    def on_button_calibration_click(self):
        self.label.setText("Button 3 clicked!")

    def on_button_close_click(self):
        self.close()
    
    def on_button_data_click(self):
        self.create_data_gathering_window()


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
        
        # NAJPIERW zapisz dane joysticka do parameters (przed update())
        parameters.joystick_left_x = data['joy_lx']
        parameters.joystick_left_y = data['joy_ly']
        parameters.joystick_left_button = data['joy_lb']
        parameters.joystick_right_x = data['joy_rx'] 
        parameters.joystick_right_y = data['joy_ry']
        parameters.joystick_right_button = data['joy_rb']
        
        # Jeśli jesteś w trybie Game, wywołaj joystick controller
        if parameters.game_state == "Game":
            self.joystick_controller.update()
            
            status_text = f"IMU Data:\n"
            status_text += f"MPU Accel: ({data['mpu_ax']:.2f}, {data['mpu_ay']:.2f}, {data['mpu_az']:.2f}) m/s²\n"
            status_text += f"MPU Gyro: ({data['mpu_gx']:.2f}, {data['mpu_gy']:.2f}, {data['mpu_gz']:.2f}) rad/s\n"
            status_text += f"Joystick L: ({data['joy_lx']}, {data['joy_ly']}) Btn: {data['joy_lb']}\n"
            status_text += f"Joystick R: ({data['joy_rx']}, {data['joy_ry']}) Btn: {data['joy_rb']}"
            self.label.setText(status_text)
        
        # Zapisuj do CSV tylko gdy jest okno data gathering i jest nagrywanie
        if self.data_window and self.data_window.is_recording:
            self.data_window.record_data(data)

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