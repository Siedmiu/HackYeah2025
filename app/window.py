from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox, QLineEdit
from PyQt5.QtCore import Qt, QTimer
import serial.tools.list_ports
import parameters
from data_gathering import DataGatheringWindow
from keybinding_dialog import KeyBindingDialog
from PyQt5.QtWidgets import QDialog
from joystick import Joystick
from gesture_detection import GestureDetectionWindow
from heurystic import HeuristicGestureWindow
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from collections import deque
import os
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_reader = None
        self.is_connected = False
        self.data_window = None
        self.joystick_controller = Joystick()
        
        # Gesture detection components
        self.gesture_model = None
        self.gesture_scaler = None
        self.gesture_label_classes = None
        self.gesture_buffer = deque(maxlen=35)  # window_size = 35
        self.max_features = 6
        self.gesture_enabled = False
        
        # Cooldown system
        self.last_gesture_time = 0
        self.cooldown_duration = 2.0  # 2 sekundy cooldown
        self.current_gesture = None
        self.gesture_confidence = 0
        
        # Timer for gesture prediction
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.predict_gesture_in_game)
        self.prediction_interval = 500  # ms
        
        # Timer to reset gesture display
        self.gesture_display_timer = QTimer()
        self.gesture_display_timer.timeout.connect(self.reset_gesture_display)
        
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

        # Nowy label dla gestów
        self.info_label = QLabel("Gesture: Waiting...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.info_label)

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

        # self.detect_btn = QPushButton("Detect Movement")
        # self.detect_btn.clicked.connect(self.open_gesture_detection)
        # self.detect_btn.setFixedSize(200, 50)  # Changed from self.button_hotkey to self.detect_btn
        # layout.addWidget(self.detect_btn, alignment=Qt.AlignCenter)  # Changed from self.button_hotkey to self.detect_btn

        self.heuristic_btn = QPushButton("Detect Gestures (Heuristic)")
        self.heuristic_btn.clicked.connect(self.open_heuristic_detector)
        self.heuristic_btn.setFixedSize(200, 50)
        
        self.detect_btn = QPushButton("Detect Movement")
        self.detect_btn.clicked.connect(self.open_gesture_detection)
        self.detect_btn.setFixedSize(200, 50)  # Changed from self.button_hotkey to self.detect_btn
        layout.addWidget(self.detect_btn, alignment=Qt.AlignCenter)  # Changed from self.button_hotkey to self.detect_btn

        # Przycisk do ładowania modelu gestów
        self.load_gesture_model_btn = QPushButton("Load Gesture Model")
        self.load_gesture_model_btn.clicked.connect(self.load_gesture_model_for_game)
        self.load_gesture_model_btn.setFixedSize(200, 50)
        layout.addWidget(self.load_gesture_model_btn, alignment=Qt.AlignCenter)

        self.button_close = QPushButton("Wyjdź")
        self.button_close.clicked.connect(self.on_button_close_click)
        self.button_close.setFixedSize(200, 50)
        layout.addWidget(self.button_close, alignment=Qt.AlignCenter)

        # Add stretch to push content to center
        layout.addStretch()

    def load_gesture_model_for_game(self):
        """Load the most recent gesture model for use in game mode"""
        try:
            # Find the most recent model file
            import glob
            model_files = glob.glob("models/*.keras") + glob.glob("models/*.h5")
            
            if not model_files:
                self.label.setText("Error: No model files found in models/")
                return
            
            # Sort by modification time, get most recent
            model_files.sort(key=os.path.getmtime, reverse=True)
            model_path = model_files[0]
            
            # Load model
            self.gesture_model = tf.keras.models.load_model(model_path)
            
            # Try to load corresponding label classes
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Extract timestamp from model name
            parts = model_name.split('_')
            if len(parts) >= 3:
                timestamp = '_'.join(parts[-2:])
                label_file = os.path.join(model_dir, f"label_classes_{timestamp}.npy")
                
                if os.path.exists(label_file):
                    self.gesture_label_classes = np.load(label_file, allow_pickle=True)
                else:
                    # Default classes if file not found
                    self.gesture_label_classes = np.array(['Jumping', 'Kucania', 'Strzal', 
                                                           'Syf', 'Udarzenia', 'Udarzenia_lewa'])
            
            # Initialize scaler
            self.gesture_scaler = StandardScaler()
            
            # Clear buffer
            self.gesture_buffer.clear()
            
            # Enable gesture detection
            self.gesture_enabled = True
            
            self.label.setText(f"Gesture model loaded:\n{os.path.basename(model_path)}")
            self.info_label.setText(f"Gesture: Ready (Classes: {len(self.gesture_label_classes)})")
            
            print(f"Gesture model loaded: {model_path}")
            print(f"Classes: {', '.join(self.gesture_label_classes)}")
            
        except Exception as e:
            self.label.setText(f"Error loading model:\n{str(e)}")
            print(f"Error loading gesture model: {e}")
            import traceback
            traceback.print_exc()

    def is_in_cooldown(self):
        """Check if we're still in cooldown period"""
        current_time = time.time()
        time_since_last = current_time - self.last_gesture_time
        return time_since_last < self.cooldown_duration

    def predict_gesture_in_game(self):
        """Predict gesture from current buffer during game mode"""
        if not self.gesture_enabled or self.gesture_model is None:
            return
        
        # Skip prediction if in cooldown
        if self.is_in_cooldown():
            remaining = self.cooldown_duration - (time.time() - self.last_gesture_time)
            self.info_label.setText(f"Gesture: {self.current_gesture} ({self.gesture_confidence:.1f}%) - Cooldown: {remaining:.1f}s")
            return
        
        if len(self.gesture_buffer) < 35:  # window_size
            self.info_label.setText(f"Gesture: Waiting... (Buffer: {len(self.gesture_buffer)}/35)")
            return
        
        try:
            # Create window from buffer
            window = np.array(list(self.gesture_buffer))
            
            # Normalize
            window_reshaped = window.reshape(-1, self.max_features)
            
            # Fit scaler on first window or use existing
            if not hasattr(self.gesture_scaler, 'mean_'):
                window_normalized = self.gesture_scaler.fit_transform(window_reshaped)
            else:
                window_normalized = self.gesture_scaler.transform(window_reshaped)
            
            window_normalized = window_normalized.reshape(1, 35, self.max_features)
            
            # Predict
            predictions = self.gesture_model.predict(window_normalized, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            gesture_name = self.gesture_label_classes[predicted_class]
            
            # FILTRUJ STRZAŁ - ignoruj ten gest całkowicie
            if gesture_name == "Strzal":
                self.info_label.setText(f"Gesture: Strzal detected but ignored")
                self.info_label.setStyleSheet(
                    "font-size: 16px; padding: 15px; background-color: #ffebee; "
                    "border: 2px solid #f44336; border-radius: 5px; color: #c62828;"
                )
                print(f"[FILTERED] Strzal gesture ignored ({confidence:.1f}%)")
                return  # Wyjdź bez wykonywania akcji
            
            # Only act if confidence is high enough
            if confidence > 70:
                # Store current gesture info
                self.current_gesture = gesture_name
                self.gesture_confidence = confidence
                self.last_gesture_time = time.time()
                
                # Update display
                color = "#4CAF50" if confidence > 80 else "#FF9800"
                self.info_label.setText(f"Gesture: {gesture_name} ({confidence:.1f}%) ✓")
                self.info_label.setStyleSheet(
                    f"font-size: 16px; padding: 15px; background-color: {color}20; "
                    f"border: 2px solid {color}; border-radius: 5px; color: {color}; font-weight: bold;"
                )
                
                # Execute gesture action
                self.execute_gesture_action(gesture_name)
                
                # Clear buffer to prevent re-detection
                self.gesture_buffer.clear()
                
                print(f"[GESTURE DETECTED] {gesture_name} with {confidence:.1f}% confidence - Cooldown started")
                
                # Start timer to reset display after cooldown
                self.gesture_display_timer.stop()
                self.gesture_display_timer.start(int(self.cooldown_duration * 1000))
            else:
                # Low confidence - show but don't act
                self.info_label.setText(f"Gesture: {gesture_name} ({confidence:.1f}%) - Low confidence")
                self.info_label.setStyleSheet(
                    "font-size: 16px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;"
                )
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()

    def reset_gesture_display(self):
        """Reset gesture display after cooldown"""
        self.gesture_display_timer.stop()
        self.info_label.setText("Gesture: Waiting...")
        self.info_label.setStyleSheet(
            "font-size: 16px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;"
        )
        self.current_gesture = None
        self.gesture_confidence = 0

    def execute_gesture_action(self, gesture_name):
        """Execute the action associated with the detected gesture"""
        if gesture_name not in parameters.gesture_bindings:
            print(f"[WARNING] Gesture {gesture_name} not found in bindings")
            return
        
        action = parameters.gesture_bindings[gesture_name]
        
        import pyautogui
        
        try:
            # SKOK - Jumping
            if gesture_name == "Jumping":
                if action == "space":
                    pyautogui.press('space')
                    print(f"[ACTION] {gesture_name} -> space")
                elif action.startswith("key_"):
                    key = action.replace("key_", "")
                    pyautogui.press(key)
                    print(f"[ACTION] {gesture_name} -> {key}")
                else:
                    pyautogui.press(action)
                    print(f"[ACTION] {gesture_name} -> {action}")
            
            # UDERZENIE PRAWĄ RĘKĄ - Udarzenia
            elif gesture_name == "Udarzenia":
                if action == "left_mouse":
                    pyautogui.click(button='left')
                    print(f"[ACTION] {gesture_name} -> left mouse click")
                elif action == "right_mouse":
                    pyautogui.click(button='right')
                    print(f"[ACTION] {gesture_name} -> right mouse click")
                elif action == "middle_mouse":
                    pyautogui.click(button='middle')
                    print(f"[ACTION] {gesture_name} -> middle mouse click")
                else:
                    # Fallback - jeśli to klawisz
                    pyautogui.press(action)
                    print(f"[ACTION] {gesture_name} -> {action}")
            
            # OGÓLNA OBSŁUGA DLA POZOSTAŁYCH (stara implementacja)
            elif action == "space":
                pyautogui.press('space')
                print(f"[ACTION] {gesture_name} -> space")
                
            elif action == "left_mouse":
                pyautogui.click(button='left')
                print(f"[ACTION] {gesture_name} -> left mouse click")
                
            elif action == "right_mouse":
                pyautogui.click(button='right')
                print(f"[ACTION] {gesture_name} -> right mouse click")
                
            elif action == "shift_toggle":
                pyautogui.press('shift')
                print(f"[ACTION] {gesture_name} -> shift toggle")
                
            elif action == "right_mouse_hold_5s":
                pyautogui.mouseDown(button='right')
                QTimer.singleShot(5000, lambda: pyautogui.mouseUp(button='right'))
                print(f"[ACTION] {gesture_name} -> right mouse hold 5s")
                
        except Exception as e:
            print(f"Error executing gesture action: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.esc_press()
        super().keyPressEvent(event)
    
    def open_heuristic_detector(self):
        self.heuristic_window = HeuristicGestureWindow(self, self.serial_reader)
        self.heuristic_window.show()

    # Funkcje obsługi przycisków
    def on_button_start_click(self):
        if not self.gesture_enabled or self.gesture_model is None:
            self.label.setText("Please load gesture model first!\nPress [Load Gesture Model]")
            return
        
        self.label.setText("Game mode activated\nPress [ESC] to cancel")
        parameters.game_state = "Game"
        
        # Reset cooldown
        self.last_gesture_time = 0
        self.current_gesture = None
        self.gesture_confidence = 0
        
        # Start gesture prediction timer
        self.gesture_timer.start(self.prediction_interval)
        
        # Reset display
        self.info_label.setText("Gesture: Waiting...")
        self.info_label.setStyleSheet(
            "font-size: 16px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;"
        )

    def on_button_hotkey_click(self):
        dialog = KeyBindingDialog(self)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            bindings = dialog.get_bindings()
            self.label.setText(f"Zapisano przypisania!\n{len(bindings)} klawiszy skonfigurowanych")
        else:
            self.label.setText("Anulowano przypisanie klawiszy")

    def open_gesture_detection(self):
        self.gesture_window = GestureDetectionWindow(self, self.serial_reader)
        self.gesture_window.show()

    def on_button_calibration_click(self):
        self.label.setText("Button 3 clicked!")

    def on_button_close_click(self):
        self.close()
    
    def on_button_data_click(self):
        self.create_data_gathering_window()

    def create_data_gathering_window(self):
        """Open data gathering window"""
        if not self.serial_reader:
            self.label.setText("Error: Serial reader not initialized!")
            return
            
        if not self.is_connected:
            self.label.setText("Error: Please connect to a device first!")
            return
        
        self.data_window = DataGatheringWindow(self, self.serial_reader)
        self.data_window.show()
    
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
            
            # Stop gesture prediction timer
            self.gesture_timer.stop()
            self.gesture_display_timer.stop()
            
            # Reset gesture state
            self.last_gesture_time = 0
            self.current_gesture = None
            self.gesture_confidence = 0
            
            # Reset info label
            self.info_label.setText("Gesture: None")
            self.info_label.setStyleSheet(
                "font-size: 16px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;"
            )

    def update_sensor_data(self, data):
        """Update GUI with sensor data from Arduino"""
        
        # Zapisz dane joysticka do parameters
        parameters.joystick_left_x = data['joy_lx']
        parameters.joystick_left_y = data['joy_ly']
        parameters.joystick_left_button = data['joy_lb']
        parameters.joystick_right_x = data['joy_rx'] 
        parameters.joystick_right_y = data['joy_ry']
        parameters.joystick_right_button = data['joy_rb']
        
        # Jeśli jesteś w trybie Game
        if parameters.game_state == "Game":
            # Wywołaj joystick controller
            self.joystick_controller.update()
            
            # Dodaj dane do bufora gestów (TYLKO jeśli nie jesteśmy w cooldown)
            if self.gesture_enabled and not self.is_in_cooldown():
                sensor_values = [
                    data.get('mpu_ax', 0.0),
                    data.get('mpu_ay', 0.0),
                    data.get('mpu_az', 0.0),
                    data.get('mpu_gx', 0.0),
                    data.get('mpu_gy', 0.0),
                    data.get('mpu_gz', 0.0),
                ]
                # Take only first 6 features
                sensor_values = sensor_values[:self.max_features]
                self.gesture_buffer.append(sensor_values)
            
            # Aktualizuj label z danymi czujników
            status_text = f"IMU Data:\n"
            status_text += f"MPU Accel: ({data['mpu_ax']:.2f}, {data['mpu_ay']:.2f}, {data['mpu_az']:.2f}) m/s²\n"
            status_text += f"MPU Gyro: ({data['mpu_gx']:.2f}, {data['mpu_gy']:.2f}, {data['mpu_gz']:.2f}) rad/s\n"
            status_text += f"Joystick L: ({data['joy_lx']}, {data['joy_ly']}) Btn: {data['joy_lb']}\n"
            status_text += f"Joystick R: ({data['joy_rx']}, {data['joy_ry']}) Btn: {data['joy_rb']}\n"
            if self.gesture_enabled:
                status_text += f"Buffer: {len(self.gesture_buffer)}/35"
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
                self.serial_reader.stop()
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