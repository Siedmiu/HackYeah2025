from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QComboBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import glob
import os
from collections import deque
from datetime import datetime

class GestureDetectionWindow(QDialog):
    def __init__(self, parent=None, serial_reader=None):
        super().__init__(parent)
        self.serial_reader = serial_reader
        self.is_detecting = False
        self.model = None
        self.scaler = None
        self.label_classes = None
        
        # Gesture to sensor mapping (must match training)
        self.gesture_sensors = {
            'Jumping': ['adxl_ax', 'adxl_ay', 'adxl_az'],
            'Kucania': ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz'],
            'Udarzenia': ['l3gd_gx', 'l3gd_gy', 'l3gd_gz', 'lsm_ax', 'lsm_ay', 'lsm_az'],
            'Udarzenia_lewa': ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz'],
            'Strzal': ['adxl_ax', 'adxl_ay', 'adxl_az', 'lsm_ax', 'lsm_ay', 'lsm_az'],
            'Syf': ['mpu_ax', 'mpu_ay', 'mpu_az', 'adxl_ax', 'adxl_ay', 'adxl_az']
        }
        
        # Buffer for sliding window (50 samples)
        self.window_size = 20
        self.max_features = 6  # Must match training padding
        self.data_buffer = deque(maxlen=self.window_size)
        
        # Detection settings
        self.prediction_interval = 500  # ms - predict every 500ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.predict_gesture)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Real-Time Gesture Detection")
        self.setGeometry(200, 200, 600, 500)
        
        layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.load_available_models()
        model_layout.addWidget(self.model_combo)
        
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        model_layout.addWidget(self.load_model_btn)
        
        layout.addLayout(model_layout)
        
        # Status
        self.model_status = QLabel("Model: Not Loaded")
        self.model_status.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.model_status)
        
        # Current prediction display
        prediction_label = QLabel("Current Prediction:")
        prediction_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(prediction_label)
        
        self.prediction_display = QLabel("---")
        self.prediction_display.setFont(QFont("Arial", 24, QFont.Bold))
        self.prediction_display.setStyleSheet(
            "padding: 20px; background-color: #e3f2fd; "
            "border: 2px solid #2196F3; border-radius: 10px; "
            "color: #1976D2;"
        )
        self.prediction_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_display)
        
        # Confidence display
        self.confidence_label = QLabel("Confidence: ---")
        self.confidence_label.setFont(QFont("Arial", 10))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
        # Buffer status
        self.buffer_label = QLabel("Buffer: 0/35 samples")
        layout.addWidget(self.buffer_label)
        
        # Prediction history
        history_label = QLabel("Prediction History:")
        layout.addWidget(history_label)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(150)
        layout.addWidget(self.history_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("background-color: green; color: white; font-size: 14px; padding: 10px;")
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: red; color: white; font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Connect to serial data if available
        if self.serial_reader:
            self.serial_reader.data_received.connect(self.on_sensor_data)
    
    def load_available_models(self):
        """Load list of available models from models directory"""
        self.model_combo.clear()
        model_files = glob.glob("models/*.keras") + glob.glob("models/*.h5")
        
        if model_files:
            for model_file in sorted(model_files):
                self.model_combo.addItem(os.path.basename(model_file), model_file)
        else:
            self.model_combo.addItem("No models found", None)
    
    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "models",
            "Model Files (*.keras *.h5);;All Files (*)"
        )
        if file_path:
            self.model_combo.addItem(os.path.basename(file_path), file_path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_combo.currentData()
        
        if not model_path or not os.path.exists(model_path):
            self.add_history("Error: Model file not found!")
            return
        
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Try to load corresponding label classes
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Extract timestamp from model name (gesture_model_20251005_051224)
            parts = model_name.split('_')
            if len(parts) >= 3:
                timestamp = '_'.join(parts[-2:])
                label_file = os.path.join(model_dir, f"label_classes_{timestamp}.npy")
                
                if os.path.exists(label_file):
                    self.label_classes = np.load(label_file, allow_pickle=True)
                else:
                    # Default classes if file not found
                    self.label_classes = np.array(['Jumping', 'Kucania', 'Strzal', 
                                                   'Syf', 'Udarzenia', 'Udarzenia_lewa'])
            
            # Initialize scaler (will be fit on-the-fly with first data)
            self.scaler = StandardScaler()
            
            self.model_status.setText(f"Model Loaded: {os.path.basename(model_path)}")
            self.model_status.setStyleSheet("padding: 10px; background-color: #c8e6c9; border-radius: 5px;")
            self.start_btn.setEnabled(True)
            self.add_history(f"Model loaded successfully: {os.path.basename(model_path)}")
            self.add_history(f"Classes: {', '.join(self.label_classes)}")
            
        except Exception as e:
            self.add_history(f"Error loading model: {str(e)}")
            self.model_status.setText("Model: Error Loading")
            self.model_status.setStyleSheet("padding: 10px; background-color: #ffcdd2; border-radius: 5px;")
    
    def start_detection(self):
        """Start real-time detection"""
        if not self.model:
            self.add_history("Error: No model loaded!")
            return
        
        if not self.serial_reader or not self.serial_reader.is_connected:
            self.add_history("Error: Serial device not connected!")
            return
        
        self.is_detecting = True
        self.data_buffer.clear()
        self.timer.start(self.prediction_interval)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_model_btn.setEnabled(False)
        
        self.add_history("Detection started...")
        self.prediction_display.setText("Collecting data...")
    
    def stop_detection(self):
        """Stop detection"""
        self.is_detecting = False
        self.timer.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_model_btn.setEnabled(True)
        
        self.prediction_display.setText("---")
        self.confidence_label.setText("Confidence: ---")
        self.add_history("Detection stopped.")
    
    def on_sensor_data(self, sensor_data):
        """Receive sensor data from serial reader"""
        if not self.is_detecting:
            return
        
        # Extract all sensor values in consistent order
        sensor_values = [
            sensor_data.get('mpu_ax', 0.0),
            sensor_data.get('mpu_ay', 0.0),
            sensor_data.get('mpu_az', 0.0),
            sensor_data.get('mpu_gx', 0.0),
            sensor_data.get('mpu_gy', 0.0),
            sensor_data.get('mpu_gz', 0.0),
            sensor_data.get('adxl_ax', 0.0),
            sensor_data.get('adxl_ay', 0.0),
            sensor_data.get('adxl_az', 0.0),
            sensor_data.get('l3gd_gx', 0.0),
            sensor_data.get('l3gd_gy', 0.0),
            sensor_data.get('l3gd_gz', 0.0),
            sensor_data.get('lsm_ax', 0.0),
            sensor_data.get('lsm_ay', 0.0),
            sensor_data.get('lsm_az', 0.0)
        ]
        
        # Take only first 6 features (max_features from training)
        sensor_values = sensor_values[:self.max_features]
        
        # Add to buffer
        self.data_buffer.append(sensor_values)
        
        # Update buffer status
        self.buffer_label.setText(f"Buffer: {len(self.data_buffer)}/{self.window_size} samples")
    
    def predict_gesture(self):
        """Make prediction from current buffer"""
        if len(self.data_buffer) < self.window_size:
            return
        
        try:
            # Create window from buffer
            window = np.array(list(self.data_buffer))
            
            # Normalize
            window_reshaped = window.reshape(-1, self.max_features)
            
            # Fit scaler on first window or use existing
            if not hasattr(self.scaler, 'mean_'):
                window_normalized = self.scaler.fit_transform(window_reshaped)
            else:
                window_normalized = self.scaler.transform(window_reshaped)
            
            window_normalized = window_normalized.reshape(1, self.window_size, self.max_features)
            
            # Predict
            predictions = self.model.predict(window_normalized, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            gesture_name = self.label_classes[predicted_class]
            
            # Update display
            self.prediction_display.setText(gesture_name)
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            # Color based on confidence
            if confidence > 80:
                color = "#4CAF50"  # Green
            elif confidence > 60:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
            
            self.prediction_display.setStyleSheet(
                f"padding: 20px; background-color: {color}20; "
                f"border: 2px solid {color}; border-radius: 10px; "
                f"color: {color}; font-weight: bold;"
            )
            
            # Add to history if confidence is high
            if confidence > 70:
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.add_history(f"[{timestamp}] {gesture_name} ({confidence:.1f}%)")
            
        except Exception as e:
            self.add_history(f"Prediction error: {str(e)}")
    
    def add_history(self, message):
        """Add message to history"""
        self.history_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.is_detecting:
            self.stop_detection()
        event.accept()