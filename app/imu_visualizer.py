from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from collections import deque
import numpy as np

class IMUVisualizerWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Data buffers (store last 500 samples)
        self.max_samples = 500
        self.timestamps = deque(maxlen=self.max_samples)

        # MPU6050 data
        self.mpu_ax = deque(maxlen=self.max_samples)
        self.mpu_ay = deque(maxlen=self.max_samples)
        self.mpu_az = deque(maxlen=self.max_samples)
        self.mpu_gx = deque(maxlen=self.max_samples)
        self.mpu_gy = deque(maxlen=self.max_samples)
        self.mpu_gz = deque(maxlen=self.max_samples)

        # ADXL345 data
        self.adxl_ax = deque(maxlen=self.max_samples)
        self.adxl_ay = deque(maxlen=self.max_samples)
        self.adxl_az = deque(maxlen=self.max_samples)

        # L3GD20 data
        self.l3gd_gx = deque(maxlen=self.max_samples)
        self.l3gd_gy = deque(maxlen=self.max_samples)
        self.l3gd_gz = deque(maxlen=self.max_samples)

        # LSM303 data
        self.lsm_ax = deque(maxlen=self.max_samples)
        self.lsm_ay = deque(maxlen=self.max_samples)
        self.lsm_az = deque(maxlen=self.max_samples)

        self.start_time = None

    def init_ui(self):
        self.setWindowTitle("IMU Data Visualizer - Live Graphs")
        self.setGeometry(100, 100, 1400, 900)

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title = QLabel("Live IMU Data Visualization (100Hz)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title)

        # Create plot widgets using pyqtgraph for better performance
        pg.setConfigOptions(antialias=True)

        # MPU6050 Accelerometer Plot
        self.mpu_accel_plot = pg.PlotWidget(title="MPU6050 Accelerometer (m/s²)")
        self.mpu_accel_plot.setLabel('left', 'Acceleration', units='m/s²')
        self.mpu_accel_plot.setLabel('bottom', 'Time', units='s')
        self.mpu_accel_plot.addLegend()
        self.mpu_ax_curve = self.mpu_accel_plot.plot(pen='r', name='X')
        self.mpu_ay_curve = self.mpu_accel_plot.plot(pen='g', name='Y')
        self.mpu_az_curve = self.mpu_accel_plot.plot(pen='b', name='Z')
        main_layout.addWidget(self.mpu_accel_plot)

        # MPU6050 Gyroscope Plot
        self.mpu_gyro_plot = pg.PlotWidget(title="MPU6050 Gyroscope (rad/s)")
        self.mpu_gyro_plot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.mpu_gyro_plot.setLabel('bottom', 'Time', units='s')
        self.mpu_gyro_plot.addLegend()
        self.mpu_gx_curve = self.mpu_gyro_plot.plot(pen='r', name='X')
        self.mpu_gy_curve = self.mpu_gyro_plot.plot(pen='g', name='Y')
        self.mpu_gz_curve = self.mpu_gyro_plot.plot(pen='b', name='Z')
        main_layout.addWidget(self.mpu_gyro_plot)

        # ADXL345 Accelerometer Plot
        self.adxl_accel_plot = pg.PlotWidget(title="ADXL345 Accelerometer (m/s²)")
        self.adxl_accel_plot.setLabel('left', 'Acceleration', units='m/s²')
        self.adxl_accel_plot.setLabel('bottom', 'Time', units='s')
        self.adxl_accel_plot.addLegend()
        self.adxl_ax_curve = self.adxl_accel_plot.plot(pen='r', name='X')
        self.adxl_ay_curve = self.adxl_accel_plot.plot(pen='g', name='Y')
        self.adxl_az_curve = self.adxl_accel_plot.plot(pen='b', name='Z')
        main_layout.addWidget(self.adxl_accel_plot)

        # L3GD20 Gyroscope Plot
        self.l3gd_gyro_plot = pg.PlotWidget(title="L3GD20 Gyroscope (rad/s)")
        self.l3gd_gyro_plot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.l3gd_gyro_plot.setLabel('bottom', 'Time', units='s')
        self.l3gd_gyro_plot.addLegend()
        self.l3gd_gx_curve = self.l3gd_gyro_plot.plot(pen='r', name='X')
        self.l3gd_gy_curve = self.l3gd_gyro_plot.plot(pen='g', name='Y')
        self.l3gd_gz_curve = self.l3gd_gyro_plot.plot(pen='b', name='Z')
        main_layout.addWidget(self.l3gd_gyro_plot)

        # Stats label
        self.stats_label = QLabel("Waiting for data...")
        self.stats_label.setStyleSheet("font-size: 12px; padding: 10px;")
        main_layout.addWidget(self.stats_label)

    def update_data(self, data):
        """Update graphs with new sensor data"""
        try:
            print(f"Visualizer received data: timestamp={data.get('timestamp')}, mpu_ax={data.get('mpu_ax')}")

            # Initialize start time on first data point
            if self.start_time is None:
                self.start_time = data['timestamp']
                print(f"Visualizer initialized with start_time: {self.start_time}")

            # Calculate relative time in seconds
            time_sec = (data['timestamp'] - self.start_time) / 1000.0
            self.timestamps.append(time_sec)

            # Store MPU6050 data
            self.mpu_ax.append(data['mpu_ax'])
            self.mpu_ay.append(data['mpu_ay'])
            self.mpu_az.append(data['mpu_az'])
            self.mpu_gx.append(data['mpu_gx'])
            self.mpu_gy.append(data['mpu_gy'])
            self.mpu_gz.append(data['mpu_gz'])

            # Store ADXL345 data
            self.adxl_ax.append(data['adxl_ax'])
            self.adxl_ay.append(data['adxl_ay'])
            self.adxl_az.append(data['adxl_az'])

            # Store L3GD20 data
            self.l3gd_gx.append(data['l3gd_gx'])
            self.l3gd_gy.append(data['l3gd_gy'])
            self.l3gd_gz.append(data['l3gd_gz'])

            # Store LSM303 data
            self.lsm_ax.append(data['lsm_ax'])
            self.lsm_ay.append(data['lsm_ay'])
            self.lsm_az.append(data['lsm_az'])

            # Update plots
            self.update_plots()

            # Update stats
            self.update_stats(data)

        except Exception as e:
            print(f"Error updating visualizer: {e}")

    def update_plots(self):
        """Update all plot curves"""
        if len(self.timestamps) < 2:
            return

        t = np.array(self.timestamps)

        # Update MPU6050 accelerometer
        self.mpu_ax_curve.setData(t, np.array(self.mpu_ax))
        self.mpu_ay_curve.setData(t, np.array(self.mpu_ay))
        self.mpu_az_curve.setData(t, np.array(self.mpu_az))

        # Update MPU6050 gyroscope
        self.mpu_gx_curve.setData(t, np.array(self.mpu_gx))
        self.mpu_gy_curve.setData(t, np.array(self.mpu_gy))
        self.mpu_gz_curve.setData(t, np.array(self.mpu_gz))

        # Update ADXL345 accelerometer
        self.adxl_ax_curve.setData(t, np.array(self.adxl_ax))
        self.adxl_ay_curve.setData(t, np.array(self.adxl_ay))
        self.adxl_az_curve.setData(t, np.array(self.adxl_az))

        # Update L3GD20 gyroscope
        self.l3gd_gx_curve.setData(t, np.array(self.l3gd_gx))
        self.l3gd_gy_curve.setData(t, np.array(self.l3gd_gy))
        self.l3gd_gz_curve.setData(t, np.array(self.l3gd_gz))

    def update_stats(self, data):
        """Update statistics display"""
        stats_text = f"Samples: {len(self.timestamps)} | "
        stats_text += f"MPU Accel Mag: {np.sqrt(data['mpu_ax']**2 + data['mpu_ay']**2 + data['mpu_az']**2):.2f} m/s² | "
        stats_text += f"MPU Gyro Mag: {np.sqrt(data['mpu_gx']**2 + data['mpu_gy']**2 + data['mpu_gz']**2):.3f} rad/s | "
        stats_text += f"Time: {self.timestamps[-1]:.2f}s"
        self.stats_label.setText(stats_text)

    def clear_data(self):
        """Clear all data buffers"""
        self.timestamps.clear()
        self.mpu_ax.clear()
        self.mpu_ay.clear()
        self.mpu_az.clear()
        self.mpu_gx.clear()
        self.mpu_gy.clear()
        self.mpu_gz.clear()
        self.adxl_ax.clear()
        self.adxl_ay.clear()
        self.adxl_az.clear()
        self.l3gd_gx.clear()
        self.l3gd_gy.clear()
        self.l3gd_gz.clear()
        self.lsm_ax.clear()
        self.lsm_ay.clear()
        self.lsm_az.clear()
        self.start_time = None
