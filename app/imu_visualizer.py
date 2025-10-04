from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from collections import deque
import numpy as np

class IMUVisualizerWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Plot window configuration
        self.window_seconds = 6.0
        self.min_y_range = 1.0
        self.range_padding = 0.25

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
        pg.setConfigOptions(antialias=False)

        # MPU6050 Accelerometer Plot
        self.mpu_accel_plot = pg.PlotWidget(title="MPU6050 Accelerometer (m/s²)")
        self.mpu_accel_plot.setLabel('left', 'Acceleration', units='m/s²')
        self.mpu_accel_plot.setLabel('bottom', 'Time', units='s')
        self.mpu_accel_plot.addLegend()
        self.mpu_ax_curve = self.mpu_accel_plot.plot(pen='r', name='X')
        self._configure_curve(self.mpu_ax_curve)
        self.mpu_ay_curve = self.mpu_accel_plot.plot(pen='g', name='Y')
        self._configure_curve(self.mpu_ay_curve)
        self.mpu_az_curve = self.mpu_accel_plot.plot(pen='b', name='Z')
        self._configure_curve(self.mpu_az_curve)
        main_layout.addWidget(self.mpu_accel_plot)

        # MPU6050 Gyroscope Plot
        self.mpu_gyro_plot = pg.PlotWidget(title="MPU6050 Gyroscope (rad/s)")
        self.mpu_gyro_plot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.mpu_gyro_plot.setLabel('bottom', 'Time', units='s')
        self.mpu_gyro_plot.addLegend()
        self.mpu_gx_curve = self.mpu_gyro_plot.plot(pen='r', name='X')
        self._configure_curve(self.mpu_gx_curve)
        self.mpu_gy_curve = self.mpu_gyro_plot.plot(pen='g', name='Y')
        self._configure_curve(self.mpu_gy_curve)
        self.mpu_gz_curve = self.mpu_gyro_plot.plot(pen='b', name='Z')
        self._configure_curve(self.mpu_gz_curve)
        main_layout.addWidget(self.mpu_gyro_plot)

        # ADXL345 Accelerometer Plot
        self.adxl_accel_plot = pg.PlotWidget(title="ADXL345 Accelerometer (m/s²)")
        self.adxl_accel_plot.setLabel('left', 'Acceleration', units='m/s²')
        self.adxl_accel_plot.setLabel('bottom', 'Time', units='s')
        self.adxl_accel_plot.addLegend()
        self.adxl_ax_curve = self.adxl_accel_plot.plot(pen='r', name='X')
        self._configure_curve(self.adxl_ax_curve)
        self.adxl_ay_curve = self.adxl_accel_plot.plot(pen='g', name='Y')
        self._configure_curve(self.adxl_ay_curve)
        self.adxl_az_curve = self.adxl_accel_plot.plot(pen='b', name='Z')
        self._configure_curve(self.adxl_az_curve)
        main_layout.addWidget(self.adxl_accel_plot)

        # L3GD20 Gyroscope Plot
        self.l3gd_gyro_plot = pg.PlotWidget(title="L3GD20 Gyroscope (rad/s)")
        self.l3gd_gyro_plot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.l3gd_gyro_plot.setLabel('bottom', 'Time', units='s')
        self.l3gd_gyro_plot.addLegend()
        self.l3gd_gx_curve = self.l3gd_gyro_plot.plot(pen='r', name='X')
        self._configure_curve(self.l3gd_gx_curve)
        self.l3gd_gy_curve = self.l3gd_gyro_plot.plot(pen='g', name='Y')
        self._configure_curve(self.l3gd_gy_curve)
        self.l3gd_gz_curve = self.l3gd_gyro_plot.plot(pen='b', name='Z')
        self._configure_curve(self.l3gd_gz_curve)
        main_layout.addWidget(self.l3gd_gyro_plot)

        # Stats label
        self.stats_label = QLabel("Waiting for data...")
        self.stats_label.setStyleSheet("font-size: 12px; padding: 10px;")
        main_layout.addWidget(self.stats_label)

    def update_data(self, data):
        """Update graphs with new sensor data"""
        try:
            # Initialize start time on first data point
            if self.start_time is None:
                self.start_time = data['timestamp']

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

        raw_t = np.fromiter(self.timestamps, dtype=float)
        if raw_t.size < 2:
            return

        mask = slice(None)
        if self.window_seconds is not None and raw_t[-1] - raw_t[0] > self.window_seconds:
            window_mask = raw_t >= (raw_t[-1] - self.window_seconds)
            if np.count_nonzero(window_mask) >= 2:
                mask = window_mask
            else:
                mask = slice(-min(2, raw_t.size), None)

        t = raw_t[mask]

        mpu_ax = np.fromiter(self.mpu_ax, dtype=float)[mask]
        mpu_ay = np.fromiter(self.mpu_ay, dtype=float)[mask]
        mpu_az = np.fromiter(self.mpu_az, dtype=float)[mask]
        mpu_gx = np.fromiter(self.mpu_gx, dtype=float)[mask]
        mpu_gy = np.fromiter(self.mpu_gy, dtype=float)[mask]
        mpu_gz = np.fromiter(self.mpu_gz, dtype=float)[mask]
        adxl_ax = np.fromiter(self.adxl_ax, dtype=float)[mask]
        adxl_ay = np.fromiter(self.adxl_ay, dtype=float)[mask]
        adxl_az = np.fromiter(self.adxl_az, dtype=float)[mask]
        l3gd_gx = np.fromiter(self.l3gd_gx, dtype=float)[mask]
        l3gd_gy = np.fromiter(self.l3gd_gy, dtype=float)[mask]
        l3gd_gz = np.fromiter(self.l3gd_gz, dtype=float)[mask]

        # Update MPU6050 accelerometer
        self.mpu_ax_curve.setData(t, mpu_ax)
        self.mpu_ay_curve.setData(t, mpu_ay)
        self.mpu_az_curve.setData(t, mpu_az)
        self._update_viewbox(self.mpu_accel_plot, t, (mpu_ax, mpu_ay, mpu_az))

        # Update MPU6050 gyroscope
        self.mpu_gx_curve.setData(t, mpu_gx)
        self.mpu_gy_curve.setData(t, mpu_gy)
        self.mpu_gz_curve.setData(t, mpu_gz)
        self._update_viewbox(self.mpu_gyro_plot, t, (mpu_gx, mpu_gy, mpu_gz))

        # Update ADXL345 accelerometer
        self.adxl_ax_curve.setData(t, adxl_ax)
        self.adxl_ay_curve.setData(t, adxl_ay)
        self.adxl_az_curve.setData(t, adxl_az)
        self._update_viewbox(self.adxl_accel_plot, t, (adxl_ax, adxl_ay, adxl_az))

        # Update L3GD20 gyroscope
        self.l3gd_gx_curve.setData(t, l3gd_gx)
        self.l3gd_gy_curve.setData(t, l3gd_gy)
        self.l3gd_gz_curve.setData(t, l3gd_gz)
        self._update_viewbox(self.l3gd_gyro_plot, t, (l3gd_gx, l3gd_gy, l3gd_gz))

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

    def _configure_curve(self, curve):
        curve.setClipToView(True)
        curve.setDownsampling(auto=True, method='peak')
        curve.setSkipFiniteCheck(True)

    def _update_viewbox(self, plot_widget, x_data, y_arrays):
        if x_data.size < 2:
            return

        plot_item = plot_widget.getPlotItem()
        view_box = plot_item.getViewBox()

        x_min = float(x_data[0])
        x_max = float(x_data[-1])
        view_box.setXRange(x_min, x_max, padding=0.01)

        max_abs = 0.0
        for arr in y_arrays:
            if arr.size:
                finite_vals = arr[np.isfinite(arr)]
                if finite_vals.size:
                    max_abs = max(max_abs, float(np.max(np.abs(finite_vals))))

        half_range = max(self.min_y_range / 2.0, max_abs * (1.0 + self.range_padding))
        view_box.setYRange(-half_range, half_range, padding=0.0)
