import sys
import signal
import serial
import serial.tools.list_ports
import csv
from datetime import datetime
from window import MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
import parameters

class SerialReader(QThread):
    data_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)  # Signal for connection status (connected, message)

    def __init__(self, port=None, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_conn = None
        self.csv_file = None
        self.csv_writer = None

    def connect(self, port):
        """Connect to a specific port"""
        self.port = port
        if not self.isRunning():
            self.start()

    def run(self):
        if not self.port:
            self.connection_status.emit(False, "No port specified")
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True

            # Reset Arduino to get fresh header (DTR pulse)
            print("Resetting Arduino...")
            self.serial_conn.setDTR(False)  # type: ignore[attr-defined]
            import time
            time.sleep(0.1)
            self.serial_conn.setDTR(True)  # type: ignore[attr-defined]
            time.sleep(2)  # Wait for Arduino to boot

            self.connection_status.emit(True, f"Connected to {self.port}")

            # Create CSV file for logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = open(f'data_{timestamp}.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Skip initial debug messages and wait for header
            header_found = False
            print("Waiting for CSV header...")
            timeout_counter = 0
            while self.running and not header_found and timeout_counter < 100:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('timestamp,'):
                    self.csv_writer.writerow(line.split(','))
                    header_found = True
                    print(f"CSV header found: {line}")
                    print(f"Header has {len(line.split(','))} columns")
                elif line:
                    print(f"Skipping: {line[:80]}")
                timeout_counter += 1

            if not header_found:
                print("⚠ Warning: CSV header not found, but continuing anyway...")
                # Write default header
                default_header = "timestamp,mpu_ax,mpu_ay,mpu_az,mpu_gx,mpu_gy,mpu_gz,adxl_ax,adxl_ay,adxl_az,l3gd_gx,l3gd_gy,l3gd_gz,lsm_ax,lsm_ay,lsm_az,joy_lx,joy_ly,joy_lb,joy_rx,joy_ry,joy_rb"
                self.csv_writer.writerow(default_header.split(','))

            # Read and process CSV data
            print("Starting data read loop...")
            data_count = 0
            while self.running:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                if line and ',' in line:
                    try:
                        values = line.split(',')
                        if len(values) != 22:
                            print(f"⚠ Received {len(values)} values (expected 22): {line[:100]}")

                        if len(values) == 22:  # Arduino sends 22 values (no temperature)
                            # Write to CSV file
                            self.csv_writer.writerow(values)
                            self.csv_file.flush()

                            # Parse data into dictionary (no mpu_temp)
                            data = {
                                'timestamp': int(values[0]),
                                'mpu_ax': float(values[1]),
                                'mpu_ay': float(values[2]),
                                'mpu_az': float(values[3]),
                                'mpu_gx': float(values[4]),
                                'mpu_gy': float(values[5]),
                                'mpu_gz': float(values[6]),
                                'adxl_ax': float(values[7]),
                                'adxl_ay': float(values[8]),
                                'adxl_az': float(values[9]),
                                'l3gd_gx': float(values[10]),
                                'l3gd_gy': float(values[11]),
                                'l3gd_gz': float(values[12]),
                                'lsm_ax': int(values[13]),
                                'lsm_ay': int(values[14]),
                                'lsm_az': int(values[15]),
                                'joy_lx': int(values[16]),
                                'joy_ly': int(values[17]),
                                'joy_lb': int(values[18]),
                                'joy_rx': int(values[19]),
                                'joy_ry': int(values[20]),
                                'joy_rb': int(values[21])
                            }

                            # Emit signal with parsed data
                            self.data_received.emit(data)
                            data_count += 1
                            if data_count % 50 == 0:  # Print every 50 packets
                                print(f"✓ {data_count} packets processed successfully")
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line[:100]} - {e}")

        except Exception as e:
            print(f"Serial error: {e}")
            self.connection_status.emit(False, f"Connection failed: {e}")
        finally:
            if self.csv_file:
                self.csv_file.close()
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.connection_status.emit(False, "Disconnected")

    def stop(self):
        self.running = False
        if self.isRunning():
            self.wait()

def find_arduino_port():
    """Automatically find Arduino port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'USB' in port.description or 'CH340' in port.description:
            return port.device
    return None

def main():
    # Initialize game state
    parameters.game_state = "Main_menu"
    
    # Find Arduino port (but don't fail if not found)
    port = find_arduino_port()

    if not port:
        print("Arduino not found. Starting in offline mode...")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        print("\nYou can connect later using the 'Connect' button in the app.")

    # Launch GUI
    app = QApplication(sys.argv)
    window = MainWindow()

    # Create serial reader thread (may start without connection)
    serial_reader = SerialReader(port)
    serial_reader.data_received.connect(lambda data: window.update_sensor_data(data))
    serial_reader.connection_status.connect(lambda connected, msg: window.update_connection_status(connected, msg))

    # Store serial reader reference
    parameters.joystick_reader = serial_reader
    
    # Pass the serial reader to the window so it can reconnect
    window.set_serial_reader(serial_reader)

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        print("\nCtrl+C detected, shutting down...")
        serial_reader.running = False
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)

    def on_about_to_quit():
        serial_reader.running = False

    app.aboutToQuit.connect(on_about_to_quit)

    # Start connection if port was found
    if port:
        print(f"Connecting to Arduino on {port}...")
        serial_reader.start()

    window.show()

    # Cleanup
    result = 0
    try:
        result = app.exec_()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting...")
    finally:
        serial_reader.stop()
        sys.exit(result)

if __name__ == "__main__":
    main()