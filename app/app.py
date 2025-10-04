import sys
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
            self.connection_status.emit(True, f"Connected to {self.port}")

            # Create CSV file for logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = open(f'data_{timestamp}.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Skip initial debug messages and wait for header
            header_found = False
            while self.running and not header_found:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('timestamp,'):
                    self.csv_writer.writerow(line.split(','))
                    header_found = True

            # Read and process CSV data
            while self.running:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                if line and ',' in line:
                    try:
                        values = line.split(',')
                        if len(values) == 23:  # Ensure we have all expected values
                            # Write to CSV file
                            self.csv_writer.writerow(values)
                            self.csv_file.flush()

                            # Parse data into dictionary
                            data = {
                                'timestamp': int(values[0]),
                                'mpu_ax': float(values[1]),
                                'mpu_ay': float(values[2]),
                                'mpu_az': float(values[3]),
                                'mpu_gx': float(values[4]),
                                'mpu_gy': float(values[5]),
                                'mpu_gz': float(values[6]),
                                'mpu_temp': float(values[7]),
                                'adxl_ax': float(values[8]),
                                'adxl_ay': float(values[9]),
                                'adxl_az': float(values[10]),
                                'l3gd_gx': float(values[11]),
                                'l3gd_gy': float(values[12]),
                                'l3gd_gz': float(values[13]),
                                'lsm_ax': float(values[14]),
                                'lsm_ay': float(values[15]),
                                'lsm_az': float(values[16]),
                                'joy_lx': int(values[17]),
                                'joy_ly': int(values[18]),
                                'joy_lb': int(values[19]),
                                'joy_rx': int(values[20]),
                                'joy_ry': int(values[21]),
                                'joy_rb': int(values[22])
                            }

                            # Emit signal with parsed data
                            self.data_received.emit(data)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line} - {e}")

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

    # Start connection if port was found
    if port:
        print(f"Connecting to Arduino on {port}...")
        serial_reader.start()

    window.show()

    # Cleanup
    result = app.exec_()
    serial_reader.stop()
    sys.exit(result)

if __name__ == "__main__":
    main()