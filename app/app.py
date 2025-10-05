import sys
import serial
import serial.tools.list_ports
import csv
from datetime import datetime
from window import MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
import parameters
import time

class SerialReader(QThread):
    data_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)

    def __init__(self, port=None, baudrate=115200, debug=True):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_conn = None
        self.csv_file = None
        self.csv_writer = None
        self.debug = debug
        
        self.column_mapping = {}
        self.expected_columns = [
            'timestamp', 
            'mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz',
            'adxl_ax', 'adxl_ay', 'adxl_az',
            'l3gd_gx', 'l3gd_gy', 'l3gd_gz',
            'lsm_ax', 'lsm_ay', 'lsm_az',
            'joy_lx', 'joy_ly', 'joy_lb',
            'joy_rx', 'joy_ry', 'joy_rb'
        ]
        
        self.default_values = {
            'timestamp': 0,
            'mpu_ax': 0.0, 'mpu_ay': 0.0, 'mpu_az': 0.0,
            'mpu_gx': 0.0, 'mpu_gy': 0.0, 'mpu_gz': 0.0,
            'adxl_ax': 0.0, 'adxl_ay': 0.0, 'adxl_az': 0.0,
            'l3gd_gx': 0.0, 'l3gd_gy': 0.0, 'l3gd_gz': 0.0,
            'lsm_ax': 0.0, 'lsm_ay': 0.0, 'lsm_az': 0.0,
            'joy_lx': 2048, 'joy_ly': 2048, 'joy_lb': 0,
            'joy_rx': 2048, 'joy_ry': 2048, 'joy_rb': 0
        }

        self.is_connected = False

        self.debug = debug

    def connect(self, port):
        self.port = port
        if not self.isRunning():
            self.start()

    def parse_header(self, header_line):
        headers = [h.strip() for h in header_line.split(',')]
        self.column_mapping = {}
        
        for idx, header in enumerate(headers):
            if header in self.expected_columns:
                self.column_mapping[header] = idx
                
        return len(self.column_mapping) > 0

    def parse_data_row(self, values):
        data = {}
        for col_name in self.expected_columns:
            if col_name in self.column_mapping:
                idx = self.column_mapping[col_name]
                if idx < len(values):
                    try:
                        if col_name == 'timestamp':
                            data[col_name] = int(values[idx])
                        elif col_name.startswith('joy_') and col_name.endswith(('x', 'y', 'b')):
                            data[col_name] = int(values[idx])
                        else:
                            data[col_name] = float(values[idx])
                    except (ValueError, IndexError) as e:
                        data[col_name] = self.default_values[col_name]
                else:
                    data[col_name] = self.default_values[col_name]
            else:
                data[col_name] = self.default_values[col_name]
        
        return data

    def run(self):
        
        if not self.port:
            self.is_connected = False
            self.connection_status.emit(False, "No port specified")
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            
            self.running = True
            self.is_connected = True

            self.connection_status.emit(True, f"Connected to {self.port}")

            # Wait for microcontroller init
            print("[DEBUG] Waiting 2 seconds for Arduino to initialize...")
            time.sleep(2)
            
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'data_{timestamp}.csv'
            self.csv_file = open(csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            header_found = False
            line_count = 0
            max_lines_to_search = 100
            
            while self.running and not header_found and line_count < max_lines_to_search:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    if 'timestamp' in line.lower() or 'joy_' in line.lower() or 'mpu_' in line.lower():
                        
                        self.csv_writer.writerow(line.split(','))
                        
                        if self.parse_header(line):
                            header_found = True
                else:
                    time.sleep(0.1)
                
                line_count += 1
            
            if not header_found:
                self.column_mapping = {col: idx for idx, col in enumerate(self.expected_columns)}

            loop_iteration = 0
            last_data_time = time.time()
            
            while self.running:
                loop_iteration += 1
                
                if time.time() - last_data_time > 5:
                    last_data_time = time.time()
                
                if self.serial_conn.in_waiting > 0:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line and ',' in line:
                            last_data_time = time.time()
                            
                            values = line.split(',')
                            
                            # Save data
                            self.csv_writer.writerow(values)
                            self.csv_file.flush()
                            
                            data = self.parse_data_row(values)

                            self.data_received.emit(data)
                                
                    except UnicodeDecodeError as e:
                        print(f"[DEBUG] Unicode decode error: {e}")
                    except Exception as e:
                        print(f"[DEBUG] Unexpected error reading line: {type(e).__name__}: {e}")
                else:
                    time.sleep(0.001)

        except serial.SerialException as e:
            print(f"[DEBUG] SERIAL EXCEPTION: {type(e).__name__}: {e}")
            self.connection_status.emit(False, f"Connection failed: {e}")
        except Exception as e:
            print(f"Serial error: {e}")
            self.is_connected = False 

            import traceback
            traceback.print_exc()
            self.connection_status.emit(False, f"Connection failed: {e}")
        finally:
            print("[DEBUG] Entering finally block - cleaning up...")
            if self.csv_file:
                self.csv_file.close()
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.is_connected = False 
                print("[DEBUG] Serial connection closed")
            print("[DEBUG] Emitting disconnected signal...")
            self.connection_status.emit(False, "Disconnected")
            print("[DEBUG] run() method finished")

    def stop(self):
        self.running = False
        self.is_connected = False

        # Waiting for thread to finish
        self.wait()

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        if 'Arduino' in port.description or 'USB' in port.description or 'CH340' in port.description:
            return port.device
    return None

def main():    
    parameters.game_state = "Main_menu"
    port = find_arduino_port()

    if not port:
        print("Arduino not found. Starting in offline mode...")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
    app = QApplication(sys.argv)
    window = MainWindow()

    serial_reader = SerialReader(port, debug=True)
    serial_reader.data_received.connect(lambda data: window.update_sensor_data(data))
    serial_reader.connection_status.connect(lambda connected, msg: window.update_connection_status(connected, msg))

    parameters.joystick_reader = serial_reader
    window.set_serial_reader(serial_reader)

    if port:
        print(f"Connecting to Arduino on {port}...")
        serial_reader.start()

    window.show()

    result = app.exec_()
    serial_reader.stop()
    
    sys.exit(result)

if __name__ == "__main__":
    main()