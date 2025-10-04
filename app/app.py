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
        
        # DYNAMICZNE mapowanie kolumn CSV
        self.column_mapping = {}  # {nazwa_kolumny: indeks}
        self.expected_columns = [
            'timestamp', 
            'mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz',
            'adxl_ax', 'adxl_ay', 'adxl_az',
            'l3gd_gx', 'l3gd_gy', 'l3gd_gz',
            'lsm_ax', 'lsm_ay', 'lsm_az',
            'joy_lx', 'joy_ly', 'joy_lb',
            'joy_rx', 'joy_ry', 'joy_rb'
        ]
        
        # Domyślne wartości dla brakujących kolumn
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

    def connect(self, port):
        """Connect to a specific port"""
        print(f"[DEBUG] connect() called with port: {port}")
        self.port = port
        if not self.isRunning():
            print("[DEBUG] Thread not running, starting...")
            self.start()
        else:
            print("[DEBUG] Thread already running!")

    def parse_header(self, header_line):
        """Parsuj nagłówek CSV i stwórz mapowanie kolumn"""
        headers = [h.strip() for h in header_line.split(',')]
        self.column_mapping = {}
        
        print(f"[DEBUG] Parsing CSV header: {headers}")
        
        for idx, header in enumerate(headers):
            if header in self.expected_columns:
                self.column_mapping[header] = idx
                print(f"[DEBUG]   ✓ Mapped '{header}' to index {idx}")
            else:
                print(f"[DEBUG]   ⚠ Unknown column '{header}' at index {idx} (will be ignored)")
        
        # Pokaż które kolumny są dostępne
        available = set(self.column_mapping.keys())
        missing = set(self.expected_columns) - available
        
        print(f"[DEBUG] Available columns ({len(available)}): {sorted(available)}")
        if missing:
            print(f"[DEBUG] Missing columns ({len(missing)}): {sorted(missing)} (will use defaults)")
        
        return len(self.column_mapping) > 0

    def parse_data_row(self, values):
        """Parsuj wiersz danych używając mapowania kolumn"""
        data = {}
        
        # Dla każdej oczekiwanej kolumny
        for col_name in self.expected_columns:
            if col_name in self.column_mapping:
                # Kolumna istnieje w CSV
                idx = self.column_mapping[col_name]
                if idx < len(values):
                    try:
                        # Parsuj wartość w zależności od typu
                        if col_name == 'timestamp':
                            data[col_name] = int(values[idx])
                        elif col_name.startswith('joy_') and col_name.endswith(('x', 'y', 'b')):
                            data[col_name] = int(values[idx])
                        else:
                            data[col_name] = float(values[idx])
                    except (ValueError, IndexError) as e:
                        print(f"[DEBUG] ⚠ Error parsing '{col_name}' value '{values[idx]}': {e}")
                        data[col_name] = self.default_values[col_name]
                else:
                    # Indeks poza zakresem - użyj domyślnej wartości
                    data[col_name] = self.default_values[col_name]
            else:
                # Kolumna nie istnieje w CSV - użyj domyślnej wartości
                data[col_name] = self.default_values[col_name]
        
        return data

    def run(self):
        print("[DEBUG] run() method started")
        
        if not self.port:
            print("[DEBUG] No port specified, emitting error signal")
            self.connection_status.emit(False, "No port specified")
            return

        try:
            print(f"[DEBUG] Attempting to open serial port: {self.port} at {self.baudrate} baud")
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[DEBUG] Serial port opened successfully")
            
            self.running = True
            self.connection_status.emit(True, f"Connected to {self.port}")

            # Odczekaj na inicjalizację Arduino
            print("[DEBUG] Waiting 2 seconds for Arduino to initialize...")
            time.sleep(2)
            
            # Wyczyść bufory
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()

            # Create CSV file for logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'data_{timestamp}.csv'
            print(f"[DEBUG] Creating CSV file: {csv_filename}")
            self.csv_file = open(csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # DYNAMICZNE wyszukiwanie nagłówka
            header_found = False
            line_count = 0
            max_lines_to_search = 100
            
            print("[DEBUG] Starting to search for header line...")
            while self.running and not header_found and line_count < max_lines_to_search:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    print(f"[DEBUG] Line {line_count}: '{line[:100]}'...")
                    
                    # Sprawdź czy to nagłówek (zawiera znane nazwy kolumn)
                    if 'timestamp' in line.lower() or 'joy_' in line.lower() or 'mpu_' in line.lower():
                        print(f"[DEBUG] ✓ HEADER FOUND: {line}")
                        
                        # Zapisz nagłówek do CSV
                        self.csv_writer.writerow(line.split(','))
                        
                        # Sparsuj nagłówek i stwórz mapowanie
                        if self.parse_header(line):
                            header_found = True
                        else:
                            print(f"[DEBUG] ✗ No valid columns found in header, continuing search...")
                    else:
                        print(f"[DEBUG] Not a header line, continuing search...")
                else:
                    time.sleep(0.1)
                
                line_count += 1
            
            if not header_found:
                print(f"[DEBUG] ⚠ WARNING: Header not found after {line_count} attempts!")
                print(f"[DEBUG] Will use default column order (old format)")
                # Ustaw domyślne mapowanie dla starego formatu
                self.column_mapping = {col: idx for idx, col in enumerate(self.expected_columns)}

            # Read and process CSV data
            print("[DEBUG] Entering main data reading loop...")
            loop_iteration = 0
            last_data_time = time.time()
            
            while self.running:
                loop_iteration += 1
                
                if loop_iteration % 100 == 0:
                    print(f"[DEBUG] Loop iteration: {loop_iteration}")
                
                if time.time() - last_data_time > 5:
                    print(f"[DEBUG] ⚠ WARNING: No data received for 5 seconds!")
                    last_data_time = time.time()
                
                if self.serial_conn.in_waiting > 0:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        
                        if loop_iteration % 50 == 0 or self.debug:
                            print(f"[DEBUG] Read line {loop_iteration}: length={len(line)}")
                        
                        if line and ',' in line:
                            last_data_time = time.time()
                            
                            values = line.split(',')
                            
                            # Zapisz surowe dane do CSV
                            self.csv_writer.writerow(values)
                            self.csv_file.flush()
                            
                            # DYNAMICZNE parsowanie danych używając mapowania kolumn
                            data = self.parse_data_row(values)
                            
                            # Debug output
                            if self.debug and loop_iteration % 10 == 0:
                                print(f"\n{'='*60}")
                                print(f"[DYNAMIC FORMAT - {len(values)} values]")
                                print(f"Timestamp: {data['timestamp']}")
                                print(f"Joystick L: X={data['joy_lx']:4d}, Y={data['joy_ly']:4d}, Btn={data['joy_lb']}")
                                print(f"Joystick R: X={data['joy_rx']:4d}, Y={data['joy_ry']:4d}, Btn={data['joy_rb']}")
                                
                                # Pokaż tylko niezerowe wartości IMU
                                imu_active = False
                                for key in ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz']:
                                    if data[key] != 0.0:
                                        imu_active = True
                                        break
                                
                                if imu_active:
                                    print(f"MPU Accel: X={data['mpu_ax']:7.2f}, Y={data['mpu_ay']:7.2f}, Z={data['mpu_az']:7.2f} m/s²")
                                    print(f"MPU Gyro:  X={data['mpu_gx']:7.2f}, Y={data['mpu_gy']:7.2f}, Z={data['mpu_gz']:7.2f} rad/s")
                                else:
                                    print("IMU: INACTIVE (using defaults)")
                                
                                print(f"{'='*60}")

                            # Emit signal with parsed data
                            self.data_received.emit(data)
                                
                    except UnicodeDecodeError as e:
                        print(f"[DEBUG] ✗ Unicode decode error: {e}")
                    except Exception as e:
                        print(f"[DEBUG] ✗ Unexpected error reading line: {type(e).__name__}: {e}")
                else:
                    time.sleep(0.001)

        except serial.SerialException as e:
            print(f"[DEBUG] ✗✗✗ SERIAL EXCEPTION: {type(e).__name__}: {e}")
            self.connection_status.emit(False, f"Connection failed: {e}")
        except Exception as e:
            print(f"[DEBUG] ✗✗✗ UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.connection_status.emit(False, f"Connection failed: {e}")
        finally:
            print("[DEBUG] Entering finally block - cleaning up...")
            if self.csv_file:
                self.csv_file.close()
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.connection_status.emit(False, "Disconnected")
            print("[DEBUG] run() method finished")

    def stop(self):
        print("[DEBUG] stop() called")
        self.running = False
        self.wait()

def find_arduino_port():
    """Automatically find Arduino port"""
    print("[DEBUG] Searching for Arduino port...")
    ports = serial.tools.list_ports.comports()
    print(f"[DEBUG] Found {len(ports)} serial ports:")
    
    for port in ports:
        print(f"[DEBUG]   - {port.device}: {port.description}")
        if 'Arduino' in port.description or 'USB' in port.description or 'CH340' in port.description:
            print(f"[DEBUG] ✓ Selected port: {port.device}")
            return port.device
    
    print("[DEBUG] No matching Arduino port found")
    return None

def main():
    print("[DEBUG] ========== APPLICATION STARTING ==========")
    
    parameters.game_state = "Main_menu"
    port = find_arduino_port()

    if not port:
        print("Arduino not found. Starting in offline mode...")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        print("\nYou can connect later using the 'Connect' button in the app.")

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
    
    print("[DEBUG] ========== APPLICATION EXITING ==========")
    sys.exit(result)

if __name__ == "__main__":
    main()