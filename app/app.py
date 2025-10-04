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
    connection_status = pyqtSignal(bool, str)  # Signal for connection status (connected, message)

    def __init__(self, port=None, baudrate=115200, debug=True):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_conn = None
        self.csv_file = None
        self.csv_writer = None

        self.is_connected = False

        self.debug = debug  # Flaga debugowania

    def connect(self, port):
        """Connect to a specific port"""
        print(f"[DEBUG] connect() called with port: {port}")
        self.port = port
        if not self.isRunning():
            print("[DEBUG] Thread not running, starting...")
            self.start()
        else:
            print("[DEBUG] Thread already running!")

    def run(self):
        print("[DEBUG] run() method started")
        
        if not self.port:
            self.is_connected = False
            print("[DEBUG] No port specified, emitting error signal")
            self.connection_status.emit(False, "No port specified")
            return

        try:
            print(f"[DEBUG] Attempting to open serial port: {self.port} at {self.baudrate} baud")
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[DEBUG] Serial port opened successfully")
            print(f"[DEBUG] Port info - is_open: {self.serial_conn.is_open}")
            
            self.running = True
            self.is_connected = True

            print(f"[DEBUG] Setting running=True, emitting connection status")
            self.connection_status.emit(True, f"Connected to {self.port}")

            # Odczekaj na inicjalizację Arduino
            print("[DEBUG] Waiting 2 seconds for Arduino to initialize...")
            time.sleep(2)
            print("[DEBUG] Arduino should be ready now")
            
            # Wyczyść bufory
            print("[DEBUG] Clearing input/output buffers...")
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            print("[DEBUG] Buffers cleared")

            # Create CSV file for logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'data_{timestamp}.csv'
            print(f"[DEBUG] Creating CSV file: {csv_filename}")
            self.csv_file = open(csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            print(f"[DEBUG] CSV file created successfully")

            # Skip initial debug messages and wait for header
            header_found = False
            line_count = 0
            max_lines_to_search = 100
            
            print("[DEBUG] Starting to search for header line...")
            while self.running and not header_found and line_count < max_lines_to_search:
                print(f"[DEBUG] Checking for data in buffer... (attempt {line_count + 1}/{max_lines_to_search})")
                print(f"[DEBUG] Bytes waiting in buffer: {self.serial_conn.in_waiting}")
                
                if self.serial_conn.in_waiting > 0:
                    print(f"[DEBUG] Reading line from serial...")
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    print(f"[DEBUG] Line {line_count}: '{line[:100]}'...")
                    
                    if line.startswith('timestamp,'):
                        print(f"[DEBUG] ✓ HEADER FOUND: {line}")
                        self.csv_writer.writerow(line.split(','))
                        header_found = True
                    else:
                        print(f"[DEBUG] Not a header line, continuing search...")
                else:
                    print(f"[DEBUG] No data in buffer, waiting 0.1s...")
                    time.sleep(0.1)
                
                line_count += 1
            
            if not header_found:
                print(f"[DEBUG] ⚠ WARNING: Header not found after {line_count} attempts!")
                print(f"[DEBUG] Continuing anyway...")
            else:
                print(f"[DEBUG] Header found successfully, proceeding to data reading loop")

            # Read and process CSV data
            print("[DEBUG] Entering main data reading loop...")
            loop_iteration = 0
            last_data_time = time.time()
            
            while self.running:
                loop_iteration += 1
                
                if loop_iteration % 100 == 0:  # Log co 100 iteracji
                    print(f"[DEBUG] Loop iteration: {loop_iteration}, running: {self.running}")
                    print(f"[DEBUG] Bytes in buffer: {self.serial_conn.in_waiting}")
                
                # Sprawdź timeout
                if time.time() - last_data_time > 5:
                    print(f"[DEBUG] ⚠ WARNING: No data received for 5 seconds!")
                    last_data_time = time.time()
                
                if self.serial_conn.in_waiting > 0:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        
                        if loop_iteration % 50 == 0 or self.debug:
                            print(f"[DEBUG] Read line {loop_iteration}: length={len(line)}, preview='{line[:50]}'")
                        
                        if line and ',' in line:
                            last_data_time = time.time()  # Resetuj timer
                            
                            values = line.split(',')
                            if self.debug and loop_iteration % 50 == 0:
                                print(f"[DEBUG] Split into {len(values)} values")
                            
                            # ========== NOWY FORMAT: 4 wartości (timestamp, joy_lx, joy_ly, joy_lb) ==========
                            if len(values) == 4:
                                print(f"[DEBUG] New simplified format detected (4 values)")
                                
                                # Write to CSV file
                                self.csv_writer.writerow(values)
                                self.csv_file.flush()
                                
                                try:
                                    # Parse simplified data - tylko lewy joystick
                                    data = {
                                        'timestamp': int(values[0]),
                                        # Wszystkie sensory IMU = 0 (nieaktywne)
                                        'mpu_ax': 0.0,
                                        'mpu_ay': 0.0,
                                        'mpu_az': 0.0,
                                        'mpu_gx': 0.0,
                                        'mpu_gy': 0.0,
                                        'mpu_gz': 0.0,
                                        'adxl_ax': 0.0,
                                        'adxl_ay': 0.0,
                                        'adxl_az': 0.0,
                                        'l3gd_gx': 0.0,
                                        'l3gd_gy': 0.0,
                                        'l3gd_gz': 0.0,
                                        'lsm_ax': 0.0,
                                        'lsm_ay': 0.0,
                                        'lsm_az': 0.0,
                                        # Lewy joystick - dane rzeczywiste
                                        'joy_lx': int(values[1]),
                                        'joy_ly': int(values[2]),
                                        'joy_lb': int(values[3]),
                                        # Prawy joystick - wartości środkowe (neutralne)
                                        'joy_rx': 2048,
                                        'joy_ry': 2048,
                                        'joy_rb': 0,
                                        # Temperatura
                                        'mpu_temp': 0.0
                                    }
                                    
                                    if self.debug and loop_iteration % 10 == 0:
                                        print(f"\n{'='*60}")
                                        print(f"[NEW FORMAT - 4 values]")
                                        print(f"Timestamp: {data['timestamp']}")
                                        print(f"Joystick L: X={data['joy_lx']:4d}, Y={data['joy_ly']:4d}, Btn={data['joy_lb']}")
                                        print(f"Joystick R: DISABLED (using default 2048, 2048)")
                                        print(f"{'='*60}")

                                    # Emit signal with parsed data
                                    self.data_received.emit(data)
                                    
                                except (ValueError, IndexError) as e:
                                    print(f"[DEBUG] ✗ Error parsing simplified format: {e}")
                                    print(f"[DEBUG] Values were: {values}")
                            
                            # ========== STARY FORMAT: 23 wartości (pełne dane) ==========
                            elif len(values) == 23:
                                if loop_iteration % 50 == 0:
                                    print(f"[DEBUG] Old full format detected (23 values)")
                                
                                # Write to CSV file
                                self.csv_writer.writerow(values)
                                self.csv_file.flush()
                                
                                try:
                                    # Parse full data - wszystkie sensory
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
                                        'lsm_ax': float(values[13]),
                                        'lsm_ay': float(values[14]),
                                        'lsm_az': float(values[15]),
                                        'joy_lx': int(values[16]),
                                        'joy_ly': int(values[17]),
                                        'joy_lb': int(values[18]),
                                        'joy_rx': int(values[19]),
                                        'joy_ry': int(values[20]),
                                        'joy_rb': int(values[21]),
                                        'mpu_temp': float(values[22])
                                    }
                                    
                                    if loop_iteration % 50 == 0:
                                        print(f"[DEBUG] Full format data parsed successfully")

                                    # WYŚWIETL SPARSOWANE DANE
                                    if self.debug and loop_iteration % 10 == 0:
                                        print(f"\n{'='*60}")
                                        print(f"[OLD FORMAT - 23 values]")
                                        print(f"Timestamp: {data['timestamp']}")
                                        print(f"Joystick L: X={data['joy_lx']:4d}, Y={data['joy_ly']:4d}, Btn={data['joy_lb']}")
                                        print(f"Joystick R: X={data['joy_rx']:4d}, Y={data['joy_ry']:4d}, Btn={data['joy_rb']}")
                                        print(f"MPU Accel: X={data['mpu_ax']:7.2f}, Y={data['mpu_ay']:7.2f}, Z={data['mpu_az']:7.2f} m/s²")
                                        print(f"MPU Gyro:  X={data['mpu_gx']:7.2f}, Y={data['mpu_gy']:7.2f}, Z={data['mpu_gz']:7.2f} rad/s")
                                        print(f"Temp: {data['mpu_temp']:.1f}°C")
                                        print(f"{'='*60}")

                                    # Emit signal with parsed data
                                    self.data_received.emit(data)
                                    
                                except (ValueError, IndexError) as e:
                                    print(f"[DEBUG] ✗ Error parsing full format: {e}")
                                    print(f"[DEBUG] Values were: {values}")
                            
                            # ========== NIEZNANY FORMAT ==========
                            else:
                                if self.debug:
                                    print(f"[DEBUG] ✗ Unsupported format: got {len(values)} values (expected 4 or 23)")
                                    print(f"[DEBUG] Line was: {line[:100]}")
                        else:
                            if loop_iteration % 100 == 0 and line:
                                print(f"[DEBUG] Line without comma: '{line[:50]}'")
                                
                    except UnicodeDecodeError as e:
                        print(f"[DEBUG] ✗ Unicode decode error: {e}")
                    except Exception as e:
                        print(f"[DEBUG] ✗ Unexpected error reading line: {type(e).__name__}: {e}")
                else:
                    # Krótkie opóźnienie gdy brak danych
                    time.sleep(0.001)

        except serial.SerialException as e:
            print(f"[DEBUG] ✗✗✗ SERIAL EXCEPTION: {type(e).__name__}: {e}")
            print(f"[DEBUG] Exception occurred at line: {sys.exc_info()[2].tb_lineno}")
            self.connection_status.emit(False, f"Connection failed: {e}")
        except Exception as e:
            print(f"Serial error: {e}")
            self.is_connected = False 

            print(f"[DEBUG] ✗✗✗ UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
            print(f"[DEBUG] Exception occurred at line: {sys.exc_info()[2].tb_lineno}")
            import traceback
            traceback.print_exc()
            self.connection_status.emit(False, f"Connection failed: {e}")
        finally:
            print("[DEBUG] Entering finally block - cleaning up...")
            if self.csv_file:
                print("[DEBUG] Closing CSV file...")
                self.csv_file.close()
                print("[DEBUG] CSV file closed")
            if self.serial_conn and self.serial_conn.is_open:
                print("[DEBUG] Closing serial connection...")
                self.serial_conn.close()
                self.is_connected = False 
                print("[DEBUG] Serial connection closed")
            print("[DEBUG] Emitting disconnected signal...")
            self.connection_status.emit(False, "Disconnected")
            print("[DEBUG] run() method finished")

    def stop(self):
        print("[DEBUG] stop() called")
        self.running = False
        self.is_connected = False

        print("[DEBUG] Waiting for thread to finish...")
        self.wait()
        print("[DEBUG] Thread finished")

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
    
    # Initialize game state
    parameters.game_state = "Main_menu"
    print(f"[DEBUG] Game state initialized: {parameters.game_state}")
    
    # Find Arduino port (but don't fail if not found)
    port = find_arduino_port()

    if not port:
        print("Arduino not found. Starting in offline mode...")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        print("\nYou can connect later using the 'Connect' button in the app.")

    # Launch GUI
    print("[DEBUG] Creating QApplication...")
    app = QApplication(sys.argv)
    print("[DEBUG] Creating MainWindow...")
    window = MainWindow()

    # Create serial reader thread (may start without connection)
    print(f"[DEBUG] Creating SerialReader with port={port}, debug=True")
    serial_reader = SerialReader(port, debug=True)
    
    print("[DEBUG] Connecting signals...")
    serial_reader.data_received.connect(lambda data: window.update_sensor_data(data))
    serial_reader.connection_status.connect(lambda connected, msg: window.update_connection_status(connected, msg))
    print("[DEBUG] Signals connected")

    # Store serial reader reference
    parameters.joystick_reader = serial_reader
    print("[DEBUG] Serial reader stored in parameters")
    
    # Pass the serial reader to the window so it can reconnect
    window.set_serial_reader(serial_reader)
    print("[DEBUG] Serial reader passed to window")

    # Start connection if port was found
    if port:
        print(f"Connecting to Arduino on {port}...")
        serial_reader.start()
        print("[DEBUG] Serial reader thread started")
    else:
        print("[DEBUG] No port found, skipping connection")

    print("[DEBUG] Showing window...")
    window.show()
    print("[DEBUG] Window shown")

    print("[DEBUG] Entering Qt event loop...")
    # Cleanup
    result = app.exec_()
    print(f"[DEBUG] Qt event loop exited with code: {result}")
    
    print("[DEBUG] Stopping serial reader...")
    serial_reader.stop()
    print("[DEBUG] Serial reader stopped")
    
    print("[DEBUG] ========== APPLICATION EXITING ==========")
    sys.exit(result)

if __name__ == "__main__":
    main()