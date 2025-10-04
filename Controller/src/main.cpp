#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_L3GD20_U.h>
#include <LSM303.h>

//========== PIN DEFINITIONS ==========
#define I2C_SDA 21
#define I2C_SCL 22
#define VRx_L 32
#define VRy_L 33
#define VRx_R 34
#define VRy_R 35
#define SW_L 12
#define SW_R 14
#define I2C_FREQ 400000  //400kHz (Fast Mode I2C)

//========== UNIFIED SENSOR OBJECTS ==========
Adafruit_MPU6050 mpu;
Adafruit_ADXL345_Unified adxl = Adafruit_ADXL345_Unified(12345);
Adafruit_L3GD20_Unified l3gd = Adafruit_L3GD20_Unified(20);
LSM303 lsm303;

//==============FUNCTIONS DEF================
float radToDeg(float rad);
float accelMagnitude(sensors_event_t &accel);

//=========================================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  pinMode(VRx_L, INPUT);
  pinMode(VRy_L, INPUT);
  pinMode(VRx_R, INPUT);
  pinMode(VRy_R, INPUT);
  pinMode(SW_L, INPUT_PULLUP);
  pinMode(SW_R, INPUT_PULLUP);

  //Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(I2C_FREQ);
  
  //MPU6050 init
  Serial.print("MPU6050 init");
  if (!mpu.begin()) {
    Serial.println("FAILED");
  } else {
    Serial.println("OK");
    
    //MPU6050 config
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G); // Accel range: ±8G
    mpu.setGyroRange(MPU6050_RANGE_250_DEG); //Gyro range: ±250°/s (360° full range)
    mpu.setFilterBandwidth(MPU6050_BAND_260_HZ); //Filter: 260 Hz (increased for higher sample rate)
  }
  
  //ADXL345 init
  Serial.print("ADXL345 init");
  if (!adxl.begin()) {
    Serial.println("FAILED!");
  } else {
    Serial.println("OK");
    
    //ADXL345 config
    adxl.setRange(ADXL345_RANGE_8_G); //Range: ±8G
    adxl.setDataRate(ADXL345_DATARATE_400_HZ); //Data rate: 400 Hz (increased sample rate)
  }
  
  //L3GD20H init (AltIMU-10 Gyro)
  Serial.print("L3GD20H init");
  l3gd.enableAutoRange(true);
  if (!l3gd.begin()) {
    Serial.println("FAILED!");
  } else {
    Serial.println("OK");
  }
  
  //LSM303 init (AltIMU-10 Accel)
  Serial.print("LSM303 init");
  lsm303.init();
  if (!lsm303.init(LSM303::device_D)) {
    Serial.println("FAILED!");
  } else {
    Serial.println("OK");
    lsm303.enableDefault();
  }
  
  Serial.println("\n=== Setup Complete ===\n");
  //CSV header
  Serial.println("timestamp,mpu_ax,mpu_ay,mpu_az,mpu_gx,mpu_gy,mpu_gz,adxl_ax,adxl_ay,adxl_az,l3gd_gx,l3gd_gy,l3gd_gz,lsm_ax,lsm_ay,lsm_az,joy_lx,joy_ly,joy_lb,joy_rx,joy_ry,joy_rb");
}

// ========== MAIN LOOP ==========
void sendCSVData() {
  // Read MPU6050
  sensors_event_t a1, g1, temp;
  mpu.getEvent(&a1, &g1, &temp);

  // Read ADXL345
  sensors_event_t a2;
  adxl.getEvent(&a2);

  // Read L3GD20H (AltIMU-10 Gyro)
  sensors_event_t g2;
  l3gd.getEvent(&g2);

  // Read LSM303 (AltIMU-10 Accel)
  lsm303.read();

  // Read Joysticks
  int xValueL = analogRead(VRx_L);
  int yValueL = analogRead(VRy_L);
  int xValueR = analogRead(VRx_R);
  int yValueR = analogRead(VRy_R);
  int buttonStateL = digitalRead(SW_L);
  int buttonStateR = digitalRead(SW_R);

  //CSV format: timestamp,mpu_ax,mpu_ay,mpu_az,mpu_gx,mpu_gy,mpu_gz,adxl_ax,adxl_ay,adxl_az,l3gd_gx,l3gd_gy,l3gd_gz,lsm_ax,lsm_ay,lsm_az,joy_lx,joy_ly,joy_lb,joy_rx,joy_ry,joy_rb
  Serial.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
    millis(),
    a1.acceleration.x, a1.acceleration.y, a1.acceleration.z,
    g1.gyro.x, g1.gyro.y, g1.gyro.z,
    a2.acceleration.x, a2.acceleration.y, a2.acceleration.z,
    g2.gyro.x, g2.gyro.y, g2.gyro.z,
    lsm303.a.x, lsm303.a.y, lsm303.a.z,
    xValueL, yValueL, buttonStateL,
    xValueR, yValueR, buttonStateR
  );
}

void loop() {
  sendCSVData();
  delay(10);  // 100Hz sampling rate (increased from 10Hz)
}

//gyro rad/s to deg/s
float radToDeg(float rad) {
  return rad * 57.2958;
}

//magnitude of acceleration vector
float accelMagnitude(sensors_event_t &accel) {
  return sqrt(accel.acceleration.x * accel.acceleration.x + accel.acceleration.y * accel.acceleration.y + accel.acceleration.z * accel.acceleration.z);
}