#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_L3GD20_U.h>
#include <Adafruit_LSM303_Accel.h>

//========== PIN DEFINITIONS ==========
#define I2C_SDA 21
#define I2C_SCL 22
#define VRx_L 32
#define VRy_L 33
#define VRx_R 34
#define VRy_R 35
#define SW_L 36
#define SW_R 37
#define I2C_FREQ 100000  //100kHz

//========== UNIFIED SENSOR OBJECTS ==========
Adafruit_MPU6050 mpu;
Adafruit_ADXL345_Unified adxl = Adafruit_ADXL345_Unified(12345);
Adafruit_L3GD20_Unified l3gd = Adafruit_L3GD20_Unified(20);
Adafruit_LSM303_Accel_Unified lsm303_accel = Adafruit_LSM303_Accel_Unified(54321);

//=========================================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
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
    mpu.setGyroRange(MPU6050_RANGE_500_DEG); //Gyro range: ±500°/s
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ); //Filter: 21 Hz
  }
  
  //ADXL345 init
  Serial.print("ADXL345 init");
  if (!adxl.begin()) {
    Serial.println("FAILED!");
  } else {
    Serial.println("OK");
    
    //ADXL345 config
    adxl.setRange(ADXL345_RANGE_16_G); //Range: ±16G
    adxl.setDataRate(ADXL345_DATARATE_100_HZ); //Data rate: 100 Hz
  }
  
  //L3GD20H init (AltIMU-10 Gyro)
  Serial.print("L3GD20H init");
  if (!l3gd.begin()) {
    Serial.println("FAILED!");
  } else {
    Serial.println("OK");
    
    //L3GD20H config
    l3gd.enableAutoRange(true); //Auto-range enabled
  }
  
  //LSM303D init (AltIMU-10 Accel)
  Serial.print("LSM303D init");
  if (!lsm303_accel.begin()) {
    Serial.println("FAILED!");
  } else {
    Serial.println("OK");
  }
  
  Serial.println("\n=== Setup Complete ===\n");
}

// ========== MAIN LOOP ==========
void loop() {
  Serial.println("========== Sensor Reading ==========");
  
  // Read MPU6050
  sensors_event_t a1, g1, temp;
  mpu.getEvent(&a1, &g1, &temp);
  
  Serial.println("\n[MPU6050]");
  Serial.printf("  Accel: X=%.2f Y=%.2f Z=%.2f m/s²\n", 
                a1.acceleration.x, a1.acceleration.y, a1.acceleration.z);
  Serial.printf("  Gyro:  X=%.2f Y=%.2f Z=%.2f rad/s (%.1f %.1f %.1f °/s)\n", 
                g1.gyro.x, g1.gyro.y, g1.gyro.z,
                radToDeg(g1.gyro.x), radToDeg(g1.gyro.y), radToDeg(g1.gyro.z));
  Serial.printf("  Temp:  %.2f °C\n", temp.temperature);
  
  // Read ADXL345
  sensors_event_t a2;
  adxl.getEvent(&a2);
  
  Serial.println("\n[ADXL345]");
  Serial.printf("  Accel: X=%.2f Y=%.2f Z=%.2f m/s²\n", 
                a2.acceleration.x, a2.acceleration.y, a2.acceleration.z);
  
  // Read L3GD20H (AltIMU-10 Gyro)
  sensors_event_t g2;
  l3gd.getEvent(&g2);
  
  Serial.println("\n[L3GD20H - AltIMU-10 Gyro]");
  Serial.printf("  Gyro:  X=%.2f Y=%.2f Z=%.2f rad/s (%.1f %.1f %.1f °/s)\n", 
                g2.gyro.x, g2.gyro.y, g2.gyro.z,
                radToDeg(g2.gyro.x), radToDeg(g2.gyro.y), radToDeg(g2.gyro.z));
  
  // Read LSM303D (AltIMU-10 Accel)
  sensors_event_t a3;
  lsm303_accel.getEvent(&a3);
  
  Serial.println("\n[LSM303D - AltIMU-10 Accel]");
  Serial.printf("  Accel: X=%.2f Y=%.2f Z=%.2f m/s²\n", 
                a3.acceleration.x, a3.acceleration.y, a3.acceleration.z);
  
  Serial.println("\n====================================\n");
  delay(1000);
}

//gyro rad/s to deg/s
float radToDeg(float rad) {
  return rad * 57.2958;
}

//magnitude of acceleration vector
float accelMagnitude(sensors_event_t &accel) {
  return sqrt(accel.acceleration.x * accel.acceleration.x + accel.acceleration.y * accel.acceleration.y + accel.acceleration.z * accel.acceleration.z);
}