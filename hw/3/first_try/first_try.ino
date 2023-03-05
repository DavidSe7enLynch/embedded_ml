// updated from the sample code for the original board by Riccardo Rizzo 

// update the library from Arduino_LSM9DS1.h to this:
#include <Arduino_BMI270_BMM150.h> 

// x, y and z are on a scale of -8192 to +8192; 
// x and z are the fraction of the gravitational pull, so not the angle; y = rotational force 
float s, ax, ay, az, gx, gy, gz;
String oBuf = "";

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU! Halted.");
    while (1);
  }

}

void loop() {
  int is_read_acc = 0;
  int is_read_gyro = 0;
  if (IMU.accelerationAvailable()) {
    is_read_acc = IMU.readAcceleration(ax, ay, az);
    is_read_gyro = IMU.readGyroscope(gx, gy, gz);
    s = IMU.accelerationSampleRate(); 
  }

  // oBuf = String(s) + " " + String(x) + " " + String(y) + " " + String(z); 
  oBuf = String (is_read_acc) + "," + String (is_read_gyro) + "," + String (ax) + "," + String (ay) + "," + String (az) + "," + String (gx) + "," + String (gy) + "," + String (gz);
  Serial.println(oBuf);

}