#include <Servo.h>

Servo jaw_servo;    // Controls retractable jaws
Servo blade_servo;  // Controls cutting blade
const int force_sensor_pin = A0;  // FSR-402 force sensor
const float force_threshold_grasp = 700;  // Calibrated for ~7N (adjust after testing)
const float force_threshold_cut = 800;    // Calibrated for ~15N (adjust after testing)

void setup() {
  Serial.begin(9600);
  jaw_servo.attach(9);   // PWM pin for jaws
  blade_servo.attach(10); // PWM pin for blade
  pinMode(force_sensor_pin, INPUT);
  Serial.println("End Effector Calibration: Send 'CAL_G' for grasp, 'CAL_C' for cut, 'TEST' to test.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    
    if (cmd == "CAL_G") {
      // Calibrate grasp (adjust jaw_servo angle for <=7N)
      Serial.println("Calibrating grasp. Testing angles...");
      for (int angle = 0; angle <= 90; angle += 10) {
        jaw_servo.write(angle);
        delay(500);
        int force_raw = analogRead(force_sensor_pin);
        float force = map_force(force_raw); // Convert to Newtons (calibrate below)
        Serial.print("Angle: "); Serial.print(angle);
        Serial.print(", Force: "); Serial.print(force, 1); Serial.println("N");
        if (force <= 7.0) {
          Serial.print("Grasp calibrated at angle "); Serial.print(angle);
          Serial.println(" for <=7N.");
          break;
        }
      }
    } else if (cmd == "CAL_C") {
      // Calibrate cut (adjust blade_servo for ~15N)
      Serial.println("Calibrating cut. Testing blade actuation...");
      blade_servo.write(90); // Actuate blade
      delay(200);
      int force_raw = analogRead(force_sensor_pin);
      float force = map_force(force_raw);
      blade_servo.write(0); // Reset blade
      Serial.print("Cut force: "); Serial.print(force, 1); Serial.println("N");
      if (force >= 10.0 && force <= 20.0) {
        Serial.println("Cut calibrated for ~15N.");
      } else {
        Serial.println("Adjust blade or sensor.");
      }
    } else if (cmd == "TEST") {
      // Test grasp, cut, open sequence
      Serial.println("Testing end effector...");
      jaw_servo.write(90); // Grasp
      delay(500);
      int force_raw = analogRead(force_sensor_pin);
      float force = map_force(force_raw);
      Serial.print("Grasp force: "); Serial.print(force, 1); Serial.println("N");
      if (force <= 7.0) {
        blade_servo.write(90); delay(200); blade_servo.write(0); // Cut
        Serial.println("Cut performed.");
        jaw_servo.write(0); // Open
        Serial.println("Jaws opened.");
        Serial.println("Test: SUCCESS");
      } else {
        Serial.println("Test: FAILED (force too high)");
      }
    }
  }
}

float map_force(int raw) {
  // Map raw ADC value (0-1023) to force in Newtons
  // Linear mapping based on FSR-402 calibration (adjust after testing)
  return raw * (10.0 / 1023.0); 
}