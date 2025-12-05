//
//#include <Servo.h>
//
//// Servos
//Servo s1; // Base
//Servo s2; // Left joint
//Servo s3; // Right joint
//Servo s4; // Claw
//
//// Correct pins from your documentation
//#define S1_PIN A1   // Base
//#define S2_PIN A0   // Left
//#define S3_PIN 8    // Right
//#define S4_PIN 9    // Claw
//
//#define STEP_DELAY 25
//#define PAUSE_DELAY 500
//
//void sweepServo(Servo &servo, const char* name) {  
//  Serial.println("=====================================");
//  Serial.print("STARTING TEST FOR: ");
//  Serial.println(name);
//  Serial.println("=====================================");
//
//  Serial.println("[Sweep Forward 0 → 180]");
//  for (int pos = 0; pos <= 180; pos++) {
//    servo.write(pos);
//    Serial.print(name);
//    Serial.print(" angle: ");
//    Serial.println(pos);
//    delay(STEP_DELAY);
//  }
//
//  delay(PAUSE_DELAY);
//
//  Serial.println("[Sweep Backward 180 → 0]");
//  for (int pos = 180; pos >= 0; pos--) {
//    servo.write(pos);
//    Serial.print(name);
//    Serial.print(" angle: ");
//    Serial.println(pos);
//    delay(STEP_DELAY);
//  }
//
//  Serial.print("FINISHED TEST FOR: ");
//  Serial.println(name);
//  Serial.println();
//  delay(PAUSE_DELAY);
//}
//
//void setup() {
//  Serial.begin(9600);
//  Serial.println("=== ROBOT ARM SLOW-MOTION SERVO TEST ===");
//  Serial.println("Initializing servos...");
//  Serial.println();
//
//  s1.attach(S1_PIN);
//  s2.attach(S2_PIN);
//  s3.attach(S3_PIN);
//  s4.attach(S4_PIN);
//
//  delay(1000);
//  Serial.println("Servos initialized. Beginning slow test...\n");
//}
//
//void loop() {
//  sweepServo(s1, "Base Servo (S1, A1)");
//  sweepServo(s2, "Left Servo (S2, A0)");
//  sweepServo(s3, "Right Servo (S3, D8)");
//  sweepServo(s4, "Claw Servo (S4, D9)");
//
//  Serial.println("=== ALL SERVOS TESTED — RESTARTING ===\n");
//  delay(2000);
//}

//
//#include <Servo.h>
//
//// Servos
//Servo s1; // Base
//Servo s2; // Left joint
//Servo s3; // Right joint
//Servo s4; // Claw
//
//// Correct pins from documentation
//#define S1_PIN A1   // Base
//#define S2_PIN A0   // Left
//#define S3_PIN 8    // Right
//#define S4_PIN 9    // Claw
//
//void setup() {
//  Serial.begin(9600);
//  delay(1000);
//
//  Serial.println("Initializing servos...");
//
//  s1.attach(S1_PIN);
//  s2.attach(S2_PIN);
//  s3.attach(S3_PIN);
//  s4.attach(S4_PIN);
//
//  delay(500);
//
//  Serial.println("Setting ALL servos to 0 degrees...");
//
//  s1.write(0);   // Base (reccomended from 0 to 180)
//  s2.write(0);   // Left controls how high the hand is (reccomended depending on how far back or front the elbow is)
//  s3.write(180);   // Right controls the elbow angle (reccomended from 45 to 180)
//  s4.write(0);   // Claw (reccomended from 0 to 100 ish otherwise it gets too strained)
//
//  Serial.println("Done. All servos set to 0°.");
//}
//
//void loop() {
//  // Nothing else to do
//}
