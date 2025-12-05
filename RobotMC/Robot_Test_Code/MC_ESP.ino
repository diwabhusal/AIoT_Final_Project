#include <Servo.h>

Servo s1, s2, s3, s4;

// Correct pins
#define S1_PIN A1
#define S2_PIN A0
#define S3_PIN 8
#define S4_PIN 9

// Store current angles
int a1 = 0;
int a2 = 90;
int a3 = 90;
int a4 = 0;

// NEW: paused state flag
bool paused = false;

// Smooth movement function
void moveSlow(Servo &servo, int &current, int target, int stepDelay = 8) {
  int step = (target > current) ? 1 : -1;

  for (int pos = current; pos != target; pos += step) {
    
    if (paused) {
      current = pos;      // save last actual position
      return;             // exit early → FREEZE in place
    }

    servo.write(pos);
    delay(stepDelay);
  }

  current = target;
}

void setup() {
  Serial.begin(115200);

  s1.attach(S1_PIN);
  s2.attach(S2_PIN);
  s3.attach(S3_PIN);
  s4.attach(S4_PIN);

  Serial.println("READY");
}

void loop() {

  if (Serial.available()) {

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // ----------------------------
    // NEW: Pause / Resume Commands
    // ----------------------------
    if (cmd == "PAUSE") {
      paused = true;
      
      // HARD FREEZE RIGHT NOW
      s1.write(a1);
      s2.write(a2);
      s3.write(a3);
      s4.write(a4);
    
      // OPTIONAL: uncomment for absolute immediate mechanical halt
      s1.detach();
      s2.detach();
      s3.detach();
      s4.detach();
    
      Serial.println("ACK: PAUSED");
      return;
    }

    if (cmd == "RESUME") {
      paused = false;
        s1.attach(S1_PIN);
        s2.attach(S2_PIN);
        s3.attach(S3_PIN);
        s4.attach(S4_PIN);
      Serial.println("ACK: RESUMED");
      return;
    }

    // If paused → ignore all movement commands
    if (paused) {
      Serial.println("ACK: IGNORED (PAUSED)");
      return;
    }

    // ----------------------------
    // Existing Servo Commands
    // ----------------------------

    // Servo commands with SLOW motion
    if (cmd.startsWith("S1:")) {
      int t = cmd.substring(3).toInt();
      moveSlow(s1, a1, t);
    }

    else if (cmd.startsWith("S2:")) {
      int t = cmd.substring(3).toInt();
      moveSlow(s2, a2, t);
    }

    else if (cmd.startsWith("S3:")) {
      int t = cmd.substring(3).toInt();
      moveSlow(s3, a3, t);
    }

    else if (cmd.startsWith("S4:")) {
      int t = cmd.substring(3).toInt();
      moveSlow(s4, a4, t);
    }

    // Claw presets
    else if (cmd == "OPEN")  moveSlow(s4, a4, 20);
    else if (cmd == "CLOSE") moveSlow(s4, a4, 90);

    // Safe pose (slow)
    else if (cmd == "SAFE") {
      moveSlow(s1, a1, 0);
      moveSlow(s2, a2, 90);
      moveSlow(s3, a3, 90);
      moveSlow(s4, a4, 0);
    }

    Serial.print("ACK: ");
    Serial.println(cmd);
  }
}
