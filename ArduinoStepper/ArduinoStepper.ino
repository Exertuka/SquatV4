#include <AccelStepper.h>

// Initialize stepper (8, 10, 9, 11 sequence for ULN2003)
AccelStepper stepper(AccelStepper::HALF4WIRE, 8, 10, 9, 11);

void setup() {
  Serial.begin(115200); 
  Serial.setTimeout(10);
  
  stepper.setMaxSpeed(2000.0); 
}

void loop() {
  // Check for new data from Python
  if (Serial.available() > 0) {
    // We now expect just one number: "Speed\n"
    int motorSpeed = Serial.parseInt();
    
    if (Serial.read() == '\n') {
      // Update Motor Speed
      stepper.setSpeed(motorSpeed);
    }
  }
  
  // Keep motor spinning
  stepper.runSpeed();
}