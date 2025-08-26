#include <Servo.h>

// Create a Servo object
Servo myServo;

// Define the pin where the servo is connected
int servoPin = 9;

// A state variable to track the position
bool currentPosition = false; // false = 0 degrees, true = 90 degrees

void setup() {
  // Attach the servo to the pin
  myServo.attach(servoPin);
  
  // Set the servo to the starting position
  myServo.write(0);
}

void loop() {
  // Move the servo to the other position
  if (currentPosition == false) {
    myServo.write(90);
    currentPosition = true;
  } else {
    myServo.write(0);
    currentPosition = false;
  }

  // Wait for 1 minute (60,000 milliseconds)
  delay(60000);
}
