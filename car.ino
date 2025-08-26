const byte enA = 11;
const byte in1 = 13;
const byte in2 = 12;

const byte enB = 3;
const byte in3 = 4;
const byte in4 = 2;

const byte enC = 5;
const byte in5 = 6;
const byte in6 = 7;

const byte enD = 10;
const byte in7 = 9;
const byte in8 = 8;




void setup(){
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);

  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  pinMode(enC, OUTPUT);
  pinMode(in5, OUTPUT);
  pinMode(in6, OUTPUT);

  pinMode(enD, OUTPUT);
  pinMode(in7, OUTPUT);
  pinMode(in8, OUTPUT);

  digitalWrite(enA, LOW);
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(enB, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  digitalWrite(enC, LOW);
  digitalWrite(in5, LOW);
  digitalWrite(in6, LOW);
  digitalWrite(enD, LOW);
  digitalWrite(in7, LOW);
  digitalWrite(in8, LOW);
 
  Serial.begin(9600);
  while (! Serial);
}

void loop(){
  // Accelerating
  for (int i = 0; i <= 255; i += 1){
    analogWrite(enA, i);
    analogWrite(enB, i);
    analogWrite(enC, i);
    analogWrite(enD, i);
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    digitalWrite(in5, LOW);
    digitalWrite(in6, HIGH);
    digitalWrite(in7, LOW);
    digitalWrite(in8, HIGH);
    delay(20);
  }

  // Decelerating
  for (int i = 255; i >= 0; i -= 1){
    analogWrite(enA, i);
    analogWrite(enB, i);
    analogWrite(enC, i);
    analogWrite(enD, i);
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    digitalWrite(in5, LOW);
    digitalWrite(in6, HIGH);
    digitalWrite(in7, LOW);
    digitalWrite(in8, HIGH);
    delay(20);
  }
}
