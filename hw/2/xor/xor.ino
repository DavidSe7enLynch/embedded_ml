void setup() {
  // put your setup code here, to run once:
  pinMode(D2, INPUT);
  pinMode(D4, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if ((digitalRead(D2) == HIGH && digitalRead(D4) == LOW)
      || (digitalRead(D2) == LOW && digitalRead(D4) == HIGH)) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}
