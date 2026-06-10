// WASP Sensor Node - ESP32 DevKit V1
// Reads: DHT11 (temp/humidity), PIR (motion), MQ-135, IR, Sound
// Outputs: JSON over Serial every 2 seconds
// Author: Paen (IoT)

#include <DHT.h>

// Pin Definitions
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define PIR_PIN 5
#define IR_PIN 18
#define SOUND_PIN 19
#define MQ135_PIN 34  // ADC pin

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();
  pinMode(PIR_PIN, INPUT);
  pinMode(IR_PIN, INPUT);
  pinMode(SOUND_PIN, INPUT);

  Serial.println("{\"status\":\"WASP Sensor Node Started\"}");
}

void loop() {
  // Read DHT11
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Handle NaN readings (sensor not ready)
  if (isnan(temperature)) temperature = 0;
  if (isnan(humidity)) humidity = 0;

  // Read digital sensors
  int motion = digitalRead(PIR_PIN);
  int ir = digitalRead(IR_PIN);
  int sound = digitalRead(SOUND_PIN);

  // Read analog sensor (MQ-135)
  int airQuality = analogRead(MQ135_PIN);

  // Build JSON output
  String json = "{";
  json += "\"temperature\":" + String(temperature, 1) + ",";
  json += "\"humidity\":" + String(humidity, 1) + ",";
  json += "\"motion\":" + String(motion) + ",";
  json += "\"ir\":" + String(ir) + ",";
  json += "\"sound\":" + String(sound) + ",";
  json += "\"air_quality\":" + String(airQuality);
  json += "}";

  Serial.println(json);

  delay(2000);  // Send every 2 seconds
}
