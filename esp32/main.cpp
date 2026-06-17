#include <Arduino.h>
#include "DHT.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <WiFi.h>

#define PIR_PIN 26
#define IR_PIN 19
#define DHT_PIN 4
#define MIC_PIN 32
#define MQ2_PIN 33
#define LED_WIFI_PIN 5
#define LED_OBSTACLE_PIN 18
#define DHTTYPE DHT11

DHT dht(DHT_PIN, DHTTYPE);

 //wifi
const char* ssid = "BuayaHaute";
const char* password = "A24AI9471";

//mqtt broker
const char* mqtt_server = "192.168.100.218";
const int mqtt_port = 1883;
const char* mqtt_topic = "sensors";

WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
const unsigned long interval = 2000;

unsigned long ledTimer = 0;
bool ledState = false;

// wifi LED blink function
void wifiBlink() {
  if (millis() - ledTimer > 300) {
    ledTimer = millis();
    ledState = !ledState;
    digitalWrite(LED_WIFI_PIN, ledState);
  }
}

//wifi setup function
void setup_wifi() {
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    wifiBlink();
  }

  digitalWrite(LED_WIFI_PIN, HIGH);
}

//wifi reconnect function
void reconnect() {

  while (!client.connected()) {

    wifiBlink();

    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);

    if (client.connect(clientId.c_str())) {
      digitalWrite(LED_WIFI_PIN, HIGH);
    } else {
      delay(1000);
    }
  }
}

//mic input smooth function
int readMic() {
  long sum = 0;
  for (int i = 0; i < 5; i++) {
    sum += analogRead(MIC_PIN);
  }
  return sum / 5;
}

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
  pinMode(IR_PIN, INPUT);
  pinMode(MIC_PIN, INPUT);
  pinMode(LED_WIFI_PIN, OUTPUT);
  pinMode(LED_OBSTACLE_PIN, OUTPUT);

  dht.begin();
  client.setServer(mqtt_server, mqtt_port);

  setup_wifi();
}

void loop() {

  //wifi reconnect
  if (WiFi.status() != WL_CONNECTED) {
    setup_wifi();
  }

  if (!client.connected()) {
    reconnect();
  }

  client.loop();

  unsigned long now = millis();

  if (now - lastMsg >= interval) {
    lastMsg = now;

    int motion = digitalRead(PIR_PIN);
    int ir_state = digitalRead(IR_PIN);
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();
    int mic_loud = readMic();
    int mq2_value = analogRead(MQ2_PIN);

    bool motion_b = (motion == HIGH);
    bool obstacle = (ir_state == LOW);

    // obstacle LED
    if (ir_state == LOW){
      digitalWrite(LED_OBSTACLE_PIN, HIGH);
    } else {
      digitalWrite(LED_OBSTACLE_PIN, LOW);
    }

    //NaN handler for DHT
    if (isnan(humidity) || isnan(temperature)) {
      humidity = 0;
      temperature = 0;
    }

    StaticJsonDocument<512> doc;

    doc["timestamp"] = now;
    doc["motion"] = motion_b;
    doc["obstacle"] = obstacle;
    doc["humidity"] = humidity;
    doc["temperature"] = temperature;
    doc["mic_level"] = mic_loud;
    doc["mq2_raw"] = mq2_value;

    char buffer[512];
    serializeJson(doc, buffer);

    client.publish(mqtt_topic, buffer, true);

    Serial.println(buffer);
  }
}