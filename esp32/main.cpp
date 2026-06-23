#include <Arduino.h>
#include "DHT.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <WiFi.h>

#define PIR_PIN         26
#define IR_PIN          19
#define DHT_PIN         4
#define MIC_PIN         32
#define MQ2_PIN         33
#define LED_WIFI_PIN    5
#define LED_OBSTACLE_PIN 18
#define DHTTYPE         DHT11

// ─── WiFi + Broker Config (paired) ───────────────────────────────────────────
// Each entry ties one WiFi network to its own Mosquitto broker.
// Add or remove rows here — count is auto-detected.
struct NetworkConfig {
  const char* ssid;
  const char* password;
  const char* brokerIP;
  int         brokerPort;
};

const NetworkConfig networks[] = {
  { "BuayaHaute",    "A24AI9471", "192.168.100.218", 1883 },
  { "iMazk",     "unimazk123",  "10.208.193.58",   1883 },
  // Add more as needed
};
const int networkCount = sizeof(networks) / sizeof(networks[0]);

const char* mqtt_topic = "sensors";

// ─── Globals ──────────────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHTTYPE);
WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg  = 0;
const unsigned long interval = 2000;
unsigned long ledTimer = 0;
bool ledState = false;

// ─── WiFi LED Blink ───────────────────────────────────────────────────────────
void wifiBlink() {
  if (millis() - ledTimer > 300) {
    ledTimer = millis();
    ledState = !ledState;
    digitalWrite(LED_WIFI_PIN, ledState);
  }
}

// ─── WiFi + Broker Connect (cycles through paired networks[]) ─────────────────
// On a successful WiFi connect, immediately points the MQTT client at that
// network's paired Mosquitto broker. Returns true if both are ready.
bool setup_wifi() {
  digitalWrite(LED_WIFI_PIN, LOW);

  for (int i = 0; i < networkCount; i++) {
    Serial.printf("[WiFi] Trying SSID: %s\n", networks[i].ssid);
    WiFi.begin(networks[i].ssid, networks[i].password);

    // Try for up to 10 seconds per network
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
      wifiBlink();
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("[WiFi] Connected to %s | IP: %s\n",
                    networks[i].ssid,
                    WiFi.localIP().toString().c_str());

      // Point MQTT at the paired broker for this WiFi
      client.setServer(networks[i].brokerIP, networks[i].brokerPort);
      Serial.printf("[MQTT] Broker set to %s:%d\n",
                    networks[i].brokerIP, networks[i].brokerPort);

      digitalWrite(LED_WIFI_PIN, HIGH);
      return true;
    }

    Serial.printf("[WiFi] Failed to connect to %s, trying next...\n", networks[i].ssid);
    WiFi.disconnect(true);
    delay(500);
  }

  Serial.println("[WiFi] All networks exhausted. Will retry on next cycle.");
  digitalWrite(LED_WIFI_PIN, LOW);
  return false;
}

// ─── MQTT Reconnect (uses broker already set by setup_wifi) ───────────────────
bool reconnect() {
  for (int attempt = 0; attempt < 3; attempt++) {
    wifiBlink();

    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);

    if (client.connect(clientId.c_str())) {
      Serial.println("[MQTT] Connected to broker.");
      digitalWrite(LED_WIFI_PIN, HIGH);
      return true;
    }

    Serial.printf("[MQTT] Attempt %d failed (state=%d), retrying...\n",
                  attempt + 1, client.state());
    delay(1000);
  }

  Serial.println("[MQTT] Broker unreachable. Will retry on next cycle.");
  return false;
}

// ─── Mic Averaging ────────────────────────────────────────────────────────────
int readMic() {
  long sum = 0;
  for (int i = 0; i < 5; i++) {
    sum += analogRead(MIC_PIN);
  }
  return sum / 5;
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(PIR_PIN,          INPUT);
  pinMode(IR_PIN,           INPUT);
  pinMode(MIC_PIN,          INPUT);
  pinMode(LED_WIFI_PIN,     OUTPUT);
  pinMode(LED_OBSTACLE_PIN, OUTPUT);

  dht.begin();
  setup_wifi();
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected. Reconnecting...");
    WiFi.disconnect(true);
    delay(500);
    if (!setup_wifi()) {
      delay(5000); // Back off before trying again
      return;
    }
  }

  // Reconnect MQTT if dropped
  if (!client.connected()) {
    if (!reconnect()) {
      delay(5000); // Back off before trying again
      return;
    }
  }

  client.loop();

  // ─── Sensor Publish Loop ────────────────────────────────────────────────────
  unsigned long now = millis();
  if (now - lastMsg >= interval) {
    lastMsg = now;

    int   motion      = digitalRead(PIR_PIN);
    int   ir_state    = digitalRead(IR_PIN);
    float humidity    = dht.readHumidity();
    float temperature = dht.readTemperature();
    int   mic_loud    = readMic();
    int   mq2_value   = analogRead(MQ2_PIN);

    bool motion_b = (motion   == HIGH);
    bool obstacle = (ir_state == LOW);

    // Obstacle LED
    digitalWrite(LED_OBSTACLE_PIN, obstacle ? HIGH : LOW);

    // NaN guard for DHT
    if (isnan(humidity) || isnan(temperature)) {
      humidity    = 0;
      temperature = 0;
    }

    StaticJsonDocument<512> doc;
    doc["timestamp"]   = now;
    doc["motion"]      = motion_b;
    doc["obstacle"]    = obstacle;
    doc["humidity"]    = humidity;
    doc["temperature"] = temperature;
    doc["mic_level"]   = mic_loud;
    doc["mq2_raw"]     = mq2_value;

    char buffer[512];
    serializeJson(doc, buffer);

    if (client.publish(mqtt_topic, buffer, true)) {
      Serial.println(buffer);
    } else {
      Serial.println("[MQTT] Publish failed.");
    }
  }
}