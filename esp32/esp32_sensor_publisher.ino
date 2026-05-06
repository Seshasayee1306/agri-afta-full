#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT11

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// Backend ingest endpoint reachable from ESP32 on the same LAN.
const char* SENSOR_ENDPOINT = "http://192.168.0.101:8000/sensor_readings";
const char* DEVICE_ID = "esp32-node-1";

DHT dht(DHTPIN, DHTTYPE);

int soilPin = 34;
int phPin = 35;

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected. IP: ");
  Serial.println(WiFi.localIP());
}

static float convertPhRawToScale14(int raw) {
  // Replace with your pH probe calibration equation if needed.
  return (raw * 14.0f) / 4095.0f;
}

static void publishToBackend(int soilValue, int phRaw, float temperature, float humidity) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, skipping publish.");
    return;
  }

  float phValue = convertPhRawToScale14(phRaw);

  HTTPClient http;
  Serial.print("Publishing to: ");
  Serial.println(SENSOR_ENDPOINT);
  http.begin(SENSOR_ENDPOINT);
  http.addHeader("Content-Type", "application/json");

  char body[320];
  snprintf(
    body,
    sizeof(body),
    "{"
    "\"device_id\":\"%s\","
    "\"soil_moisture\":%d,"
    "\"temperature\":%.2f,"
    "\"humidity\":%.2f,"
    "\"ph\":%.2f,"
    "\"soil_raw\":%d,"
    "\"ph_raw\":%d"
    "}",
    DEVICE_ID,
    soilValue,
    temperature,
    humidity,
    phValue,
    soilValue,
    phRaw
  );

  int code = http.POST((uint8_t*)body, strlen(body));
  String response = http.getString();
  http.end();

  Serial.print("POST /sensor_readings -> ");
  Serial.println(code);
  Serial.println(response);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 STARTED");
  dht.begin();
  connectWifi();
}

void loop() {
  int soilValue = analogRead(soilPin);
  int phValue = analogRead(phPin);

  delay(2000); // DHT11 needs enough interval between reads
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  Serial.println("------ Sensor Data ------");
  Serial.print("Soil Moisture Raw: ");
  Serial.println(soilValue);
  Serial.print("pH Raw Value: ");
  Serial.println(phValue);
  Serial.print("Temperature: ");
  Serial.println(temperature);
  Serial.print("Humidity: ");
  Serial.println(humidity);
  Serial.println("-------------------------");

  if (!isnan(temperature) && !isnan(humidity)) {
    publishToBackend(soilValue, phValue, temperature, humidity);
  } else {
    Serial.println("DHT ERROR! Skipping publish.");
  }

  delay(1000);
}
