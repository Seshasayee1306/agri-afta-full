#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>

// Set to 1 after you add TensorFlow Lite for Microcontrollers library
// and generate esp32/model_data.h + esp32/afta_feature_stats.h.
#define USE_TFLM 0

#if USE_TFLM
#include "model_data.h"
#include "afta_feature_stats.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif

// ------------------------------------------------------------
// ESP32 Hybrid Inference Flow
// ------------------------------------------------------------
// 1) Run local AFTA prediction on ESP32 (TinyML).
// 2) Send afta_prediction + sensor/context data to backend.
// 3) Backend computes stage model and final irrigation decision.
// ------------------------------------------------------------

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* BACKEND_EDGE_ENDPOINT = "http://YOUR_BACKEND_IP:8000/predict_edge_afta";
const unsigned long PREDICTION_INTERVAL_MS = 10000UL;

unsigned long lastPredictionMs = 0;

struct SensorInput {
  float soilMoisture;
  float temperature;
  float humidity;
  float rainfall;
  float ph;
  float ndvi;
  int daysAfterSowing;
  const char* sowingDate;
  const char* currentDate;
  const char* region;
  const char* cropType;
  const char* soilType;
};

#if USE_TFLM
namespace {
const int kTensorArenaSize = 24 * 1024;
uint8_t tensorArena[kTensorArenaSize];

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInput = nullptr;
TfLiteTensor* tflOutput = nullptr;
bool tflReady = false;
}

static bool initTinyMl() {
  if (tflReady) return true;

  tflModel = tflite::GetModel(afta_model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLM schema version mismatch");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterpreter(
      tflModel,
      resolver,
      tensorArena,
      kTensorArenaSize
  );

  tflInterpreter = &staticInterpreter;

  TfLiteStatus allocStatus = tflInterpreter->AllocateTensors();
  if (allocStatus != kTfLiteOk) {
    Serial.println("TFLM AllocateTensors failed");
    return false;
  }

  tflInput = tflInterpreter->input(0);
  tflOutput = tflInterpreter->output(0);

  if (!tflInput || !tflOutput) {
    Serial.println("TFLM tensor binding failed");
    return false;
  }

  if (tflInput->type != kTfLiteInt8 || tflOutput->type != kTfLiteInt8) {
    Serial.println("Expected int8 in/out model");
    return false;
  }

  tflReady = true;
  return true;
}

static int8_t quantizeInput(float normalized) {
  int32_t q = (int32_t)lroundf((normalized / AFTA_INPUT_SCALE) + AFTA_INPUT_ZERO_POINT);
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

static float dequantizeOutput(int8_t qOut) {
  return ((float)qOut - (float)AFTA_OUTPUT_ZERO_POINT) * AFTA_OUTPUT_SCALE;
}
#endif

static int runTinyMlModel(const SensorInput& in) {
#if USE_TFLM
  if (!initTinyMl()) {
    Serial.println("TinyML init failed; fallback heuristic used");
    return (in.soilMoisture < 35.0f && in.rainfall < 8.0f) ? 1 : 0;
  }

  float raw[AFTA_NUM_FEATURES] = {
      in.soilMoisture,
      in.temperature,
      in.humidity,                // soil_humidity proxy
      12.0f,                      // hour fallback
      120.0f,                     // dayofyear fallback
      in.temperature,             // air_temp proxy
      in.humidity,                // air_humidity proxy
      in.rainfall,
      in.ph,
      0.0f,                       // nitrogen fallback
      0.0f,                       // phosphorus fallback
      0.0f                        // potassium fallback
  };

  for (int i = 0; i < AFTA_NUM_FEATURES; i++) {
    float norm = (raw[i] - AFTA_FEATURE_MEANS[i]) / AFTA_FEATURE_STDS[i];
    tflInput->data.int8[i] = quantizeInput(norm);
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("TinyML Invoke failed; fallback heuristic used");
    return (in.soilMoisture < 35.0f && in.rainfall < 8.0f) ? 1 : 0;
  }

  int8_t qOut = tflOutput->data.int8[0];
  float prob = dequantizeOutput(qOut);
  prob = fmaxf(0.0f, fminf(1.0f, prob));

  Serial.print("Local AFTA probability: ");
  Serial.println(prob, 4);
  return (prob >= 0.5f) ? 1 : 0;
#else
  // Compile-time fallback when TFLM is disabled.
  return (in.soilMoisture < 35.0f && in.rainfall < 8.0f) ? 1 : 0;
#endif
}

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

static void sendEdgePredictionToBackend(const SensorInput& in, int localAftaPrediction) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected; skipping backend stage computation.");
    return;
  }

  HTTPClient http;
  http.begin(BACKEND_EDGE_ENDPOINT);
  http.addHeader("Content-Type", "application/json");

  char body[900];
  snprintf(
      body,
      sizeof(body),
      "{"
      "\"afta_prediction\":%d,"
      "\"sowing_date\":\"%s\","
      "\"current_date\":\"%s\","
      "\"region\":\"%s\","
      "\"crop_type\":\"%s\","
      "\"soil_type\":\"%s\","
      "\"soil_moisture\":%.3f,"
      "\"temperature\":%.3f,"
      "\"humidity\":%.3f,"
      "\"rainfall\":%.3f,"
      "\"ph\":%.3f,"
      "\"context\":{\"ndvi\":%.3f}"
      "}",
      localAftaPrediction,
      in.sowingDate,
      in.currentDate,
      in.region,
      in.cropType,
      in.soilType,
      in.soilMoisture,
      in.temperature,
      in.humidity,
      in.rainfall,
      in.ph,
      in.ndvi
  );

  int code = http.POST((uint8_t*)body, strlen(body));
  String response = http.getString();
  http.end();

  Serial.print("Backend status: ");
  Serial.println(code);
  Serial.print("Backend response: ");
  Serial.println(response);
}

static SensorInput readSensorInput() {
  // TODO: Replace with real sensor reads.
  SensorInput input;
  input.soilMoisture = 28.0f;
  input.temperature = 34.0f;
  input.humidity = 32.0f;
  input.rainfall = 2.0f;
  input.ph = 6.4f;
  input.ndvi = 0.42f;
  input.daysAfterSowing = 38;
  input.sowingDate = "2026-01-20";
  input.currentDate = "2026-03-06";
  input.region = "Coimbatore";
  input.cropType = "Paddy";
  input.soilType = "Loamy";
  return input;
}

static void runPredictionCycle() {
  SensorInput input = readSensorInput();

  int aftaPrediction = runTinyMlModel(input);

  Serial.println();
  Serial.println("=== ESP32 Local AFTA Prediction ===");
  Serial.print("AFTA prediction: ");
  Serial.println(aftaPrediction == 1 ? "Needs water" : "No irrigation");

  sendEdgePredictionToBackend(input, aftaPrediction);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 STARTED");
  connectWifi();
  runPredictionCycle();
  lastPredictionMs = millis();
}

void loop() {
  if (millis() - lastPredictionMs >= PREDICTION_INTERVAL_MS) {
    runPredictionCycle();
    lastPredictionMs = millis();
  }
}
