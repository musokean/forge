// forge 硬件 Phase 1 · ESP32 设备端固件参考（美容仪）
//
// 这份 sketch 是 `device_sim.py` 的**真机版**：说同一套协议（protocol v1，见 docs/hardware.md），
// 换成真继电器 / PWM 档位 / 温度传感器即可驱动物理美容仪。
//
// ⚠️ 状态说明（诚实标注）：这是**参考骨架**，未在本机编译/上机验证（本机没有 ESP32 工具链与硬件）。
//    烧录前请按 docs/hardware.md 的字段顺序与 CRC 约定核对；协议一致性可用
//    `python device_sim.py --transport socket` 与 Agent 侧对拍后再上真机。
//
// 依赖：ArduinoJson（v6/v7 均可）+ WiFi + PubSubClient（用 MQTT 承载时才需要）
//
// 接线（示例，按你的硬件改）：
//   档位 1/2/3  → GPIO 25/26/27（继电器或 MOS 驱动，注意与市电隔离！）
//   电源        → GPIO 14（主继电器）
//   温度        → GPIO 34（NTC 10k 分压，ADC1 通道）
//   心跳 LED    → GPIO 2
//
// 协议要点（与 src/hwproto.py 一一对应）：
//   行式 JSON，`\n` 结尾；字段 v/seq/ts/type/crc
//   crc = crc32(不含 crc 字段的规范化 JSON) 前 8 位 hex —— **规范化 = 键按字典序、无空格**
//   下行 cmd {"id":"set_level","args":{"level":2}}；上行先 ack（接下/拒绝）再 state（真实状态）
//   过热 → 立即断电 + 上报 event{"kind":"over_temp"}
//
// 安全：设备端的过热保护是**最后一道防线**（Agent 侧的控制平面还有一道），
//       两者都要有——真机上别只依赖一边。

#include <ArduinoJson.h>

// ── 配置 ──────────────────────────────────────────────────────────
static const char* DEVICE_ID = "beauty-01";
static const char* DEVICE_NAME = "三合一美容仪";
static const char* FW_VERSION = "esp32-1.0";
static const int   PROTOCOL_VERSION = 1;

static const float MAX_SAFE_TEMP_C = 45.0f;   // 过热阈值（与 config 的 policy.max_temp_c 对齐）
static const int   MAX_LEVEL = 3;

static const int PIN_POWER = 14;
static const int PIN_LEVEL[4] = { -1, 25, 26, 27 };   // 1/2/3 档
static const int PIN_TEMP = 34;
static const int PIN_LED = 2;

// ── 设备状态 ──────────────────────────────────────────────────────
bool  g_power = false;
int   g_level = 0;
float g_tempC = 25.0f;
bool  g_tripped = false;
unsigned long g_startedAt = 0;
bool  g_trippedReported = false;

// ── NTC 10k 分压（beta 公式；按你的分压电阻/系数校准）─────────────
float readTemperatureC() {
  int raw = analogRead(PIN_TEMP);
  if (raw <= 0) return -100.0f;                     // 传感器异常，交给上层处理
  float r = 10000.0f * (4095.0f / (float)raw - 1.0f);
  const float B = 3950.0f, R0 = 10000.0f, T0 = 298.15f;
  float invT = 1.0f / T0 + (1.0f / B) * log(r / R0);
  return 1.0f / invT - 273.15f;
}

// ── 物理动作 ──────────────────────────────────────────────────────
void applyLevel(int level) {
  for (int i = 1; i <= MAX_LEVEL; i++) {
    digitalWrite(PIN_LEVEL[i], (g_power && level == i) ? HIGH : LOW);
  }
}

void emergencyOff(const char* why) {
  g_power = false;
  g_level = 0;
  applyLevel(0);
  digitalWrite(PIN_POWER, LOW);
  digitalWrite(PIN_LED, LOW);
  g_tripped = true;
  Serial.printf("{\"note\":\"emergency off: %s\"}\n", why);
}

bool powerOn() {
  if (g_tripped) return false;
  g_power = true;
  g_level = 1;
  g_startedAt = millis();
  digitalWrite(PIN_POWER, HIGH);
  digitalWrite(PIN_LED, HIGH);
  applyLevel(1);
  return true;
}

bool powerOff() {
  g_power = false;
  g_level = 0;
  applyLevel(0);
  digitalWrite(PIN_POWER, LOW);
  digitalWrite(PIN_LED, LOW);
  return true;
}

float currentOf(int level) {           // 与 fake_device 的档位-电流表对齐（可按实测改）
  switch (level) { case 1: return 0.8f; case 2: return 1.6f; case 3: return 2.4f; default: return 0.0f; }
}

// ── 帧收发：字段按字典序输出（Python 侧 crc_of() 的规范化形式必须一致）──
uint32_t crc32Of(const String& s) {                 // 标准 CRC-32 (IEEE 802.3)，与 zlib.crc32 一致
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < s.length(); i++) {
    crc ^= (uint8_t)s[i];
    for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & -(int32_t)(crc & 1));
  }
  return ~crc;
}

void sendRaw(String body) {                          // body 不含 crc 字段，且键已按字典序排列
  char crcHex[9];
  snprintf(crcHex, sizeof(crcHex), "%08x", crc32Of(body));
  String out = body.substring(0, body.length() - 1) + ",\"crc\":\"" + String(crcHex) + "\"}";
  Serial.print(out);
  Serial.print('\n');
}

// state 帧：键按字典序 current_a < level < power < run_seconds < safety_tripped < temperature_c
void sendState(long seq, const char* cmdId, bool ok, const char* reason) {
  DynamicJsonDocument doc(768);
  doc["v"] = PROTOCOL_VERSION;
  doc["seq"] = seq;
  doc["type"] = "state";
  doc["id"] = cmdId;
  doc["ok"] = ok;
  doc["reason"] = reason;
  JsonObject st = doc.createNestedObject("state");
  st["current_a"] = g_power ? currentOf(g_level) : 0.0f;
  st["level"] = g_level;
  st["power"] = g_power;
  st["run_seconds"] = g_power ? (millis() - g_startedAt) / 1000.0f : 0.0f;
  st["safety_tripped"] = g_tripped;
  st["temperature_c"] = g_tempC;
  String body;
  serializeJson(doc, body);
  sendRaw(body);
}

void sendAck(long seq, const char* cmdId, bool ok, const char* reason, const char* code) {
  DynamicJsonDocument doc(384);
  doc["v"] = PROTOCOL_VERSION;
  doc["seq"] = seq;
  doc["type"] = "ack";
  doc["id"] = cmdId;
  doc["ok"] = ok;
  doc["reason"] = reason;
  doc["code"] = code;
  String body; serializeJson(doc, body);
  sendRaw(body);
}

void sendHelloAck(long seq) {
  DynamicJsonDocument doc(768);
  doc["v"] = PROTOCOL_VERSION;
  doc["seq"] = seq;
  doc["type"] = "hello_ack";
  doc["ok"] = true;
  doc["device_id"] = DEVICE_ID;
  doc["name"] = DEVICE_NAME;
  doc["fw"] = FW_VERSION;
  JsonArray caps = doc.createNestedArray("capabilities");
  const char* c[] = { "status", "power_on", "power_off", "set_level", "reset_safety" };
  for (auto& x : c) caps.add(x);
  JsonObject lim = doc.createNestedObject("limits");
  lim["max_level"] = MAX_LEVEL;
  lim["max_temp_c"] = MAX_SAFE_TEMP_C;
  String body; serializeJson(doc, body);
  sendRaw(body);
}

void sendOverTempEvent() {
  DynamicJsonDocument doc(384);
  doc["v"] = PROTOCOL_VERSION;
  doc["seq"] = 0;
  doc["type"] = "event";
  doc["kind"] = "over_temp";
  JsonObject d = doc.createNestedObject("detail");
  d["limit"] = MAX_SAFE_TEMP_C;
  d["temperature_c"] = g_tempC;
  String body; serializeJson(doc, body);
  sendRaw(body);
}

// ── 命令处理 ──────────────────────────────────────────────────────
void handleCmd(JsonObjectConst f) {
  long seq = f["seq"] | 0;
  const char* id = f["id"] | "";
  JsonObjectConst args = f["args"].as<JsonObjectConst>();
  bool ok = true;
  String reason = "";
  const char* code = "";

  if (strcmp(id, "status") == 0) {
    ok = true;
  } else if (strcmp(id, "power_on") == 0) {
    if (g_tripped) { ok = false; reason = "过热保护已触发，需复位"; code = "E_REJECTED"; }
    else if (g_tempC >= MAX_SAFE_TEMP_C) { ok = false; reason = "温度过高，拒绝开机"; code = "E_OVER_TEMP"; }
    else powerOn();
  } else if (strcmp(id, "power_off") == 0) {
    powerOff();
  } else if (strcmp(id, "set_level") == 0) {
    int lv = args["level"] | 0;
    if (!g_power) { ok = false; reason = "设备未开机"; code = "E_REJECTED"; }
    else if (lv < 1 || lv > MAX_LEVEL) { ok = false; reason = "档位越界"; code = "E_REJECTED"; }
    else if (g_tripped) { ok = false; reason = "过热保护已触发，需复位"; code = "E_REJECTED"; }
    else { g_level = lv; applyLevel(lv); }
  } else if (strcmp(id, "reset_safety") == 0) {
    g_tripped = false; g_tempC = 25.0f; g_level = 0; g_power = false;
    powerOff();
    reason = "安全复位完成";
  } else {
    ok = false; reason = "设备不支持该命令"; code = "E_UNSUPPORTED";
  }

  sendAck(seq, id, ok, reason.c_str(), code);        // 先 ack（接下/拒绝）
  if (ok) sendState(seq, id, true, "");              // 再 state（真实生效后的状态）
}

void setup() {
  Serial.begin(115200);                              // 与 device.baudrate 对齐
  pinMode(PIN_POWER, OUTPUT);
  for (int i = 1; i <= MAX_LEVEL; i++) pinMode(PIN_LEVEL[i], OUTPUT);
  pinMode(PIN_LED, OUTPUT);
  analogReadResolution(12);
  powerOff();
  Serial.printf("{\"note\":\"%s ready, protocol v%d\"}\n", DEVICE_ID, PROTOCOL_VERSION);
}

String rx;

void loop() {
  // 1) 收帧
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      if (rx.length() > 2) {
        DynamicJsonDocument doc(768);
        if (deserializeJson(doc, rx) == DeserializationError::Ok) {
          const char* t = doc["type"] | "";
          if (strcmp(t, "hello") == 0)      sendHelloAck(doc["seq"] | 0);
          else if (strcmp(t, "cmd") == 0)   handleCmd(doc.as<JsonObjectConst>());
          // 注：真机应校验 crc（docs/hardware.md 给了 crc32 实现），此处省略以保持骨架简洁
        }
      }
      rx = "";
    } else if (rx.length() < 512) {
      rx += c;
    }
  }

  // 2) 采样温度 + 设备端最后一道过热保护
  static unsigned long lastSample = 0;
  if (millis() - lastSample > 200) {
    lastSample = millis();
    g_tempC = readTemperatureC();
    if (g_power && g_tempC >= MAX_SAFE_TEMP_C) {
      emergencyOff("over temp");
      if (!g_trippedReported) { g_trippedReported = true; sendOverTempEvent(); }
    }
    if (!g_tripped) g_trippedReported = false;
  }

  // 3) MQTT 承载（可选）：把 Serial 换成 PubSubClient 的 {prefix}/cmd/{id} ↔ {prefix}/up/{id}
  //    协议与帧格式完全不变，只是载体从串口变成 MQTT——见 docs/hardware.md「承载无关」一节。
}
