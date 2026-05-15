#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

/* ============================================================
 * 0) PCA9685 기본설정
 * ============================================================ */
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN 150
#define SERVOMAX 600
#define SAFE_MIN_DEG 5
#define SAFE_MAX_DEG 180

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 서보 개수 (0~5 = 6채널)
const uint8_t NUM_SERVO = 6;

/* ============================================================
 * 1) 링크 길이(mm)
 * ============================================================ */
const float L_BASE = 80.0f;
const float L1 = 110.0f;
const float L2 = 105.0f;
const float L_TOOL = 105.0f;

/* ============================================================
 * 1-1) 높이/거리 오차 보정용 바이어스 (mm)
 * ============================================================ */
const float Y_BIAS_MM = +200.0f;  // 높이 보정
const float X_BIAS_MM = +60.0f;   // z(앞/뒤) 보정

/* ============================================================
 * 1-2) 그리퍼 각도 설정
 * ============================================================ */
const float GRIPPER_OPEN_DEG  = 10.0f;   // 기본 오픈
const float GRIPPER_CLOSE_DEG = 100.0f; // 자동 채집 시 닫기 각도

// 그리퍼 논리각 최대 (move/Set 모두 이 값 기준으로 클램프)
const float GRIPPER_MAX_LOGICAL_DEG = 120.0f;

/* ============================================================
 * 2) 채널 매핑
 * ============================================================ */
const uint8_t CH_BASE    = 0;  // 베이스 회전
const uint8_t CH_SHOULDER= 1;  // 어깨 (2번축)
const uint8_t CH_ELBOW   = 2;  // 팔꿈치
const uint8_t CH_WRIST   = 3;  // 손목 피치 (4번축)
const uint8_t CH_LOL     = 4;  // 손목 롤 (회전축)
const uint8_t CH_GRIPPER = 5;  // 그리퍼

/* ============================================================
 * 3) 초기 Servo Offset (초기 자세 75/80/65/75/0/0 기준)
 * ============================================================ */
float SERVO_OFFSET_DEG[NUM_SERVO] = {
  -15.0f,  // CH_BASE
  +10.0f,  // CH_SHOULDER
  -25.0f,  // CH_ELBOW
  -15.0f,  // CH_WRIST
   0.0f,   // CH_LOL
   0.0f    // CH_GRIPPER
};

/* ============================================================
 * 4) 현재 논리각 + 부드러운 이동 설정
 * ============================================================ */
float curDeg[NUM_SERVO] = { 90, 90, 90, 90, 0, 0 };

//          BASE, SHOULDER, ELBOW, WRIST, LOL, GRIPPER
int STEP_DEG_CH[NUM_SERVO]      = { 2,   1,   1,   2,   5,   2 };
int STEP_DELAY_MS_CH[NUM_SERVO] = { 35,  50,  45,  35,  15,  30 };

/* ============================================================
 * 4-1) 거리 구간별 어깨/손목 보정값 (수정됨!)
 * ============================================================ */
void applyDistanceBasedAdjust(float z_mm, float& shDeg, float& wrDeg) {
    
    // z_mm 값은 (입력 cm * 10 + 60) 입니다.
    // 17cm -> 230mm
    // 20cm -> 260mm

    // ============================================================
    // [1] 17cm ~ 19cm 구간 (230mm ~ 259mm)
    // "전 코드처럼 잘 잡게" -> 과감하게 내림 (-12도)
    // 기존 -5도로는 부족해서 여전히 높았으므로 더 내립니다.
    // ============================================================
    if (z_mm >= 230.0f && z_mm < 260.0f) {
        shDeg += -12.0f; // 어깨를 확 내려서 높이 낮춤
        wrDeg += +8.0f;  // 어깨가 내려간 만큼 손목을 들어 수평 유지
    }
    
    // 더 가까운 거리 (16cm 이하 등) 안전장치
    else if (z_mm < 230.0f) {
        if (z_mm >= 170.0f) { shDeg += -5.0f; wrDeg += 0.0f; }
    }

    // ============================================================
    // [2] 20cm 이상 구간 (260mm ~ )
    // "4cm씩 위로 올려줌" -> +12도 적용 (1도당 약 0.3~0.4cm 상승)
    // ============================================================
    
    // 20cm ~ 21cm (260 ~ 280mm)
    else if (z_mm >= 260.0f && z_mm < 280.0f) {
        shDeg += 12.0f;  // 약 4~5cm 상승 효과
        wrDeg += -12.0f; // 손목 수평 유지
    }
    
    // 22cm 이상 (280mm ~ )
    else if (z_mm >= 280.0f) {
        // 거리가 아주 멀면 처짐 보상을 위해 조금 더 들어줌
        shDeg += 15.0f;  
        wrDeg += -15.0f;
    }

    // 보정 후 안전 범위 클램프
    if (shDeg < 0.0f)  shDeg = 0.0f;
    if (shDeg > 90.0f) shDeg = 90.0f;

    if (wrDeg < 0.0f)  wrDeg = 0.0f;
    if (wrDeg > 90.0f) wrDeg = 90.0f;
}

/* ============================================================
 * 5) deg → pulse 변환
 * ============================================================ */
uint16_t degToPulse(float deg) {
    if (deg < 0)   deg = 0;
    if (deg > 180) deg = 180;
    return (uint16_t)(SERVOMIN + (SERVOMAX - SERVOMIN) * (deg / 180.0f));
}

/* ============================================================
 * 6) 서보 제어
 * ============================================================ */
void setServoLogicalDeg(uint8_t ch, float logicalDeg) {
    if (ch >= NUM_SERVO) return;

    float limited = logicalDeg;

    if (ch == CH_GRIPPER) {
        if (limited < 0.0f)                     limited = 0.0f;
        if (limited > GRIPPER_MAX_LOGICAL_DEG) limited = GRIPPER_MAX_LOGICAL_DEG;
    }
    else {
        if (limited < SAFE_MIN_DEG) limited = SAFE_MIN_DEG;
        if (limited > SAFE_MAX_DEG) limited = SAFE_MAX_DEG;
    }

    float realDeg = limited + SERVO_OFFSET_DEG[ch];
    uint16_t pulse = degToPulse(realDeg);
    pwm.setPWM(ch, 0, pulse);

    curDeg[ch] = limited;
}

void moveServoLogicalDegSmooth(uint8_t ch, float target) {
    if (ch >= NUM_SERVO) return;

    float limited = target;

    if (ch == CH_GRIPPER) {
        if (limited < 0.0f)                     limited = 0.0f;
        if (limited > GRIPPER_MAX_LOGICAL_DEG) limited = GRIPPER_MAX_LOGICAL_DEG;
    }
    else {
        if (limited < SAFE_MIN_DEG) limited = SAFE_MIN_DEG;
        if (limited > SAFE_MAX_DEG) limited = SAFE_MAX_DEG;
    }

    float from = curDeg[ch];
    float to   = limited;
    int step   = STEP_DEG_CH[ch];
    if (step < 1) step = 1;
    int dir = (to >= from) ? 1 : -1;

    while (true) {
        setServoLogicalDeg(ch, from);
        if (from == to) break;
        from += dir * step;
        if ((dir > 0 && from > to) || (dir < 0 && from < to)) from = to;
        delay(STEP_DELAY_MS_CH[ch]);
    }
}

/* ============================================================
 * 7) 역기구학(IK)
 * ============================================================ */
bool solveIK(float x, float y, float z,
    float& baseDeg, float& shDeg,
    float& elDeg, float& wrDeg) {
    
    // 1) 1번축
    float theta_base_rad = atan2(x, z);
    float theta_base_deg = theta_base_rad * 180.0f / M_PI;
    float servo1 = 90.0f - theta_base_deg;

    // 2) 수평 거리 r, 어깨 기준 높이 y_rel
    float r = sqrt(x * x + z * z);
    float y_rel = y - L_BASE;

    // 3) 어깨에서 목표점까지 거리 R
    float R = sqrt(r * r + y_rel * y_rel);

    // 4) 로봇팔 최대 길이
    float L23 = L2 + L_TOOL;
    float maxReach = L1 + L23;
    float minReach = fabs(L1 - L23);

    // [자동 거리 조절] 최대 사거리 Clamping
    if (R > maxReach) {
        Serial.println("[IK WARN] Target too far! Clamping to max reach.");
        float scale = (maxReach - 0.5f) / R; 
        r *= scale;
        y_rel *= scale;
        R = sqrt(r * r + y_rel * y_rel);
    }
    else if (R < minReach) {
        Serial.println("[IK WARN] Target too close!");
        return false;
    }

    // 5) 엘보 관절
    float cos_elbow = (L1 * L1 + L23 * L23 - R * R) / (2.0f * L1 * L23);
    if (cos_elbow > 1.0f)  cos_elbow = 1.0f;
    if (cos_elbow < -1.0f) cos_elbow = -1.0f;

    float beta3_rad = acos(cos_elbow);
    float beta3_deg = beta3_rad * 180.0f / M_PI;

    // 6) 숄더 각도
    float gamma_rad = atan2(y_rel, r);
    float cos_delta = (L1 * L1 + R * R - L23 * L23) / (2.0f * L1 * R);
    if (cos_delta > 1.0f)  cos_delta = 1.0f;
    if (cos_delta < -1.0f) cos_delta = -1.0f;
    float delta_rad = acos(cos_delta);

    float theta_shoulder_rad = gamma_rad + delta_rad;
    float theta_shoulder_deg = theta_shoulder_rad * 180.0f / M_PI;

    // 7) 서보각
    float servo2 = theta_shoulder_deg; 
    if (servo2 < 0.0f) servo2 = 0.0f;
    if (servo2 > 90.0f) servo2 = 90.0f;

    float servo3 = 180.0f - beta3_deg;
    if (servo3 < 90.0f)  servo3 = 90.0f;
    if (servo3 > 180.0f) servo3 = 180.0f;

    // 8) 4번축
    float q1_deg = theta_shoulder_deg; 
    float q2_deg = 180.0f - beta3_deg; 
    float toolTarget_deg = 90.0f;
    float wristIK_deg = toolTarget_deg - (q1_deg + q2_deg);

    float servo4 = wristIK_deg;
    if (servo4 < 0.0f)  servo4 = 0.0f;
    if (servo4 > 90.0f) servo4 = 90.0f;

    // 9) 1번축 클램프
    if (servo1 < 0.0f)   servo1 = 0.0f;
    if (servo1 > 180.0f) servo1 = 180.0f;

    baseDeg = servo1;
    shDeg   = servo2;
    elDeg   = servo3;
    wrDeg   = servo4;

    return true;
}

/* ============================================================
 * 8-1) 좌표 이동
 * ============================================================ */
bool gotoXYZ_mm(float x, float y, float z) {
    float baseDeg, shDeg, elDeg, wrDeg;

    if (!solveIK(x, y, z, baseDeg, shDeg, elDeg, wrDeg)) {
        Serial.println("[IK] Unreachable (Critical)");
        return false;
    }

    // [거리 보정 적용] 17~19cm는 내리고, 20cm+는 올림
    applyDistanceBasedAdjust(z, shDeg, wrDeg);

    Serial.print("[IK] base="); Serial.print(baseDeg);
    Serial.print(" sh=");       Serial.print(shDeg);
    Serial.print(" el=");       Serial.print(elDeg);
    Serial.print(" wr=");       Serial.println(wrDeg);

    moveServoLogicalDegSmooth(CH_BASE,    baseDeg);
    moveServoLogicalDegSmooth(CH_SHOULDER,shDeg);
    moveServoLogicalDegSmooth(CH_ELBOW,   elDeg);
    moveServoLogicalDegSmooth(CH_WRIST,   wrDeg);

    return true;
}

/* ============================================================
 * 8-2) 시퀀스 함수
 * ============================================================ */

void goHomePose();

// SEQ1: 좌표로 가서 채집
void sequence_pickFromXYZ(float x, float y, float z) {
    Serial.println("[SEQ1] Home -> Pick at target XYZ");

    Serial.println("[SEQ1] Wait 1 second before arm motion");
    delay(1000);

    moveServoLogicalDegSmooth(CH_GRIPPER, GRIPPER_OPEN_DEG);

    // [보정 1] 더 깊숙이 잡기
    float capture_depth_offset = 25.0f; 
    float target_z = z + capture_depth_offset;

    // [보정 2] 경유점 이동
    float approach_offset = 50.0f;
    if (y > 250.0f) approach_offset = 0.0f; 

    Serial.println("[SEQ1] Approach Position...");
    
    if (gotoXYZ_mm(x, y + approach_offset, target_z)) {
        delay(200); 
    }

    // 실제 목표 좌표로 이동 (보정 적용됨)
    if (!gotoXYZ_mm(x, y, target_z)) {
        Serial.println("[SEQ1] IK failed, abort pick.");
        return;
    }

    delay(300);

    Serial.println("[SEQ1] Close gripper (pick)");
    moveServoLogicalDegSmooth(CH_GRIPPER, GRIPPER_CLOSE_DEG);

    Serial.println("[SEQ1] Wait 1s then rotate wrist roll 180");
    delay(1000);
    moveServoLogicalDegSmooth(CH_LOL, 180.0f);

    delay(1000);
    moveServoLogicalDegSmooth(CH_BASE, 180.0f);
    delay(500);
    moveServoLogicalDegSmooth(CH_SHOULDER, 60.0f);
    delay(500);
    moveServoLogicalDegSmooth(CH_WRIST, 90.0f);
    delay(1000);
    moveServoLogicalDegSmooth(CH_GRIPPER, GRIPPER_OPEN_DEG);

    delay(2000);
    goHomePose();

    moveServoLogicalDegSmooth(CH_GRIPPER, GRIPPER_OPEN_DEG);

    delay(300);

    Serial1.println("s");
    Serial.println("[SEQ1] Sent 's' to Raspberry Pi");
}

/* ============================================================
 * 9) 파싱 함수
 * ============================================================ */
void parseXYZLine(const String& s, float& xm, float& ym, float& zm) {
    int ix = s.indexOf('x');
    int iy = s.indexOf('y');
    int iz = s.indexOf('z');
    if (ix < 0 || iy < 0 || iz < 0) return;

    String sx = s.substring(ix + 2, iy); sx.trim();
    String sy = s.substring(iy + 2, iz); sy.trim();
    String sz = s.substring(iz + 2);     sz.trim();

    float x_cm = sx.toFloat();
    float y_cm = sy.toFloat();
    float z_cm = sz.toFloat();

    xm = x_cm * 10.0f;
    ym = y_cm * 10.0f;
    zm = z_cm * 10.0f;

    xm += 1.0f;
    ym -= 5.0f;
    ym += Y_BIAS_MM;
    zm += X_BIAS_MM;
}

/* ============================================================
 * 10) 명령 처리
 * ============================================================ */
void handleCommandLine(const String& line) {
    String cmd = line;
    cmd.trim();
    if (cmd.length() == 0) return;

    if (cmd.indexOf('x') >= 0 &&
        cmd.indexOf('y') >= 0 &&
        cmd.indexOf('z') >= 0) {

        float x, y, z;
        parseXYZLine(cmd, x, y, z);

        Serial.print("[XYZmm] ");
        Serial.print(x); Serial.print(", ");
        Serial.print(y); Serial.print(", ");
        Serial.println(z);

        sequence_pickFromXYZ(x, y, z);
        return;
    }

    Serial.print("[WARN] Unknown: ");
    Serial.println(cmd);
}

/* ============================================================
 * 11) 초기 자세
 * ============================================================ */
void goHomePose() {
    Serial.println("[HOME] Go home pose");
    moveServoLogicalDegSmooth(CH_BASE, 90);
    moveServoLogicalDegSmooth(CH_SHOULDER, 90);
    moveServoLogicalDegSmooth(CH_ELBOW, 90);
    moveServoLogicalDegSmooth(CH_WRIST, 90);
    moveServoLogicalDegSmooth(CH_LOL, 0);
    moveServoLogicalDegSmooth(CH_GRIPPER, GRIPPER_OPEN_DEG);
}

/* ============================================================
 * 12) SETUP
 * ============================================================ */
String rxLinePc;
String rxLinePi;

void setup() {
    Serial.begin(38400);
    Serial1.begin(38400);

    Wire.begin();
    pwm.begin();
    pwm.setPWMFreq(50);

    delay(500);
    goHomePose();

    Serial.println("[INIT] Robot Arm Ready");
}

/* ============================================================
 * 13) LOOP
 * ============================================================ */
void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (rxLinePc.length() > 0) {
                Serial.print("[RX PC] "); Serial.println(rxLinePc);
                handleCommandLine(rxLinePc);
                rxLinePc = "";
            }
        }
        else {
            rxLinePc += c;
            if (rxLinePc.length() > 80) rxLinePc = "";
        }
    }

    while (Serial1.available()) {
        char c = Serial1.read();
        if (c == '\n' || c == '\r') {
            if (rxLinePi.length() > 0) {
                Serial.print("[RX Pi] "); Serial.println(rxLinePi);
                handleCommandLine(rxLinePi);
                rxLinePi = "";
            }
        }
        else {
            rxLinePi += c;
            if (rxLinePi.length() > 80) rxLinePi = "";
        }
    }
}