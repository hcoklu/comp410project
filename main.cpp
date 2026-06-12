#include <iostream>
#include <windows.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#ifdef _DEBUG
#pragma comment(lib, "opencv_world4130d.lib")
#else
#pragma comment(lib, "opencv_world4130.lib")
#endif

// GLEW & GLFW
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// =====================================================================
// YOLOv8-POSE CONSTANTS (COCO 17 Keypoints + 1 Virtual Joint)
// =====================================================================
#define NOSE        0
#define L_EYE       1
#define R_EYE       2
#define L_EAR       3
#define R_EAR       4
#define L_SHOULDER  5
#define R_SHOULDER  6
#define L_ELBOW     7
#define R_ELBOW     8
#define L_WRIST     9
#define R_WRIST     10
#define L_HIP       11
#define R_HIP       12
#define L_KNEE      13
#define R_KNEE      14
#define L_ANKLE     15
#define R_ANKLE     16
#define NECK        17 // Virtual joint (L_SHOULDER + R_SHOULDER midpoint)
#define NUM_PARTS   18

const int POSE_INPUT_W = 640;
const int POSE_INPUT_H = 640;
const float POSE_THRESHOLD = 0.40f;

// =====================================================================
// TIME-DEPENDENT JOINT TARGETS
// =====================================================================
struct JointTarget {
    float minAngle, maxAngle;
    bool active;
};

struct ExerciseJointTargets {
    JointTarget lKnee, rKnee, lElbow, rElbow, lShoulder, rShoulder;
};

const float TIGHT_TOL = 12.0f;
const float MEDIUM_TOL = 20.0f;

static JointTarget makeTarget(float centerDeg, float tol, bool active) {
    return { centerDeg - tol, centerDeg + tol, active };
}

ExerciseJointTargets getTargetsForStateAndTime(int state, float time) {
    ExerciseJointTargets t;
    t.lKnee = t.rKnee = t.lElbow = t.rElbow = t.lShoulder = t.rShoulder = { 0, 180, false };

    switch (state) {
    case 1: {
        float pc = (std::sin(time * 4.0f) + 1.0f) * 0.5f;
        float elbowCenter = 90.0f + pc * 70.0f;
        float shoulderCenter = 50.0f + pc * 40.0f;
        t.lElbow = makeTarget(elbowCenter, TIGHT_TOL, true);
        t.rElbow = makeTarget(elbowCenter, TIGHT_TOL, true);
        t.lShoulder = makeTarget(shoulderCenter, MEDIUM_TOL, true);
        t.rShoulder = makeTarget(shoulderCenter, MEDIUM_TOL, true);
        break;
    }
    case 2: {
        float sa = (std::sin(time * 3.0f) + 1.0f) * 0.5f;
        float kneeCenter = 175.0f - sa * 115.0f;
        t.lKnee = makeTarget(kneeCenter, TIGHT_TOL, true);
        t.rKnee = makeTarget(kneeCenter, TIGHT_TOL, true);
        t.lShoulder = makeTarget(90.0f, MEDIUM_TOL, true);
        t.rShoulder = makeTarget(90.0f, MEDIUM_TOL, true);
        break;
    }
    case 3: {
        t.lElbow = makeTarget(90.0f, MEDIUM_TOL, true);
        t.rElbow = makeTarget(90.0f, MEDIUM_TOL, true);
        t.lShoulder = makeTarget(80.0f, MEDIUM_TOL, true);
        t.rShoulder = makeTarget(80.0f, MEDIUM_TOL, true);
        t.lKnee = makeTarget(175.0f, MEDIUM_TOL, true);
        t.rKnee = makeTarget(175.0f, MEDIUM_TOL, true);
        break;
    }
    case 4: {
        float lp = (std::sin(time * 4.0f) + 1.0f) * 0.5f;
        t.lKnee = makeTarget(175.0f - lp * 85.0f, TIGHT_TOL, true);
        t.rKnee = makeTarget(175.0f - lp * 40.0f, TIGHT_TOL, true);
        break;
    }
    case 5: {
        float fp = (std::sin(time * 3.5f) + 1.0f) * 0.5f;
        t.lElbow = makeTarget(110.0f - fp * 40.0f, MEDIUM_TOL, true);
        t.rElbow = makeTarget(110.0f - fp * 40.0f, MEDIUM_TOL, true);
        t.lShoulder = makeTarget(30.0f + fp * 60.0f, MEDIUM_TOL, true);
        t.rShoulder = makeTarget(30.0f + fp * 60.0f, MEDIUM_TOL, true);
        break;
    }
    case 6: {
        float jj = (std::sin(time * 3.5f) + 1.0f) * 0.5f;
        t.lShoulder = makeTarget(20.0f + jj * 140.0f, MEDIUM_TOL, true);
        t.rShoulder = makeTarget(20.0f + jj * 140.0f, MEDIUM_TOL, true);
        break;
    }
    case 7: {
        float kc = std::max(0.0f, std::sin(time * 2.5f));
        t.rKnee = makeTarget(175.0f - kc * 120.0f, MEDIUM_TOL, true);
        t.lKnee = makeTarget(175.0f, TIGHT_TOL, true);

        break;
    }
    default: break;
    }
    return t;
}

glm::vec4 getFeedbackColor(float angle, JointTarget t) {
    if (!t.active) return glm::vec4(0.55f, 0.55f, 0.6f, 1.0f);
    if (angle < 0)  return glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
    float mid = (t.minAngle + t.maxAngle) / 2.0f;
    float range = (t.maxAngle - t.minAngle) / 2.0f;
    float dist = std::abs(angle - mid);
    if (dist <= range)             return glm::vec4(0.15f, 0.9f, 0.35f, 1.0f);
    else if (dist <= range * 1.4f) return glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
    else                           return glm::vec4(0.95f, 0.15f, 0.15f, 1.0f);
}

// =====================================================================
// SHADERS
// =====================================================================
const char* vertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aNormal;\n"
"uniform mat4 u_mvpMatrix;\n"
"uniform mat4 u_modelMatrix;\n"
"out vec3 vNormal;\n"
"out vec3 vFragPos;\n"
"void main(){\n"
"   gl_Position = u_mvpMatrix * vec4(aPos, 1.0);\n"
"   vFragPos  = vec3(u_modelMatrix * vec4(aPos, 1.0));\n"
"   vNormal   = mat3(transpose(inverse(u_modelMatrix))) * aNormal;\n"
"}\0";

const char* fragmentShaderSource =
"#version 330 core\n"
"in vec3 vNormal;\n"
"in vec3 vFragPos;\n"
"out vec4 FragColor;\n"
"uniform vec4 u_color;\n"
"uniform vec3 u_lightDir;\n"
"uniform float u_ambient;\n"
"void main(){\n"
"   vec3 norm = normalize(vNormal);\n"
"   float diff = max(dot(norm, normalize(-u_lightDir)), 0.0);\n"
"   float light = u_ambient + (1.0 - u_ambient) * diff;\n"
"   FragColor = vec4(u_color.rgb * light, u_color.a);\n"
"}\n\0";

const char* quadVertSrc =
"#version 330 core\n"
"layout(location=0) in vec2 aPos;\n"
"layout(location=1) in vec2 aTexCoord;\n"
"out vec2 TexCoord;\n"
"void main(){ gl_Position = vec4(aPos,0.0,1.0); TexCoord = aTexCoord; }\0";

const char* quadFragSrc =
"#version 330 core\n"
"in vec2 TexCoord;\n"
"out vec4 FragColor;\n"
"uniform sampler2D webcamTex;\n"
"uniform float alpha;\n"
"void main(){ FragColor = vec4(texture(webcamTex,TexCoord).rgb, alpha); }\n\0";

// =====================================================================
// GLOBALS
// =====================================================================
struct Keypoint { float x, y; bool detected; };
Keypoint keypoints[NUM_PARTS];
Keypoint smoothKeypoints[NUM_PARTS];

float flashAlpha = 0.0f;
glm::vec3 flashColor = { 0.15f, 0.90f, 0.35f };
float flashTarget = 0.0f;

const float SMOOTH = 0.5f;

unsigned int shaderProgram, VAO, VBO, EBO;
int mvpLoc, modelLoc, colorLoc, lightDirLoc, ambientLoc;
unsigned int webcamTexture, quadVAO, quadVBO;
unsigned int quadShaderProgram;

float cameraYaw = 30.0f, cameraPitch = 12.0f, cameraRadius = 6.4f;
bool isDragging = false;
double lastMouseX = 0.0, lastMouseY = 0.0;
bool showBones = false;
bool showWebcam = true;
bool poseEnabled = false;
float animSpeed = 1.0f;

float fpsSmooth = 0.0f;
float inferenceMs = 0.0f;
int   detectedJoints = 0;

int keypointMissCount[NUM_PARTS] = { 0 };
const int MISS_THRESHOLD = 4;

std::mutex keypointsMutex;
std::atomic<bool> poseThreadRunning(false);
std::atomic<bool> newFrameReady(false);
cv::Mat poseFrame;
std::mutex frameMutex;

int lastDetectedState = 0;
float lastDetectedTime = 0.0f;
int   candidateState = 0;
float candidateStartTime = 0.0f;
const float STATE_HOLD_SECONDS = 3.0f;
const float STATE_CONFIRM_SECONDS = 0.4f;

std::atomic<bool> cameraThreadRunning(false);
std::atomic<bool> newRawFrameAvailable(false);
cv::Mat rawFrame;
std::mutex rawFrameMutex;

int currentAnimationState = 0;

// =====================================================================
// EGZERSİZ TESPİTİ İÇİN GEÇMİŞ BUFFER — temporal smoothing
// Her egzersiz için son N frame'deki raw state oyları tutulur.
// =====================================================================
static const int VOTE_WINDOW = 8;  // kaç frame'in oyu sayılır
static std::deque<int> stateVoteBuffer;

// =====================================================================
// ANGLE CALC
// =====================================================================
float calcAngle(Keypoint& a, Keypoint& b, Keypoint& c) {
    if (!a.detected || !b.detected || !c.detected) return -1.0f;
    float ax = a.x - b.x, ay = a.y - b.y;
    float cx = c.x - b.x, cy = c.y - b.y;
    float dot = ax * cx + ay * cy;
    float mag = std::sqrt((ax * ax + ay * ay) * (cx * cx + cy * cy));
    if (mag < 0.0001f) return -1.0f;
    return glm::degrees(std::acos(std::max(-1.0f, std::min(1.0f, dot / mag))));
}

// =====================================================================
// POSE NET
// =====================================================================
cv::dnn::Net poseNet;
bool poseNetLoaded = false;

void loadPoseNet(const std::string& onnxPath) {
    try {
        poseNet = cv::dnn::readNetFromONNX(onnxPath);
        if (!poseNet.empty()) {
            poseNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            poseNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            poseNetLoaded = true;
            std::cout << "YOLOv8 Pose model is active!" << std::endl;
        }
    }
    catch (...) {
        std::cerr << "YOLOv8 Pose cannot be activated, ." << std::endl;
        poseNetLoaded = false;
    }
}

void poseThreadFunc() {
    while (poseThreadRunning) {
        if (!newFrameReady) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
        cv::Mat localFrame;
        { std::lock_guard<std::mutex> lock(frameMutex); localFrame = poseFrame.clone(); newFrameReady = false; }
        if (localFrame.empty() || !poseNetLoaded) continue;

        cv::Mat blob = cv::dnn::blobFromImage(localFrame, 1.0 / 255.0,
            cv::Size(POSE_INPUT_W, POSE_INPUT_H), cv::Scalar(0, 0, 0), true, false);
        poseNet.setInput(blob);
        cv::Mat output;
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            output = poseNet.forward();
            auto t1 = std::chrono::high_resolution_clock::now();
            inferenceMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        }
        catch (...) { continue; }

        int rows = output.size[1];
        int cols = output.size[2];
        bool transposed = false;
        if (rows > cols) { std::swap(rows, cols); transposed = true; }

        float* data = (float*)output.data;
        int   best_anchor = -1;
        float max_score = 0.0f;
        for (int i = 0; i < cols; i++) {
            float score = transposed ? data[i * rows + 4] : data[4 * cols + i];
            if (score > max_score) { max_score = score; best_anchor = i; }
        }

        std::lock_guard<std::mutex> lock(keypointsMutex);
        for (int i = 0; i < NUM_PARTS; i++) keypoints[i].detected = false;

        if (max_score > POSE_THRESHOLD && best_anchor != -1) {
            for (int k = 0; k < 17; k++) {
                int ax = 5 + k * 3, ay = 5 + k * 3 + 1, ac = 5 + k * 3 + 2;
                float kx = transposed ? data[best_anchor * rows + ax] : data[ax * cols + best_anchor];
                float ky = transposed ? data[best_anchor * rows + ay] : data[ay * cols + best_anchor];
                float k_conf = transposed ? data[best_anchor * rows + ac] : data[ac * cols + best_anchor];
                if (k_conf > 0.5f) {
                    keypoints[k].x = kx / POSE_INPUT_W;
                    keypoints[k].y = ky / POSE_INPUT_H;
                    keypoints[k].detected = true;
                }
            }
            if (keypoints[L_SHOULDER].detected && keypoints[R_SHOULDER].detected) {
                keypoints[NECK].x = (keypoints[L_SHOULDER].x + keypoints[R_SHOULDER].x) * 0.5f;
                keypoints[NECK].y = (keypoints[L_SHOULDER].y + keypoints[R_SHOULDER].y) * 0.5f;
                keypoints[NECK].detected = true;
            }
        }

        for (int i = 0; i < NUM_PARTS; i++) {
            if (keypoints[i].detected) {
                keypointMissCount[i] = 0;
                if (!smoothKeypoints[i].detected) {
                    smoothKeypoints[i] = keypoints[i];
                }
                else {
                    smoothKeypoints[i].x = smoothKeypoints[i].x * (1 - SMOOTH) + keypoints[i].x * SMOOTH;
                    smoothKeypoints[i].y = smoothKeypoints[i].y * (1 - SMOOTH) + keypoints[i].y * SMOOTH;
                    smoothKeypoints[i].detected = true;
                }
            }
            else {
                keypointMissCount[i]++;
                if (keypointMissCount[i] >= MISS_THRESHOLD)
                    smoothKeypoints[i].detected = false;
            }
        }

        int cnt = 0;
        for (int i = 0; i < NUM_PARTS; i++)
            if (smoothKeypoints[i].detected) cnt++;
        detectedJoints = cnt;
    }
}

void cameraThreadFunc() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { cameraThreadRunning = false; return; }
    while (cameraThreadRunning) {
        cv::Mat t; cap >> t;
        if (!t.empty()) {
            std::lock_guard<std::mutex> lock(rawFrameMutex);
            rawFrame = t.clone(); newRawFrameAvailable = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    cap.release();
}


int detectExerciseState() {
    if (!poseNetLoaded) return 0;

    int raw = 0;
    {
        std::lock_guard<std::mutex> lock(keypointsMutex);

        float lk = calcAngle(keypoints[L_HIP], keypoints[L_KNEE], keypoints[L_ANKLE]);
        float rk = calcAngle(keypoints[R_HIP], keypoints[R_KNEE], keypoints[R_ANKLE]);
        float le = calcAngle(keypoints[L_SHOULDER], keypoints[L_ELBOW], keypoints[L_WRIST]);
        float re = calcAngle(keypoints[R_SHOULDER], keypoints[R_ELBOW], keypoints[R_WRIST]);

        bool torsoHorizontal = false;
        if (keypoints[NECK].detected &&
            keypoints[L_HIP].detected && keypoints[R_HIP].detected)
        {
            float hipMX = (keypoints[L_HIP].x + keypoints[R_HIP].x) * 0.5f;
            float hipMY = (keypoints[L_HIP].y + keypoints[R_HIP].y) * 0.5f;
            float dx = hipMX - keypoints[NECK].x;
            float dy = hipMY - keypoints[NECK].y;
            float angle = std::abs(std::atan2(std::abs(dy), std::abs(dx)));
            torsoHorizontal = (angle < glm::radians(45.0f));
        }

        // ── 2) Lateral torso
        if (torsoHorizontal) {
            float avgElbow = -1.0f;
            if (le > 0 && re > 0) avgElbow = (le + re) / 2.0f;
            else if (le > 0)      avgElbow = le;
            else if (re > 0)      avgElbow = re;

            if (avgElbow > 0 && avgElbow < 110.0f)
                raw = 1;
            else
                raw = 3;
        }
        // ── 3) Jumping Jack: 
        else if (keypoints[L_SHOULDER].detected && keypoints[L_ELBOW].detected &&
            keypoints[R_SHOULDER].detected && keypoints[R_ELBOW].detected)
        {
            bool lArmUp = keypoints[L_ELBOW].y < keypoints[L_SHOULDER].y - 0.03f;
            bool rArmUp = keypoints[R_ELBOW].y < keypoints[R_SHOULDER].y - 0.03f;

            if (lArmUp && rArmUp) {
                raw = 6; // Jumping Jack
            }
            // ── 4) Lunge: 
            else if (lk > 0 && rk > 0 && std::abs(lk - rk) > 30.0f) {
                raw = 4;
            }
            // ── 5) Squat:
            else if (lk > 0 && rk > 0 &&
                lk < 140.0f && rk < 140.0f &&
                std::abs(lk - rk) <= 35.0f) {
                raw = 2;
            }
            else {
                raw = 0;
            }
        }
        else if (lk > 0 || rk > 0) {
            float avgKnee = (lk > 0 && rk > 0) ? (lk + rk) / 2.0f
                : (lk > 0 ? lk : rk);
            bool twoLegs = (lk > 0 && rk > 0);

            if (twoLegs && std::abs(lk - rk) > 30.0f)
                raw = 4; // Lunge
            else if (avgKnee < 130.0f)
                raw = 2; // Squat
            else
                raw = 0;
        }
        else {
            raw = 0;
        }
    }

    stateVoteBuffer.push_back(raw);
    if ((int)stateVoteBuffer.size() > VOTE_WINDOW)
        stateVoteBuffer.pop_front();

    int voteCounts[9] = { 0 };
    for (int v : stateVoteBuffer) if (v >= 0 && v < 9) voteCounts[v]++;
    int votedState = 0;
    int maxVotes = 0;
    for (int i = 0; i < 9; i++) {
        if (voteCounts[i] > maxVotes) { maxVotes = voteCounts[i]; votedState = i; }
    }
    int majorityThreshold = (int)(VOTE_WINDOW * 0.6f);
    if (maxVotes < majorityThreshold) votedState = lastDetectedState;

    float now = (float)glfwGetTime();

    if (votedState != candidateState) {
        candidateState = votedState;
        candidateStartTime = now;
    }

    if (candidateState != lastDetectedState &&
        (now - candidateStartTime) >= STATE_CONFIRM_SECONDS)
    {
        if (candidateState == 0 && (now - lastDetectedTime) < STATE_HOLD_SECONDS)
            return lastDetectedState;
        lastDetectedState = candidateState;
        lastDetectedTime = now;
    }

    if (lastDetectedState != 0 && raw != 0)
        lastDetectedTime = now;

    return lastDetectedState;
}

// =====================================================================
// OPENGL
// =====================================================================
void checkCompileErrors(unsigned int shader, std::string type) {
    int success; char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) { glGetShaderInfoLog(shader, 1024, NULL, infoLog); std::cerr << infoLog << std::endl; }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) { glGetProgramInfoLog(shader, 1024, NULL, infoLog); std::cerr << infoLog << std::endl; }
    }
}

void setupCube() {
    float v[] = {
        -0.5f,-0.5f,-0.5f,0,0,-1,  0.5f,-0.5f,-0.5f,0,0,-1,  0.5f,0.5f,-0.5f,0,0,-1,  -0.5f,0.5f,-0.5f,0,0,-1,
        -0.5f,-0.5f, 0.5f,0,0, 1,  0.5f,-0.5f, 0.5f,0,0, 1,  0.5f,0.5f, 0.5f,0,0, 1,  -0.5f,0.5f, 0.5f,0,0, 1,
        -0.5f,-0.5f,-0.5f,-1,0,0, -0.5f,-0.5f,0.5f,-1,0,0, -0.5f,0.5f,0.5f,-1,0,0, -0.5f,0.5f,-0.5f,-1,0,0,
         0.5f,-0.5f,-0.5f, 1,0,0,  0.5f,-0.5f,0.5f, 1,0,0,  0.5f,0.5f,0.5f, 1,0,0,  0.5f,0.5f,-0.5f, 1,0,0,
        -0.5f,-0.5f,-0.5f,0,-1,0,  0.5f,-0.5f,-0.5f,0,-1,0,  0.5f,-0.5f,0.5f,0,-1,0, -0.5f,-0.5f,0.5f,0,-1,0,
        -0.5f, 0.5f,-0.5f,0, 1,0,  0.5f, 0.5f,-0.5f,0, 1,0,  0.5f, 0.5f,0.5f,0, 1,0, -0.5f, 0.5f,0.5f,0, 1,0,
    };
    unsigned int idx[] = {
        0,1,2,2,3,0,  4,5,6,6,7,4,  8,9,10,10,11,8,
        12,13,14,14,15,12, 16,17,18,18,19,16, 20,21,22,22,23,20
    };
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO); glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);                 glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
}

void setupQuad() {
    float qv[] = { -1,-1,0,1, 1,-1,1,1, 1,1,1,0, -1,1,0,0 };
    unsigned int qi[] = { 0,1,2,2,3,0 };
    unsigned int qEBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO); glGenBuffers(1, &qEBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO); glBufferData(GL_ARRAY_BUFFER, sizeof(qv), qv, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, qEBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(qi), qi, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);                 glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    unsigned int qvs = glCreateShader(GL_VERTEX_SHADER);   glShaderSource(qvs, 1, &quadVertSrc, NULL); glCompileShader(qvs);
    unsigned int qfs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(qfs, 1, &quadFragSrc, NULL); glCompileShader(qfs);
    quadShaderProgram = glCreateProgram(); glAttachShader(quadShaderProgram, qvs); glAttachShader(quadShaderProgram, qfs); glLinkProgram(quadShaderProgram);
    glDeleteShader(qvs); glDeleteShader(qfs);
    glGenTextures(1, &webcamTexture); glBindTexture(GL_TEXTURE_2D, webcamTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void updateWebcamTexture(cv::Mat& frame) {
    if (frame.empty()) return;
    cv::Mat rgb; cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB); cv::flip(rgb, rgb, 1);
    glBindTexture(GL_TEXTURE_2D, webcamTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void drawCubeWithTransform(glm::mat4 vp, glm::mat4 model, glm::vec4 color) {
    glm::mat4 mvp = vp * model;
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform4fv(colorLoc, 1, glm::value_ptr(color));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

// =====================================================================
// INPUT CALLBACKS
// =====================================================================
void mouse_button_callback(GLFWwindow* w, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) { isDragging = false; return; }
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) { isDragging = true; glfwGetCursorPos(w, &lastMouseX, &lastMouseY); }
        else isDragging = false;
    }
}
void cursor_position_callback(GLFWwindow* w, double x, double y) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (isDragging) {
        cameraYaw += (float)((x - lastMouseX) * 0.5f);
        cameraPitch += (float)((lastMouseY - y) * 0.5f);
        lastMouseX = x; lastMouseY = y;
        if (cameraPitch > 89) cameraPitch = 89;
        if (cameraPitch < -89) cameraPitch = -89;
    }
}
void key_callback(GLFWwindow* w, int key, int sc, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_0) currentAnimationState = 0;
        else if (key == GLFW_KEY_1) currentAnimationState = 1;
        else if (key == GLFW_KEY_2) currentAnimationState = 2;
        else if (key == GLFW_KEY_3) currentAnimationState = 3;
        else if (key == GLFW_KEY_4) currentAnimationState = 4;
        else if (key == GLFW_KEY_5) currentAnimationState = 5;
        else if (key == GLFW_KEY_6) currentAnimationState = 6;
        else if (key == GLFW_KEY_7) currentAnimationState = 7;
        else if (key == GLFW_KEY_8) currentAnimationState = 8;
        else if (key == GLFW_KEY_P) showBones = !showBones;
        else if (key == GLFW_KEY_W) showWebcam = !showWebcam;
        else if (key == GLFW_KEY_C) poseEnabled = !poseEnabled;
        else if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, true);
    }
}

// =====================================================================
// UI THEME
// =====================================================================
void SetupImGuiStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 10; s.FrameRounding = 7; s.PopupRounding = 7; s.GrabRounding = 7; s.ScrollbarRounding = 7;
    s.WindowBorderSize = 1; s.FrameBorderSize = 0;
    s.WindowPadding = { 14,14 }; s.FramePadding = { 10,6 }; s.ItemSpacing = { 8,7 };
    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg] = { 0.07f,0.08f,0.10f,0.97f };
    c[ImGuiCol_ChildBg] = { 0.10f,0.11f,0.13f,1.00f };
    c[ImGuiCol_Border] = { 0.18f,0.20f,0.24f,1.00f };
    c[ImGuiCol_TitleBg] = { 0.10f,0.12f,0.15f,1.00f };
    c[ImGuiCol_TitleBgActive] = { 0.12f,0.15f,0.18f,1.00f };
    c[ImGuiCol_Button] = { 0.14f,0.18f,0.23f,1.00f };
    c[ImGuiCol_ButtonHovered] = { 0.15f,0.72f,0.42f,0.85f };
    c[ImGuiCol_ButtonActive] = { 0.12f,0.90f,0.40f,1.00f };
    c[ImGuiCol_FrameBg] = { 0.13f,0.15f,0.18f,1.00f };
    c[ImGuiCol_FrameBgHovered] = { 0.18f,0.20f,0.24f,1.00f };
    c[ImGuiCol_CheckMark] = { 0.15f,1.00f,0.42f,1.00f };
    c[ImGuiCol_SliderGrab] = { 0.15f,0.80f,0.42f,1.00f };
    c[ImGuiCol_SliderGrabActive] = { 0.20f,1.00f,0.50f,1.00f };
    c[ImGuiCol_Text] = { 0.93f,0.93f,0.95f,1.00f };
    c[ImGuiCol_TextDisabled] = { 0.50f,0.52f,0.56f,1.00f };
    c[ImGuiCol_Separator] = { 0.20f,0.22f,0.26f,1.00f };
    c[ImGuiCol_Header] = { 0.15f,0.72f,0.40f,0.35f };
    c[ImGuiCol_HeaderHovered] = { 0.15f,0.85f,0.45f,0.70f };
    c[ImGuiCol_HeaderActive] = { 0.15f,1.00f,0.45f,1.00f };
    c[ImGuiCol_PlotHistogram] = { 0.15f,0.80f,0.42f,1.00f };
}

static void SectionHeader(const char* lbl) {
    ImGui::Spacing();
    ImGui::TextColored({ 0.15f,0.90f,0.45f,1 }, "%s", lbl);
    ImGui::Separator(); ImGui::Spacing();
}

std::string getExeDir() {
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string s(path);
    return s.substr(0, s.find_last_of("\\/") + 1);
}

// =====================================================================
// MAIN
// =====================================================================
int main() {
    loadPoseNet(getExeDir() + "yolov8n-pose.onnx");
    for (int i = 0; i < NUM_PARTS; i++) { keypoints[i] = { 0,0,false }; smoothKeypoints[i] = { 0,0,false }; }

    if (!glfwInit()) return -1;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height,
        "FitTrack 3D - Exercise Coaching System", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    setupCube(); setupQuad();
    glEnable(GL_DEPTH_TEST);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetKeyCallback(window, key_callback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    SetupImGuiStyle();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    io.FontGlobalScale = 1.25f;

    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, NULL); glCompileShader(vs); checkCompileErrors(vs, "VERTEX");
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, NULL); glCompileShader(fs); checkCompileErrors(fs, "FRAGMENT");
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs); glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram); checkCompileErrors(shaderProgram, "PROGRAM");
    glUseProgram(shaderProgram);

    mvpLoc = glGetUniformLocation(shaderProgram, "u_mvpMatrix");
    modelLoc = glGetUniformLocation(shaderProgram, "u_modelMatrix");
    colorLoc = glGetUniformLocation(shaderProgram, "u_color");
    lightDirLoc = glGetUniformLocation(shaderProgram, "u_lightDir");
    ambientLoc = glGetUniformLocation(shaderProgram, "u_ambient");

    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.8f));
    glUniform3fv(lightDirLoc, 1, glm::value_ptr(lightDir));
    glUniform1f(ambientLoc, 0.35f);

    glm::vec4 skinColor(0.96f, 0.82f, 0.70f, 1);
    glm::vec4 shirtColor(0.15f, 0.45f, 0.65f, 1);
    glm::vec4 pantsColor(0.12f, 0.18f, 0.32f, 1);
    glm::vec4 shoeColor(0.90f, 0.90f, 0.92f, 1);
    glm::vec4 hairColor(0.22f, 0.14f, 0.08f, 1);
    glm::vec4 eyeColor(0.05f, 0.05f, 0.08f, 1);
    glm::vec4 shoulderCol(0.20f, 0.85f, 0.50f, 1);
    glm::vec4 elbowCol(0.90f, 0.35f, 0.90f, 1);
    glm::vec4 hipCol(0.25f, 0.55f, 0.95f, 1);
    glm::vec4 kneeCol(0.95f, 0.85f, 0.15f, 1);
    glm::vec4 ankleCol(0.95f, 0.50f, 0.15f, 1);
    glm::vec4 neckCol(0.96f, 0.82f, 0.70f, 1);
    glm::vec4 boneColor(0.88f, 0.92f, 1.00f, 1);

    cameraThreadRunning = true; std::thread cameraThread(cameraThreadFunc);
    poseThreadRunning = true;   std::thread poseThread(poseThreadFunc);

    struct ExInfo { const char* name; ImVec4 color; };
    ExInfo exInfo[] = {
        {"Idle",         {0.60f,0.60f,0.65f,1}},
        {"Push-up",      {0.25f,0.75f,1.00f,1}},
        {"Squat",        {0.25f,1.00f,0.45f,1}},
        {"Plank",        {1.00f,0.80f,0.20f,1}},
        {"Lunge",        {1.00f,0.50f,0.20f,1}},
        {"Chest Fly",    {1.00f,0.30f,0.75f,1}},
        {"Jumping Jack", {0.30f,1.00f,1.00f,1}},
        {"Kick",         {1.00f,0.30f,0.30f,1}},
        {"Live Pose",    {0.80f,0.80f,1.00f,1}},
    };
    const char* exKeys[] = { "0","1","2","3","4","5","6","7","8" };

    // =====================================================================
    // MAIN LOOP
    // =====================================================================
    while (!glfwWindowShouldClose(window)) {
        cv::Mat currentFrame; bool updateTexture = false;
        {
            std::lock_guard<std::mutex> lock(rawFrameMutex);
            if (newRawFrameAvailable) { currentFrame = rawFrame.clone(); newRawFrameAvailable = false; updateTexture = true; }
        }
        if (updateTexture && !currentFrame.empty()) {
            if (poseEnabled) { std::lock_guard<std::mutex> lock(frameMutex); poseFrame = currentFrame.clone(); newFrameReady = true; }
            updateWebcamTexture(currentFrame);
        }

        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        int winW, winH; glfwGetFramebufferSize(window, &winW, &winH);
        glViewport(0, 0, winW, winH);
        glClearColor(0.10f, 0.11f, 0.13f, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        float time = (float)glfwGetTime() * animSpeed;
        int as = currentAnimationState;
        if (poseEnabled && as == 0) as = detectExerciseState();

        // LEFT PANEL
        float leftW = 310;
        ImGui::SetNextWindowPos({ 10,10 }, ImGuiCond_Always);
        ImGui::SetNextWindowSize({ leftW,(float)winH - 20 }, ImGuiCond_Always);
        ImGui::Begin("##ctrl", nullptr,
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

        ImGui::SetWindowFontScale(1.5f);
        ImGui::TextColored({ 0.20f,0.95f,0.50f,1 }, "FitTrack 3D");
        ImGui::SetWindowFontScale(0.9f);
        ImGui::TextColored({ 0.55f,0.57f,0.62f,1 }, "Exercise Coaching System");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        SectionHeader("WORKOUT MODE");
        float bw = (leftW - 28 - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
        for (int i = 0; i < 9; i++) {
            if (i % 2 != 0) ImGui::SameLine();
            char lbl[64]; snprintf(lbl, sizeof(lbl), "%s %s##ex%d", exKeys[i], exInfo[i].name, i);
            bool active = (currentAnimationState == i);
            ImVec4 ac = exInfo[i].color;
            if (active) ImGui::PushStyleColor(ImGuiCol_Button, { ac.x * 0.7f,ac.y * 0.7f,ac.z * 0.7f,1 });
            if (ImGui::Button(lbl, { bw,42 })) currentAnimationState = i;
            if (active) ImGui::PopStyleColor();
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        SectionHeader("SETTINGS");
        ImGui::Checkbox(" Pose Detection (C)", &poseEnabled);
        ImGui::Checkbox(" Webcam Feed   (W)", &showWebcam);
        ImGui::Checkbox(" X-Ray Bones   (P)", &showBones);
        ImGui::Spacing();
        ImGui::Text("Animation Speed");
        ImGui::SetNextItemWidth(leftW - 28);
        ImGui::SliderFloat("##spd", &animSpeed, 0.2f, 3.0f, "%.1fx");
        ImGui::Spacing();
        ImGui::Text("Camera Radius");
        ImGui::SetNextItemWidth(leftW - 28);
        ImGui::SliderFloat("##rad", &cameraRadius, 2.0f, 15.0f, "%.1f");

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        SectionHeader("JOINT ANGLES");
        auto drawBar = [&](const char* name, int a, int b, int c) {
            float angle = -1;
            if (poseEnabled) { std::lock_guard<std::mutex> lock(keypointsMutex); angle = calcAngle(keypoints[a], keypoints[b], keypoints[c]); }
            if (angle >= 0) {
                ImGui::Text("%-14s", name); ImGui::SameLine();
                char buf[16]; snprintf(buf, sizeof(buf), "%.0f", angle);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, { 0.15f,0.80f,0.42f,1 });
                ImGui::ProgressBar(angle / 180.0f, { -1,14 }, buf);
                ImGui::PopStyleColor();
            }
            else { ImGui::TextDisabled("%-14s  --", name); }
            };
        drawBar("L Shoulder", NECK, L_SHOULDER, L_ELBOW);
        drawBar("R Shoulder", NECK, R_SHOULDER, R_ELBOW);
        drawBar("L Elbow", L_SHOULDER, L_ELBOW, L_WRIST);
        drawBar("R Elbow", R_SHOULDER, R_ELBOW, R_WRIST);
        drawBar("L Knee", L_HIP, L_KNEE, L_ANKLE);
        drawBar("R Knee", R_HIP, R_KNEE, R_ANKLE);

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextDisabled("Drag background to orbit");
        ImGui::TextDisabled("Keys 0-8 select exercise");
        ImGui::End();

        // TOP BADGE
        {
            float bw2 = 340, bh = 80;
            ImGui::SetNextWindowPos({ (float)winW / 2 - bw2 / 2,14 }, ImGuiCond_Always);
            ImGui::SetNextWindowSize({ bw2,bh }, ImGuiCond_Always);
            ImGui::Begin("##badge", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);
            ImGui::SetWindowFontScale(1.1f);
            float tw = ImGui::CalcTextSize("CURRENT EXERCISE").x;
            ImGui::SetCursorPosX((bw2 - tw) * 0.5f);
            ImGui::TextColored({ 0.65f,0.67f,0.72f,1 }, "CURRENT EXERCISE");
            ImGui::SetWindowFontScale(1.9f);
            const char* nm = exInfo[as].name;
            tw = ImGui::CalcTextSize(nm).x;
            ImGui::SetCursorPosX((bw2 - tw) * 0.5f);
            ImGui::TextColored(exInfo[as].color, "%s", nm);
            ImGui::SetWindowFontScale(1.0f);
            ImGui::End();
        }

        // RIGHT PANEL — webcam + form
        float rightW = 400, rightX = (float)winW - rightW - 10;
        if (showWebcam) {
            ImGui::SetNextWindowPos({ rightX,10 }, ImGuiCond_Always);
            ImGui::SetNextWindowSize({ rightW,300 }, ImGuiCond_Always);
            ImGui::Begin("##cam", nullptr,
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
            ImGui::TextColored({ 0.20f,0.95f,0.50f,1 }, "  LIVE CAMERA FEED");
            ImGui::Separator(); ImGui::Spacing();
            float iw = rightW - 20, ih = iw * 0.75f;
            ImGui::Image((ImTextureID)(intptr_t)webcamTexture, { iw,ih });
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImVec2 ip = ImGui::GetItemRectMin(), is = ImGui::GetItemRectSize();
            if (poseEnabled) {
                std::lock_guard<std::mutex> lock(keypointsMutex);
                for (int i = 0; i < NUM_PARTS; i++) {
                    if (!keypoints[i].detected) continue;
                    float px = ip.x + keypoints[i].x * is.x, py = ip.y + keypoints[i].y * is.y;
                    dl->AddCircleFilled({ px,py }, 5, IM_COL32(40, 230, 100, 255));
                    dl->AddCircle({ px,py }, 7, IM_COL32(255, 255, 255, 180), 12, 1.5f);
                }
            }
            ImGui::End();
        }

        if (poseEnabled && poseNetLoaded) {
            float formY = showWebcam ? 316 : 10;
            ImGui::SetNextWindowPos({ rightX,formY }, ImGuiCond_Always);
            ImGui::SetNextWindowSize({ rightW,(float)winH - formY - 100 }, ImGuiCond_Always);
            ImGui::Begin("##form", nullptr,
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
            ImGui::TextColored({ 0.95f,0.30f,0.30f,1 }, "  FORM ANALYSIS");
            ImGui::Separator(); ImGui::Spacing();

            ExerciseJointTargets tgt = getTargetsForStateAndTime(as, time);
            glm::vec4 lsc, rsc, lec, rec, lkc, rkc;
            {
                std::lock_guard<std::mutex> lock(keypointsMutex);
                lsc = getFeedbackColor(calcAngle(keypoints[NECK], keypoints[L_SHOULDER], keypoints[L_ELBOW]), tgt.lShoulder);
                rsc = getFeedbackColor(calcAngle(keypoints[NECK], keypoints[R_SHOULDER], keypoints[R_ELBOW]), tgt.rShoulder);
                lec = getFeedbackColor(calcAngle(keypoints[L_SHOULDER], keypoints[L_ELBOW], keypoints[L_WRIST]), tgt.lElbow);
                rec = getFeedbackColor(calcAngle(keypoints[R_SHOULDER], keypoints[R_ELBOW], keypoints[R_WRIST]), tgt.rElbow);
                lkc = getFeedbackColor(calcAngle(keypoints[L_HIP], keypoints[L_KNEE], keypoints[L_ANKLE]), tgt.lKnee);
                rkc = getFeedbackColor(calcAngle(keypoints[R_HIP], keypoints[R_KNEE], keypoints[R_ANKLE]), tgt.rKnee);
            }
            auto isRed = [](glm::vec4 c) {return c.r > 0.8f && c.g < 0.3f; };
            struct W { const char* n; glm::vec4 col; };
            std::vector<W> warns;
            if (isRed(lsc)) warns.push_back({ "! Left Shoulder: Adjust angle",lsc });
            if (isRed(rsc)) warns.push_back({ "! Right Shoulder: Adjust angle",rsc });
            if (isRed(lec)) warns.push_back({ "! Left Elbow: Check your form",lec });
            if (isRed(rec)) warns.push_back({ "! Right Elbow: Check your form",rec });
            if (isRed(lkc)) warns.push_back({ "! Left Knee: Unsafe position",lkc });
            if (isRed(rkc)) warns.push_back({ "! Right Knee: Unsafe position",rkc });
            if (warns.empty()) ImGui::TextColored({ 0.15f,0.95f,0.45f,1 }, "  Perfect Form! Keep going.");
            else for (auto& w : warns) { ImGui::TextColored({ w.col.r,w.col.g,w.col.b,1 }, "%s", w.n); ImGui::Spacing(); }
            ImGui::End();
        }

        // 3D SKELETON RENDERING
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)winW / (float)winH, 0.1f, 100.0f);
        float cx2 = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
        float cy2 = cameraRadius * sin(glm::radians(cameraPitch));
        float cz2 = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
        glm::mat4 view = glm::lookAt(
            glm::vec3(cx2, cy2 + 1.0f, cz2),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 viewProj = proj * view;

        glm::vec4 lShoulderCol = shoulderCol, rShoulderCol = shoulderCol;
        glm::vec4 lElbowCol = elbowCol, rElbowCol = elbowCol;
        glm::vec4 lKneeCol = kneeCol, rKneeCol = kneeCol;
        glm::vec4 lHipCol = hipCol, rHipCol = hipCol;
        glm::vec4 lAnkleCol = ankleCol, rAnkleCol = ankleCol;

        if (poseEnabled && poseNetLoaded) {
            ExerciseJointTargets tgt = getTargetsForStateAndTime(as, time);
            std::lock_guard<std::mutex> lock(keypointsMutex);
            lKneeCol = getFeedbackColor(calcAngle(keypoints[L_HIP], keypoints[L_KNEE], keypoints[L_ANKLE]), tgt.lKnee);
            rKneeCol = getFeedbackColor(calcAngle(keypoints[R_HIP], keypoints[R_KNEE], keypoints[R_ANKLE]), tgt.rKnee);
            lElbowCol = getFeedbackColor(calcAngle(keypoints[L_SHOULDER], keypoints[L_ELBOW], keypoints[L_WRIST]), tgt.lElbow);
            rElbowCol = getFeedbackColor(calcAngle(keypoints[R_SHOULDER], keypoints[R_ELBOW], keypoints[R_WRIST]), tgt.rElbow);
            lShoulderCol = getFeedbackColor(calcAngle(keypoints[NECK], keypoints[L_SHOULDER], keypoints[L_ELBOW]), tgt.lShoulder);
            rShoulderCol = getFeedbackColor(calcAngle(keypoints[NECK], keypoints[R_SHOULDER], keypoints[R_ELBOW]), tgt.rShoulder);
            glm::vec4 grayCol(0.55f, 0.55f, 0.6f, 1.0f);
            lHipCol = rHipCol = lAnkleCol = rAnkleCol = grayCol;
        }

        float torsoY = 1.5f, torsoZ = 0, torsoPitch = 0, headPitch = 0;
        float lSA = 0, rSA = 0, lSRoll = 0, rSRoll = 0, lSYaw = 0, rSYaw = 0;
        float lEA = 0, rEA = 0, lTA = 0, rTA = 0, lTRoll = 0, rTRoll = 0, lCA = 0, rCA = 0;
        float lAA = glm::radians(180.0f), rAA = glm::radians(180.0f);
        float symS = 0, symE = 0, symT = 0, symC = 0, symA = 0;

        if (as == 1) {
            float pc = (std::sin(time * 4) + 1) * 0.5f;
            torsoPitch = glm::radians(-65.0f) - (1.3f - pc) * glm::radians(12.0f);
            float ay = -1.2f, az = 0.5f;
            torsoY = ay + 2.7f * cos(torsoPitch); torsoZ = az + 2.7f * sin(torsoPitch);
            headPitch = -glm::radians(15.0f); symT = 0; symC = 0; symA = glm::radians(90.0f) - torsoPitch;
            float hy = ay, hz = az - 3.1f, sy = torsoY + 0.75f * cos(torsoPitch), sz = torsoZ + 0.75f * sin(torsoPitch);
            float dy = hy - sy, dz = hz - sz, D = sqrt(dy * dy + dz * dz); if (D > 1.199f)D = 1.199f;
            symS = atan2(dz, -dy) - acos(D / 1.2f) - torsoPitch;
            symE = 3.14159f - acos(1.0f - (D * D) / 0.72f);
        }
        else if (as == 2) {
            float sa = (sin(time * 3) + 1) * 0.5f;
            torsoPitch = 0; headPitch = 0; torsoY = 1.5f - sa * 0.7f; torsoZ = 0;
            symS = sa * glm::radians(90.0f); symE = 0;
            symT = sa * glm::radians(90.0f); symC = -sa * glm::radians(120.0f);
            symA = glm::radians(180.0f) + sa * glm::radians(20.0f);
        }
        else if (as == 3) {
            torsoPitch = glm::radians(-80.0f); torsoY = 0.3f; torsoZ = 0.5f; headPitch = -glm::radians(10.0f);
            symS = -torsoPitch; symE = glm::radians(90.0f);
            symT = glm::radians(8.0f); symC = -glm::radians(5.0f); symA = glm::radians(-75.0f);
        }
        else if (as == 4) {
            float lp = (sin(time * 4) + 1) * 0.5f;
            torsoPitch = 0; headPitch = 0; torsoY = 1.5f - lp * 0.65f; torsoZ = 0;
            float hy = torsoY - 1.1f, tl = 0.9f;
            auto calcLeg = [&](float tz, float ty, float& ta, float& ca, float& aa) {
                float dy = ty - hy, dz = tz, D = sqrt(dy * dy + dz * dz); if (D > 1.799f)D = 1.799f;
                float ia = acos((D / 2) / tl); ta = atan2(dz, -dy) + ia; ca = -2 * ia; aa = glm::radians(180.0f) - (ta + ca); };
            calcLeg(0.9f, -1.2f, rTA, rCA, rAA); calcLeg(-1.1f, -1.05f, lTA, lCA, lAA);
            lAA += glm::radians(35.0f);
            lSA = glm::radians(25.0f); lEA = glm::radians(60.0f);
            rSA = -glm::radians(25.0f); rEA = glm::radians(45.0f);
        }
        else if (as == 5) {
            float fp = (sin(time * 3.5f) + 1) * 0.5f;
            torsoPitch = 0; headPitch = 0; torsoY = 1.0f; torsoZ = 0;
            symT = glm::radians(90.0f); symC = -glm::radians(90.0f); symA = glm::radians(180.0f); symS = 0;
            lSRoll = glm::radians(-70.0f); rSRoll = glm::radians(70.0f);
            lSYaw = -fp * glm::radians(90.0f); rSYaw = fp * glm::radians(90.0f);
            symE = glm::radians(70.0f) - fp * glm::radians(60.0f);
        }
        else if (as == 6) {
            float jj = (sin(time * 3.5f) + 1) * 0.5f;
            torsoPitch = 0; headPitch = 0; torsoY = 1.5f + jj * 0.2f; torsoZ = 0;
            lSRoll = jj * glm::radians(-160.0f); rSRoll = jj * glm::radians(160.0f);
            symS = 0; symE = 0;
            lTRoll = jj * glm::radians(-45.0f); rTRoll = jj * glm::radians(45.0f);
            symT = 0; symC = 0; symA = glm::radians(180.0f);
        }
        else if (as == 7) {
            float kc = std::max(0.0f, sin(time * 2.5f));
            torsoPitch = kc * glm::radians(-5.0f); headPitch = 0; torsoY = 1.5f; torsoZ = 0;
            lSA = kc * glm::radians(40.0f); rSA = -kc * glm::radians(30.0f);
            lEA = kc * glm::radians(20.0f); rEA = kc * glm::radians(20.0f);
            lTA = 0; lCA = 0; lAA = glm::radians(180.0f);
            rTA = kc * glm::radians(80.0f); rCA = -kc * glm::radians(70.0f);
            rAA = glm::radians(180.0f) + kc * glm::radians(20.0f);
        }
        else if (as == 8) {
            std::lock_guard<std::mutex> kL(keypointsMutex);
            torsoPitch = 0; headPitch = 0; torsoZ = 0;
            lSRoll = 0; rSRoll = 0; lSYaw = 0; rSYaw = 0; lTRoll = 0; rTRoll = 0;

            auto kpAngleRad = [](int a, int b, int c) -> float {
                if (!smoothKeypoints[a].detected || !smoothKeypoints[b].detected || !smoothKeypoints[c].detected) return 0.0f;
                float ax = smoothKeypoints[a].x - smoothKeypoints[b].x, ay = smoothKeypoints[a].y - smoothKeypoints[b].y;
                float cx = smoothKeypoints[c].x - smoothKeypoints[b].x, cy = smoothKeypoints[c].y - smoothKeypoints[b].y;
                float dot = ax * cx + ay * cy, mag = std::sqrt((ax * ax + ay * ay) * (cx * cx + cy * cy));
                if (mag < 0.0001f) return 0.0f;
                return std::acos(std::max(-1.0f, std::min(1.0f, dot / mag)));
                };

            {
                static float smoothTorsoY = 1.5f;
                float targetY = 1.5f;
                if (smoothKeypoints[L_HIP].detected && smoothKeypoints[R_HIP].detected)
                    targetY = 2.2f - (smoothKeypoints[L_HIP].y + smoothKeypoints[R_HIP].y) * 0.5f * 3.5f;
                else if (smoothKeypoints[L_HIP].detected)
                    targetY = 2.2f - smoothKeypoints[L_HIP].y * 3.5f;
                else if (smoothKeypoints[R_HIP].detected)
                    targetY = 2.2f - smoothKeypoints[R_HIP].y * 3.5f;
                smoothTorsoY = smoothTorsoY * 0.85f + targetY * 0.15f;
                torsoY = std::clamp(smoothTorsoY, -0.5f, 3.5f);
            }

            if (smoothKeypoints[NECK].detected &&
                smoothKeypoints[L_HIP].detected && smoothKeypoints[R_HIP].detected)
            {
                float hipMidX = (smoothKeypoints[L_HIP].x + smoothKeypoints[R_HIP].x) * 0.5f;
                float hipMidY = (smoothKeypoints[L_HIP].y + smoothKeypoints[R_HIP].y) * 0.5f;
                float dx = hipMidX - smoothKeypoints[NECK].x;
                float dy = hipMidY - smoothKeypoints[NECK].y;

                torsoPitch = std::clamp(std::atan2(dx, dy) * 0.5f,
                    glm::radians(-50.0f), glm::radians(50.0f));
            }

            // Left arm
            if (smoothKeypoints[L_SHOULDER].detected && smoothKeypoints[L_ELBOW].detected) {
                float dx = smoothKeypoints[L_ELBOW].x - smoothKeypoints[L_SHOULDER].x;
                float dy = smoothKeypoints[L_ELBOW].y - smoothKeypoints[L_SHOULDER].y;
                // Roll: kol yukarı/aşağı
                lSRoll = std::clamp(std::atan2(-dy, 0.001f) * 0.9f,
                    glm::radians(-170.0f), glm::radians(10.0f));
                // Yaw: kol öne/arkaya (dx ile yaklaşım)
                lSYaw = std::clamp(-dx * glm::pi<float>() * 1.2f,
                    glm::radians(-60.0f), glm::radians(60.0f));
                lSA = std::clamp(kpAngleRad(NECK, L_SHOULDER, L_ELBOW) - glm::radians(90.0f),
                    glm::radians(-30.0f), glm::radians(90.0f));
            }
            // Right arm
            if (smoothKeypoints[R_SHOULDER].detected && smoothKeypoints[R_ELBOW].detected) {
                float dx = smoothKeypoints[R_ELBOW].x - smoothKeypoints[R_SHOULDER].x;
                float dy = smoothKeypoints[R_ELBOW].y - smoothKeypoints[R_SHOULDER].y;
                rSRoll = std::clamp(std::atan2(-dy, 0.001f) * 0.9f,
                    glm::radians(-10.0f), glm::radians(170.0f));
                rSYaw = std::clamp(-dx * glm::pi<float>() * 1.2f,
                    glm::radians(-60.0f), glm::radians(60.0f));
                rSA = std::clamp(-(kpAngleRad(NECK, R_SHOULDER, R_ELBOW) - glm::radians(90.0f)),
                    glm::radians(-90.0f), glm::radians(30.0f));
            }

            // ── Elbows
            lEA = std::clamp(glm::pi<float>() - kpAngleRad(L_SHOULDER, L_ELBOW, L_WRIST),
                0.0f, glm::pi<float>());
            rEA = std::clamp(glm::pi<float>() - kpAngleRad(R_SHOULDER, R_ELBOW, R_WRIST),
                0.0f, glm::pi<float>());

            // ── Legs
            auto calcLegAngles = [&](int hip, int knee, int ankle,
                float& thigh, float& calf, float& ankleA, float& roll)
                {
                    thigh = 0; calf = 0; ankleA = glm::radians(180.0f); roll = 0;
                    if (!smoothKeypoints[hip].detected || !smoothKeypoints[knee].detected) return;

                    float dx = smoothKeypoints[knee].x - smoothKeypoints[hip].x;
                    float dy = smoothKeypoints[knee].y - smoothKeypoints[hip].y;

                    thigh = std::clamp(std::atan2(dy, 0.001f),
                        glm::radians(-10.0f), glm::radians(110.0f));
                    roll = std::clamp(dx * glm::pi<float>() * 2.0f,
                        glm::radians(-60.0f), glm::radians(60.0f));

                    if (!smoothKeypoints[ankle].detected) return;

                    float kneeRad = kpAngleRad(hip, knee, ankle);
                    calf = std::clamp(-(glm::pi<float>() - kneeRad),
                        glm::radians(-150.0f), 0.0f);
                    ankleA = std::clamp(glm::radians(180.0f) - thigh - calf,
                        glm::radians(60.0f), glm::radians(270.0f));
                };

            calcLegAngles(L_HIP, L_KNEE, L_ANKLE, lTA, lCA, lAA, lTRoll);
            calcLegAngles(R_HIP, R_KNEE, R_ANKLE, rTA, rCA, rAA, rTRoll);
        }
        else { torsoPitch = 0; headPitch = 0; torsoY = 1.5f; torsoZ = 0; symS = 0; symE = 0; symT = 0; symC = 0; symA = glm::radians(180.0f); }

        if (as != 4 && as != 7 && as != 8) { lSA = rSA = symS; lEA = rEA = symE; lTA = rTA = symT; lCA = rCA = symC; lAA = rAA = symA; }

        const float GS = 0.75f;
        float jSz = poseEnabled ? 0.26f : 0.16f;
        float eSz = poseEnabled ? 0.23f : 0.14f;
        float hSz = poseEnabled ? 0.22f : 0.14f;
        float aSz = poseEnabled ? 0.20f : 0.12f;
        glm::mat4 root = glm::scale(glm::mat4(1), { GS,GS,GS });

        auto drawLimb = [&](glm::mat4 flesh, glm::mat4 bone, glm::vec4 col) {
            if (showBones) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                drawCubeWithTransform(viewProj, flesh, col);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                drawCubeWithTransform(viewProj, bone, boneColor);
            }
            else {
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                drawCubeWithTransform(viewProj, flesh, col);
            }
            };

        auto T = [](glm::mat4 m, glm::vec3 v) {return m * glm::translate(glm::mat4(1), v); };
        auto R = [](glm::mat4 m, float a, glm::vec3 ax) {return m * glm::rotate(glm::mat4(1), a, ax); };
        auto S = [](glm::mat4 m, glm::vec3 v) {return m * glm::scale(glm::mat4(1), v); };
        glm::vec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);

        // TORSO
        glm::mat4 torso = T(R(root, torsoPitch, X), { 0,torsoY,torsoZ });
        drawLimb(S(torso, { 1.05f,1.8f,0.55f }), S(torso, { 0.8f,1.7f,0.2f }), shirtColor);
        // NECK + HEAD
        glm::mat4 neck = T(torso, { 0,0.9f,0 });
        drawCubeWithTransform(viewProj, S(neck, { 0.18f,0.18f,0.18f }), neckCol);
        glm::mat4 hp = R(neck, headPitch, X);
        drawLimb(S(T(hp, { 0,0.38f,0 }), { 0.72f,0.72f,0.72f }), S(T(hp, { 0,0.38f,0 }), { 0.42f,0.42f,0.42f }), skinColor);
        drawCubeWithTransform(viewProj, S(T(hp, { 0,0.72f,0 }), { 0.74f,0.22f,0.74f }), hairColor);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        drawCubeWithTransform(viewProj, S(T(hp, { -0.14f,0.44f,-0.37f }), { 0.12f,0.10f,0.10f }), eyeColor);
        drawCubeWithTransform(viewProj, S(T(hp, { 0.14f,0.44f,-0.37f }), { 0.12f,0.10f,0.10f }), eyeColor);
        drawCubeWithTransform(viewProj, S(T(hp, { 0,0.26f,-0.37f }), { 0.22f,0.06f,0.06f }), glm::vec4(0.6f, 0.25f, 0.20f, 1));

        float ual = 0.62f, fal = 0.60f, thl = 0.90f, cal = 0.90f;

        // LEFT ARM
        glm::mat4 lSJ = T(torso, { -0.67f,0.72f,0 });
        drawCubeWithTransform(viewProj, S(lSJ, { jSz,jSz,jSz }), lShoulderCol);
        glm::mat4 lUAP = R(R(R(lSJ, lSYaw, Y), lSRoll, Z), lSA, X);
        drawLimb(S(T(lUAP, { 0,-ual / 2,0 }), { 0.30f,ual,0.30f }), S(T(lUAP, { 0,-ual / 2,0 }), { 0.08f,ual,0.08f }), shirtColor);
        glm::mat4 lEJ = T(lUAP, { 0,-ual,0 });
        drawCubeWithTransform(viewProj, S(lEJ, { eSz,eSz,eSz }), lElbowCol);
        glm::mat4 lFAP = R(lEJ, lEA, X);
        drawLimb(S(T(lFAP, { 0,-fal / 2,0 }), { 0.25f,fal,0.25f }), S(T(lFAP, { 0,-fal / 2,0 }), { 0.06f,fal,0.06f }), skinColor);
        drawCubeWithTransform(viewProj, S(T(lFAP, { 0,-fal,0 }), { 0.28f,0.22f,0.20f }), skinColor);

        // RIGHT ARM
        glm::mat4 rSJ = T(torso, { 0.67f,0.72f,0 });
        drawCubeWithTransform(viewProj, S(rSJ, { jSz,jSz,jSz }), rShoulderCol);
        glm::mat4 rUAP = R(R(R(rSJ, rSYaw, Y), rSRoll, Z), rSA, X);
        drawLimb(S(T(rUAP, { 0,-ual / 2,0 }), { 0.30f,ual,0.30f }), S(T(rUAP, { 0,-ual / 2,0 }), { 0.08f,ual,0.08f }), shirtColor);
        glm::mat4 rEJ = T(rUAP, { 0,-ual,0 });
        drawCubeWithTransform(viewProj, S(rEJ, { eSz,eSz,eSz }), rElbowCol);
        glm::mat4 rFAP = R(rEJ, rEA, X);
        drawLimb(S(T(rFAP, { 0,-fal / 2,0 }), { 0.25f,fal,0.25f }), S(T(rFAP, { 0,-fal / 2,0 }), { 0.06f,fal,0.06f }), skinColor);
        drawCubeWithTransform(viewProj, S(T(rFAP, { 0,-fal,0 }), { 0.28f,0.22f,0.20f }), skinColor);

        // LEFT LEG
        glm::mat4 lHJ = T(torso, { -0.27f,-0.92f,0 });
        drawCubeWithTransform(viewProj, S(lHJ, { hSz,hSz,hSz }), lHipCol);
        glm::mat4 lTP = R(R(lHJ, lTRoll, Z), lTA, X);
        drawLimb(S(T(lTP, { 0,-thl / 2,0 }), { 0.42f,thl,0.42f }), S(T(lTP, { 0,-thl / 2,0 }), { 0.1f,thl,0.1f }), pantsColor);
        glm::mat4 lKJ = T(lTP, { 0,-thl,0 });
        drawCubeWithTransform(viewProj, S(lKJ, { jSz,jSz,jSz }), lKneeCol);
        glm::mat4 lCP = R(lKJ, lCA, X);
        drawLimb(S(T(lCP, { 0,-cal / 2,0 }), { 0.38f,cal,0.38f }), S(T(lCP, { 0,-cal / 2,0 }), { 0.08f,cal,0.08f }), pantsColor);
        glm::mat4 lAJ = T(lCP, { 0,-cal,0 });
        drawCubeWithTransform(viewProj, S(lAJ, { aSz,aSz,aSz }), lAnkleCol);
        glm::mat4 lFP = R(lAJ, lAA, X);
        drawLimb(S(T(lFP, { 0,-0.05f,0.18f }), { 0.34f,0.16f,0.44f }), S(T(lFP, { 0,-0.05f,0.18f }), { 0.1f,0.05f,0.2f }), shoeColor);

        // RIGHT LEG
        glm::mat4 rHJ = T(torso, { 0.27f,-0.92f,0 });
        drawCubeWithTransform(viewProj, S(rHJ, { hSz,hSz,hSz }), rHipCol);
        glm::mat4 rTP = R(R(rHJ, rTRoll, Z), rTA, X);
        drawLimb(S(T(rTP, { 0,-thl / 2,0 }), { 0.42f,thl,0.42f }), S(T(rTP, { 0,-thl / 2,0 }), { 0.1f,thl,0.1f }), pantsColor);
        glm::mat4 rKJ = T(rTP, { 0,-thl,0 });
        drawCubeWithTransform(viewProj, S(rKJ, { jSz,jSz,jSz }), rKneeCol);
        glm::mat4 rCP = R(rKJ, rCA, X);
        drawLimb(S(T(rCP, { 0,-cal / 2,0 }), { 0.38f,cal,0.38f }), S(T(rCP, { 0,-cal / 2,0 }), { 0.08f,cal,0.08f }), pantsColor);
        glm::mat4 rAJ = T(rCP, { 0,-cal,0 });
        drawCubeWithTransform(viewProj, S(rAJ, { aSz,aSz,aSz }), rAnkleCol);
        glm::mat4 rFP = R(rAJ, rAA, X);
        drawLimb(S(T(rFP, { 0,-0.05f,0.18f }), { 0.34f,0.16f,0.44f }), S(T(rFP, { 0,-0.05f,0.18f }), { 0.1f,0.05f,0.2f }), shoeColor);

        if (poseEnabled && poseNetLoaded && as != 0) {
            ExerciseJointTargets tgt = getTargetsForStateAndTime(as, time);

            int totalActive = 0, totalRed = 0;
            {
                std::lock_guard<std::mutex> lock(keypointsMutex);

                auto check = [&](float angle, JointTarget t) {
                    if (!t.active) return;
                    totalActive++;
                    glm::vec4 c = getFeedbackColor(angle, t);
                    if (c.r > 0.8f && c.g < 0.3f) totalRed++;
                    };
                check(calcAngle(keypoints[NECK], keypoints[L_SHOULDER], keypoints[L_ELBOW]), tgt.lShoulder);
                check(calcAngle(keypoints[NECK], keypoints[R_SHOULDER], keypoints[R_ELBOW]), tgt.rShoulder);
                check(calcAngle(keypoints[L_SHOULDER], keypoints[L_ELBOW], keypoints[L_WRIST]), tgt.lElbow);
                check(calcAngle(keypoints[R_SHOULDER], keypoints[R_ELBOW], keypoints[R_WRIST]), tgt.rElbow);
                check(calcAngle(keypoints[L_HIP], keypoints[L_KNEE], keypoints[L_ANKLE]), tgt.lKnee);
                check(calcAngle(keypoints[R_HIP], keypoints[R_KNEE], keypoints[R_ANKLE]), tgt.rKnee);
            }

            if (totalActive > 0) {
                float dt = ImGui::GetIO().DeltaTime;

                if (totalRed == 0) {
                    flashColor = { 0.10f, 0.85f, 0.30f };
                    flashTarget = 0.22f;
                }
                else if (totalRed >= totalActive) {
                    flashColor = { 0.85f, 0.10f, 0.10f };
                    flashTarget = 0.38f;
                }
                else {
                    flashColor = { 0.90f, 0.55f, 0.05f };
                    flashTarget = 0.28f;
                }

                float speed = (flashTarget > flashAlpha) ? 2.5f : 1.2f;
                flashAlpha += (flashTarget - flashAlpha) * speed * dt;
                flashAlpha = std::clamp(flashAlpha, 0.0f, 0.25f);
            }
            else {
                flashAlpha *= 0.92f;
            }
        }
        else {
            flashAlpha *= 0.90f;
        }

        // ── Overlay
        if (flashAlpha > 0.005f) {
            ImDrawList* bgDraw = ImGui::GetBackgroundDrawList();

            // Color
            ImU32 edgeCol = IM_COL32(
                (int)(flashColor.r * 255),
                (int)(flashColor.g * 255),
                (int)(flashColor.b * 255),
                (int)(flashAlpha * 255));
            ImU32 centerCol = IM_COL32(0, 0, 0, 0);
            float W = (float)winW, H = (float)winH;
            float thickness = W * 0.12f;

            bgDraw->AddRectFilledMultiColor(
                { 0,0 }, { W, thickness },
                edgeCol, edgeCol, centerCol, centerCol);

            bgDraw->AddRectFilledMultiColor(
                { 0, H - thickness }, { W, H },
                centerCol, centerCol, edgeCol, edgeCol);
            bgDraw->AddRectFilledMultiColor(
                { 0,0 }, { thickness, H },
                edgeCol, centerCol, centerCol, edgeCol);
            bgDraw->AddRectFilledMultiColor(
                { W - thickness,0 }, { W, H },
                centerCol, edgeCol, edgeCol, centerCol);
        }

        {
            float dt = ImGui::GetIO().DeltaTime;
            float fps = dt > 0.0f ? 1.0f / dt : 0.0f;
            fpsSmooth = fpsSmooth * 0.92f + fps * 0.08f;

            ImVec4 fpsCol;
            if (fpsSmooth >= 55.0f) fpsCol = { 0.15f, 0.90f, 0.40f, 1.0f };
            else if (fpsSmooth >= 30.0f) fpsCol = { 1.00f, 0.80f, 0.10f, 1.0f };
            else                         fpsCol = { 0.95f, 0.20f, 0.20f, 1.0f };

            ImVec4 infCol = inferenceMs < 30.0f
                ? ImVec4{ 0.15f, 0.90f, 0.40f, 1.0f }
            : ImVec4{ 1.00f, 0.65f, 0.10f, 1.0f };

            float panelW = 260.0f, panelH = 58.0f;
            float panelX = (float)winW - panelW - 10.0f;
            float panelY = (float)winH - panelH - 10.0f;

            ImGui::SetNextWindowPos({ panelX, panelY }, ImGuiCond_Always);
            ImGui::SetNextWindowSize({ panelW, panelH }, ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.55f); 
            ImGui::Begin("##perf", nullptr,
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoNav);

            //FPS
            ImGui::TextDisabled("FPS ");
            ImGui::SameLine(0, 0);
            ImGui::TextColored(fpsCol, "%-5.0f", fpsSmooth);
            ImGui::SameLine(0, 12);

            //Pose inference time
            ImGui::TextDisabled("Pose ");
            ImGui::SameLine(0, 0);
            if (poseEnabled && poseNetLoaded)
                ImGui::TextColored(infCol, "%-5.1fms", inferenceMs);
            else
                ImGui::TextDisabled("--    ");

            ImGui::TextDisabled("Joints ");
            ImGui::SameLine(0, 0);
            if (poseEnabled) {
                ImVec4 jCol = detectedJoints >= 12
                    ? ImVec4{ 0.15f, 0.90f, 0.40f, 1.0f }
                : ImVec4{ 1.00f, 0.65f, 0.10f, 1.0f };
                ImGui::TextColored(jCol, "%d / %d", detectedJoints, NUM_PARTS);
            }
            else {
                ImGui::TextDisabled("--");
            }

            ImGui::End();
        }
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext();
    cameraThreadRunning = false; if (cameraThread.joinable()) cameraThread.join();
    poseThreadRunning = false;   if (poseThread.joinable())   poseThread.join();
    glDeleteVertexArrays(1, &VAO); glDeleteBuffers(1, &VBO); glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram); glDeleteProgram(quadShaderProgram);
    glDeleteTextures(1, &webcamTexture);
    glfwTerminate();
    return 0;
}