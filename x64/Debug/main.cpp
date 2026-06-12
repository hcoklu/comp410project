#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// GLEW & GLFW
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// =====================================================================
// POSE ESTIMATION CONSTANTS
// =====================================================================
// COCO model body part indices
#define NOSE        0
#define NECK        1
#define R_SHOULDER  2
#define R_ELBOW     3
#define R_WRIST     4
#define L_SHOULDER  5
#define L_ELBOW     6
#define L_WRIST     7
#define R_HIP       8
#define R_KNEE      9
#define R_ANKLE     10
#define L_HIP       11
#define L_KNEE      12
#define L_ANKLE     13
#define NUM_PARTS   14

const int POSE_INPUT_W = 368;
const int POSE_INPUT_H = 368;
const float POSE_THRESHOLD = 0.1f;

// =====================================================================
// SHADERS
// =====================================================================
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 u_mvpMatrix;\n"
"void main(){ gl_Position = u_mvpMatrix * vec4(aPos, 1.0); }\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"uniform vec4 u_color;\n"
"void main(){ FragColor = u_color; }\n\0";

const char* quadVertSrc = "#version 330 core\n"
"layout(location=0) in vec2 aPos;\n"
"layout(location=1) in vec2 aTexCoord;\n"
"out vec2 TexCoord;\n"
"void main(){ gl_Position = vec4(aPos, 0.0, 1.0); TexCoord = aTexCoord; }\0";

const char* quadFragSrc = "#version 330 core\n"
"in vec2 TexCoord;\n"
"out vec4 FragColor;\n"
"uniform sampler2D webcamTex;\n"
"uniform float alpha;\n"
"void main(){ FragColor = vec4(texture(webcamTex, TexCoord).rgb, alpha); }\n\0";

// =====================================================================
// GLOBALS
// =====================================================================
unsigned int shaderProgram, VAO, VBO, EBO;
int mvpLoc, colorLoc;
unsigned int webcamTexture, quadVAO, quadVBO;
unsigned int quadShaderProgram;

float cameraYaw = 45.0f, cameraPitch = 15.0f, cameraRadius = 6.4f;
bool isDragging = false;
double lastMouseX = 0.0, lastMouseY = 0.0;
bool showBones = false;
bool showWebcam = true;
bool poseEnabled = false; // C tuşu: kameradan pose al

// Animation state: 0=Idle,1=Pushup,2=Squat,3=Plank,4=Lunge,5=Fly,6=JJ,7=Kick
int currentAnimationState = 0;

// Detected keypoints from pose estimation (normalized 0..1)
// -1 means not detected
struct Keypoint { float x, y; bool detected; };
Keypoint keypoints[NUM_PARTS];

// =====================================================================
// ANGLE CALCULATION
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
// POSE DETECTION
// =====================================================================
cv::dnn::Net poseNet;
bool poseNetLoaded = false;

void loadPoseNet(const std::string& xmlPath, const std::string& binPath) {
    try {
        poseNet = cv::dnn::readNet(xmlPath, binPath);
        if (!poseNet.empty()) {
            poseNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            poseNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            poseNetLoaded = true;
            std::cout << "Pose model yuklendi!" << std::endl;
        }
    }
    catch (...) {
        std::cerr << "Pose model yuklenemedi, animasyon modunda devam ediliyor." << std::endl;
        poseNetLoaded = false;
    }
}

void detectPose(cv::Mat& frame) {
    if (!poseNetLoaded || frame.empty()) return;

    int h = frame.rows, w = frame.cols;
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
        cv::Size(POSE_INPUT_W, POSE_INPUT_H),
        cv::Scalar(0, 0, 0), false, false);

    poseNet.setInput(blob);
    cv::Mat output;
    try {
        output = poseNet.forward();
    }
    catch (...) {
        return;
    }

    int outH = output.size[2];
    int outW = output.size[3];

    for (int i = 0; i < NUM_PARTS; i++) {
        cv::Mat heatMap(outH, outW, CV_32F, output.ptr(0, i));
        cv::Point maxLoc;
        double maxVal;
        cv::minMaxLoc(heatMap, nullptr, &maxVal, nullptr, &maxLoc);

        if (maxVal > POSE_THRESHOLD) {
            keypoints[i].x = (float)maxLoc.x / outW;
            keypoints[i].y = (float)maxLoc.y / outH;
            keypoints[i].detected = true;
        }
        else {
            keypoints[i].detected = false;
        }
    }
}

// Check if person is doing a squat based on knee angle
std::string detectExercise() {
    if (!poseNetLoaded) return "";

    float leftKneeAngle = calcAngle(keypoints[L_HIP], keypoints[L_KNEE], keypoints[L_ANKLE]);
    float rightKneeAngle = calcAngle(keypoints[R_HIP], keypoints[R_KNEE], keypoints[R_ANKLE]);
    float leftElbowAngle = calcAngle(keypoints[L_SHOULDER], keypoints[L_ELBOW], keypoints[L_WRIST]);

    float avgKnee = -1.0f;
    if (leftKneeAngle > 0 && rightKneeAngle > 0)
        avgKnee = (leftKneeAngle + rightKneeAngle) / 2.0f;
    else if (leftKneeAngle > 0) avgKnee = leftKneeAngle;
    else if (rightKneeAngle > 0) avgKnee = rightKneeAngle;

    if (avgKnee > 0 && avgKnee < 120.0f) return "SQUAT";
    if (leftElbowAngle > 0 && leftElbowAngle < 100.0f) return "PUSHUP";
    return "IDLE";
}

// =====================================================================
// FEEDBACK COLORS based on pose
// =====================================================================
glm::vec4 getJointFeedbackColor(int partA, int joint, int partB, float minAngle, float maxAngle) {
    float angle = calcAngle(keypoints[partA], keypoints[joint], keypoints[partB]);
    if (angle < 0) return glm::vec4(1.0f, 1.0f, 0.0f, 1.0f); // yellow = not detected
    if (angle >= minAngle && angle <= maxAngle)
        return glm::vec4(0.0f, 1.0f, 0.0f, 1.0f); // green = correct
    return glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);     // red = incorrect
}

// =====================================================================
// OPENGL SETUP
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
    float vertices[] = {
        -0.5f,-0.5f,-0.5f, 0.5f,-0.5f,-0.5f, 0.5f,0.5f,-0.5f, -0.5f,0.5f,-0.5f,
        -0.5f,-0.5f, 0.5f, 0.5f,-0.5f, 0.5f, 0.5f,0.5f, 0.5f, -0.5f,0.5f, 0.5f
    };
    unsigned int indices[] = {
        0,1,2,2,3,0, 1,5,6,6,2,1, 7,6,5,5,4,7,
        4,0,3,3,7,4, 4,5,1,1,0,4, 3,2,6,6,7,3
    };
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO); glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

void setupQuad() {
    float quadVerts[] = {
        -1.0f,-1.0f, 0.0f,1.0f,
         1.0f,-1.0f, 1.0f,1.0f,
         1.0f, 1.0f, 1.0f,0.0f,
        -1.0f, 1.0f, 0.0f,0.0f,
    };
    unsigned int quadIdx[] = { 0,1,2,2,3,0 };
    unsigned int qEBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO); glGenBuffers(1, &qEBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, qEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    unsigned int qv = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(qv, 1, &quadVertSrc, NULL); glCompileShader(qv);
    unsigned int qf = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(qf, 1, &quadFragSrc, NULL); glCompileShader(qf);
    quadShaderProgram = glCreateProgram();
    glAttachShader(quadShaderProgram, qv); glAttachShader(quadShaderProgram, qf);
    glLinkProgram(quadShaderProgram);
    glDeleteShader(qv); glDeleteShader(qf);

    glGenTextures(1, &webcamTexture);
    glBindTexture(GL_TEXTURE_2D, webcamTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void updateWebcamTexture(cv::Mat& frame) {
    if (frame.empty()) return;
    cv::Mat rgb; cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::flip(rgb, rgb, 0);
    glBindTexture(GL_TEXTURE_2D, webcamTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void drawWebcamBackground(float alpha) {
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(quadShaderProgram);
    glUniform1i(glGetUniformLocation(quadShaderProgram, "webcamTex"), 0);
    glUniform1f(glGetUniformLocation(quadShaderProgram, "alpha"), alpha);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, webcamTexture);
    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(shaderProgram);
}

void drawCubeWithTransform(glm::mat4 vp, glm::mat4 model, glm::vec4 color) {
    glm::mat4 mvp = vp * model;
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform4fv(colorLoc, 1, glm::value_ptr(color));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

void mouse_button_callback(GLFWwindow* w, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) { isDragging = true; glfwGetCursorPos(w, &lastMouseX, &lastMouseY); }
        else isDragging = false;
    }
}

void cursor_position_callback(GLFWwindow* w, double x, double y) {
    if (isDragging) {
        cameraYaw += (float)((x - lastMouseX) * 0.5f);
        cameraPitch += (float)((lastMouseY - y) * 0.5f);
        lastMouseX = x; lastMouseY = y;
        if (cameraPitch > 89.0f) cameraPitch = 89.0f;
        if (cameraPitch < -89.0f) cameraPitch = -89.0f;
    }
}

void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_0) currentAnimationState = 0;
        else if (key == GLFW_KEY_1) currentAnimationState = 1;
        else if (key == GLFW_KEY_2) currentAnimationState = 2;
        else if (key == GLFW_KEY_3) currentAnimationState = 3;
        else if (key == GLFW_KEY_4) currentAnimationState = 4;
        else if (key == GLFW_KEY_5) currentAnimationState = 5;
        else if (key == GLFW_KEY_6) currentAnimationState = 6;
        else if (key == GLFW_KEY_7) currentAnimationState = 7;
        else if (key == GLFW_KEY_P) showBones = !showBones;
        else if (key == GLFW_KEY_W) showWebcam = !showWebcam;
        else if (key == GLFW_KEY_C) {
            poseEnabled = !poseEnabled;
            std::cout << (poseEnabled ? "Pose detection ACIK" : "Pose detection KAPALI") << std::endl;
        }
    }
}

// =====================================================================
// MAIN
// =====================================================================
int main() {
    // Init webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Webcam acilamadi!" << std::endl; return -1; }
    std::cout << "Webcam acildi!" << std::endl;
    std::cout << "Tuslar: 0-7 animasyon | W: webcam | P: x-ray | C: pose detection" << std::endl;

    // Load pose model
    std::string basePath = "C:/Users/hacer/OneDrive - Koc Universitesi/My Computer/Masaustu/COMP410/";
    loadPoseNet(basePath + "pose.xml", basePath + "pose.bin");

    // Init keypoints
    for (int i = 0;i < NUM_PARTS;i++) keypoints[i] = { 0,0,false };

    // GLFW
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 600,
        "Exercise App | 0-7:Animasyon | C:PoseDetect | W:Webcam | P:XRay", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    setupCube(); setupQuad();
    glEnable(GL_DEPTH_TEST);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetKeyCallback(window, key_callback);

    // Compile shaders
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, NULL); glCompileShader(vs); checkCompileErrors(vs, "VERTEX");
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, NULL); glCompileShader(fs); checkCompileErrors(fs, "FRAGMENT");
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs); glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram); checkCompileErrors(shaderProgram, "PROGRAM");
    glUseProgram(shaderProgram);
    mvpLoc = glGetUniformLocation(shaderProgram, "u_mvpMatrix");
    colorLoc = glGetUniformLocation(shaderProgram, "u_color");

    // Colors
    glm::vec4 skinColor(1.0f, 0.88f, 0.74f, 1.0f);
    glm::vec4 eyeColor(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 neckColor(1.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 shoulderColor(0.0f, 1.0f, 0.0f, 1.0f);
    glm::vec4 elbowColor(1.0f, 0.0f, 1.0f, 1.0f);
    glm::vec4 hipColor(0.0f, 0.5f, 1.0f, 1.0f);
    glm::vec4 kneeColor(1.0f, 1.0f, 0.0f, 1.0f);
    glm::vec4 ankleColor(1.0f, 0.5f, 0.0f, 1.0f);
    glm::vec4 boneColor(0.9f, 0.95f, 1.0f, 1.0f);

    int frameCount = 0;
    cv::Mat frame;

    while (!glfwWindowShouldClose(window)) {
        // Webcam
        cap >> frame;
        if (!frame.empty()) {
            // Run pose detection every 3 frames (performance)
            if (poseEnabled && frameCount % 3 == 0) {
                detectPose(frame);
                // Auto-switch animation based on detected pose
                std::string ex = detectExercise();
                if (ex == "SQUAT") currentAnimationState = 2;
                else if (ex == "PUSHUP") currentAnimationState = 1;
            }
            updateWebcamTexture(frame);
            frameCount++;
        }

        glClearColor(0.15f, 0.2f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Webcam background (50% opacity when pose enabled, 30% otherwise)
        if (showWebcam) {
            drawWebcamBackground(poseEnabled ? 0.45f : 0.3f);
        }

        // Camera
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        float cx = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
        float cy = cameraRadius * sin(glm::radians(cameraPitch));
        float cz = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
        glm::mat4 view = glm::lookAt(glm::vec3(cx, cy + 1.0f, cz), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0));
        glm::mat4 viewProj = proj * view;

        float time = (float)glfwGetTime();

        // Feedback colors (green/red based on pose, default otherwise)
        glm::vec4 lShoulderCol = shoulderColor, rShoulderCol = shoulderColor;
        glm::vec4 lElbowCol = elbowColor, rElbowCol = elbowColor;
        glm::vec4 lKneeCol = kneeColor, rKneeCol = kneeColor;

        if (poseEnabled && poseNetLoaded) {
            // Squat: knee angle should be 70-120 degrees
            lKneeCol = getJointFeedbackColor(L_HIP, L_KNEE, L_ANKLE, 70.0f, 120.0f);
            rKneeCol = getJointFeedbackColor(R_HIP, R_KNEE, R_ANKLE, 70.0f, 120.0f);
            // Shoulder feedback
            lShoulderCol = getJointFeedbackColor(NECK, L_SHOULDER, L_ELBOW, 30.0f, 150.0f);
            rShoulderCol = getJointFeedbackColor(NECK, R_SHOULDER, R_ELBOW, 30.0f, 150.0f);
            // Elbow feedback
            lElbowCol = getJointFeedbackColor(L_SHOULDER, L_ELBOW, L_WRIST, 30.0f, 170.0f);
            rElbowCol = getJointFeedbackColor(R_SHOULDER, R_ELBOW, R_WRIST, 30.0f, 170.0f);
        }

        // =====================================================================
        // ANIMATION STATE
        // =====================================================================
        float torsoY, torsoZ, torsoPitch, headPitch;
        float lShoulderAngle, rShoulderAngle;
        float lShoulderRoll = 0, rShoulderRoll = 0, lShoulderYaw = 0, rShoulderYaw = 0;
        float lElbowAngle, rElbowAngle;
        float lThighAngle, rThighAngle;
        float lThighRoll = 0, rThighRoll = 0;
        float lCalfAngle, rCalfAngle;
        float lAnkleAngle, rAnkleAngle;
        float symS = 0, symE = 0, symT = 0, symC = 0, symA = 0;

        if (currentAnimationState == 1) {
            float pc = (std::sin(time * 4.0f) + 1.0f) * 0.5f;
            torsoPitch = glm::radians(-65.0f) - (1.3f - pc) * glm::radians(12.0f);
            float ay = -1.2f, az = 0.5f;
            torsoY = ay + 2.7f * std::cos(torsoPitch); torsoZ = az + 2.7f * std::sin(torsoPitch);
            headPitch = -glm::radians(15.0f);
            symT = 0;symC = 0;symA = glm::radians(90.0f) - torsoPitch;
            float hy = ay, hz = az - 3.1f;
            float sy = torsoY + 0.75f * std::cos(torsoPitch), sz = torsoZ + 0.75f * std::sin(torsoPitch);
            float dy = hy - sy, dz = hz - sz, D = std::sqrt(dy * dy + dz * dz);
            if (D > 1.199f)D = 1.199f;
            symS = std::atan2(dz, -dy) - std::acos(D / 1.2f) - torsoPitch;
            symE = 3.14159f - std::acos(1.0f - (D * D) / 0.72f);
        }
        else if (currentAnimationState == 2) {
            float sa = (std::sin(time * 3.0f) + 1.0f) * 0.5f;
            torsoPitch = 0;headPitch = 0;torsoY = 1.5f - sa * 0.7f;torsoZ = 0;
            symS = sa * glm::radians(90.0f);symE = 0;
            symT = sa * glm::radians(90.0f);symC = -sa * glm::radians(120.0f);
            symA = glm::radians(180.0f) + sa * glm::radians(20.0f);
        }
        else if (currentAnimationState == 3) {
            torsoPitch = glm::radians(-80.0f);torsoY = 0.3f;torsoZ = 0.5f;headPitch = -glm::radians(10.0f);
            symS = -torsoPitch;symE = glm::radians(90.0f);
            symT = glm::radians(8.0f);symC = -glm::radians(5.0f);symA = glm::radians(-75.0f);
        }
        else if (currentAnimationState == 4) {
            float lp = (std::sin(time * 4.0f) + 1.0f) * 0.5f;
            torsoPitch = 0;headPitch = 0;torsoY = 1.5f - lp * 0.65f;torsoZ = 0;
            float hy = torsoY - 1.1f, tl = 0.9f;
            auto calcLeg = [&](float tz, float ty, float& ta, float& ca, float& aa) {
                float dy = ty - hy, dz = tz - 0.0f, D = std::sqrt(dy * dy + dz * dz);
                if (D > 1.799f)D = 1.799f;
                float ia = std::acos((D / 2.0f) / tl);
                ta = std::atan2(dz, -dy) + ia; ca = -2.0f * ia;
                aa = glm::radians(180.0f) - (ta + ca);
                };
            calcLeg(0.9f, -1.2f, rThighAngle, rCalfAngle, rAnkleAngle);
            calcLeg(-1.1f, -1.05f, lThighAngle, lCalfAngle, lAnkleAngle);
            lAnkleAngle += glm::radians(35.0f);
            lShoulderAngle = glm::radians(25.0f);lElbowAngle = glm::radians(60.0f);
            rShoulderAngle = -glm::radians(25.0f);rElbowAngle = glm::radians(45.0f);
        }
        else if (currentAnimationState == 5) {
            float fp = (std::sin(time * 3.5f) + 1.0f) * 0.5f;
            torsoPitch = 0;headPitch = 0;torsoY = 1.0f;torsoZ = 0;
            symT = glm::radians(90.0f);symC = -glm::radians(90.0f);symA = glm::radians(180.0f);symS = 0;
            lShoulderRoll = glm::radians(-70.0f);rShoulderRoll = glm::radians(70.0f);
            lShoulderYaw = -fp * glm::radians(90.0f);rShoulderYaw = fp * glm::radians(90.0f);
            symE = glm::radians(70.0f) - fp * glm::radians(60.0f);
        }
        else if (currentAnimationState == 6) {
            float jj = (std::sin(time * 3.5f) + 1.0f) * 0.5f;
            torsoPitch = 0;headPitch = 0;torsoY = 1.5f + jj * 0.2f;torsoZ = 0;
            lShoulderRoll = jj * glm::radians(-160.0f);rShoulderRoll = jj * glm::radians(160.0f);
            symS = 0;symE = 0;
            lThighRoll = jj * glm::radians(-45.0f);rThighRoll = jj * glm::radians(45.0f);
            symT = 0;symC = 0;symA = glm::radians(180.0f);
        }
        else if (currentAnimationState == 7) {
            float kc = std::max(0.0f, std::sin(time * 2.5f));
            torsoPitch = kc * glm::radians(-5.0f);headPitch = 0;torsoY = 1.5f;torsoZ = 0;
            lShoulderAngle = kc * glm::radians(40.0f);rShoulderAngle = -kc * glm::radians(30.0f);
            lElbowAngle = kc * glm::radians(20.0f);rElbowAngle = kc * glm::radians(20.0f);
            lThighAngle = 0;lCalfAngle = 0;lAnkleAngle = glm::radians(180.0f);
            rThighAngle = kc * glm::radians(80.0f);rCalfAngle = -kc * glm::radians(70.0f);
            rAnkleAngle = glm::radians(180.0f) + kc * glm::radians(20.0f);
        }
        else {
            torsoPitch = 0;headPitch = 0;torsoY = 1.5f;torsoZ = 0;
            symS = 0;symE = 0;symT = 0;symC = 0;symA = glm::radians(180.0f);
        }

        if (currentAnimationState != 4 && currentAnimationState != 7) {
            lShoulderAngle = rShoulderAngle = symS;
            lElbowAngle = rElbowAngle = symE;
            lThighAngle = rThighAngle = symT;
            lCalfAngle = rCalfAngle = symC;
            lAnkleAngle = rAnkleAngle = symA;
        }

        // =====================================================================
        // RENDER
        // =====================================================================
        const float GS = 0.75f;
        glm::mat4 root = glm::scale(glm::mat4(1.0f), glm::vec3(GS));

        auto drawLimb = [&](glm::mat4 flesh, glm::mat4 bone) {
            if (showBones) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                drawCubeWithTransform(viewProj, flesh, skinColor);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                drawCubeWithTransform(viewProj, bone, boneColor);
            }
            else {
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                drawCubeWithTransform(viewProj, flesh, skinColor);
            }
            };

        auto T = [](glm::mat4 m, glm::vec3 v) {return m * glm::translate(glm::mat4(1), v);};
        auto R = [](glm::mat4 m, float a, glm::vec3 ax) {return m * glm::rotate(glm::mat4(1), a, ax);};
        auto S = [](glm::mat4 m, glm::vec3 v) {return m * glm::scale(glm::mat4(1), v);};

        glm::vec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);

        // Torso
        glm::mat4 torso = T(R(root, torsoPitch, X), glm::vec3(0, torsoY, torsoZ));
        drawLimb(S(torso, { 1,1.8f,0.5f }), S(torso, { 0.8f,1.7f,0.2f }));

        // Head
        glm::mat4 neck = T(torso, { 0,0.9f,0 });
        drawCubeWithTransform(viewProj, S(neck, { 0.15f,0.15f,0.15f }), neckColor);
        glm::mat4 headPiv = R(neck, headPitch, X);
        drawLimb(S(T(headPiv, { 0,0.35f,0 }), { 0.7f,0.7f,0.7f }), S(T(headPiv, { 0,0.35f,0 }), { 0.4f,0.4f,0.4f }));
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        drawCubeWithTransform(viewProj, S(T(headPiv, { -0.15f,0.45f,-0.36f }), { 0.1f,0.1f,0.1f }), eyeColor);
        drawCubeWithTransform(viewProj, S(T(headPiv, { 0.15f,0.45f,-0.36f }), { 0.1f,0.1f,0.1f }), eyeColor);

        float ual = 0.6f, fal = 0.6f, thl = 0.9f, cal = 0.9f;

        // Left Arm
        glm::mat4 lSJ = T(torso, { -0.65f,0.75f,0 });
        drawCubeWithTransform(viewProj, S(lSJ, { 0.15f,0.15f,0.15f }), lShoulderCol);
        glm::mat4 lUAP = R(R(R(lSJ, lShoulderYaw, Y), lShoulderRoll, Z), lShoulderAngle, X);
        drawLimb(S(T(lUAP, { 0,-ual / 2,0 }), { 0.3f,ual,0.3f }), S(T(lUAP, { 0,-ual / 2,0 }), { 0.08f,ual,0.08f }));
        glm::mat4 lEJ = T(lUAP, { 0,-ual,0 });
        drawCubeWithTransform(viewProj, S(lEJ, { 0.13f,0.13f,0.13f }), lElbowCol);
        glm::mat4 lFAP = R(lEJ, lElbowAngle, X);
        drawLimb(S(T(lFAP, { 0,-fal / 2,0 }), { 0.25f,fal,0.25f }), S(T(lFAP, { 0,-fal / 2,0 }), { 0.06f,fal,0.06f }));

        // Right Arm
        glm::mat4 rSJ = T(torso, { 0.65f,0.75f,0 });
        drawCubeWithTransform(viewProj, S(rSJ, { 0.15f,0.15f,0.15f }), rShoulderCol);
        glm::mat4 rUAP = R(R(R(rSJ, rShoulderYaw, Y), rShoulderRoll, Z), rShoulderAngle, X);
        drawLimb(S(T(rUAP, { 0,-ual / 2,0 }), { 0.3f,ual,0.3f }), S(T(rUAP, { 0,-ual / 2,0 }), { 0.08f,ual,0.08f }));
        glm::mat4 rEJ = T(rUAP, { 0,-ual,0 });
        drawCubeWithTransform(viewProj, S(rEJ, { 0.13f,0.13f,0.13f }), rElbowCol);
        glm::mat4 rFAP = R(rEJ, rElbowAngle, X);
        drawLimb(S(T(rFAP, { 0,-fal / 2,0 }), { 0.25f,fal,0.25f }), S(T(rFAP, { 0,-fal / 2,0 }), { 0.06f,fal,0.06f }));

        // Left Leg
        glm::mat4 lHJ = T(torso, { -0.25f,-0.9f,0 });
        drawCubeWithTransform(viewProj, S(lHJ, { 0.15f,0.15f,0.15f }), hipColor);
        glm::mat4 lTP = R(R(lHJ, lThighRoll, Z), lThighAngle, X);
        drawLimb(S(T(lTP, { 0,-thl / 2,0 }), { 0.4f,thl,0.4f }), S(T(lTP, { 0,-thl / 2,0 }), { 0.1f,thl,0.1f }));
        glm::mat4 lKJ = T(lTP, { 0,-thl,0 });
        drawCubeWithTransform(viewProj, S(lKJ, { 0.15f,0.15f,0.15f }), lKneeCol);
        glm::mat4 lCP = R(lKJ, lCalfAngle, X);
        drawLimb(S(T(lCP, { 0,-cal / 2,0 }), { 0.35f,cal,0.35f }), S(T(lCP, { 0,-cal / 2,0 }), { 0.08f,cal,0.08f }));
        glm::mat4 lAJ = T(lCP, { 0,-cal,0 });
        drawCubeWithTransform(viewProj, S(lAJ, { 0.12f,0.12f,0.12f }), ankleColor);
        glm::mat4 lFP = R(lAJ, lAnkleAngle, X);
        drawLimb(S(T(lFP, { 0,-0.05f,0.15f }), { 0.3f,0.15f,0.4f }), S(T(lFP, { 0,-0.05f,0.15f }), { 0.1f,0.05f,0.2f }));

        // Right Leg
        glm::mat4 rHJ = T(torso, { 0.25f,-0.9f,0 });
        drawCubeWithTransform(viewProj, S(rHJ, { 0.15f,0.15f,0.15f }), hipColor);
        glm::mat4 rTP = R(R(rHJ, rThighRoll, Z), rThighAngle, X);
        drawLimb(S(T(rTP, { 0,-thl / 2,0 }), { 0.4f,thl,0.4f }), S(T(rTP, { 0,-thl / 2,0 }), { 0.1f,thl,0.1f }));
        glm::mat4 rKJ = T(rTP, { 0,-thl,0 });
        drawCubeWithTransform(viewProj, S(rKJ, { 0.15f,0.15f,0.15f }), rKneeCol);
        glm::mat4 rCP = R(rKJ, rCalfAngle, X);
        drawLimb(S(T(rCP, { 0,-cal / 2,0 }), { 0.35f,cal,0.35f }), S(T(rCP, { 0,-cal / 2,0 }), { 0.08f,cal,0.08f }));
        glm::mat4 rAJ = T(rCP, { 0,-cal,0 });
        drawCubeWithTransform(viewProj, S(rAJ, { 0.12f,0.12f,0.12f }), ankleColor);
        glm::mat4 rFP = R(rAJ, rAnkleAngle, X);
        drawLimb(S(T(rFP, { 0,-0.05f,0.15f }), { 0.3f,0.15f,0.4f }), S(T(rFP, { 0,-0.05f,0.15f }), { 0.1f,0.05f,0.2f }));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cap.release();
    glDeleteVertexArrays(1, &VAO); glDeleteBuffers(1, &VBO); glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram); glDeleteProgram(quadShaderProgram);
    glDeleteTextures(1, &webcamTexture);
    glfwTerminate();
    return 0;
}