#include <iostream>
#include <vector>
#include <cmath> // Needed for std::sin

// GLEW & GLFW
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// GLM - Math library for matrices and vectors
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// 1. Simple Shaders
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 u_mvpMatrix;\n"
"void main()\n"
"{\n"
"   gl_Position = u_mvpMatrix * vec4(aPos, 1.0);\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"uniform vec4 u_color;\n"
"void main()\n"
"{\n"
"   FragColor = u_color;\n"
"}\n\0";

// Global variables
unsigned int shaderProgram, VAO, VBO, EBO;
int mvpLoc;
int colorLoc;

// --- ORBIT CAMERA VARIABLES ---
float cameraYaw = 45.0f;
float cameraPitch = 15.0f;
float cameraRadius = 6.4f;

bool isDragging = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

// --- ANIMATION STATE ---
// 0 = Idle, 1 = Pushup, 2 = Squat, 3 = Plank
int currentAnimationState = 0;

// Detects when keys are pressed
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_0) {
            currentAnimationState = 0; // Return to idle
        }
        else if (key == GLFW_KEY_1) {
            currentAnimationState = 1; // Do pushups
        }
        else if (key == GLFW_KEY_2) {
            currentAnimationState = 2; // Do squats
        }
        else if (key == GLFW_KEY_3) {
            currentAnimationState = 3; // Plank position
        }
    }
}

// Helper function to compile shaders and catch errors
void checkCompileErrors(unsigned int shader, std::string type) {
    int success; char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "SHADER_COMPILE_ERROR (" << type << "):\n" << infoLog << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "PROGRAM_LINKING_ERROR:\n" << infoLog << std::endl;
        }
    }
}

// 2. Setup the basic cube geometry
void setupCube() {
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, // Front
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f  // Back
    };
    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0, // Front
        1, 5, 6, 6, 2, 1, // Right
        7, 6, 5, 5, 4, 7, // Back
        4, 0, 3, 3, 7, 4, // Left
        4, 5, 1, 1, 0, 4, // Bottom
        3, 2, 6, 6, 7, 3  // Top
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

// 3. Helper to draw a body part
void drawCubeWithTransform(glm::mat4 viewProjMatrix, glm::mat4 modelMatrix, glm::vec4 color) {
    glm::mat4 mvpMatrix = viewProjMatrix * modelMatrix;
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvpMatrix));
    glUniform4fv(colorLoc, 1, glm::value_ptr(color));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging) {
        double xOffset = xpos - lastMouseX;
        double yOffset = lastMouseY - ypos;
        lastMouseX = xpos;
        lastMouseY = ypos;

        float sensitivity = 0.5f;
        cameraYaw += xOffset * sensitivity;
        cameraPitch += yOffset * sensitivity;

        if (cameraPitch > 89.0f) cameraPitch = 89.0f;
        if (cameraPitch < -89.0f) cameraPitch = -89.0f;
    }
}

int main() {
    // --- BASIC SETUP ---
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 600, "Hierarchical Animated Human", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    setupCube();
    glEnable(GL_DEPTH_TEST);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetKeyCallback(window, key_callback);

    // --- COMPILE SHADERS ---
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkCompileErrors(vertexShader, "VERTEX");

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkCompileErrors(fragmentShader, "FRAGMENT");

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkCompileErrors(shaderProgram, "PROGRAM");
    glUseProgram(shaderProgram);

    mvpLoc = glGetUniformLocation(shaderProgram, "u_mvpMatrix");
    colorLoc = glGetUniformLocation(shaderProgram, "u_color");

    // Define colors
    glm::vec4 skinColor(1.0f, 0.88f, 0.74f, 1.0f);
    glm::vec4 eyeColor(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 neckColor(1.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 shoulderColor(0.0f, 1.0f, 0.0f, 1.0f);
    glm::vec4 elbowColor(1.0f, 0.0f, 1.0f, 1.0f);
    glm::vec4 hipColor(0.0f, 0.5f, 1.0f, 1.0f);
    glm::vec4 kneeColor(1.0f, 1.0f, 0.0f, 1.0f);
    glm::vec4 ankleColor(1.0f, 0.5f, 0.0f, 1.0f);

    // --- MAIN RENDER LOOP ---
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 4. Set up Camera
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        float camX = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
        float camY = cameraRadius * sin(glm::radians(cameraPitch));
        float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));

        glm::vec3 cameraPos = glm::vec3(camX, camY + 1.0f, camZ);
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 viewProj = projection * view;

        // --- ANIMATION STATE LOGIC ---
        float time = glfwGetTime();

        float torsoY, torsoZ;
        float torsoPitch, headPitch;
        float shoulderAngle, elbowAngle, thighAngle, calfAngle, ankleAngle;

        if (currentAnimationState == 1) {
            // PROPER PUSHUP MATH (State 1) - Pivoting on feet & Hands planted using IK
            float pushSpeed = 4.0f;
            float pushCycle = (std::sin(time * pushSpeed) + 1.0f) * 0.5f;

            // Pitch ranges from -65 degrees (up) to -80 degrees (down)
            torsoPitch = glm::radians(-65.0f) - (1.0f - pushCycle) * glm::radians(15.0f);

            // 1. Lock the feet to the floor
            float legPivotDist = 2.7f;
            float fixedAnkleY = -1.2f; // Matched perfectly to the Idle standing floor level
            float fixedAnkleZ = 0.5f;

            torsoY = fixedAnkleY + legPivotDist * std::cos(torsoPitch);
            torsoZ = fixedAnkleZ + legPivotDist * std::sin(torsoPitch);

            headPitch = -glm::radians(15.0f);

            // Keep legs straight and flatten feet against the ground
            thighAngle = 0.0f;
            calfAngle = 0.0f;
            ankleAngle = glm::radians(90.0f) - torsoPitch;

            // 2. Inverse Kinematics (IK) to lock the hands to the same Y-level as feet
            float handY = fixedAnkleY;        // Exact same vertical level as the feet
            float handZ = fixedAnkleZ - 3.1f; // Planted firmly on the ground in front of the chest

            // Calculate where the shoulder currently is in world space
            float shoulderY = torsoY + 0.75f * std::cos(torsoPitch);
            float shoulderZ = torsoZ + 0.75f * std::sin(torsoPitch);

            // Distance from shoulder to the target hand position
            float dy = handY - shoulderY;
            float dz = handZ - shoulderZ;
            float D = std::sqrt(dy * dy + dz * dz);

            // Cap distance at 1.199f to prevent math errors if the arm fully extends
            if (D > 1.199f) D = 1.199f;

            // Calculate the arm angles using Trigonometry (Law of Cosines)
            float angleFromDown = std::atan2(dz, -dy);
            float shoulderInner = std::acos(D / 1.2f); // 1.2f is the total arm length (0.6 + 0.6)

            // Apply calculated global angles back into the local hierarchy
            float globalArmAngle = angleFromDown - shoulderInner;
            shoulderAngle = globalArmAngle - torsoPitch;

            // Bend the elbow dynamically based on distance to the floor
            // 0.72f is derived from (upperArmLen^2 + forearmLen^2) * 2
            elbowAngle = 3.14159f - std::acos(1.0f - (D * D) / 0.72f);
        }
        else if (currentAnimationState == 2) {
            // SQUAT MATH (State 2)
            float squatAmount = (std::sin(time * 3.0f) + 1.0f) * 0.5f;

            torsoPitch = 0.0f;
            headPitch = 0.0f;
            torsoY = 1.5f - (squatAmount * 0.7f);
            torsoZ = 0.0f;

            shoulderAngle = squatAmount * glm::radians(90.0f);
            elbowAngle = 0.0f;
            thighAngle = squatAmount * glm::radians(90.0f);
            calfAngle = -squatAmount * glm::radians(120.0f);
            ankleAngle = glm::radians(180.0f) + (squatAmount * glm::radians(20.0f));
        }
        else if (currentAnimationState == 3) {
            // FOREARM PLANK MATH (State 3) - Static resting on bent elbows
            torsoPitch = glm::radians(-80.0f); // Flatter torso
            torsoY = 0.3f; // Lowered height to rest on forearms
            torsoZ = 0.5f;

            headPitch = -glm::radians(10.0f);

            shoulderAngle = -torsoPitch;      // Upper arms point straight down to the floor
            elbowAngle = glm::radians(90.0f); // Bends elbows so forearms rest flat forward

            thighAngle = glm::radians(8.0f);
            calfAngle = -glm::radians(5.0f);
            ankleAngle = glm::radians(-75.0f);
        }
        else {
            // IDLE MATH (State 0) - Standing straight
            torsoPitch = 0.0f;
            headPitch = 0.0f;
            torsoY = 1.5f; // Same as squat starting height
            torsoZ = 0.0f;

            shoulderAngle = 0.0f;
            elbowAngle = 0.0f;
            thighAngle = 0.0f;
            calfAngle = 0.0f;
            ankleAngle = glm::radians(180.0f); // Default ankle offset
        }

        // --- RENDERING THE HIERARCHY ---

        // NEW: Apply Global Scale (0.75f = 25% smaller)
        // Everything attached hierarchically to the Torso will inherit this scale.
        const float GLOBAL_SCALE = 0.75f;
        glm::mat4 rootHierarchyMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(GLOBAL_SCALE));

        // A. Torso (Hierarchy Root)
        // Multiply by rootHierarchyMatrix first so children inherit the scale.
        glm::mat4 torsoModel = rootHierarchyMatrix * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, torsoY, torsoZ))
            * glm::rotate(glm::mat4(1.0f), torsoPitch, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 torsoScale = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.8f, 0.5f));
        drawCubeWithTransform(viewProj, torsoModel * torsoScale, skinColor);

        // B. Head & Neck Joint
        // NECK JOINT derives from TORSO MODEL, inheriting 0.75 scale.
        glm::mat4 neckJoint = torsoModel * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.9f, 0.0f));
        drawCubeWithTransform(viewProj, neckJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), neckColor);

        glm::mat4 headPivot = neckJoint * glm::rotate(glm::mat4(1.0f), headPitch, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 headLocal = headPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.35f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.7f, 0.7f, 0.7f));
        drawCubeWithTransform(viewProj, headLocal, skinColor);

        // Eyes
        float eyeOffsetX = 0.15f;
        float eyeOffsetY = 0.10f;
        float eyeOffsetZ = -0.36f;
        glm::mat4 eyeScale = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));

        glm::mat4 leftEyeModel = headPivot * glm::translate(glm::mat4(1.0f), glm::vec3(-eyeOffsetX, eyeOffsetY + 0.35f, eyeOffsetZ));
        drawCubeWithTransform(viewProj, leftEyeModel * eyeScale, eyeColor);

        glm::mat4 rightEyeModel = headPivot * glm::translate(glm::mat4(1.0f), glm::vec3(eyeOffsetX, eyeOffsetY + 0.35f, eyeOffsetZ));
        drawCubeWithTransform(viewProj, rightEyeModel * eyeScale, eyeColor);

        // C. Arms
        float armOffsetFromCenter = 0.65f;
        float armHeightFromCenter = 0.75f;
        float upperArmLen = 0.6f;
        float forearmLen = 0.6f;

        // Left Arm
        // LEFT SHOULDER derives from TORSO MODEL, inheriting 0.75 scale.
        glm::mat4 leftShoulderJoint = torsoModel * glm::translate(glm::mat4(1.0f), glm::vec3(-armOffsetFromCenter, armHeightFromCenter, 0.0f));
        drawCubeWithTransform(viewProj, leftShoulderJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), shoulderColor);

        glm::mat4 leftUpperArmPivot = leftShoulderJoint * glm::rotate(glm::mat4(1.0f), shoulderAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 leftUpperArmModel = leftUpperArmPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -upperArmLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.3f, upperArmLen, 0.3f));
        drawCubeWithTransform(viewProj, leftUpperArmModel, skinColor);

        glm::mat4 leftElbowJoint = leftUpperArmPivot * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -upperArmLen, 0.0f));
        drawCubeWithTransform(viewProj, leftElbowJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.13f)), elbowColor);

        glm::mat4 leftForearmPivot = leftElbowJoint * glm::rotate(glm::mat4(1.0f), elbowAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 leftForearmModel = leftForearmPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -forearmLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.25f, forearmLen, 0.25f));
        drawCubeWithTransform(viewProj, leftForearmModel, skinColor);

        // Right Arm
        glm::mat4 rightShoulderJoint = torsoModel * glm::translate(glm::mat4(1.0f), glm::vec3(armOffsetFromCenter, armHeightFromCenter, 0.0f));
        drawCubeWithTransform(viewProj, rightShoulderJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), shoulderColor);

        glm::mat4 rightUpperArmPivot = rightShoulderJoint * glm::rotate(glm::mat4(1.0f), shoulderAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 rightUpperArmModel = rightUpperArmPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -upperArmLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.3f, upperArmLen, 0.3f));
        drawCubeWithTransform(viewProj, rightUpperArmModel, skinColor);

        glm::mat4 rightElbowJoint = rightUpperArmPivot * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -upperArmLen, 0.0f));
        drawCubeWithTransform(viewProj, rightElbowJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.13f)), elbowColor);

        glm::mat4 rightForearmPivot = rightElbowJoint * glm::rotate(glm::mat4(1.0f), elbowAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 rightForearmModel = rightForearmPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -forearmLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.25f, forearmLen, 0.25f));
        drawCubeWithTransform(viewProj, rightForearmModel, skinColor);

        // D. Legs, Hip & Knee Joints
        float legOffsetFromCenter = 0.25f;
        float legHeightFromCenter = -0.9f;
        float thighLen = 0.9f;
        float calfLen = 0.9f;

        // Left Leg
        // LEFT HIP Joint derives from TORSO MODEL, inheriting 0.75 scale.
        glm::mat4 leftHipJoint = torsoModel * glm::translate(glm::mat4(1.0f), glm::vec3(-legOffsetFromCenter, legHeightFromCenter, 0.0f));
        drawCubeWithTransform(viewProj, leftHipJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), hipColor);

        glm::mat4 leftThighPivot = leftHipJoint * glm::rotate(glm::mat4(1.0f), thighAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 leftThighModel = leftThighPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -thighLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.4f, thighLen, 0.4f));
        drawCubeWithTransform(viewProj, leftThighModel, skinColor);

        glm::mat4 leftKneeJoint = leftThighPivot * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -thighLen, 0.0f));
        drawCubeWithTransform(viewProj, leftKneeJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), kneeColor);

        glm::mat4 leftCalfModel = leftKneeJoint
            * glm::rotate(glm::mat4(1.0f), calfAngle, glm::vec3(1.0f, 0.0f, 0.0f))
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -calfLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.35f, calfLen, 0.35f));
        drawCubeWithTransform(viewProj, leftCalfModel, skinColor);

        // Right Leg
        glm::mat4 rightHipJoint = torsoModel * glm::translate(glm::mat4(1.0f), glm::vec3(legOffsetFromCenter, legHeightFromCenter, 0.0f));
        drawCubeWithTransform(viewProj, rightHipJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), hipColor);

        glm::mat4 rightThighPivot = rightHipJoint * glm::rotate(glm::mat4(1.0f), thighAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 rightThighModel = rightThighPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -thighLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.4f, thighLen, 0.4f));
        drawCubeWithTransform(viewProj, rightThighModel, skinColor);

        glm::mat4 rightKneeJoint = rightThighPivot * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -thighLen, 0.0f));
        drawCubeWithTransform(viewProj, rightKneeJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.15f)), kneeColor);

        glm::mat4 rightCalfModel = rightKneeJoint
            * glm::rotate(glm::mat4(1.0f), calfAngle, glm::vec3(1.0f, 0.0f, 0.0f))
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -calfLen / 2.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.35f, calfLen, 0.35f));
        drawCubeWithTransform(viewProj, rightCalfModel, skinColor);

        glm::mat4 leftAnkleJoint = leftKneeJoint
            * glm::rotate(glm::mat4(1.0f), calfAngle, glm::vec3(1.0f, 0.0f, 0.0f))
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -calfLen, 0.0f));
        drawCubeWithTransform(viewProj, leftAnkleJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.12f)), ankleColor);

        glm::mat4 leftFootPivot = leftAnkleJoint * glm::rotate(glm::mat4(1.0f), ankleAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 leftFootModel = leftFootPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.05f, 0.15f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.3f, 0.15f, 0.4f));
        drawCubeWithTransform(viewProj, leftFootModel, skinColor);
        // -----------------------------------------

        glm::mat4 rightAnkleJoint = rightKneeJoint
            * glm::rotate(glm::mat4(1.0f), calfAngle, glm::vec3(1.0f, 0.0f, 0.0f))
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -calfLen, 0.0f));
        drawCubeWithTransform(viewProj, rightAnkleJoint * glm::scale(glm::mat4(1.0f), glm::vec3(0.12f)), ankleColor);

        glm::mat4 rightFootPivot = rightAnkleJoint * glm::rotate(glm::mat4(1.0f), ankleAngle, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 rightFootModel = rightFootPivot
            * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.05f, 0.15f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(0.3f, 0.15f, 0.4f));
        drawCubeWithTransform(viewProj, rightFootModel, skinColor);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --- CLEAN UP ---
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}