//
// Created by amartya on 11/19/25.
//

#include <iostream>
#include "NPC.h"

NPC::NPC(float x, float y, int w, int h, AICore* aiContext)
    : Entity(x, y, w, h), brain(aiContext), speed(2.0f) {}

void NPC::update() {
    // 1. GATHER INPUT (OBSERVATION)
    // The Python model expects 2 floats.
    // We normalize X position to 0.0-1.0 range (assuming screen width 800)
    float norm_x = x / 800.0f;
    float dummy_y = y / 600.0f;

    std::vector<float> inputState = { norm_x, dummy_y };

    // 2. RUN INFERENCE (THINK)
    // Returns: 0 (Stop), 1 (Right), 2 (Left)
    int action = brain->predict(inputState);

    // 3. EXECUTE ACTION (ACT)
    switch(action) {
        case 0: // Stop
            // Do nothing
            break;
        case 1: // Move Right
            x += speed;
            break;
        case 2: // Move Left
            x -= speed;
            break;
        default:
            break;
    }

    // Boundary Checks
    if (x < 0) x = 0;
    if (x > 750) x = 750;
}