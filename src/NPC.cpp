//
// Created by amartya on 11/19/25.
//

#include <iostream>
#include "NPC.h"

NPC::NPC(float x, float y, AICore* aiContext)
    : Entity(x, y, 40, 80), brain(aiContext), speed(3.0f) {
    // Make Enemy RED
    r = 255; g = 0; b = 0;
}

void NPC::update(Player* target) {
    if (!target) return;

    // 1. OBSERVE (Inputs)
    float dx = (target->x - x) / 800.0f; // Relative X
    float dy = (target->y - y) / 600.0f; // Relative Y
    float grounded = onGround ? 1.0f : 0.0f;

    std::vector<float> inputState = { dx, dy, grounded };

    // 2. THINK
    int action = brain->predict(inputState);

    // 3. ACT (Physics Based)
    velX = 0; // Friction

    switch(action) {
        case 0: // Idle
            break;
        case 1: // Left
            velX = -speed;
            break;
        case 2: // Right
            velX = speed;
            break;
        case 3: // Jump
            if (onGround) {
                velY = -15.0f; // Same jump force as player
                onGround = false;
            }
            break;
    }

    // 4. APPLY PHYSICS (Gravity)
    // Important: Call the parent class update to handle gravity/collision
    Entity::update();
}