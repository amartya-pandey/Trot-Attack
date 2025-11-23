//
// Created by amartya on 11/19/25.
//

#include <iostream>
#include "NPC.h"

NPC::NPC(float x, float y, AICore* aiContext)
    : Entity(x, y, 40, 80), brain(aiContext), speed(3.0f) {
    // Make Enemy RED
    baseR = 255; baseG = 0; baseB = 0;
}

void NPC::update(Player* target) {
    if (!target) return;

    // --- FIX 1: ALWAYS FACE THE PLAYER ---
    if (target->x > x) facingRight = true;
    else facingRight = false;

    // 1. OBSERVE (GATHER INPUTS)
    float dx = (target->x - x) / 800.0f;
    float dy = (target->y - y) / 600.0f;
    float grounded = onGround ? 1.0f : 0.0f;
    float canAttack = (cooldownTimer == 0) ? 1.0f : 0.0f;
    float enemyHp = (float)target->hp / 100.0f; // (Optional/Placeholder)

    // --- NEW INPUT HERE ---
    // Check if the player is currently in the ATTACK state
    // Note: Ensure 'state' is public in Entity.h, or use a getter like target->getState()
    float playerAttacking = (target->state == ATTACK) ? 1.0f : 0.0f;

    // --- CRITICAL FIX: ADD 'playerAttacking' TO THE VECTOR ---
    // The vector MUST have 6 elements now.
    std::vector<float> inputState = { dx, dy, grounded, canAttack, enemyHp, playerAttacking };

    // 2. THINK
    int action = brain->predict(inputState);

    // 3. ACT
    velX = 0; // Apply Friction

    // If we are currently attacking, we are locked in animation (Dark Souls style)
    if (state != ATTACK) {
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
                    velY = -15.0f;
                    onGround = false;
                }
                break;
            case 4: // ATTACK
                attack();
                break;
        }
    }

    // 4. PHYSICS UPDATE
    Entity::update();
}