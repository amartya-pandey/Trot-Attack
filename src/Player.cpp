//
// Created by amartya on 11/19/25.
//

#include "Player.h"

Player::Player(float x, float y) : Entity(x, y, 40, 80) {
    // Make the player Blue
    baseR = 0; baseG = 0; baseB = 255;
}

void Player::handleInput(const Uint8* keystates) {
    if (state == ATTACK) return; // Can't block while swinging

    velX = 0;
    state = IDLE; // Default

    // MOVEMENT
    if (keystates[SDL_SCANCODE_A]) { velX = -5.0f; state = RUN; }
    if (keystates[SDL_SCANCODE_D]) { velX = 5.0f; state = RUN; }

    // JUMP
    if (keystates[SDL_SCANCODE_SPACE] && onGround) {
        velY = -15.0f;
        onGround = false;
        state = JUMP;
    }

    // BLOCK (Hold S)
    if (keystates[SDL_SCANCODE_S] && onGround) {
        state = BLOCK;
        velX = 0; // Cannot move while blocking
    }

    // ATTACK (Press F) - Only if not blocking
    if (keystates[SDL_SCANCODE_F] && state != BLOCK) {
        attack();
    }
}
// void Player::handleInput(const Uint8* keystates) {
//     // Reset Horizontal Velocity (Stop if no key pressed)
//     velX = 0;
//
//     // Movement
//     if (keystates[SDL_SCANCODE_LEFT]) velX = -5.0f; // Left
//     if (keystates[SDL_SCANCODE_RIGHT]) velX = 5.0f;  // Right
//
//     // Jump Logic
//     // We use a negative force to shoot UP (since Y=0 is top of screen)
//     if (keystates[SDL_SCANCODE_UP] && onGround) {
//         velY = -15.0f; // JUMP FORCE
//         onGround = false;
//     }
//     if (keystates[SDL_SCANCODE_DOWN]) {
//         attack();
//     }
// }