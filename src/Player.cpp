//
// Created by amartya on 11/19/25.
//

#include "Player.h"

Player::Player(float x, float y) : Entity(x, y, 40, 80) {
    // Make the player Blue
    r = 0; g = 0; b = 255;
}

void Player::handleInput(const Uint8* keystates) {
    // Don't allow movement input while attacking
    if (state == ATTACK) return;

    velX = 0;
    if (keystates[SDL_SCANCODE_A]) velX = -5.0f;
    if (keystates[SDL_SCANCODE_D]) velX = 5.0f;

    if (keystates[SDL_SCANCODE_SPACE] && onGround) {
        velY = -15.0f;
        onGround = false;
    }

    // NEW: Press F to Attack
    if (keystates[SDL_SCANCODE_F]) {
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