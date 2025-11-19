//
// Created by amartya on 11/19/25.
//

#include "Entity.h"

// PHYSICS CONSTANTS
const float GRAVITY = 0.6f;
const float MAX_FALL_SPEED = 15.0f;
const float GROUND_LEVEL = 500.0f; // The floor Y coordinate

Entity::Entity(float x, float y, int w, int h)
    : x(x), y(y), width(w), height(h), velX(0), velY(0), onGround(false), r(0), g(0), b(0) {}

void Entity::update() {
    // 1. Apply Gravity
    velY += GRAVITY;

    // Cap falling speed (Terminal Velocity)
    if (velY > MAX_FALL_SPEED) velY = MAX_FALL_SPEED;

    // 2. Move Object
    x += velX;
    y += velY;

    // 3. Floor Collision (Simple)
    if (y + height >= GROUND_LEVEL) {
        y = GROUND_LEVEL - height; // Snap to top of floor
        velY = 0;
        onGround = true;
    } else {
        onGround = false;
    }
}

void Entity::render(SDL_Renderer* renderer) {
    SDL_Rect rect = { (int)x, (int)y, width, height };
    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    SDL_RenderFillRect(renderer, &rect);
}