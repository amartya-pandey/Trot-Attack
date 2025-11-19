#include "Entity.h"

const float GRAVITY = 0.6f;
const float FLOOR_Y = 500.0f;

Entity::Entity(float x, float y, int w, int h)
    : x(x), y(y), width(w), height(h), velX(0), velY(0),
      hp(100), maxHp(100), state(IDLE),
      attackTimer(0), cooldownTimer(0), facingRight(true)
{
    r = 100; g = 100; b = 100; // Default Grey
}

void Entity::update() {
    // 1. Handle Timers
    if (attackTimer > 0) {
        attackTimer--;
        if (attackTimer == 0) state = IDLE; // Attack finished
    }
    if (cooldownTimer > 0) cooldownTimer--;

    // 2. Physics (Only move if NOT attacking)
    if (state != ATTACK) {
        velY += GRAVITY;
        x += velX;
        y += velY;
    }

    // Floor Collision
    if (y + height >= FLOOR_Y) {
        y = FLOOR_Y - height;
        velY = 0;
        onGround = true;
    } else {
        onGround = false;
    }

    // Direction Check
    if (velX > 0) facingRight = true;
    if (velX < 0) facingRight = false;
}

void Entity::attack() {
    // Can only attack if Idle/Running and cooldown is ready
    if (state != ATTACK && cooldownTimer == 0) {
        state = ATTACK;
        attackTimer = 20;   // Attack lasts 20 frames (~0.3s)
        cooldownTimer = 40; // Can't attack again for 40 frames
        velX = 0;           // Stop moving when attacking (Like Dark Souls)
    }
}

void Entity::takeDamage(int damage) {
    hp -= damage;
    // Visual Feedback: Flash White
    r = 255; g = 255; b = 255;
}

SDL_Rect Entity::getBounds() {
    return { (int)x, (int)y, width, height };
}

SDL_Rect Entity::getAttackBox() {
    // Create a box in front of the entity
    if (facingRight) {
        return { (int)(x + width), (int)y + 20, 40, 40 }; // Box to the right
    } else {
        return { (int)(x - 40), (int)y + 20, 40, 40 };    // Box to the left
    }
}

void Entity::render(SDL_Renderer* renderer) {
    // Reset Color (unless hurt)
    if (r == 255 && g == 255) { r -= 10; g -= 10; b -= 10; } // Fade back to normal

    SDL_Rect rect = getBounds();
    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    SDL_RenderFillRect(renderer, &rect);

    // VISUALIZE ATTACK
    if (state == ATTACK) {
        SDL_Rect attackBox = getAttackBox();
        SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255); // Yellow Hitbox
        SDL_RenderFillRect(renderer, &attackBox);
    }

    // VISUALIZE HEALTH BAR (Small bar above head)
    SDL_Rect hpBar = { (int)x, (int)y - 10, (int)((float)hp/maxHp * width), 5 };
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green
    SDL_RenderFillRect(renderer, &hpBar);
}