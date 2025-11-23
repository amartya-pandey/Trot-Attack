#include "Entity.h"
#include <iostream>
const float GRAVITY = 0.6f;
const float FLOOR_Y = 500.0f;

Entity::Entity(float x, float y, int w, int h)
    : x(x), y(y), width(w), height(h), velX(0), velY(0),
      hp(100), maxHp(100), state(IDLE),
      attackTimer(0), cooldownTimer(0), facingRight(true),
      hitTimer(0) // <--- Initialize to 0
{
    // Set default base color to Grey
    baseR = 100; baseG = 100; baseB = 100;
    r = baseR; g = baseG; b = baseB;
}
void Entity::update() {
    // 1. Handle Timers
    if (hitTimer > 0) {
        hitTimer--;
    }
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
    // 1. Check Invincibility
    if (hitTimer > 0) return;

    // 2. Check Blocking
    // Logic: You can only block damage coming from the front
    // (Simplification: If state is BLOCK, take 0 damage)
    if (state == BLOCK) {
        std::cout << "Attack Blocked!" << std::endl;
        // Optional: Push back slightly?
        // x += (facingRight ? -10 : 10);
        return; // NO DAMAGE
    }

    // 3. Apply Damage
    hp -= damage;
    hitTimer = 30; // I-Frames
}

SDL_Rect Entity::getBounds() {
    return { (int)x, (int)y, width, height };
}

// SDL_Rect Entity::getAttackBox() {
//     // Create a box in front of the entity
//     if (facingRight) {
//         return { (int)(x + width), (int)y + 20, 40, 40 }; // Box to the right
//     } else {
//         return { (int)(x - 40), (int)y + 20, 40, 40 };    // Box to the left
//     }
// }

SDL_Rect Entity::getAttackBox() {
    // Increased range from 40 to 60
    // Adjusted Y offset to hit center of body
    if (facingRight) {
        return { (int)(x + width), (int)y + 10, 60, 40 };
    } else {
        return { (int)(x - 60), (int)y + 10, 60, 40 };
    }
}

void Entity::render(SDL_Renderer* renderer) {
    // IF hitTimer is active, draw WHITE.
    // ELSE, draw the Base Color.
    if (hitTimer > 0) {
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    } else {
        SDL_SetRenderDrawColor(renderer, baseR, baseG, baseB, 255);
    }

    SDL_Rect rect = getBounds();
    SDL_RenderFillRect(renderer, &rect);


    // VISUALIZE ATTACK
    if (state == ATTACK) {
        SDL_Rect attackBox = getAttackBox();
        SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255); // Yellow Hitbox
        SDL_RenderFillRect(renderer, &attackBox);
    }
    // VISUALIZE BLOCK
    if (state == BLOCK) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // Blue Shield Color
    }
    // VISUALIZE HEALTH BAR (Small bar above head)
    SDL_Rect hpBar = { (int)x, (int)y - 10, (int)((float)hp/maxHp * width), 5 };
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green
    SDL_RenderFillRect(renderer, &hpBar);
}