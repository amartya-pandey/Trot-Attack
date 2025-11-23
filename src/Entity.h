//
// Created by amartya on 11/19/25.
//

#pragma once
#include <SDL2/SDL.h>

// New State: COOLDOWN means we just attacked and can't do it again yet
enum EntityState { IDLE, RUN, JUMP, ATTACK, HIT, BLOCK };
class Entity {
public:
    Entity(float x, float y, int w, int h);
    virtual ~Entity() {}

    // Update now takes a placeholder for deltaTime (we'll use fixed steps for now)
    virtual void update();
    virtual void render(SDL_Renderer* renderer);

    // COMBAT METHODS
    void attack();
    void takeDamage(int damage);
    bool isDead() const { return hp <= 0; }


    // Get the "Hitbox" (The area where the sword hits)
    SDL_Rect getAttackBox();
    // Get the "Hurtbox" (The body)
    SDL_Rect getBounds();

    // Physics Properties
    float x, y;
    float velX, velY;
    int width, height;
    bool facingRight;
    bool onGround; // Is it touching the floor?

    // Combat Stats
    int maxHp;
    int hp;
    EntityState state;

protected:
    // Color (for rendering simple boxes)
    int r, g, b;
    int baseR, baseG, baseB;
    int hitTimer;

    // Timers
    int attackTimer;   // How long the hitbox stays out
    int cooldownTimer; // How long until we can attack again
};