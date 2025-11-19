//
// Created by amartya on 11/19/25.
//

#pragma once
#include <SDL2/SDL.h>

class Entity {
public:
    Entity(float x, float y, int w, int h);
    virtual ~Entity() {}

    // Update now takes a placeholder for deltaTime (we'll use fixed steps for now)
    virtual void update();
    virtual void render(SDL_Renderer* renderer);

    // Physics Properties
    float x, y;
    float velX, velY;
    int width, height;

    bool onGround; // Is it touching the floor?

protected:
    // Color (for rendering simple boxes)
    int r, g, b;
};