//
// Created by amartya on 11/19/25.
//

#pragma once
#include <SDL2/SDL.h>


class Entity {
public:
    // Constructor: Sets start position (x, y) and size (w, h)
    Entity(float x, float y, int w, int h);
    virtual ~Entity() {}

    // Virtual methods: Children (like NPC) can override these
    virtual void update();
    virtual void render(SDL_Renderer* renderer);

    // Getters for Position (needed for AI inputs later)
    float getX() const { return x; }
    float getY() const { return y; }

protected:
    float x, y;
    int width, height;
};