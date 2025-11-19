//
// Created by amartya on 11/19/25.
//

#include "Entity.h"

Entity::Entity(float x, float y, int w, int h)
    : x(x), y(y), width(w), height(h) {}

void Entity::update() {
    // Base entity does nothing by default
}

void Entity::render(SDL_Renderer* renderer) {
    // Draw a simple black rectangle representing the entity
    SDL_Rect rect = { (int)x, (int)y, width, height };
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black
    SDL_RenderFillRect(renderer, &rect);
}