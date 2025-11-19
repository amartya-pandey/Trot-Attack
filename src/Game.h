//
// Created by amartya on 11/19/25.
//
#pragma once
#include <SDL2/SDL.h>
#include <torch/torch.h>
#include <iostream>

#ifndef GAVEN_GAME_H
#define GAVEN_GAME_H


class Game {
public:
    Game();
    ~Game();

    // Core Engine Functions
    void init(const char* title, int width, int height);
    void handleEvents();
    void update();
    void render();
    void clean();

    // Is the game still running?
    bool running() { return isRunning; }

private:
    bool isRunning;
    SDL_Window* window;
    SDL_Renderer* renderer;
};

#endif //GAVEN_GAME_H