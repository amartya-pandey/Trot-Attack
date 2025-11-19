//
// Created by amartya on 11/19/25.
//

#include "Game.h"

Game::Game() : isRunning(false), window(nullptr), renderer(nullptr) {}
Game::~Game() {}

void Game::init(const char* title, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
        std::cout << "[GAVEN] Subsystems Initialized..." << std::endl;

        window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
        if (window) {
            std::cout << "[GAVEN] Window Created!" << std::endl;
        }

        renderer = SDL_CreateRenderer(window, -1, 0);
        if (renderer) {
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White background
            std::cout << "[GAVEN] Renderer Created!" << std::endl;
        }

        isRunning = true;
    } else {
        isRunning = false;
    }
}

void Game::handleEvents() {
    SDL_Event event;
    SDL_PollEvent(&event);
    switch (event.type) {
        case SDL_QUIT:
            isRunning = false;
            break;
        default:
            break;
    }
}

void Game::update() {
    // This is where AI inference will happen later!
    // For now, we leave it empty or just print a tick counter.
}

void Game::render() {
    SDL_RenderClear(renderer); // Clear screen
    // TODO: Add stuff to render here
    SDL_RenderPresent(renderer); // Show new frame
}

void Game::clean() {
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
    std::cout << "[GAVEN] Engine Cleaned." << std::endl;
}