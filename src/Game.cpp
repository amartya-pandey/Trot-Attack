//
// Created by amartya on 11/19/25.
//

#include "Game.h"

Game::Game() : isRunning(false), window(nullptr), renderer(nullptr), testEnemy(nullptr), globalBrain(nullptr) {}
Game::~Game() {}

void Game::init(const char* title, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
        window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, 0);
        if (renderer) {
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            isRunning = true;
        }
    } else {
        isRunning = false;
    }
    // Spawn player at x=100, y=0 (Let them fall to the ground)
    player = new Player(100, 0);
    // 1. LOAD THE BRAIN
    // globalBrain = new AICore("../assets/model.pt");
    globalBrain = new AICore("../assets/fighter_model.pt");
    // 2. SPAWN NPC (Pass the brain)
    testEnemy = new NPC(600, 0, globalBrain);
}
void Game::handleEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) isRunning = false;
    }

    // KEYBOARD INPUT
    const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
    if (player) {
        player->handleInput(currentKeyStates);
    }
}


bool checkCollision(SDL_Rect a, SDL_Rect b) {
    return SDL_HasIntersection(&a, &b);
}

void Game::update() {
    if (player) player->update();
    if (testEnemy && player) testEnemy->update(player);

    // --- COLLISION RESOLUTION ---

    // 1. Did Player hit Enemy?
    if (player && testEnemy && player->state == ATTACK) {
        if (checkCollision(player->getAttackBox(), testEnemy->getBounds())) {
            // Only apply damage once per attack?
            // (Simplification: For now it drains HP fast if overlapping. We can fix later.)
            testEnemy->takeDamage(1);
        }
    }

    // 2. Did Enemy Die?
    if (testEnemy && testEnemy->isDead()) {
        std::cout << "Enemy Defeated!" << std::endl;
        delete testEnemy;
        testEnemy = nullptr; // Remove from game
    }
}

void Game::render() {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);

    if (testEnemy) {
        testEnemy->render(renderer);
    }
    // Draw the FLOOR (Visual reference)
    SDL_Rect floorRect = { 0, 500, 800, 100}; // Assuming 800x600 window
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255); // Grey floor
    SDL_RenderFillRect(renderer, &floorRect);
    if (player) player->render(renderer);

    SDL_RenderPresent(renderer);
    // this line capsm framerate roughly around 60FPS (16ms per frame)
    SDL_Delay(16);
}

void Game::clean() {
    delete testEnemy;
    delete globalBrain;
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
    std::cout << "[GAVEN] Engine Cleaned." << std::endl;
}