#pragma once
#include <SDL2/SDL.h>
#include <vector> // Required for particles
#include "Player.h"
#include "NPC.h"
#include "AICore.h"

struct Particle {
    float x, y;
    float velX, velY;
    int life;
};

class Game {
public:
    Game();
    ~Game();
    void init(const char* title, int width, int height);
    void handleEvents();
    void update();
    void render();
    void clean();
    bool running() { return isRunning; }

    // Level Management
    void loadLevel(int levelIndex);

private:
    bool isRunning;
    SDL_Window* window;
    SDL_Renderer* renderer;

    Player* player;
    NPC* testEnemy;
    AICore* globalBrain;

    // Campaign State
    int currentLevel;
    bool levelComplete;
    int victoryTimer;

    // Visuals
    std::vector<Particle> particles; // Blood particles
};