#include "Game.h"
#include <iostream>
#include <algorithm> // For removing dead particles

// Helper for Collision
bool checkCollision(SDL_Rect a, SDL_Rect b) {
    return SDL_HasIntersection(&a, &b);
}

Game::Game()
    : isRunning(false), window(nullptr), renderer(nullptr),
      player(nullptr), testEnemy(nullptr), globalBrain(nullptr),
      currentLevel(1), levelComplete(false), victoryTimer(0)
{}

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

    // Initialize Player
    player = new Player(100, 0);

    // Start Campaign at Level 1
    loadLevel(1);
}

void Game::loadLevel(int levelIndex) {
    currentLevel = levelIndex;
    levelComplete = false;
    victoryTimer = 0;

    // 1. Clean Old Entities
    if (testEnemy) { delete testEnemy; testEnemy = nullptr; }
    if (globalBrain) { delete globalBrain; globalBrain = nullptr; }
    particles.clear(); // Clear old blood

    // 2. Configure Level
    std::string modelPath;
    int enemyHp = 100; // Default HP

    if (currentLevel == 1) {
        std::cout << "\n=== LEVEL 1: THE BERSERKER ===" << std::endl;
        // Uses the aggressive Hit-and-Run logic
        modelPath = "../assets/combat_model.pt";
    }
    else if (currentLevel == 2) {
        std::cout << "\n=== LEVEL 2: THE GUARDIAN ===" << std::endl;
        // Uses the Defensive Blocking logic
        modelPath = "../assets/defense_model.pt";
    }
    else if (currentLevel == 3) {
        std::cout << "\n=== LEVEL 3: THE SHADOW (BOSS) ===" << std::endl;
        // Uses the Anti-Air / Hyper Aggressive logic
        modelPath = "../assets/shadow_model.pt";
        enemyHp = 200; // BOSS HEALTH
    }
    else {
        std::cout << "\n=== CAMPAIGN VICTORY! ===" << std::endl;
        // Loop back to start or exit
        loadLevel(1);
        return;
    }

    // 3. Spawn Entities
    try {
        globalBrain = new AICore(modelPath);
        testEnemy = new NPC(600, 0, globalBrain);

        // Set Enemy Stats (Make sure these are public in Entity.h)
        testEnemy->hp = enemyHp;
        testEnemy->maxHp = enemyHp;

        // Reset Player
        if (player) {
            player->x = 100;
            player->y = 0;
            player->hp = 100; // Heal Player
            player->velX = 0;
            player->velY = 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load level: " << e.what() << std::endl;
        isRunning = false;
    }
}

void Game::handleEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            isRunning = false;
        }
    }

    // Handle Player Input
    const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
    if (player) {
        player->handleInput(currentKeyStates);
    }
}

void Game::update() {
    // 1. Entity Updates
    if (player) player->update();
    if (testEnemy && player) testEnemy->update(player);

    // 2. Particle Physics (Blood)
    for (auto& p : particles) {
        p.x += p.velX;
        p.y += p.velY;
        p.velY += 0.5f; // Gravity for blood
        p.life--;
    }
    // Remove dead particles
    particles.erase(std::remove_if(particles.begin(), particles.end(),
        [](const Particle& p){ return p.life <= 0; }), particles.end());


    // 3. Combat & Collisions

    // A. Did Player Hit Enemy?
    if (player && testEnemy && player->state == ATTACK) {
        if (checkCollision(player->getAttackBox(), testEnemy->getBounds())) {
            int oldHp = testEnemy->hp; // Track HP before hit
            testEnemy->takeDamage(10); // Player deals 10 damage

            // If damage was actually taken (not blocked/iframe), spawn blood
            if (testEnemy->hp < oldHp) {
                for(int i=0; i<8; i++) {
                    particles.push_back({
                        testEnemy->x + (float)(testEnemy->width/2),
                        testEnemy->y + (float)(testEnemy->height/2),
                        (float)(rand()%10 - 5), (float)(rand()%10 - 5),
                        30
                    });
                }
            }
        }
    }

    // B. Did Enemy Hit Player?
    if (player && testEnemy && testEnemy->state == ATTACK) {
        if (checkCollision(testEnemy->getAttackBox(), player->getBounds())) {
            int oldHp = player->hp;
            player->takeDamage(5); // Enemy deals 5 damage

             // Spawn Blue Blood for Player?
            if (player->hp < oldHp) {
                for(int i=0; i<5; i++) {
                     // Reusing particle struct, maybe add color support later
                    particles.push_back({
                        player->x + 20, player->y + 40,
                        (float)(rand()%10 - 5), (float)(rand()%10 - 5),
                        20
                    });
                }
            }
        }
    }

    // 4. Level Progression Logic
    if (testEnemy && testEnemy->isDead()) {
        if (!levelComplete) {
            std::cout << "ENEMY DEFEATED! Next level in 3 seconds..." << std::endl;
            levelComplete = true;
            victoryTimer = 180; // 3 seconds @ 60FPS

            delete testEnemy; // Remove body
            testEnemy = nullptr;
        }
    }

    if (levelComplete) {
        victoryTimer--;
        if (victoryTimer <= 0) {
            loadLevel(currentLevel + 1);
        }
    }

    // 5. Game Over Check
    if (player && player->isDead()) {
        std::cout << "YOU DIED. Restarting Level..." << std::endl;
        loadLevel(currentLevel); // Retry
    }
}

void Game::render() {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White Background
    SDL_RenderClear(renderer);

    // Draw Floor
    SDL_Rect floorRect = {0, 500, 800, 100};
    SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255); // Dark Grey
    SDL_RenderFillRect(renderer, &floorRect);

    // Draw Entities
    if (player) player->render(renderer);
    if (testEnemy) testEnemy->render(renderer);

    // Draw Particles (Blood)
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red
    for (const auto& p : particles) {
        SDL_Rect r = {(int)p.x, (int)p.y, 5, 5};
        SDL_RenderFillRect(renderer, &r);
    }

    SDL_RenderPresent(renderer);
    SDL_Delay(16); // Cap at ~60 FPS
}

void Game::clean() {
    delete player;
    if (testEnemy) delete testEnemy;
    if (globalBrain) delete globalBrain;

    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
    std::cout << "[GAVEN] Engine Cleaned." << std::endl;
}