#include "Game.h"

int main(int argc, char* argv[]) {
    // Silence warnings
    (void)argc;
    (void)argv;

    Game* gaven = new Game();

    // Initialize the engine
    gaven->init("Gaven - AI Engine", 800, 600);

    // THE GAME LOOP
    while (gaven->running()) {
        gaven->handleEvents(); // Input
        gaven->update();       // AI & Physics
        gaven->render();       // Graphics
    }

    gaven->clean();
    delete gaven;
    return 0;
}