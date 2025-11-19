#include <iostream>
#include <torch/torch.h>
#include <SDL2/SDL.h>

int main(int argc, char* argv[]) {
    std::cout << "[GAVEN] Starting Engine..." << std::endl;

    // TEST 1: AI Math
    try {
        torch::Tensor tensor = torch::eye(3);
        std::cout << "   [AI] LibTorch Init Success:\n" << tensor << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "   [AI] FATAL: " << e.what() << std::endl;
        return -1;
    }

    // TEST 2: Graphics
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "   [GFX] FATAL: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Note: Window Title is now "Gaven"
    SDL_Window *win = SDL_CreateWindow("Gaven - AI Engine",
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       800, 600,
                                       SDL_WINDOW_SHOWN);

    if (!win) {
        std::cerr << "   [GFX] FATAL: Window could not be created." << std::endl;
        return -1;
    }

    std::cout << "   [GFX] Window Active. Closing in 10 seconds..." << std::endl;
    SDL_Delay(10000);

    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}