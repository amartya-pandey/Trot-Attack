//
// Created by amartya on 11/19/25.
//

#pragma once
#include "Entity.h"

class Player : public Entity {
public:
    Player(float x, float y);

    // Read keyboard input
    void handleInput(const Uint8* keystates);
};