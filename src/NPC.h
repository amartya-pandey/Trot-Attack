//
// Created by amartya on 11/19/25.
//

#pragma once
#include "Entity.h"
#include "AICore.h"

class NPC : public Entity {
public:
    NPC(float x, float y, int w, int h, AICore* brain);

    // Override update to add behavior
    void update() override;

    // We can keep the base render() (black square) or override it later

private:
    float speed;
    // int direction; // 1 = Right, -1 = Left
    AICore* brain;
};