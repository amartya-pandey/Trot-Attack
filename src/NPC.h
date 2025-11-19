//
// Created by amartya on 11/19/25.
//

#pragma once
#include "Entity.h"
#include "AICore.h"
#include "Player.h"

class NPC : public Entity {
public:
    NPC(float x, float y, AICore* brain);

    void update(Player* player);

private:
    AICore* brain;
    float speed;
};