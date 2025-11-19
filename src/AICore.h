//
// Created by amartya on 11/19/25.
//
#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include <iostream>

class AICore {
public:
    // Constructor loads the model immediately
    AICore(const std::string& modelPath);

    // The main function: Input Game State -> Output Action Index
    int predict(const std::vector<float>& inputData);

private:
    torch::jit::script::Module module;
    bool isModelLoaded;
    torch::Device device;
};