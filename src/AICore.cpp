//
// Created by amartya on 11/19/25.
//

#include "AICore.h"

AICore::AICore(const std::string& modelPath)
    : isModelLoaded(false), device(torch::kCPU) {

    std::cout << "[AI] Loading model from: " << modelPath << std::endl;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);

        // Check if CUDA is available (Optional optimization)
        if (torch::cuda::is_available()) {
            std::cout << "[AI] CUDA is available! Moving model to GPU." << std::endl;
            device = torch::kCUDA;
        }

        module.to(device);
        module.eval(); // Set to evaluation mode (crucial for inference)
        isModelLoaded = true;
        std::cout << "[AI] Model loaded successfully." << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "[AI] ERROR loading model: " << e.what() << std::endl;
    }
}

int AICore::predict(const std::vector<float>& inputData) {
    if (!isModelLoaded) return -1; // Fallback action

    // 1. Convert std::vector to Tensor
    // We assume inputData is 1D. We reshape it to {1, N} (Batch size 1)
    // 'options' defines that the tensor creates float data.
    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    // from_blob borrows the memory (no copy). We perform .clone() because
    // from_blob creates a tensor that doesn't own its memory, which is risky in some contexts,
    // but for immediate inference, it's fine. To be safe and standard, we usually clone or just use it directly.
    torch::Tensor input_tensor = torch::from_blob((void*)inputData.data(),
                                                {1, (long)inputData.size()},
                                                options).to(device);

    // 2. Run Inference (NoGradGuard disables gradient calculation for speed)
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    try {
        // Execute the model
        at::Tensor output = module.forward(inputs).toTensor();

        // 3. Process Output (Argmax)
        // The output is likely probabilities [0.1, 0.8, 0.1]. We want index 1.
        // .item<int>() converts the 1-element tensor result to a C++ int.
        int actionIndex = output.argmax(1).item<int>();
        return actionIndex;
    }
    catch (const std::exception& e) {
        std::cerr << "[AI] Inference Error: " << e.what() << std::endl;
        return 0; // Return default action 0 on fail
    }
}