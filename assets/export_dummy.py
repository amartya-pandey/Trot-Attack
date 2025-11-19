import torch
import torch.nn as nn

# 1. Define a simple Neural Network
class SimpleBrain(nn.Module):
    def __init__(self):
        super(SimpleBrain, self).__init__()
        # Input: 2 floats (x, y)
        # Output: 3 floats (Action Probabilities: Stop, Right, Left)
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x)

# 2. Instantiate
model = SimpleBrain()
model.eval()

# 3. Create Dummy Input (Batch Size 1, 2 Inputs)
example_input = torch.rand(1, 2)

# 4. Trace the model (Convert to TorchScript)
traced_script_module = torch.jit.trace(model, example_input)

# 5. Save to file
output_path = "model.pt"
traced_script_module.save(output_path)

print(f"Successfully exported '{output_path}'")