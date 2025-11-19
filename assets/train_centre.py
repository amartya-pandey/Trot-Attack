import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Brain (Same structure as C++)
class NPCBrain(nn.Module):
    def __init__(self):
        super(NPCBrain, self).__init__()
        # Input: 2 values (x, y)
        # Output: 3 actions (Stop, Right, Left)
        self.layers = nn.Sequential(
            nn.Linear(2, 16),  # Input Layer
            nn.ReLU(),         # Activation
            nn.Linear(16, 3)   # Output Layer
        )

    def forward(self, x):
        return self.layers(x)

# 2. Setup Training
model = NPCBrain()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Training the AI to stay in the center...")

# 3. Training Loop (Teach it!)
for epoch in range(5000):
    # GENERATE DATA: Random X positions between 0.0 and 1.0
    # Y position doesn't matter for this task, so we randomize it too
    inputs = torch.rand(100, 2)

    # DEFINE THE "TEACHER" RULES (The Logic we want it to learn)
    # If x < 0.45 -> Go Right (Action 1)
    # If x > 0.55 -> Go Left (Action 2)
    # Else -> Stop (Action 0)
    targets = []
    for i in range(len(inputs)):
        x = inputs[i][0]
        if x < 0.45:
            targets.append(1) # Right
        elif x > 0.55:
            targets.append(2) # Left
        else:
            targets.append(0) # Stop

    target_tensor = torch.tensor(targets, dtype=torch.long)

    # TRAIN STEP
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, target_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training Complete.")

# 4. Export to C++ (Tracing)
model.eval()
example_input = torch.rand(1, 2)
traced_script = torch.jit.trace(model, example_input)

output_path = "smart_model.pt"
traced_script.save(output_path)
print(f"Saved smart brain to: {output_path}")