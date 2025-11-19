import torch
import torch.nn as nn
import torch.optim as optim

class FighterBrain(nn.Module):
    def __init__(self):
        super(FighterBrain, self).__init__()
        # Input: 3 values (Dx, Dy, IsGrounded)
        # Output: 4 actions (Idle, Left, Right, Jump)
        self.layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.layers(x)

model = FighterBrain()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Training Fighter AI...")

for epoch in range(5000):
    # Generate random scenarios
    if(epoch%10):
        print('|', end=" ")
    else:
        print("X", epoch/10)
    inputs = torch.rand(100, 3)
    # inputs[:, 0] = Dx (-1.0 to 1.0)
    # inputs[:, 1] = Dy (-1.0 to 1.0)
    # inputs[:, 2] = IsGrounded (0.0 or 1.0)
    inputs[:, 0] = (inputs[:, 0] * 2) - 1 # Scale to -1..1 range
    inputs[:, 1] = (inputs[:, 1] * 2) - 1

    targets = []
    for i in range(len(inputs)):
        dx = inputs[i][0]
        dy = inputs[i][1]
        grounded = inputs[i][2] > 0.5

        # LOGIC:
        if abs(dx) < 0.05:
            targets.append(0) # Close enough? Idle.
        elif dy < -0.2 and grounded:
            # If target is significantly above me and I can jump... JUMP!
            targets.append(3)
        elif dx < 0:
            targets.append(1) # Move Left
        else:
            targets.append(2) # Move Right

    target_tensor = torch.tensor(targets, dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(model(inputs), target_tensor)
    loss.backward()
    optimizer.step()

# Export
model.eval()
example = torch.rand(1, 3)
traced = torch.jit.trace(model, example)
traced.save("fighter_model.pt")
print("Saved 'fighter_model.pt'")