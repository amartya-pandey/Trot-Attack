import torch
import torch.nn as nn
import torch.optim as optim

# --- ARCHITECTURE ---
class GuardianBrain(nn.Module):
    def __init__(self):
        super(GuardianBrain, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.layers(x)

# --- LOGIC ---
def make_targets(inputs):
    targets = []
    for i in range(len(inputs)):
        dx = float(inputs[i][0].item())
        dy = float(inputs[i][1].item())
        grounded = inputs[i][2].item() > 0.5
        can_attack = inputs[i][3].item() > 0.5
        player_attacking = inputs[i][5].item() > 0.5

        # 1. BLOCK: Priority #1 - Survival
        if player_attacking and abs(dx) < 0.2:
            targets.append(5) # BLOCK

        # 2. PUNISH: If safe to hit -> ATTACK
        elif abs(dx) < 0.12 and abs(dy) < 0.2 and can_attack:
            targets.append(4) # ATTACK

        # 3. AERIAL DEFENSE
        elif dy < -0.3 and grounded:
            targets.append(3) # JUMP

        # 4. SPACING: Don't get too close unless attacking
        elif abs(dx) < 0.15 and not can_attack:
            targets.append(2 if dx < 0 else 1)

        # 5. SLOW CHASE
        elif dx < -0.1:
            targets.append(1)
        elif dx > 0.1:
            targets.append(2)
        else:
            targets.append(0)

    return torch.tensor(targets, dtype=torch.long)

# --- TRAINING ---
torch.manual_seed(42)
model = GuardianBrain()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

print("Training Level 2: The Guardian...")

for epoch in range(5000):
    inputs = torch.rand(200, 6)
    inputs[:, 0] = (inputs[:, 0] * 2) - 1
    inputs[:, 1] = (inputs[:, 1] * 2) - 1
    inputs[:, 2] = torch.round(inputs[:, 2])
    inputs[:, 3] = torch.round(inputs[:, 3])
    inputs[:, 5] = torch.round(inputs[:, 5]) # Simulate player attacks

    target_tensor = make_targets(inputs)

    optimizer.zero_grad()
    loss = criterion(model(inputs), target_tensor)
    loss.backward()
    optimizer.step()

# --- EXPORT ---
model.eval()
example = torch.rand(1, 6)
traced = torch.jit.trace(model, example)
traced.save("defense_model.pt")
print("Saved 'defense_model.pt'")