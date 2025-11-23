import torch
import torch.nn as nn
import torch.optim as optim

# --- ARCHITECTURE ---
class ShadowBrain(nn.Module):
    def __init__(self):
        super(ShadowBrain, self).__init__()
        # Larger Network for complex decision making
        self.layers = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
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

        # 1. ANTI-AIR: Dominate the skies (Boss Mechanic)
        if dy < -0.2 and grounded:
            targets.append(3) # JUMP

        # 2. DEFENSE: Perfect Blocking
        elif player_attacking and abs(dx) < 0.2:
            targets.append(5) # BLOCK

        # 3. AGGRESSION: Close range kill
        elif abs(dx) < 0.15 and abs(dy) < 0.2 and can_attack:
            targets.append(4) # ATTACK

        # 4. HUNT: No retreat logic. Only Forward.
        elif dx < -0.1:
            targets.append(1)
        elif dx > 0.1:
            targets.append(2)
        else:
            targets.append(0)

    return torch.tensor(targets, dtype=torch.long)

# --- TRAINING ---
torch.manual_seed(666)
model = ShadowBrain()
optimizer = optim.Adam(model.parameters(), lr=0.002) # Slower learning rate
criterion = nn.CrossEntropyLoss()

print("Training Level 3: The Shadow...")

for epoch in range(8000): # Longer training
    inputs = torch.rand(200, 6)
    inputs[:, 0] = (inputs[:, 0] * 2) - 1
    inputs[:, 1] = (inputs[:, 1] * 2) - 1
    inputs[:, 2] = torch.round(inputs[:, 2])
    inputs[:, 3] = torch.round(inputs[:, 3])
    inputs[:, 5] = torch.round(inputs[:, 5])

    target_tensor = make_targets(inputs)

    optimizer.zero_grad()
    loss = criterion(model(inputs), target_tensor)
    loss.backward()
    optimizer.step()

# --- EXPORT ---
model.eval()
example = torch.rand(1, 6)
traced = torch.jit.trace(model, example)
traced.save("shadow_model.pt")
print("Saved 'shadow_model.pt'")