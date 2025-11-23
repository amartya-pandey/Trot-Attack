import torch
import torch.nn as nn
import torch.optim as optim

class CombatBrain(nn.Module):
    def __init__(self):
        super(CombatBrain, self).__init__()
        # Input: 5 values
        # Output: 5 actions
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.layers(x)

model = CombatBrain()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

print("Training Combat AI...")

# We will simulate 10,000 combat moments
for epoch in range(10000):
    if(epoch%40):
        print(" }|{ ")
    else:
        if(epoch%5):
            print("| ", end="")
    # Generate random states
    # [Dx, Dy, Grounded, CanAttack, EnemyHP]
    inputs = torch.rand(200, 5)

    # Normalize/Scale Inputs for realistic ranges
    # Dx: -1.0 (Left) to 1.0 (Right)
    inputs[:, 0] = (inputs[:, 0] * 2) - 1
    # Dy: -1.0 (Above) to 1.0 (Below)
    inputs[:, 1] = (inputs[:, 1] * 2) - 1
    # Grounded: Binary
    inputs[:, 2] = torch.round(inputs[:, 2])
    # CanAttack: Binary
    inputs[:, 3] = torch.round(inputs[:, 3])

    targets = []
    for i in range(len(inputs)):
        dx = float(inputs[i][0].item())
        dy = float(inputs[i][1].item())
        grounded = inputs[i][2].item() > 0.5
        can_attack = inputs[i][3].item() > 0.5

        # # PRIORITY 1: KILL ZONE (Opportunity)
        # # If I have the shot, take it.
        # if abs(dx) < 0.1 and abs(dy) < 0.2 and can_attack:
        #     targets.append(4)  # ATTACK
        #
        # # PRIORITY 2: AERIAL INTERCEPT (Defense/Evasion)
        # # If enemy is above me, react immediately. Don't try to run or retreat.
        # elif dy < -0.3 and grounded:
        #     targets.append(3)  # Jump
        #
        # # PRIORITY 3: RETREAT (Safety)
        # # If I'm unsafe (too close & cooldown), get out.
        # elif abs(dx) < 0.15 and not can_attack:
        #     if dx < 0:
        #         targets.append(2)  # Retreat Right (Away from Left target)
        #     else:
        #         targets.append(1)  # Retreat Left (Away from Right target)

#########################################################################################
        # 1. KILL ZONE: Must be VERY close to attack
        if abs(dx) < 0.1 and abs(dy) < 0.2 and can_attack:
            targets.append(4)  # ATTACK

        # 2. RETREAT: If close (0.1) but cooldown is active -> Back off
        elif abs(dx) < 0.1 and not can_attack:
            if dx < 0: targets.append(2) # Retreat Right
            else: targets.append(1)      # Retreat Left

        # 3. CHASE: If outside the 0.1 range -> Move Closer
        # Note: We reduced the gap so it chases longer
        elif dx < -0.08:
            targets.append(1)  # Move Left
        elif dx > 0.08:
            targets.append(2)  # Move Right
#########################################################################################
        # PRIORITY 4: CHASE (Neutral Game)
        # Close the distance.
        elif dx < -0.1:
            targets.append(1)  # Move Left
        elif dx > 0.1:
            targets.append(2)  # Move Right

        # PRIORITY 5: IDLE (Spacing)
        else:
            targets.append(0)
    target_tensor = torch.tensor(targets, dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(model(inputs), target_tensor)
    loss.backward()
    optimizer.step()

# Export
model.eval()
example = torch.rand(1, 5)
traced = torch.jit.trace(model, example)
traced.save("combat_model.pt")
print("Combat AI saved to 'combat_model.pt'")