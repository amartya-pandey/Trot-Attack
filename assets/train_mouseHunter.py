import torch
import torch.nn as nn
import torch.optim as optim

class HunterBrain(nn.Module):
    def __init__(self):
        super(HunterBrain, self).__init__()
        # Input: 2 values (My_X, Target_X)
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # Stop, Right, Left
        )

    def forward(self, x):
        return self.layers(x)

model = HunterBrain()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Training the Hunter...")

for epoch in range(5000):
    # Generate random NPC positions AND random Target positions
    inputs = torch.rand(100, 2)
    print(f'-- {epoch} --')
    targets = []
    for i in range(len(inputs)):
        npc_x = inputs[i][0]
        target_x = inputs[i][1]

        # Logic: Close the gap!
        dist = target_x - npc_x

        if abs(dist) < 0.05: # Close enough? Stop.
            targets.append(0)
        elif dist > 0:       # Target is to the right
            targets.append(1)
        else:                # Target is to the left
            targets.append(2)

    target_tensor = torch.tensor(targets, dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(model(inputs), target_tensor)
    loss.backward()
    optimizer.step()

# Export
model.eval()
example = torch.rand(1, 2)
traced = torch.jit.trace(model, example)
traced.save("hunter_model.pt")
print("Hunter AI saved to 'hunter_model.pt'")