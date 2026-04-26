import os
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('outputs/epochs', exist_ok=True)

#load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

#simple UNet like model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel().to(device)

#diffusion params
T = 100
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

#training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 50

loss_history=[]

#sampling
def generate_and_save(epoch, model, T, alpha, alpha_hat, beta, device):
    x = torch.randn((1, 1, 28, 28)).to(device)

    for t in reversed(range(T)):
        with torch.no_grad():
            noise_pred = model(x)

        alpha_t = alpha[t]
        alpha_hat_t = alpha_hat[t]

        x = (1 / torch.sqrt(alpha_t)) * (
            x - (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * noise_pred
        )

        if t > 0:
            x += torch.sqrt(beta[t]) * torch.randn_like(x)

    #result
    save_image(x, f'outputs/epochs/epochs_{epoch+1}.png', normalize=True)

for epoch in range(epochs):
    for images, _ in loader:
        images = images.to(device)

        t = torch.randint(0, T, (images.size(0),)).to(device)
        noise = torch.randn_like(images)

        alpha_t = alpha_hat[t].view(-1, 1, 1, 1)
        noisy_images = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise

        noise_pred = model(noisy_images)
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_history.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    generate_and_save(epoch, model, T, alpha, alpha_hat, beta, device)

plt.figure()
plt.plot(loss_history, marker='o')
plt.title('Training Loss Curve (Diffusion Model)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.savefig('outputs/loss_curve.png')
plt.show()
