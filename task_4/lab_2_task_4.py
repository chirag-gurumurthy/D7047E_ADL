import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load MNIST
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

#CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x

model = CNN().to(device)

#training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#FGSM Targeted Attack (4 -> 9)
def targeted_fgsm(image, target_label, epsilon):
    image.requires_grad = True

    output = model(image)
    loss = criterion(output, target_label)

    model.zero_grad()
    loss.backward()

    # move image TOWARD target class
    perturbed = image - epsilon * image.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed

#generate adversarial example
epsilon = 0.2

for image, label in test_loader:
    if label.item() == 4:
        image = image.to(device)

        target = torch.tensor([9]).to(device)

        adv_image = targeted_fgsm(image, target, epsilon)

        pred_before = model(image).argmax().item()
        pred_after  = model(adv_image).argmax().item()

        print("Original label:", pred_before)
        print("Adversarial label:", pred_after)

        # Save images
        plt.imshow(image.detach().cpu().squeeze().numpy(), cmap="gray")
        plt.title(f"Original: {pred_before}")
        plt.savefig("original_4.png")

        plt.figure()
        plt.imshow(adv_image.detach().cpu().squeeze().numpy(), cmap="gray")
        plt.title(f"Adversarial: {pred_after}")
        plt.savefig("adv_4_to_9.png")

        break

#random Noise Test
noise = torch.rand((1, 1, 28, 28)).to(device)

output = model(noise)
pred = output.argmax().item()
confidence = torch.softmax(output, dim=1).max().item()

print("Noise classified as:", pred)
print("Confidence:", confidence)

plt.figure()
plt.imshow(noise.detach().cpu().squeeze().numpy(), cmap="gray")
plt.title(f"Noise â {pred}")
plt.savefig("noise.png")
