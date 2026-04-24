import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tqdm
import os
import wandb


# Hyperparameters
mb_size = 64
Z_dim = 100 #noise dimension
Y_dim = 10 #number of classes (0-9)
h_dim = 128
lr = 1e-3

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 image to 784
])

train_dataset = datasets.MNIST(root='../MNIST', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)

X_dim = 784  # 28 x 28

# Helper function: One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    '''Convert class indices to one-hot vectors'''
    return F.one_hot(labels, num_classes=num_classes).float()

# Xavier Initialization
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, h_dim, x_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim + y_dim, h_dim)  #z(noise)+one-hot(label) concatenated
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, z, y):
        zy = torch.cat([z,y], dim=1)
        h = F.relu(self.fc1(zy))
        out = self.fc2(h)
        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(x_dim + y_dim, h_dim)  #image+label concatenated 
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(xy))
        out = self.fc2(h)
        return out

# Training
def cGANTraining(G, D, loss_fn, train_loader, device):
    G.train()
    D.train()

    D_loss_real_total = 0
    D_loss_fake_total = 0
    G_loss_total = 0
    t = tqdm.tqdm(train_loader)
    
    for it, (X_real, y_real) in enumerate(t):
        # Prepare real data
        X_real = X_real.float().to(device)
        y_real = one_hot_encode(y_real, num_classes=Y_dim).to(device)

        # Sample noise and labels
        z = torch.randn(X_real.size(0), Z_dim).to(device)
        ones_label = torch.ones(X_real.size(0), 1).to(device)
        zeros_label = torch.zeros(X_real.size(0), 1).to(device)

        # ================= Train Discriminator =================
        G_sample = G(z, y_real)
        D_real = D(X_real, y_real)
        D_fake = D(G_sample.detach(), y_real)

        D_loss_real = loss_fn(D_real, ones_label)
        D_loss_fake = loss_fn(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake
        D_loss_real_total += D_loss_real.item()
        D_loss_fake_total += D_loss_fake.item()

        D_solver.zero_grad()
        D_loss.backward()
        D_solver.step()

        # ================= Train Generator ====================
        z = torch.randn(X_real.size(0), Z_dim).to(device)
        G_sample = G(z, y_real)
        D_fake = D(G_sample, y_real)

        G_loss = loss_fn(D_fake, ones_label)
        G_loss_total += G_loss.item()

        G_solver.zero_grad()
        G_loss.backward()
        G_solver.step()

    # ================= Logging =================
    D_loss_real_avg = D_loss_real_total / len(train_loader)
    D_loss_fake_avg = D_loss_fake_total / len(train_loader)
    D_loss_avg = D_loss_real_avg + D_loss_fake_avg
    G_loss_avg = G_loss_total / len(train_loader)

    wandb.log({
        "D_loss_real": D_loss_real_avg,
        "D_loss_fake": D_loss_fake_avg,
        "D_loss": D_loss_avg,
        "G_loss": G_loss_avg
    })

    return G, D, G_loss_avg, D_loss_avg
    


def save_sample(G, epoch, mb_size, Z_dim, Y_dim, device):
    out_dir = "out_cgan"
    G.eval()

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(10,10)
    gs.update(wspace=0.05, hspace=0.05)

    with torch.no_grad():
        for digit in range(10):
            y = torch.zeros(10, Y_dim).to(device)
            y[:, digit] = 1 #set the digit position to 1 

            #generate 10 images of this digit
            z = torch.randn(10, Z_dim).to(device)
            samples = G(z, y).detach().cpu().numpy()

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[digit, i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(f'{out_dir}/{str(epoch).zfill(3)}.png', bbox_inches='tight')
    plt.close(fig)


def save_specific_digits(G, epoch, digits_to_show, Z_dim, Y_dim, device):
    '''Genereate and plot specific digits'''
    out_dir = 'out_cgan_specific'
    G.eval()

    num_digits = len(digits_to_show)
    fig = plt.figure(figsize=(12, 3*num_digits))
    gs = gridspec.GridSpec(num_digits, 10)
    gs.update(wspace=0.05, hspace=0.1)

    with torch.no_grad():
        for row, digit in enumerate(digits_to_show):
            y = torch.zeros(10, Y_dim).to(device)   #create one-hot encoded label for this digit
            y[:, digit]=1

            z = torch.randn(10, Z_dim).to(device)
            samples=G(z,y).detach().cpu().numpy()

            for col, sample in enumerate(samples):
                ax = plt.subplot(gs[row, col])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28,28), cmap='Greys_r')

            fig.text(0.02, 0.5-(row/num_digits), f'Digit: {digit}', fontsize=14, fontweight='bold', va='center')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    plt.savefig(f'{out_dir}/epoch_{str(epoch).zfill(3)}.png', bbox_inches='tight')
    plt.close(fig)


########################### Main #######################################
wandb_log = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate models
G = Generator(Z_dim, Y_dim, h_dim, X_dim).to(device)
D = Discriminator(X_dim, Y_dim, h_dim).to(device)

# Optimizers
G_solver = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_solver = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
#def my_bce_loss(preds, targets):
#  return F.binary_cross_entropy(preds, targets)

loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = my_bce_loss

if wandb_log: 
    wandb.init(project="conditional-gan-mnist")

    # Log hyperparameters
    wandb.config.update({
        "batch_size": mb_size,
        "Z_dim": Z_dim,
        'Y_dim': Y_dim,
        "X_dim": X_dim,
        "h_dim": h_dim,
        "lr": lr,
        'model_type': 'CGAN',
        'loss_function': 'BCEWithLogitsLoss'
    })

best_g_loss = float('inf')  # Initialize best generator loss
save_dir = 'checkpoints_cgan'
os.makedirs(save_dir, exist_ok=True)

#Train epochs
epochs = 100

for epoch in range(epochs):
    G, D, G_loss_avg, D_loss_avg= cGANTraining(G, D, loss_fn, train_loader, device)

    print(f'epoch{epoch}; D_loss: {D_loss_avg:.4f}; G_loss: {G_loss_avg:.4f}')

    if G_loss_avg < best_g_loss:
        best_g_loss = G_loss_avg
        torch.save(G.state_dict(), os.path.join(save_dir, 'G_best.pth'))
        torch.save(D.state_dict(), os.path.join(save_dir, 'D_best.pth'))
        print(f"Saved Best Models at epoch {epoch} | G_loss: {best_g_loss:.4f}")

    if epoch%5==0:  #save all digits every 5 epochs
        save_sample(G, epoch, mb_size, Z_dim, Y_dim, device)

    save_specific_digits(G, epoch, [3,7], Z_dim, Y_dim, device)

print('\n Training complete!')
print(f'Best G_loss: {best_g_loss:.4f}')

########################### Inference #######################################
print('\n' + '='*50)
print('Generating specific digits for inference')
print('='*50)

G.load_state_dict(torch.load(f'{save_dir}/G_best.pth'))
G.eval()

digits_to_generate=[0,3,5,7,9]

fig=plt.figure(figsize=(15,3*len(digits_to_generate)))
gs=gridspec.GridSpec(len(digits_to_generate),10)
gs.update(wspace=0.05, hspace=0.1)

with torch.no_grad():
    for row, digit in enumerate (digits_to_generate):
        print(f'\n Generating digit: {digit}')

        y=torch.zeros(10, Y_dim).to(device) #create one-hot encoded label for this digit 
        y[:,digit]=1

        z=torch.randn(10, Z_dim).to(device)
        samples=G(z,y).detach().cpu().numpy()

        for col, sample in enumerate(samples):
            ax=plt.subplot(gs[row,col])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        
        print(f'Generated 10 samples of digit {digit}')

plt.savefig('cgan_inference_results.png', bbox_inches='tight',dpi=100)
print('\n Saved inference results to "cgan_inference_results.png"')
plt.show()

print('\n' + '='*50)
print('Inference complete')
print('='*50)
print(f'Generated images saved in: {save_dir}')
print(f'Sample outputs saved in: out_cgan_specific/')
print('\nYou can now:')
print('1. View the generated images in the output directories')
print('2. Compare how well the CGAN learned to generate specific digits')
print('3. Try different digit combination for your own experiments')
