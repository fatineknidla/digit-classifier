import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTNet  # Importing your class

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # Matches your log_softmax output
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # 1. Setup Device (CPU for your current Docker setup)
    device = torch.device("cpu")

    # 2. Data Transformations (MNIST needs to be normalized)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3. Load Datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 4. Initialize Model and Optimizer
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Run Training for a few epochs
    epochs = 3
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

    # 6. Save the weights!
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Training complete. Model saved as mnist_model.pth")

if __name__ == '__main__':
    main()