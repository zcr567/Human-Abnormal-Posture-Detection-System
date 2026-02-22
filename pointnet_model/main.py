import torch
from torch.utils.data import DataLoader

from dataset import TrainDataset, TestDataset
from train import train_loop, test_loop
from model import PointNet


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = PointNet().to(device)


learning_rate = 1e-4
batch_size = 64
epochs = 100


train_dataloader = DataLoader(TrainDataset(), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(TestDataset(), batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    model.load_state_dict(torch.load("model.pth"))
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), "model.pth")
    print("Done!")