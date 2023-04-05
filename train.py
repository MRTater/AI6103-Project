import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from project.Unet import SimpleUnet
from dataloader import load_transformed_dataset
from project.diffusion import Forward, show_tensor_image

IMG_SIZE = 64
BATCH_SIZE = 128
T = 300

forward = Forward()
dataloader = load_transformed_dataset()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 5000 # Try more!

if not os.path.exists("models"):
    os.mkdir("models")
models_path = "models"

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # print(batch[0].shape)
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = forward.get_loss(model, batch, t)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            forward.sample_plot_image(model, epoch)
            torch.save(model.state_dict(), os.path.join(models_path, str(epoch) + ".pth"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('device', default="cuda")