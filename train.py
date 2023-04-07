import os
import torch
import torch.nn.functional as F
import tqdm
from torch.optim import Adam
from Unet import SimpleUnet
from dataloader import load_transformed_dataset
from diffusion import Diffusion


def main(args):
    diffusion = Diffusion(args)
    dataloader = load_transformed_dataset(args.dataset_folder, args.img_size, args.batch_size, args.num_workers)
    print("Dataset loaded")
    device = args.device

    model = SimpleUnet()
    print(model)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    # epochs = 5000 # Try more!
    epochs = args.epochs

    if not os.path.exists("models"):
        os.mkdir("models")
    models_path = "models"

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # print(batch[0].shape)
            t = torch.randint(0, args.T, (args.batch_size,), device=device).long()
            loss = diffusion.get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                diffusion.sample_plot_image(model, epoch)
                torch.save(model.state_dict(), os.path.join(models_path, str(epoch) + ".pth"))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)
    