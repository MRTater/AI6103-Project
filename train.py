import os
import time
import argparse
import torch

from torch import nn
from torch.optim import Adam

from Unet import SimpleUnet
from dataloader import load_transformed_dataset
from diffusion import Diffusion


def main(args):
    diffusion = Diffusion(args)
    dataloader = load_transformed_dataset(args.img_size, args.batch_size, args.num_workers)
    print("Dataset loaded")
    model = SimpleUnet()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    model.to(args.device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    if not os.path.exists("models"):
        os.mkdir("models")
    models_path = "models"

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, args.T, (args.batch_size,), device=args.device).long()
            loss = diffusion.get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_time = (args.epochs - epoch - 1) * epoch_duration
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 1 == 0:  # for logging
            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                f"Time per epoch: {epoch_duration:.2f}seconds | "
                f"Remaining time: {remaining_time/3600:.2f}hours | "
                f"Current LR: {current_lr:.5f}"
            )
            diffusion.sample_plot_image(model, epoch)

        if epoch % 20 == 0:  # for saving models
            torch.save(model.state_dict(), os.path.join(models_path, str(epoch) + ".pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    
    print("Experiment Hyperparameters:")
    print('-' * 20)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('-' * 20)
    main(args)
    