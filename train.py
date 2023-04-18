import os
import time
import argparse
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from Unet import SimpleUnet
from dataloader import load_transformed_dataset, load_FaceDataset
from diffusion import Diffusion


def main(args):
    diffusion = Diffusion(args)
    # dataloader = load_FaceDataset(args.dataset_folder, args.img_size, args.batch_size, args.num_workers)
    dataloader = load_transformed_dataset(args.dataset_folder, args.img_size, args.batch_size, args.num_workers)
    print("Dataset loaded")
    device = args.device
    
    # Activation fuction
    if args.activation == "relu":
        activation_function = nn.ReLU()
    elif args.activation == "silu":
        activation_function = nn.SiLU()
    model = SimpleUnet(activation_function, args.use_self_attention)
    optimizer = Adam(model.parameters(), lr=0.001)
    # Setup Cosine scheduler for LR
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)
    else:
        scheduler = None

    # If resume the training
    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        args.start_epoch = 0

    model.to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    # print(model)
    
    if not os.path.exists("models"):
        os.mkdir("models")
    models_path = "models"

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # print(batch[0].shape)
            t = torch.randint(0, args.T, (args.batch_size,), device=device).long()
            loss = diffusion.get_loss(model, batch, t)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, os.path.join(models_path, str(epoch) + ".pth"))
        
        if scheduler:
            scheduler.step()  # Update the learning rate after each epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'], help='Beta schedule to use: "linear" or "cosine"')
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "silu"], help="Activation function to use in the U-Net model (default: relu)")
    parser.add_argument('--use_self_attention', action='store_true', help="Enable self-attention in the U-Net model")

    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['cosine'], help='Learning rate scheduler to use: "cosine"')
    parser.add_argument('--resume_from', type=str, default=None, help="Resume training from a saved model")
    args = parser.parse_args()
    
    print("Experiment Hyperparameters:")
    print('-' * 20)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('-' * 20)
    main(args)
    