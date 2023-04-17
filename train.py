import os
import time
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.optim import Adam
from Unet_old import SimpleUnet
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

    # model = SimpleUnet(activation_function, args.use_self_attention)
    model = SimpleUnet(activation_function)
    # print(model)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = args.epochs

    if not os.path.exists("models"):
        os.mkdir("models")
    models_path = "models"

    for epoch in range(epochs):
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
        remaining_time = (epochs - epoch - 1) * epoch_duration

        if epoch % 1 == 0:  # for debugging
            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                f"Time per epoch: {epoch_duration:.2f}seconds | "
                f"Remaining time: {remaining_time/3600:.2f}hours"
            )
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
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'], help='Beta schedule to use: "linear" or "cosine"')
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "silu"], help="Activation function to use in the U-Net model (default: relu)")
    parser.add_argument('--use_self_attention', action='store_true', help="Enable self-attention in the U-Net model")
    args = parser.parse_args()
    print("Experiment Hyperparameters:")
    print('-' * 20)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('-' * 20)
    main(args)
    