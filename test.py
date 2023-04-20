import argparse
import torch
from Unet import SimpleUnet
from diffusion import Diffusion
from torch.optim import Adam

def main(args):
    diffusion = Diffusion(args)
    model = SimpleUnet(args.activation, args.use_self_attention, args.use_skip_connection)
    optimizer = Adam(model.parameters(), lr=0.001)

    model.to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Model loaded")
    print(model)

    diffusion.sample_plot_image(model, train="test")
    print("Test complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "silu"], help="Activation function to use in the U-Net model (default: relu)")
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'], help='Beta schedule to use: "linear" or "cosine"')
    parser.add_argument('--use_self_attention', action='store_true', help="Enable self-attention in the U-Net model")
    parser.add_argument('--use_skip_connection', action='store_true', help="Enable skip-connection blockwise")
    parser.add_argument('--model_path', type=str, default=None, help="The model used to inference")
    args = parser.parse_args()
    
    print("Inference Hyperparameters:")
    print('-' * 20)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('-' * 20)
    main(args)