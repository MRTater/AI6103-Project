import math
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms 
import numpy as np
from pathlib import Path


class Diffusion():
    def __init__(self, args) -> None:
        # Define beta schedule
        self.args = args
        # self.betas = self.linear_beta_schedule(timesteps=self.args.T)
        if self.args.beta_schedule == 'linear':
            self.betas = self.linear_beta_schedule(timesteps=self.args.T)
        elif self.args.beta_schedule == 'cosine':
            print("cosine beta schedule invoked")
            self.betas = self.cosine_beta_schedule(timesteps=self.args.T)
        else:
            raise ValueError("Invalid beta_schedule. Choose 'linear' or 'cosine'.")

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def cosine_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        # Calculate t in the range of [0, pi/2] for each timestep
        t = torch.linspace(0, math.pi / 2, timesteps)

        # Calculate the cosine beta schedule using the cosine function
        betas = (end - start) * (1 - torch.cos(t)) + start
        return betas

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cuda"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape) 
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        # print(sqrt_alphas_cumprod_t.shape) # 128, 1, 1
        # print(x_0.shape) # 3, 64, 64
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        result = mean + variance
        result = torch.clamp(result, -1.0, 1.0)  # Clamp the values
        return result, noise.to(device)

    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def sample_timestep(self, model, x, t):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample_plot_image(self, model, epoch="sample", train="train"):
        # Sample noise
        img_size = self.args.img_size
        img = torch.randn((1, 3, img_size, img_size), device="cuda")
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.args.T/num_images)

        for i in range(0,self.args.T)[::-1]:
            t = torch.full((1,), i, device="cuda", dtype=torch.long)
            img = self.sample_timestep(model, img, t)
            if i % stepsize == 0:
                plt.subplot(1, num_images, math.floor(i/stepsize+1))
                show_tensor_image(img.detach().cpu())
        img_folder = os.path.join("result", train) + "/"
        if not os.path.exists(img_folder):
            Path(img_folder).mkdir(parents=True, exist_ok=True)
            # os.makedirs(img_folder)
        plt.savefig(os.path.join(img_folder, str(epoch)+".png"))

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

