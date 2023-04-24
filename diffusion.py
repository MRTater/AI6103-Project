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
        if self.args.beta_schedule == 'cosine':
            self.beta = self.beta_schedule_linear(time_steps=self.args.T)
        elif self.args.beta_schedule == 'linear':
            self.beta = self.beta_schedule_cos(time_steps=self.args.T)
        else:
            raise ValueError("Wrong beta_schedule parameter. Only 'linear' or 'cosine'.")
        
        # print("betas:")
        # print(self.betas)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.beta
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.posterior_variance = self.beta * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def beta_schedule_linear(self, time_steps, start=0.0001, end=0.02):
        return torch.linspace(start, end, time_steps)

    def beta_schedule_cos(self, time_steps, start=0.0001, end=0.02):
        # Calculate t in the range of [0, pi/2] for each timestep
        # t = torch.linspace(0, math.pi / 2, timesteps)

        # Calculate the cosine beta schedule using the cosine function
        # betas = (end - start) * (1 - torch.cos(t)) + start
        # return betas

        max_beta = 0.999
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(time_steps):
            t1 = i / time_steps
            t2 = (i + 1) / time_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.Tensor(betas)

    def get_index(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_process_sample(self, x_0, t, device="cuda"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_one_minus_alphas_cumprod_t = self.get_index(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        sqrt_alphas_cumprod_t = self.get_index(self.sqrt_alphas_cumprod, t, x_0.shape) 
        # mean + variance
        # print(sqrt_alphas_cumprod_t.shape) # 128, 1, 1
        # print(x_0.shape) # 3, 64, 64
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        outcome = variance + mean
        outcome = torch.clamp(outcome, -1.0, 1.0)  # Clamp the values
        return outcome, noise.to(device)


    def get_loss(self, model, x_0, t):
        noise_x, noise = self.forward_process_sample(x_0, t)
        noise_pred = model(noise_x, t)
        return F.l1_loss(noise, noise_pred)
    
    # L2 loss can lead to smoother denoising results, as it tends to penalize large deviations more heavily.
    # def get_loss(self, model, x_0, t):
    #     x_noisy, noise = self.forward_diffusion_sample(x_0, t)
    #     noise_pred = model(x_noisy, t)
    #     return F.mse_loss(noise, noise_pred)

    @torch.no_grad()
    def sample_timestep(self, model, x, t, index):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        sqrt_recip_alphas_t = self.get_index(self.sqrt_recip_alphas, t, x.shape)
        # Call model (current image - noise prediction)
        pred_noise = model(x, t)

        model_mean = sqrt_recip_alphas_t * (
            x - (self.get_index(self.beta, t, x.shape)) * pred_noise / (self.get_index(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        ))
        )

        variance = self.get_index(self.posterior_variance, t, x.shape)
        
        if index != 0:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(variance) * noise 
        else:
            return model_mean
            
        
    @torch.no_grad()
    def plot_sample_images(self, model, train="train", epoch="sample"):
        # Sample noise
        size = self.args.img_size
        image = torch.randn((1, 3, size, size), device=self.args.device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        divisor = 10
        step_size = int(self.args.T/divisor)

        for step in range(0,self.args.T)[::-1]:
            t = torch.full((1,), step, device=self.args.device, dtype=torch.long)
            image = self.sample_timestep(model, image, t, step)
            if step % step_size == 0:
                plt.subplot(1, divisor, math.floor(step/step_size+1))
                show_image(image.detach().cpu())
        save_dir = os.path.join("result", train) + "/"
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_dir, str(epoch)+".png"))
        plt.close()

def show_image(image):
    transforms = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda x: x * 255.),
        transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    image = transforms(image)
    plt.imshow(image)


