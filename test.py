import torch
import matplotlib.pyplot as plt
import numpy as np
from Unet import SimpleUnet
from diffusion import Diffusion

model = SimpleUnet()
model.to("cuda")
model.load_state_dict(torch.load("models/4900.pth"))   # 加载模型文件
model.eval()
diffusion = Diffusion()

diffusion.sample_plot_image(model, train="test")
print("Test complete")
