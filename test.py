import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusion import SimpleUnet
from utils import Forward

model = SimpleUnet()
model.to("cuda")
model.load_state_dict(torch.load("models/4900.pth"))   # 加载模型文件
model.eval()
diffusion = Forward()

diffusion.sample_plot_image(model)
print("Test complete")
