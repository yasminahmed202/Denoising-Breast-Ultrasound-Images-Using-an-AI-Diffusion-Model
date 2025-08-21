import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def mse_score(pred, target):
    return F.mse_loss(pred, target).item()

def rmse_score(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()

def psnr_score(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return psnr(target_np, pred_np, data_range=1.0)

def ssim_score(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(target_np, pred_np, data_range=1.0)

