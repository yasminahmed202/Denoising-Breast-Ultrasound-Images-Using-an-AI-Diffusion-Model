"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import pandas as pd
import time
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from LIDCLoader import load_LIDC

# Reproducibility
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    return (img - _min) / (_max - _min)

def show_tensor_images(image, mask, output, num, title=None):
    if output is None:
        print(f"Skipping visualization — output is None for {title}")
        return

    to_pil = transforms.ToPILImage()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(to_pil(image.squeeze().cpu()), cmap="gray")
    axes[0].set_title('Image', fontsize=10)
    
    axes[1].imshow(to_pil(mask.squeeze().cpu()), cmap="gray")
    axes[1].set_title('Ground Truth', fontsize=10)

    axes[2].imshow(to_pil(output), cmap="gray")
    axes[2].set_title('Output', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.savefig(f'./output_images/Output_{num}.png')
    plt.close()

def main():
    args = create_argparser().parse_args()
    logger.configure()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location=device))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    os.makedirs('./output', exist_ok=True)
    os.makedirs('./output_images', exist_ok=True)
    os.makedirs('./output_metrics', exist_ok=True)

    # Load test data
    ds = load_LIDC(image_size=224, combine_train_val=True, mode='Train')
    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    data = iter(datal)
    print(f" Loaded dataset length: {len(ds)}")

    # Prepare output dataframe
    df = pd.DataFrame(columns=['title', 'mse', 'rmse', 'psnr', 'ssim'])
    cnt = 1

    while True:
        try:
            b, mask, image_path, mask_path = next(data)
        except StopIteration:
            break

        b, mask = b.to(device), mask.to(device)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1).to(device)

        slice_ID = os.path.basename(image_path[0]).split(".")[0]
        title = os.path.basename(mask_path[0]).split(".")[0]

        logger.log(f" Sampling {slice_ID}...")
        sys.stdout.flush()
        start_time = time.time()

        tensor_list = []

        try:
            with th.no_grad():
                for _ in range(args.num_ensemble):
                    model_kwargs = {}
                    sample_fn = (
                        diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                    )

                    sample, x_noisy, org = sample_fn(
                        model,
                        (args.batch_size, 3, args.image_size, args.image_size),
                        img,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )
                    s = sample.clone().detach()
                    tensor_list.append(s.squeeze().cpu())
        except Exception as e:
            print(f" Error during sampling {slice_ID}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Find the best sample with lowest MSE
        best_mse = float('inf')
        best_output = None
        best_metrics = {}

        for s in tensor_list:
            s_np = np.clip(s.numpy(), 0, 1)
            mask_np = np.clip(mask.squeeze().cpu().numpy(), 0, 1)

            mse_val = np.mean((s_np - mask_np) ** 2)
            if mse_val < best_mse:
                best_mse = mse_val
                best_output = s

                rmse_val = np.sqrt(mse_val)
                psnr_val = compare_psnr(mask_np, s_np, data_range=1.0)
                try:
                    ssim_val = compare_ssim(mask_np, s_np, data_range=1.0)
                except ValueError:
                    print(f"SSIM calculation failed for {slice_ID}: {e}")
                    ssim_val = float('nan')

                best_metrics = {
                    'mse': mse_val,
                    'rmse': rmse_val,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                }
            
        if best_output is None:
            print(f"Skipping {slice_ID} — no valid outputs.")
            continue

        df = pd.concat([df, pd.DataFrame([[
            title,
            best_metrics['mse'],
            best_metrics['rmse'],
            best_metrics['psnr'],
            best_metrics['ssim']
        ]], columns=['title', 'mse', 'rmse', 'psnr', 'ssim'])], ignore_index=True)

        row = pd.DataFrame([[
            title,
            best_metrics['mse'],
            best_metrics['rmse'],
            best_metrics['psnr'],
            best_metrics['ssim']
        ]], columns=['title', 'mse', 'rmse', 'psnr', 'ssim'])

        row.to_csv('./output_metrics/evaluation_metrics.csv', mode='a', index=False, header=(cnt == 1))

        show_tensor_images(b, mask, best_output, cnt, title=slice_ID)
        th.save(best_output, f'./output/{slice_ID}_output.pt')
        
        print(f" Sample {cnt} ({slice_ID}) done in {time.time() - start_time:.2f}s | MSE: {best_mse:.4f}")
        sys.stdout.flush()
        cnt += 1

def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
