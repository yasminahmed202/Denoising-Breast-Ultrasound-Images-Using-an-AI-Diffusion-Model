import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.signal import wiener
from scipy.ndimage import median_filter
import csv


def gaussian_blur(img, ksize=5):
    # Apply Gaussian blur with kernel size ksize x ksize
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)
    return blurred


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    # Apply bilateral filter - edge preserving smoothing
    # Note: cv2.bilateralFilter expects uint8 images
    img_uint8 = np.uint8(np.clip(img * 255, 0, 255))
    filtered = cv2.bilateralFilter(img_uint8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return filtered.astype(np.float32) / 255.0


def median_filter_denoise(img, size=3):
    # Median filter is often better for impulse noise, but included here for completeness
    return median_filter(img, size=size)


def nl_means_denoise(img, patch_size=7, patch_distance=11, h=0.1):
    sigma_est = np.mean(estimate_sigma(img, channel_axis=None))
    denoised = denoise_nl_means(img,
                                h=h * sigma_est,
                                patch_size=patch_size,
                                patch_distance=patch_distance,
                                fast_mode=True,
                                channel_axis=None)
    return denoised


def wiener_filter(img, mysize=(5, 5)):
    # Wiener filter (requires grayscale float image)
    denoised = wiener(img, mysize=mysize)
    # Wiener filter can produce values outside [0,1], clip them
    return np.clip(denoised, 0, 1)


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def rmse(img1, img2):
    return np.sqrt(mse(img1, img2))


def evaluate_metrics(denoised, reference):
    mse_val = mse(denoised, reference)
    rmse_val = rmse(denoised, reference)
    psnr_val = psnr(reference, denoised, data_range=1.0)
    ssim_val = ssim(reference, denoised, data_range=1.0)
    return mse_val, rmse_val, psnr_val, ssim_val


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


def load_and_resize(image_path, size=(512, 512)):
    img = Image.open(image_path).convert('L')
    img = img.resize(size, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    return img


if __name__ == "__main__":

    meta_path ="/data/home/bt24001/new_model/data_Image/meta.csv"
    meta_df = pd.read_csv(meta_path)
    test_files = set(os.path.basename(path) for path in meta_df[meta_df['data_split'] == 'Test']['original_image'])
    print(f"Number of test files: {len(test_files)}")

    noisy_dir = '/data/home/bt24001/new_model/data_Image/Mask'
    clean_dir = '/data/home/bt24001/new_model/data_Image/Images'

    save_folder = 'denoised_results_gaussian'
    os.makedirs(save_folder, exist_ok=True)

    metrics_file = os.path.join(save_folder, 'gaussian_noise_metrics.csv')
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Method', 'MSE', 'RMSE', 'PSNR', 'SSIM'])

        for filename in os.listdir(noisy_dir):
            #noisy_path = os.path.join(noisy_dir, filename)
            #clean_path = os.path.join(clean_dir, filename)
            if filename not in test_files:
                continue
            print(f"Processing test file: {filename}")

            noisy_path = os.path.join(noisy_dir, filename)
            clean_path = os.path.join(clean_dir, filename)  # Assumes same filename for clean image

            noisy_img = load_and_resize(noisy_path)
            clean_img = load_and_resize(clean_path)

            print(filename)
            print("  Exact equality:", np.array_equal(noisy_img, clean_img))
            print("  Max abs diff:", np.max(np.abs(noisy_img - clean_img)))


            # Apply Gaussian noise filters
            gauss_img = gaussian_blur(noisy_img, ksize=5)
            bilateral_img = bilateral_filter(noisy_img, d=9, sigma_color=75, sigma_space=75)
            median_img = median_filter_denoise(noisy_img, size=3)
            nlm_img = nl_means_denoise(noisy_img, h=0.1)
            wiener_img = wiener_filter(noisy_img, mysize=(5, 5))

            # Save denoised images
            base_name = os.path.splitext(filename)[0]
            save_image(gauss_img, os.path.join(save_folder, f'{base_name}_gaussian_blur.png'))
            save_image(bilateral_img, os.path.join(save_folder, f'{base_name}_bilateral.png'))
            save_image(median_img, os.path.join(save_folder, f'{base_name}_median.png'))
            save_image(nlm_img, os.path.join(save_folder, f'{base_name}_nlm.png'))
            save_image(wiener_img, os.path.join(save_folder, f'{base_name}_wiener.png'))

            # Evaluate metrics
            gauss_metrics = evaluate_metrics(gauss_img, clean_img)
            bilateral_metrics = evaluate_metrics(bilateral_img, clean_img)
            median_metrics = evaluate_metrics(median_img, clean_img)
            nlm_metrics = evaluate_metrics(nlm_img, clean_img)
            wiener_metrics = evaluate_metrics(wiener_img, clean_img)

            print(f"Metrics for {filename} (MSE, RMSE, PSNR, SSIM):")
            print(f"  Gaussian Blur: MSE={gauss_metrics[0]:.6f}, RMSE={gauss_metrics[1]:.6f}, PSNR={gauss_metrics[2]:.2f}, SSIM={gauss_metrics[3]:.4f}")
            print(f"  Bilateral:     MSE={bilateral_metrics[0]:.6f}, RMSE={bilateral_metrics[1]:.6f}, PSNR={bilateral_metrics[2]:.2f}, SSIM={bilateral_metrics[3]:.4f}")
            print(f"  Median Filter: MSE={median_metrics[0]:.6f}, RMSE={median_metrics[1]:.6f}, PSNR={median_metrics[2]:.2f}, SSIM={median_metrics[3]:.4f}")
            print(f"  NLM Filter:    MSE={nlm_metrics[0]:.6f}, RMSE={nlm_metrics[1]:.6f}, PSNR={nlm_metrics[2]:.2f}, SSIM={nlm_metrics[3]:.4f}")
            print(f"  Wiener Filter: MSE={wiener_metrics[0]:.6f}, RMSE={wiener_metrics[1]:.6f}, PSNR={wiener_metrics[2]:.2f}, SSIM={wiener_metrics[3]:.4f}")

            # Write metrics to CSV
            writer.writerow([filename, 'Gaussian Blur', *gauss_metrics])
            writer.writerow([filename, 'Bilateral Filter', *bilateral_metrics])
            writer.writerow([filename, 'Median Filter', *median_metrics])
            writer.writerow([filename, 'NLM Filter', *nlm_metrics])
            writer.writerow([filename, 'Wiener Filter', *wiener_metrics])

    print(f"\nSaved all Gaussian noise filter evaluation metrics to {metrics_file}")
