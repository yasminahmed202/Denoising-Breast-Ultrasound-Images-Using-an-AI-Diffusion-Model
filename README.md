# Denoising-Breast-Ultrasound-Images-Using-an-AI-Diffusion-Model

This repository is created for my Masters Dissertation. This code is a denoising diffusion probabilistic model to denoise Breast Ultrasound images.

# Data
The data is organised as Mask and Image file. The Image file contains clean images and the Mask file containing noisy imgaes of the clean image. The data is not uploaded but can be provided when requested at yasminahmed202@gmail.com. 

MODEL_FLAGS="--num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
SAMPLE_FLAGS="--data_dir ./data/testing --model_path ./results/emasavedmodel_0.9999_100000.pt --num_ensemble 5 --batch_size 1 --num_samples 10"
