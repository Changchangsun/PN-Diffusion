# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Optional
from thop import profile
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.audio_diffusion import Mel
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm

from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline
from torchsummary import summary
import inspect
# import pdb
import torch.nn as nn
logger = get_logger(__name__)


def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    print("accelerator")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(
                args.dataset_name,
                storage_options=args.dataset_config_name)["train"]
                # storage_options=args.dataset_config_name)["test"]

        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
    if args.dataset_name_test is not None:
        if os.path.exists(args.dataset_name_test):
            dataset_test = load_from_disk(
                args.dataset_name_test,
                storage_options=args.dataset_config_name)["test"]
    # Determine image resolution
    resolution = dataset[0]["image"].height, dataset[0]["image"].width

    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])

    def transforms(examples):
        if args.vae is not None and vqvae.config["in_channels"] == 3:
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
        else:
            images = [augmentations(image) for image in examples["image"]]###
        if args.encodings is not None:
            
            encoding = [encodings[file.split(".")[0].split("/")[-1]] for file in examples["audio_file"]]
            encoding_neg = [encodings_neg[file.split(".")[0].split("/")[-1]] for file in examples["audio_file"]]
            encoding_motion = [encodings_motion[file.split(".")[0].split("/")[-1]] for file in examples["audio_file"]]
            encoding_motion_neg = [encodings_motion_neg[file.split(".")[0].split("/")[-1]] for file in examples["audio_file"]]

            return {"input": images, "encoding": encoding,"encoding_motion":encoding_motion,"encoding_neg":encoding_neg,"encoding_motion_neg":encoding_motion_neg}
        return {"input": images}
    
    def transforms_test(examples):
        if args.vae is not None and vqvae.config["in_channels"] == 3:
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
        else:
            images = [augmentations(image) for image in examples["image"]]###
        if args.encodings_test is not None:
            
            encoding_test = [encodings_test[file.split(".")[0].split("/")[-1]] for file in examples["audio_file"]]
            encoding_motion_test = [encodings_motion_test[file.split(".")[0].split("/")[-1]] for file in examples["audio_file"]]

            return {"input": images, "encoding": encoding_test,"encoding_motion":encoding_motion_test}
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)
    
    
    dataset_test.set_transform(transforms_test)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.train_batch_size, shuffle=True)

    if args.encodings is not None:
        encodings = {}
        encodings_neg = {}
        files = os.listdir(args.encodings)
        for file in files:
            name = file.split(".")[0]
            feas = np.load(os.path.join(args.encodings,name+".npy"))
            encodings[name] = feas.astype(np.float32)
            feas_neg = np.load(os.path.join(args.encodings_negative,name+".npy"))
            encodings_neg[name] = feas_neg.astype(np.float32)

    if args.encodings_test is not None:
        encodings_test = {}
        encodings_test_neg = {}
        files = os.listdir(args.encodings_test)
        for file in files:
            name = file.split(".")[0]
            feas = np.load(os.path.join(args.encodings_test,name+".npy"))
            encodings_test[name] = feas.astype(np.float32)
            feas_neg_test = np.load(os.path.join(args.encodings_negative_test,name+".npy"))
            encodings_test_neg[name] = feas_neg_test.astype(np.float32)


    if args.encodings_motion is not None:
        encodings_motion = {}
        encodings_motion_neg = {}
        files = os.listdir(args.encodings_motion)
        for file in files:
            name = file.split(".")[0]
            feas = np.load(os.path.join(args.encodings_motion,name+".npy"))
            encodings_motion[name] = feas.astype(np.float32)
            r_e_m = np.flip(feas, axis=0)
            encodings_motion_neg[name] = r_e_m.copy().astype(np.float32)
        
    if args.encodings_motion_test is not None:
        encodings_motion_test = {}
        encodings_motion_test_neg = {}
        files = os.listdir(args.encodings_motion_test)
        for file in files:
            name = file.split(".")[0]
            feas = np.load(os.path.join(args.encodings_motion_test,name+".npy"))
            encodings_motion_test[name] = feas.astype(np.float32)
            r_e_m = np.flip(feas, axis=0)
            encodings_motion_test_neg[name] = r_e_m.copy().astype(np.float32)

        
    vqvae = None
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae) ####
        except EnvironmentError:
            vqvae = AudioDiffusionPipeline.from_pretrained(args.vae).vqvae
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = vqvae.encode(
                torch.zeros((1, 1) +
                            resolution)).latent_dist.sample().shape[2:]

    if args.from_pretrained is not None:
        pipeline = AudioDiffusionPipeline.from_pretrained(args.from_pretrained)
        mel = pipeline.mel
        model = pipeline.unet
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")

        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae

    else:
        if args.encodings is None:
            model = UNet2DModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

        else:
            model = UNet2DConditionModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                cross_attention_dim=3072, #2048+1024
                # encoder_hid_dim  = 1024,#only motion
                # encoder_hid_dim  = 2048,#only rgb
            )

    
    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps)
    total_params = sum(p.numel() for p in model.parameters())
        

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(output_dir).name,
                                           token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        repo = Repository(output_dir, clone_from=repo_name)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    mel = Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    unet = accelerator.unwrap_model(model)
    pipeline = AudioDiffusionPipeline(
                    vqvae=vqvae,
                    unet=unet,
                    mel=mel,
                    scheduler=noise_scheduler,
                )
    
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        all_loss = 0
        all_loss_neg = 0
        all_loss_com = 0

        value = 0.5
        margin = torch.tensor(value)

        for step, batch in enumerate(train_dataloader):
            
            clean_images = batch["input"]

            if vqvae is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(
                        clean_images).latent_dist.sample()
                clean_images = clean_images * 0.18215

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz, ),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)
            with accelerator.accumulate(model):
                
                if args.encodings is not None:
                    noise_output = model(noisy_images, timesteps, batch["encoding"],batch["encoding_motion"],batch["encoding_neg"],batch["encoding_motion_neg"])  ##########encodings
                    noise_pred, noise_pred_neg = noise_output[0]["sample"],noise_output[1]["sample"]
                else:
                    noise_pred = model(noisy_images, timesteps)["sample"]

            
                loss_1 = F.mse_loss(noise_pred, noise)
                loss_2  = F.mse_loss(noise_pred_neg, -noise)
                loss = args.alpha*loss_1+(1-args.alpha)*loss_2
                
                
                accelerator.backward(loss)
                all_loss +=loss

                if accelerator.sync_gradients: 
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "all_loss":all_loss.detach().item(),
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if ((epoch + 1) % args.save_model_epochs == 0
                    or (epoch + 1) % args.save_images_epochs == 0
                    or epoch == args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.copy_to(unet.parameters())
                pipeline = AudioDiffusionPipeline(
                    vqvae=vqvae,
                    unet=unet,
                    mel=mel,
                    scheduler=noise_scheduler,
                )

            if (
                    epoch + 1
            ) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
                pipeline.save_pretrained(save_path)
                # pipeline.save_pretrained(output_dir)

                # save the model
                if args.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Epoch {epoch}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            if (epoch + 1) % args.save_images_epochs == 0:
                generator = torch.Generator(
                    device=clean_images.device).manual_seed(42)

                if args.encodings_test is not None:
                    random.seed(42)
                    print("len(encodings_test)",len(encodings_test.keys()))#
                    all_indices = list(range(len(encodings_test)))

                    sampled_indices = random.sample(all_indices, args.eval_batch_size)
                    condition_keys = encodings_test.keys()
                    list_keys = [list(condition_keys)[i] for i in sampled_indices]  
                    for i in list_keys:
                        print(i)

                    selected_encoding = [list(encodings_test.values())[i] for i in sampled_indices]
                    selected_encoding_motion = [list(encodings_motion_test.values())[i] for i in sampled_indices]
                    selected_encoding_neg = [list(encodings_test_neg.values())[i] for i in sampled_indices]
                    selected_encoding_motion_neg = [list(encodings_motion_test_neg.values())[i] for i in sampled_indices]
                    encoding = torch.stack( 
                                           [torch.tensor(arr) for arr in
                        selected_encoding
                        ]
                        ).to(clean_images.device)
                    encoding_motion = torch.stack( 
                                           [torch.tensor(arr) for arr in
                        selected_encoding_motion
                        ]
                        ).to(clean_images.device)
                    
                    encoding_neg = torch.stack( 
                                           [torch.tensor(arr) for arr in
                        selected_encoding_neg
                        ]
                        ).to(clean_images.device)
                    encoding_motion_neg = torch.stack( 
                                           [torch.tensor(arr) for arr in
                        selected_encoding_motion_neg
                        ]
                        ).to(clean_images.device)
                        
                else:
                    encoding = None
                    encoding_motion  =None
    
                images, (sample_rate, audios) = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    return_dict=False,
                    encoding=encoding, 
                    encoding_motion=encoding_motion,
                    encoding_neg=encoding_neg, 
                    encoding_motion_neg=encoding_motion_neg,
                ) 
                images = np.array([
                    np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                        (len(image.getbands()), image.height, image.width))
                    for image in images
                ])
                accelerator.trackers[0].writer.add_images(
                    "test_samples", images, epoch)
                for _, audio in enumerate(audios):
                    accelerator.trackers[0].writer.add_audio(
                        f"test_audio_{_}",
                        normalize(audio),
                        epoch,
                        sample_rate=sample_rate,
                    )
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_name_test", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddpm",
                        help="ddpm or ddim")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to encoding",
    )
    parser.add_argument(
        "--encodings_negative",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to encoding",
    )
    parser.add_argument(
        "--encodings_test",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to encoding",
    )
    #####
    parser.add_argument(
        "--encodings_motion",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to motion encoding",
    )
    parser.add_argument(
        "--encodings_motion_test",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to motion encoding",
    )

    ##hyper-parameter
    parser.add_argument("--alpha", type=float, default=1)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )
    main(args)


"""
CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file config/accelerate_local.yaml scripts/train_unet_aist_negative_re.py 
--dataset_name /
--dataset_name_test /
--hop_length 512 
--output_dir / 
--train_batch_size 32 
--num_epochs 100 
--gradient_accumulation_steps 1 
--learning_rate 1e-4 
--lr_warmup_steps 500 
--mixed_precision no 
--vae /
--encodings /
--encodings_test /
--encodings_motion / 
--encodings_motion_test /

"""