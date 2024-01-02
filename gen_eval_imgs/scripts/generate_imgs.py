import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import json
import torch 
import argparse

import random
from PIL import Image
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


class CustomDataset(Dataset):
    def __init__(self, file_path):
        # 读取json文件
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本数据
        sample = self.data[idx]

        # 提取文本
        sample_id =sample['id']
        sample_text = sample['text']

    
        return {"id": sample_id,
                "text": sample_text
            }

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# 4. schedule
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

device = "cuda:2"

vae.to(device)
text_encoder.to(device)
unet.to(device)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# img_save_dir = "/media/sdb/liutao/datasets/rm_images/imgs"



def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 创建成功")
    else:
        print(f"文件夹 {folder_path} 已经存在") 


def extract_index_dir(image_path):
    index = image_path.split('/')[-1]
    return int(index)


def gen_from_last(image_dir, pattern = "seed42", nums = 10):
    image_index_dirs = sorted(os.listdir(image_dir), key=lambda x: int(x))
    image_index_dirs = [os.path.join(image_dir, img_dir) for img_dir in image_index_dirs]

    for image_dir in image_index_dirs:
        images = os.listdir(image_dir)
        pattern_count = 0

        for image_name in images:
            if pattern in image_name:
                pattern_count += 1
        
        if pattern_count < nums:
            print(f"从 {image_dir} 开始下载……")
            break

    # print(extract_index_dir(image_dir))
    return extract_index_dir(image_dir)
                
    

    



def run_inference(g, prompts, num_images_per_prompt, device):

    batch_size = len(prompts)
    # 1. get input_ids
    text_inputs = tokenizer(
        prompts, 
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    guidance_scale = 7.5
    do_classifier_free_guidance = guidance_scale > 1.0
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask.to(device)

    # 2.get prompt embedding
    prompt_embeds = text_encoder(
                text_input_ids.to(device),
                # attention_mask=None,
            )
    prompt_embeds = prompt_embeds[0]
    # [2, 77, 768]
    bs_embed, seq_len, _ = prompt_embeds.shape
    # print(prompt_embeds.shape) 
    # [2, 154, 768]
    prompt_embeds = prompt_embeds.repeat(1,num_images_per_prompt, 1)

    # [4, 77, 768]
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # do_classifier_free_guidance
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
    
    attention_mask = uncond_input.attention_mask.to(device)
    negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                # attention_mask=None,
            )
    negative_prompt_embeds = negative_prompt_embeds[0]
    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    # to avoid doing two forward passes
    # [8, 77, 768]
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = unet.config.in_channels
    shape = (batch_size * num_images_per_prompt, num_channels_latents, 64, 64)
    latents = torch.randn(shape, generator = g, device = device)
   
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - 50 * scheduler.order

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # (1, 512, 512, 3)
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    

    return pil_images




def generate_images(args):
   
    # 2. load prompt dataset

    # 创建自定义数据集
    dataset = CustomDataset(args.prompt_json_path)
    
    # 创建dataloader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 使用 len 函数获取 dataloader 的长度
    dataloader_length = len(data_loader)

    print("Dataloader的长度:", dataloader_length,"\t batch_size:", args.batch_size)

    # print(data_loader)

    # 3. load model
    
    model = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16)
    model.to(device)

    # append judge：generate from last
    pattern = f"seed{args.seed}"
    step_start = gen_from_last(args.img_save_dir, pattern=pattern, nums = 10)
    
    progress_bar = tqdm(range(step_start, dataloader_length))
    progress_bar.set_description("Generating Images……")

    # 4. aplly  prompt pair (complex-simple) to generate images
    json_data = []
    for step ,batch in enumerate(data_loader):
        # gen from last
        if step + 1 < step_start:
            continue
        
        ids = batch["id"]
        prompts = batch["text"]

        # print(prompts)
        # # b. generate images
        # 返回的是存储图像地址的 list
        g = torch.Generator(device=device).manual_seed(args.seed)
        images = run_inference(g, prompts, args.num_images_per_prompt, device)
        # return image_path_list
        ids = [id.item() for id in ids for _ in range(args.num_images_per_prompt)]  # [id] * num_images_per_prompt

        # print(ids)
        generations = []
        prompt_dir = f"{args.img_save_dir}/{ids[0]}/"
        create_folder_if_not_exists(prompt_dir)

        for i, image in enumerate(images):
            image_idx = i % args.num_images_per_prompt
            image_name = f"id{ids[0]}_seed{args.seed}_{image_idx}.png"
            save_img_path = prompt_dir + image_name

            generations.append(save_img_path)
            image.save(save_img_path)
        
        # 写入 entry
        entry = {}
        entry["ids"] = ids[0]
        entry["text"] = prompts[0]
        entry["generations"] = generations
        json_data.append(entry)

        if (step+1) % 2000 == 0:
            # 将选择的数据写入新的JSON文件
            save_json_path = f"/media/sdb/liutao/datasets/rm_images/seed{args.seed}_data_{step+1}.json"
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        progress_bar.update(1)
        
        
        # print(images)
        # break
            




if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/khf/liutao/sd1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inferencing.")
    parser.add_argument(
        "--prompt_json_path",
        default="/media/sdb/liutao/refl_base/ImageReward/data/refl_data_v2.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--img_save_dir",
        default="/media/sdb/liutao/datasets/rm_images/images",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be the name of the model and should correspond to the name specified in `model`.",
    )
    parser.add_argument(
        "--model",
        default="default",
        type=str,
        help="""default(["sd1-4", "sd2-1"]), all or any specified model names splitted with comma(,).""",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (per device) for the prompt dataloader.",
    )
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        default=10,
        type=int,
        help="Num of images generated for each prompt.",
    )


    args = parser.parse_args()

    create_folder_if_not_exists(args.img_save_dir)


    generate_images(args)
    # gen_from_last(args.img_save_dir)
    # extract_index_dir("/media/sdb/liutao/datasets/rm_images/images/6749")
            

