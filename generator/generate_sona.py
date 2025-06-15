
import argparse
import logging
import random
import json
import os
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn.functional as F

from pipelines import SONAPipeline
from datasets import load_dataset
import utils.util as util
from utils.prompt_utils import (
    get_ood_list,
    select_random_word2prompt,
)

from accelerate import Accelerator



def main(args):
    random.seed(args.new_seed)
    
    save_dir = args.save_dir
    # Initialize Accelerator
    accelerator = Accelerator()

    # Load datasets and create dataloaders
    dataset = load_dataset(args.dataset, args.datadir, split="generate", resolution=args.resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )

    # Load model
    model_id = args.pretrained
    pipe = SONAPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Prepare model and dataloader for DDP setting
    pipe.to(accelerator.device)
    dataloader = accelerator.prepare(dataloader)

    ood_word_list, idx2class = get_ood_list(dataset)
    time_list = list(range(args.ddim_steps))
    for batch in tqdm(dataloader):
        x, y, file_names = batch

        editing_prompts = select_random_word2prompt(y, idx2class, file_names, ood_word_list)
        
        stop_steps = random.choice(time_list)
        generated_images = pipe(inputs=x,
                                batch_size=x.shape[0],
                                output_type='pil',
                                num_inference_steps=args.ddim_steps,
                                num_early_stop_steps=stop_steps,
                                guidance_scale=args.w,
                                editing_prompt=editing_prompts,
                                reverse_editing_direction=[False, True, False], # True : (-), False : (+)
                                semantic_direction=[False, True, True], # True : semantic, False : nuisance
                                edit_warmup_steps=[20,10,10],
                                gradual_type = None,
                                edit_guidance_scale=[10, 10, 10],
                                edit_threshold=[0.1, 0.9, 0.9],
                                edit_momentum_scale=0.3,
                                edit_mom_beta=0.6,
                                edit_weights=[1,1,1]).images


        util.save_images(generated_images, file_names, save_dir)
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

        break


if __name__ == "__main__":
    data_root = '/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--datadir", type=str, default=os.path.join(data_root, 'train'))
    parser.add_argument("--new_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--pretrained", type=str, default="stabilityai/stable-diffusion-2-base")

    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--stop_steps", type=int, default=20)
    parser.add_argument("--w", type=int, default=7.5)

    parser.add_argument("--save_dir", type=str, default='samples')

    args = parser.parse_args()
    main(args)