import argparse
import logging
import random
import os
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn.functional as F

from pipelines import StableDiffusionSDPipeline
from datasets import load_dataset


from accelerate import Accelerator
import utils.util as util


def main(args):
    
    random.seed(42)
    save_dir = os.path.join(args.save_dir, 'png')

    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load datasets and create dataloaders
    dataset = load_dataset(args.dataset, args.datadir, split="generate", resolution=args.resolution)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             drop_last=False)

    num_classes = dataset.num_classes

    # Create word dict
    idx2folder = dict(zip(dataset.class_to_idx.values(), dataset.class_to_idx.keys()))
    print(idx2folder)
    folder2class = util.read_classnames('./utils/classnames.txt')
    word_dict = util.map_idx2class(idx2folder, folder2class)
    print(word_dict)
    

    # Load model
    model_id = args.pretrained
    pipe = StableDiffusionSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Prepare model and dataloader for DDP setting
    pipe.to(accelerator.device)
    dataloader = accelerator.prepare(dataloader)

    
    # Generate SD-outlier
    for batch in tqdm(dataloader):
        x, y, file_names = batch

        # SD sampling
        generated_images = pipe("a photo of a cylindrical or oval shape with various shape wings and jet engines", 
                                inputs=x, 
                                output_type='pil',
                                num_inference_steps=args.ddim_steps, 
                                num_early_stop_steps=args.stop_steps, 
                                guidance_scale=args.w).images

        util.save_images(generated_images, file_names, save_dir)
        
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet200")
    parser.add_argument("--datadir", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=768)

    parser.add_argument("--pretrained", type=str, default="stabilityai/stable-diffusion-2-1") 

    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--stop_steps", type=int, default=20)
    parser.add_argument("--w", type=int, default=7.5)

    parser.add_argument("--save_dir", type=str, default="./temp")


    args = parser.parse_args()
    main(args)