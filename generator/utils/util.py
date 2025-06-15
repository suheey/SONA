import argparse
import logging
import random
import os
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn.functional as F

random.seed(42)

IMAGENET_PROMPT_TEMPLATES = [
    "not a photo of a {}",
]

def select_random_time_step(time_list, rank):
    seed = torch.randint(0, 1000, (1,)).item() * rank
    random.seed(seed)   
    return random.choice(time_list)

def select_random_word2prompt(y_batch, word_list, rank):
    seed = torch.randint(0, 1000, (1,)).item() * rank
    random.seed(seed)    
    prompts = list()
    for y in y_batch:
        prompt = random.choice(IMAGENET_PROMPT_TEMPLATES).format(random.choice(word_list))
        prompts.append(prompt)
    return prompts

def select_random_label_rank(y_batch, num_classes, rank):
    seed = torch.randint(0, 1000, (1,)).item() * rank
    random.seed(seed)
    return torch.tensor([random.choice([c for c in range(num_classes) if c != y.item()]) for y in y_batch])

def select_random_label(y_batch, num_classes): 
    return torch.tensor([random.choice([c for c in range(num_classes) if c !=y.item()]) for y in y_batch])

def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames

def map_idx2class(idx2folder, folder2class):
    output = {}
    for idx, folder in idx2folder.items():
        if folder in folder2class:
            output[idx] = folder2class[folder]
    return output

def find_ood_words(source_dict, target_dict):
    missing_items = []
    for key, value in source_dict.items():
        if key not in target_dict.values():
            missing_items.append(value)
    return missing_items


def word2prompt(y_batch, word_dict):
    prompts = list()
    for y in y_batch:

        prompt = random.choice(IMAGENET_PROMPT_TEMPLATES).format(word_dict[str(y.item())])
        prompts.append(prompt)
    return prompts

def desc2prompt(y_batch, desc_dict):
    prompts = list()
    for y in y_batch:
        prompts.append(desc_dict[str(y.item())])
    return prompts

def save_images(image_list, path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(image_list):
        data_path = os.path.join(output_dir, os.path.basename(os.path.dirname(path[i])))
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        file_name, _ = os.path.splitext(os.path.basename(path[i]))
        new_file_name = file_name + '.png'     
        image_path = os.path.join(data_path, new_file_name)
        image.save(image_path)

# usage: save_dir_jpeg = os.path.join(args.save_dir, 'jpeg')
def save_images_jpeg(image_list, path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(image_list):
        data_path = os.path.join(output_dir, os.path.basename(os.path.dirname(path[i])))
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
   
        image_path = os.path.join(data_path, f"{os.path.basename(path[i])}")
        image.save(image_path)
