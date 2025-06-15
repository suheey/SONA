import os
import random
import hashlib

import utils.util as util

IMAGENET_PROMPT_TEMPLATES = [
    "a photo of a {}",
]


def get_seed_from_filename(filename: str) -> int:
    """Generate a deterministic seed from a filename."""
    hash_object = hashlib.sha256(filename.encode())
    hash_digest = hash_object.hexdigest()
    seed = int(hash_digest, 16) % (10**10)
    return seed


def get_ood_list(dataset):
    """Return a list of OOD words and id-to-class mapping for prompts."""
    idx2folder = {v: k for k, v in dataset.class_to_idx.items()}
    folder2class = util.read_classnames('./utils/classnames.txt')
    idx2class = {idx: folder2class[folder] for idx, folder in idx2folder.items()}
    ood_word_list = util.find_ood_words(folder2class, idx2folder)
    return ood_word_list, idx2class


def select_random_word2prompt(y_batch, id_word_dict, file_names, ood_word_list):
    """Return prompts for ID and OOD words using deterministic seeds."""
    extracted_names = [os.path.basename(name) for name in file_names]

    id_editing_prompts, ood_editing_prompts = [], []
    for idx, y in enumerate(y_batch):
        seed = get_seed_from_filename(extracted_names[idx])
        random.seed(seed)

        id_prompt = random.choice(IMAGENET_PROMPT_TEMPLATES).format(id_word_dict[y.item()])
        id_editing_prompts.append(id_prompt)
        ood_prompt = random.choice(IMAGENET_PROMPT_TEMPLATES).format(random.choice(ood_word_list))
        ood_editing_prompts.append(ood_prompt)

    editing_prompts = id_editing_prompts + id_editing_prompts + ood_editing_prompts
    return editing_prompts
