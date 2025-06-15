import os
import torch
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

root_folder = '/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/stepwise_val/'
file_folder = '/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/val/'

sample_limit = 50

clip_embeddings = []
labels = []
color_map = plt.cm.get_cmap('tab20', 10)






for folder_idx, folder_name in enumerate(os.listdir(file_folder)):
    if folder_idx == 10:  # Only consider the first 20 folders
        break
    
    folder_path = os.path.join(file_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    image_names = [file_name.split('.')[0] for file_name in os.listdir(folder_path)][:sample_limit]
    # if file_name.endswith('.JPEG')

    for i, image_name in enumerate(image_names):
        label = folder_idx  # Assign each folder a unique label
        labels.extend([label] * 7)  # Each image has 21 points
        clip_embeddings_per_image = []
        
        for j in range(7):
            image_path = os.path.join(root_folder, str(j), folder_name, f'{image_name}.png')
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            pooled_output = outputs.pooler_output  
            clip_embeddings_per_image.append(pooled_output)
        
        # image_path = os.path.join('/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/val/', folder_name, f'{image_name}.JPEG')
        # image = Image.open(image_path)
        # inputs = processor(images=image, return_tensors="pt")
        # outputs = model(**inputs)
        # pooled_output = outputs.pooler_output  
        # clip_embeddings_per_image.append(pooled_output)

        clip_embeddings_per_image = torch.cat(clip_embeddings_per_image, 0).detach().numpy()
        clip_embeddings.append(clip_embeddings_per_image)

clip_embeddings = np.concatenate(clip_embeddings, axis=0)

tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(clip_embeddings)

plt.figure(figsize=(10, 8))
offset = 0
for folder_idx, folder_name in enumerate(os.listdir(file_folder)):
    if folder_idx == 10:  # Only consider the first 20 folders
        break
    
    folder_path = os.path.join(file_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    image_names = [file_name.split('.')[0] for file_name in os.listdir(folder_path)][:sample_limit]
    # if file_name.endswith('.JPEG')

    for i, image_name in enumerate(image_names):
        num_points = 7  # Each image plus the original
        plt.scatter(embeddings_2d[offset:offset+num_points, 0], embeddings_2d[offset:offset+num_points, 1], 
                    label=f'Folder: {folder_name}', color=color_map(folder_idx), alpha=0.7)
        offset += num_points

plt.title('TSNE Visualization of CLIP Embeddings with Arrows and Indices')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.savefig(f'clip_embeddings_visualization_all_folders.png')
plt.show()