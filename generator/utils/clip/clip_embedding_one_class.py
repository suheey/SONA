import os
import torch
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoProcessor, CLIPVisionModel
import matplotlib.cm as cm

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

folder_name = 'n01443537'
folder_path = f'/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/stepwise_val/0/{folder_name}'


image_names = [file_name.split('.')[0] for file_name in os.listdir(folder_path) if file_name.endswith('.png')][:10]  # 폴더 내의 모든 PNG 파일 이름 가져오기

clip_embeddings = []
colors = cm.get_cmap('tab10', len(image_names))
colors = [colors(i) for i in range(len(image_names))]
for color, image_name in zip(colors, image_names):
    clip_embeddings_per_image = []
    for i in range(20):
        root = f'/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/stepwise_val/{i}/{folder_name}'
        image = Image.open(os.path.join(root, f'{image_name}.png'))
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output  
        clip_embeddings_per_image.append(pooled_output)
    
    # root = f'/workspace/sanghyu.yoon/lab-di/squads/ood_detection/dataset/ImageNet-200/val/{folder_name}'
    # image = Image.open(os.path.join(root, f'{image_name}.JPEG'))
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
for i, (image_name, color) in enumerate(zip(image_names, colors)):
    num_points = 21  # Each image plus the original
    plt.scatter(embeddings_2d[offset:offset+num_points, 0], embeddings_2d[offset:offset+num_points, 1], label=image_name, color=color)
    for j, (x, y) in enumerate(embeddings_2d[offset:offset+num_points]):
        txt = 20 - j
        if txt < 0:
            plt.text(x, y, 'org', color='blue', fontsize=12)
        else:
            plt.text(x, y, str(txt), color='blue', fontsize=12)

    for j in range(num_points - 1):
        plt.arrow(embeddings_2d[offset+j, 0], embeddings_2d[offset+j, 1], 
                  embeddings_2d[offset+j+1, 0] - embeddings_2d[offset+j, 0], embeddings_2d[offset+j+1, 1] - embeddings_2d[offset+j, 1],
                  width=0.1, head_width=5, head_length=5, color='red')
    offset += num_points

plt.title(f'TSNE Visualization of CLIP Embeddings with Arrows and Indices\nFolder: {folder_name}')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
# plt.legend()
plt.savefig(f'clip_embeddings_visualization_{folder_name}.png')
plt.show()