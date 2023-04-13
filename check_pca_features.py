import torch
import torch.nn as nn
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.models import build_segmentor
from mmcv import Config
from mmcv.runner import load_checkpoint
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from IPython.display import display as ipython_display
from sklearn.metrics.pairwise import cosine_similarity

image = Image.open("/content/pths/Lenna.png").convert("RGB")
image = image.resize((256, 256))
image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input[0]

    def close(self):
        self.hook.remove()

def process_model(cfg_path, checkpoint_path, image_tensor, use_original_hook=False):
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    normalized_image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]
    image_input = normalized_image_tensor.unsqueeze(0)

    if use_original_hook:
        hook = Hook(model.decode_head.fusion_conv)

        def custom_forward(model, x):
            with torch.no_grad():
                model.eval()
                x = model.backbone(x)
                x = [model.decode_head.convs[i](x[i]) for i in range(len(x))]

                target_size = x[0].size()[2:]
                for i in range(1, len(x)):
                    x[i] = F.interpolate(x[i], size=target_size, mode='bilinear', align_corners=False)

                x = torch.cat(x, dim=1)
                y = model.decode_head.fusion_conv(x)
                z = model.decode_head.dropout(y)
                z = model.decode_head.conv_seg(y)
            hook.output = x
            return y

    else:
        hook = Hook(model.decode_head.conv_seg)

        def custom_forward(model, x):
            with torch.no_grad():
                model.eval()
                x = model.backbone(x)
                x = [model.decode_head.convs[i](x[i]) for i in range(len(x))]

                target_size = x[0].size()[2:]
                out = 0
                for i in range(1, len(x)):
                    x[i] = F.interpolate(x[i], size=target_size, mode='bilinear', align_corners=False)
                    out += x[i]

                x = torch.cat(x, dim=1)
                y = model.decode_head.dropout(x)
                y = model.decode_head.conv_seg(out)
            hook.output = x
            return x

    fusion_input = custom_forward(model, image_input)
    fusion_output = hook.output

    hook.close()

    reshaped_tensor = fusion_output.squeeze(0)
    return reshaped_tensor

models = [
    {
        'cfg_path': 'configs/segformer/segformer_mit-b2_512x512_160k_ade20k_ws.py',
        'checkpoint_path': '../pths/iter_160000.pth',
        'label': 'SegFormer-B2-ADE20K (WSNet)',
        'use_original_hook': False,
    },
    {
        'cfg_path': 'configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py',
        'checkpoint_path': '../pths/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth',
        'label': 'SegFormer-B2-ADE20K',
        'use_original_hook': True,
    },
    {
        'cfg_path': 'configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py',
        'checkpoint_path': '../pths/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
        'label': 'SegFormer-B2-Cityscape',
        'use_original_hook': True,
    },
    {
        'cfg_path': 'configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py',
        'checkpoint_path': '../pths/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
        'label': 'SegFormer-B5-ADE20K',
        'use_original_hook': True,
    },
    # Add more models if necessary
]

input_tensors = []
for model in models:
    reshaped_tensor = process_model(
        cfg_path=model['cfg_path'],
        checkpoint_path=model['checkpoint_path'],
        image_tensor=image_tensor,
        use_original_hook=model['use_original_hook']
    )
    input_tensors.append(reshaped_tensor)

threshold = 1e-5

def plot_histogram(models, input_image):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'orange', 'green', 'red']
    bar_width = 0.2

    for idx, input_tensor in enumerate(input_tensors):
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group in groups:
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            pca = PCA(n_components=256)
            pca.fit(filter_kernel_np)

            explained_variance = np.sum(pca.explained_variance_[i] for i in range(1))
            total_variance = np.sum(pca.explained_variance_)

            if total_variance > 0 and explained_variance * 100 > threshold:
                explained_variance_ratios.append(explained_variance / total_variance * 100)
            else:
                explained_variance_ratios.append(0)
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            label = models[idx]['label'] if group_idx == 0 else None
            x_positions = np.arange(n) + idx * bar_width + group_idx * len(input_tensors) * bar_width * 1.5
            ax.bar(x_positions, group_ratios, width=bar_width, color=colors[idx], label=label)

    #ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    ax.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 0.95))
    ax.set_xticks(np.arange(0, 4 * len(input_tensors) * bar_width * 1.5, len(input_tensors) * bar_width * 1.5) + 1.5 * bar_width)
    ax.set_xticklabels([f'Stage {i}' for i in range(1, 5)])
    ax.set(xlabel='Features from encoder stages', ylabel='Variance explained by PC1 [%]')

    plt.tight_layout()
    plt.savefig('histogram_multiple_models_with_stages.png')
    plt.close()

    ipython_display(Image.open('histogram_multiple_models_with_stages.png'))

plot_histogram(models, image_tensor)

def plot_histogram_2(models, input_image):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)  
    colors = ['blue', 'orange', 'green', 'red']
    group_labels = []
    bar_width = 0.2

    for idx, input_tensor in enumerate(input_tensors):
        ax = axs[idx // 2, idx % 2]
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group in groups:
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            pca = PCA(n_components=256)
            pca.fit(filter_kernel_np)

            explained_variance = np.sum(pca.explained_variance_[i] for i in range(1))
            total_variance = np.sum(pca.explained_variance_)

            if total_variance > 0 and explained_variance * 100 > threshold:
                explained_variance_ratios.append(explained_variance / total_variance * 100)
            else:
                explained_variance_ratios.append(0)
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            label = f'Model {idx + 1}, Stage {group_idx + 1}' if idx == 0 else None
            x_positions = np.arange(n) + idx * bar_width + group_idx * len(input_tensors) * bar_width
            ax.bar(x_positions, group_ratios, width=bar_width, color=colors[idx], label=label)

    ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    ax.legend()
    ax.set_xticks(np.arange(0, 4 * len(input_tensors) * bar_width, len(input_tensors) * bar_width) + 1.5 * bar_width)
    ax.set_xticklabels([f'Stage {i}' for i in range(1, 5)])
    ax.set(xlabel='Stages', ylabel='Variance explained by PC1 [%]')

    plt.tight_layout()
    plt.savefig('histogram_combined.png')
    plt.show()
    plt.close()

plot_histogram_2(models, image_tensor)

