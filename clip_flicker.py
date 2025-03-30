import open_clip
from PIL import Image
import torch
import os
import pandas as pd
# 加载数据集
# 加载预训练模型和预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',device=device)
model.eval()  # 将模型设置为评估模式

# 数据集路径
data_path = "/mnt/nvme0n1/lizijing/flickr8k"
images_path = os.path.join(data_path, "Images")
captions_path = os.path.join(data_path, "captions.txt")

# 加载描述性文本
captions_df = pd.read_csv(captions_path, sep=",", names=["image", "caption"])

# 将描述性文本存储为字典
captions_dict = {}
for _, row in captions_df.iterrows():
    image_name = row["image"]
    caption = row["caption"]
    if image_name not in captions_dict:
        captions_dict[image_name] = []
    captions_dict[image_name].append(caption)

# 示例：查看一张图像及其描述
image_name = "1000268201_693b08cb0e.jpg"
captions_for_image = captions_dict[image_name]

# 预处理图像
image_path = os.path.join(images_path, image_name)
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

# 编码文本
text = open_clip.tokenize(captions_for_image).to(device)

# 计算特征
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)

# 归一化特征
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 计算相似度
similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

print("Similarity:", similarity)


import matplotlib.pyplot as plt

# 可视化相似度
plt.imshow(similarity, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.xlabel("Text Captions")
plt.ylabel("Images")
plt.title("Similarity between Text Captions and Image")
plt.xticks(range(len(captions_for_image)), captions_for_image, rotation=90)
plt.yticks([0], [image_name])
plt.show()
plt.savefig('show_relation.png')