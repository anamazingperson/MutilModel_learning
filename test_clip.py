import open_clip
from PIL import Image

# 加载预训练模型和预处理函数
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # 将模型设置为评估模式

# 获取分词器
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 加载和预处理图像
image = Image.open("dog.png")
image = preprocess(image).unsqueeze(0)  # 添加批量维度

# 编码文本
text = tokenizer(["a diagram", "a cat"])
text_features = model.encode_text(text)

# 编码图像
image_features = model.encode_image(image)

# 计算相似度
similarity = (image_features @ text_features.T).softmax(dim=-1)

# 输出相似度
print("Similarity:", similarity)