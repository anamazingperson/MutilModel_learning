import os
import kagglehub

# 指定保存数据集的路径
save_path = "/mnt/lizijing"

# 如果文件夹不存在，则创建它
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 下载数据集
path = kagglehub.dataset_download("adityajn105/flickr8k", path=save_path)

print("Path to dataset files:", path)