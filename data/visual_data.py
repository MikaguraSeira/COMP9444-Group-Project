import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

def visualize_local_dataset(data_dir="breast_cancer_data"):
    # 1. 遍历文件夹，找到所有图片路径
    image_paths = []
    labels = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # 只读取常见图片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 过滤掉 mask 图片（因为它们是用来做分割的，不是原图）
                if 'mask' not in file.lower():
                    image_paths.append(os.path.join(root, file))
                    # 图片所在的文件夹名称就是它的类别标签 (benign, malignant, normal)
                    labels.append(os.path.basename(root))

    if not image_paths:
        print(f"在 {data_dir} 中没有找到图片，请检查文件夹路径是否正确！")
        return

    # 2. 绘制类别分布柱状图
    label_counts = Counter(labels)
    classes = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['#4C72B0', '#DD8452', '#55A868'])
    plt.title('Class Distribution of Ultrasound Images (Excluding Masks)')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Images')
    
    # 在柱状图上方显示具体数字
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')
        
    plt.savefig('class_distribution.png')  # 保存柱状图

    # 3. 随机抽取并展示每个类别的样本图片
    samples_per_class = 3
    num_classes = len(classes)
    
    # 动态创建画布
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 3 * num_classes))
    fig.suptitle('Sample Images per Class', fontsize=16)
    
    for class_idx, class_name in enumerate(classes):
        # 找出当前类别的所有图片路径
        class_imgs = [img for img, lbl in zip(image_paths, labels) if lbl == class_name]
        
        # 随机抽取3张（如果不够3张就全取）
        sampled_imgs = random.sample(class_imgs, min(samples_per_class, len(class_imgs)))
        
        for sample_idx, img_path in enumerate(sampled_imgs):
            # 打开图片并转换为灰度图展示
            img = Image.open(img_path).convert('L')
            
            # 获取对应的子图位置
            ax = axes[class_idx][sample_idx] if num_classes > 1 else axes[sample_idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(class_name)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig('sample_images_per_class.png')  # 保存图片


if __name__ == "__main__":
    # 请确保 "breast_cancer_data" 是你用终端下载的数据集文件夹名称
    # 如果你下载在了当前目录（"."），请将参数改为 data_dir="."
    visualize_local_dataset(data_dir="breast_cancer_data")