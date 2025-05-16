import random
from pathlib import Path
from PIL import Image
import sys
import argparse
from tqdm import tqdm
from PIL import Image

# python 3.gerenate_data.py ../data/Novel_data_UTF8_processed/alice.txt

# 假设 generate_base_image 和 add_noise_and_reduce_resolution 已在 generate_base_add_noise 模块中定义
sys.path.append(str(Path('../function')))  # 添加当前目录到模块搜索路径

from generate_base_add_noise import generate_base_image, add_noise_and_reduce_resolution, binarize_image  # 导入生成函数

def process_text_file(txt_file_path):
    """
    处理单个 txt 文件，将文本内容转化为图像并保存到指定目录。
    
    :param txt_file_path: 单个 txt 文件的路径
    """
    # 设置文件夹路径
    output_folder = Path("../data/full_pic_low_res")
    base_output_folder = output_folder / "base"
    noise_output_folder = output_folder / "noise"
    base_output_folder.mkdir(parents=True, exist_ok=True)  # 确保 base 文件夹存在
    noise_output_folder.mkdir(parents=True, exist_ok=True)  # 确保 noise 文件夹存在

    # 生成随机文件名
    def generate_random_filename():
        return f"{random.randint(10000000000, 99999999999)}"

    # 处理文本以插入随机空格作为段落缩进
    def format_text_with_indent(lines):
        formatted_text = []
        for line in lines:
            # 如果检测到连续的换行，插入 1-6 个空格作为缩进
            if line == "\n":
                formatted_text.append("\n" + " " * random.randint(0, 6))
            else:
                formatted_text.append(line)
        return "".join(formatted_text)

    # 逐个处理文件
    with open(txt_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

        # 每 40 行生成一张图片
        for i in tqdm(range(0, len(lines), 30)):
            # 将 40 行内容合并成一段文本，并添加缩进
            text_chunk = format_text_with_indent(lines[i:i+30]).strip()
            if not text_chunk:
                continue  # 如果文本为空，跳过

            # 生成基础图片
            preset_value = random.randint(1, 8)

            scale = random.uniform(0.3, 1.1)
            # image.resize((int(image.width * scale), int(image.height * scale)), Image.ANTIALIAS)

            base_image = generate_base_image(text_chunk, preset=preset_value)
            new_size = (int(base_image.width * scale), int(base_image.height * scale))

            # 随机生成文件编号，避免重复
            filename_base = generate_random_filename()

            # 保存基础图像
            base_image_path = base_output_folder / f"{filename_base}_0.jpg"
            

            # 生成对应的四个脏图
            for level in range(1, 5):
                processed_image = add_noise_and_reduce_resolution(base_image, preset=f"{level}_level")
                if random.random() < 0.1:  # 10% 的概率
                    processed_image = binarize_image(processed_image)
                noise_image_path = noise_output_folder / f"{filename_base}_{level}.jpg"
                processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
                processed_image.save(noise_image_path, format="JPEG")
            
            base_image = base_image.resize(new_size, Image.Resampling.LANCZOS)
            base_image.save(base_image_path, format="JPEG")

    print(f"图片生成完成，源文件：{txt_file_path}")

# 解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文本文件转换为图像并添加噪声效果")
    parser.add_argument("txt_file_path", type=str, help="要处理的 txt 文件路径")
    args = parser.parse_args()

    # 调用处理函数
    process_text_file(args.txt_file_path)
