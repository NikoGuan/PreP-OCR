"""
本代码主要实现以下功能：
1. 从指定文件夹中随机选取两张原始图像，并将它们预处理（等宽、寻找分界白行、裁剪出分界线附近的区域）后上下拼接，
   生成一张基础图像（base_image）。
2. 基于基础图像生成4个不同level的“脏图”（添加噪声和降分辨率处理），
   保存后将这4张脏图存入列表processed_images中。
3. 随后对4张脏图进行两种随机拼接操作：
   - 左右拼接：随机选取相邻两张图，将左图取左半部分、右图取右半部分拼接为一张图（lr_merged）。
   - 上下拼接：随机选取相邻两张图，将上图取上半部分、下图取下半部分拼接为一张图（ud_merged）。
4. 最后从4个level的文件中随机选取两个，分别用左右拼接和上下拼接后的图覆盖对应的level文件，
   以增加数据多样性。基础图像也会保存。
   
需要注意：
- 本代码依赖于外部模块 generate_base_add_noise 中定义的函数：
  generate_base_image, add_noise_and_reduce_resolution, binarize_image
- 输出目录在../data/pinjie下，其中base/存放基础图像，noise/存放脏图和替换后的图。
"""
from PIL import Image
import numpy as np
import os
import random
from pathlib import Path
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

from generate_base_add_noise import generate_base_image, add_noise_and_reduce_resolution, binarize_image 

def find_closest_white_line_to_middle(img_gray, white_threshold=250):
    """
    以图像中间行为起点，上下同时扩散，
    寻找距离中间最近的一条全白行（像素值都 >= white_threshold）。
    
    返回：
    - white_line_y: 找到的白行行号 (0 <= y < height)。
      如果没找到，返回 None。
    """
    h, w = img_gray.shape
    m = h // 2  # 中间行
    max_d = max(m, h - m)  # 从中间到顶部或底部的最大偏移

    for d in range(max_d + 1):
        # 先检查上方 (m - d)
        up_line = m - d
        if up_line >= 0:
            if np.all(img_gray[up_line, :] >= white_threshold):
                return up_line
        
        # 再检查下方 (m + d)
        down_line = m + d
        if down_line < h:
            if np.all(img_gray[down_line, :] >= white_threshold):
                return down_line
    
    # 如果全图都没有找到全白行，则返回 None
    return None

def resize_to_same_width(img1, img2):
    """
    将 img2 等比例缩放到与 img1 相同宽度。
    返回 (img1, resized_img2)。
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    if w1 != w2:
        new_h2 = int(h2 * (w1 / w2))
        # Pillow 新版推荐用 LANCZOS
        img2 = img2.resize((w1, new_h2), resample=Image.Resampling.LANCZOS)
    return img1, img2

def pick_two_random_jpg_files(folder_path):
    # 1. 获取文件夹下所有文件（不包含子文件夹）
    all_files = os.listdir(folder_path)
    
    # 2. 过滤出以 .jpg 结尾的文件 (也可根据实际情况改成 .JPG / .jpeg 等)
    jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]
    
    # 3. 如果文件夹中 jpg 文件少于2个，就直接返回全部或抛出异常
    if len(jpg_files) < 2:
        raise ValueError("该文件夹下的 JPG 文件少于 2 个，无法随机选取 2 个。")
    
    # 4. 随机选取2个文件
    selected = random.sample(jpg_files, 2)
    
    # 5. 转为绝对路径（也可保留相对路径）
    selected_full_paths = [os.path.join(folder_path, f) for f in selected]
    return selected_full_paths

def generate_random_filename():
    return f"{random.randint(10000000000, 99999999999)}"

def main():
    # ---------------- 主流程 ----------------
    # 你两张图片的路径
    folder_path = "../data/full_pic_low_res/base"
    two_files = pick_two_random_jpg_files(folder_path)
    img_path_1 = two_files[0]
    img_path_2 = two_files[1]
    # img_path_1 = "../data/full_pic_low_res/base/97764590890_0.jpg"
    # img_path_2 = "../data/full_pic_low_res/base/98038681385_0.jpg"

    # 打开图像
    img1 = Image.open(img_path_1)
    img2 = Image.open(img_path_2)

    # （可选）先让两个图等宽
    img1, img2 = resize_to_same_width(img1, img2)

    # 分别转灰度图，便于检测全白行
    img1_gray = np.array(img1.convert("L"))
    img2_gray = np.array(img2.convert("L"))

    # 找到各自“离中间最近”的白行
    white_line1 = find_closest_white_line_to_middle(img1_gray, white_threshold=250)
    white_line2 = find_closest_white_line_to_middle(img2_gray, white_threshold=250)

    # 如果没找到白行，做一个兜底（此时可能说明没有明显白行）
    if white_line1 is None:
        white_line1 = img1_gray.shape[0] // 2
        print("图1 未找到白行，默认使用中间行:", white_line1)
    if white_line2 is None:
        white_line2 = img2_gray.shape[0] // 2
        print("图2 未找到白行，默认使用中间行:", white_line2)

    # ---------------- 只保留分界线附近的区域 ----------------
    # 定义在分割线附近保留多少“边距”像素（可以自行调整）
    margin_top_img1 = 400   # 第1张图，在其“白行”之上保留 200 像素
    margin_bottom_img2 = 400  # 第2张图，在其“白行”之下保留 200 像素

    # 计算安全上下边界，避免越界
    upper_start_img1 = max(0, white_line1 - margin_top_img1)  # 第一张图下半部分的起点
    lower_end_img1   = white_line1  # 裁到白行为止

    upper_start_img2 = white_line2  # 第二张图上半部分从白行开始
    lower_end_img2   = min(img2.height, white_line2 + margin_bottom_img2)

    # 裁剪后，img1_sub 只包含分割线“往上一点点”的区域
    # 注：Pillow crop (left, upper, right, lower)
    img1_sub = img1.crop((0, upper_start_img1, img1.width, lower_end_img1))

    # img2_sub 只包含分割线“往下一点点”的区域
    img2_sub = img2.crop((0, upper_start_img2, img2.width, lower_end_img2))

    # ---------------- 合并裁剪后的子图 ----------------
    new_width = img1_sub.width
    new_height = img1_sub.height + img2_sub.height
    merged_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))

    # 把第一张子图贴到上面
    merged_img.paste(img1_sub, (0, 0))
    # 把第二张子图贴到下面
    merged_img.paste(img2_sub, (0, img1_sub.height))

    # 最终保存
    out_path = "merged_result.jpg"
    merged_img

    base_image = merged_img



    output_folder = Path("../data/pinjie")
    base_output_folder = output_folder / "base"
    noise_output_folder = output_folder / "noise"
    base_output_folder.mkdir(parents=True, exist_ok=True)  # 确保 base 文件夹存在
    noise_output_folder.mkdir(parents=True, exist_ok=True)  # 确保 noise 文件夹存在

    # 随机生成文件编号，避免重复
    filename_base = generate_random_filename()

    # 保存基础图像
    base_image_path = base_output_folder / f"{filename_base}_0.jpg"


    # 生成对应的四个脏图，存到 processed_images 列表
    processed_images = []
    for level in range(1, 5):
        # 对 base_image 加噪声/降分辨率
        pimg = add_noise_and_reduce_resolution(base_image, preset=f"{level}_level")
        
        # 10% 的概率再做二值化
        if random.random() < 0.1:
            pimg = binarize_image(pimg)
        
        # 保存原始脏图
        noise_image_path = noise_output_folder / f"{filename_base}_{level}.jpg"
        if pimg.size == base_image.size:
            pimg.save(noise_image_path, format="JPEG")
        else:
            print("尺寸不同")
        
        # 也将这张图存到一个列表中，等待后续“左右拼接”
        processed_images.append(pimg)

    # ---------------------------
    # 1. 左右随机拼接
    # 随机选取相邻两张图
    left_idx = random.randint(0, 2)  
    right_idx = left_idx + 1        
    left_img = processed_images[left_idx]
    right_img = processed_images[right_idx]

    # Step 1: 保证高度一致（若不一致，对右图做等高缩放）
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    if h1 != h2:
        # 等比例缩放 right_img 到和 left_img 相同高度
        right_img = right_img.resize((int(w2 * (h1 / h2)), h1), Image.Resampling.LANCZOS)
        w2, h2 = right_img.size

    # Step 2: 分别计算左右裁剪宽度
    #  - left_part_width = w1 // 2
    #  - right_part_width = w2 - (w2 // 2)
    #    这样相当于 w2//2 + (w2 - w2//2) = w2，不会缺失像素

    left_part_width = w1 // 2
    right_part_width = w2 - (w2 // 2)

    # Step 3: 分别裁剪出左右部分
    #  左图 [0, 0, left_part_width, h1]
    #  右图 [w2 - right_part_width, 0, w2, h2]
    left_part = left_img.crop((0, 0, left_part_width, h1))
    right_part = right_img.crop((w2 - right_part_width, 0, w2, h2))

    # Step 4: 创建新图，并将左右部分贴上去
    lr_merged_width = left_part_width + right_part_width
    lr_merged_height = h1  # 与两图统一后的高度一致
    lr_merged = Image.new("RGB", (lr_merged_width, lr_merged_height), (255, 255, 255))
    lr_merged.paste(left_part, (0, 0))
    lr_merged.paste(right_part, (left_part_width, 0))


    # ---------------------------
    # 2. 上下随机拼接
    # 同样随机选取相邻两张图
    top_idx = random.randint(0, 2)  
    bottom_idx = top_idx + 1       
    top_img = processed_images[top_idx]
    bottom_img = processed_images[bottom_idx]

    # Step 1: 保证宽度一致（若不一致，对 bottom_img 做等宽缩放）
    w_top, h_top = top_img.size
    w_bottom, h_bottom = bottom_img.size
    if w_top != w_bottom:
        # 等比例缩放 bottom_img 到与 top_img 同宽
        bottom_img = bottom_img.resize((w_top, int(h_bottom * (w_top / w_bottom))), Image.Resampling.LANCZOS)
        w_bottom, h_bottom = bottom_img.size

    # Step 2: 分别计算上下裁剪高度
    #  - top_part_height = h_top // 2
    #  - bottom_part_height = h_bottom - (h_bottom // 2)

    top_part_height = h_top // 2
    bottom_part_height = h_bottom - (h_bottom // 2)

    # Step 3: 分别裁剪出上下部分
    top_part = top_img.crop((0, 0, w_top, top_part_height))
    bottom_part = bottom_img.crop((0, h_bottom - bottom_part_height, w_bottom, h_bottom))

    # Step 4: 创建新图，并将上下部分贴上去
    ud_merged_width = w_top  # 与两图统一后的宽度
    ud_merged_height = top_part_height + bottom_part_height
    ud_merged = Image.new("RGB", (ud_merged_width, ud_merged_height), (255, 255, 255))
    ud_merged.paste(top_part, (0, 0))
    ud_merged.paste(bottom_part, (0, top_part_height))


    # ---------------------------
    # 3. 随机替换两个 level 文件
    # 从 4 个 level 中随机选两个（对应的文件： filename_base_1.jpg ~ filename_base_4.jpg）
    replace_indices = random.sample(range(4), 2)
    # 随机决定哪个拼接结果替换哪一个（例如，先将左右拼接图 lr_merged 替换到其中一个，ud_merged 替换到另外一个）
    # 这里直接顺序替换：
    #    replace_indices[0] 对应的 level 替换为 lr_merged
    #    replace_indices[1] 对应的 level 替换为 ud_merged

    # 这里，level 文件命名格式为: f"{filename_base}_{level}.jpg"，其中 level 从 1 开始
    level_to_replace_lr = replace_indices[0] + 1  # 索引转换为 level（1~4）
    level_to_replace_ud = replace_indices[1] + 1

    # 构造保存路径（注意 noise_output_folder 已经定义好）
    lr_replace_path = noise_output_folder / f"{filename_base}_{level_to_replace_lr}.jpg"
    ud_replace_path = noise_output_folder / f"{filename_base}_{level_to_replace_ud}.jpg"

    if lr_merged.size == base_image.size and ud_merged.size == base_image.size:
        lr_merged.save(lr_replace_path, format="JPEG")
        ud_merged.save(ud_replace_path, format="JPEG")
    else:
        print("上下拼接尺寸不同")

    # print(f"左右拼接图保存并替换到: {lr_replace_path}")
    # print(f"上下拼接图保存并替换到: {ud_replace_path}")

    base_image_path = base_output_folder / f"{filename_base}_0.jpg"
    base_image.save(base_image_path, format="JPEG")

for i in tqdm(range(250)):
    main()
    
print("完成")