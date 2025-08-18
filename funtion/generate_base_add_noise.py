import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageOps
from PIL import Image, ImageEnhance

#Base
def generate_base_image(text, preset=1, **kwargs):
    """
    生成带有指定文本的底图，并支持多行文本、行距、字间距、页边距、文字旋转角度和轻微错位变形。

    :param text: 要显示的文本，支持多行。
    :param language: 语言，默认为"eng"。
    :param font_size: 字体大小,默认为50。
    :param bg_color: 背景颜色，默认为白色。
    :param text_color: 文本颜色，默认为黑色。
    :param line_spacing: 行距的最大值，控制各行之间的间隔。
    :param char_spacing: 字间距的最大值，控制每个字符之间的间隔。
    :param angle: 文字旋转角度范围，默认为-输入值到+输入值之间随机选择。
    :param margin: 页边距，可以是整数（四边相同）或元组（左、上、右、下）。
    :return: 带有指定文本的底图(PIL Image对象)。
    """
    base_parameters = {
        1: {"char_shift": False, "apply_seg": False, "apply_bend": False},
        2: {"char_shift": True, "apply_seg": False, "apply_bend": False},
        3: {"char_shift": False, "apply_seg": False, "apply_bend": True},
        4: {"char_shift": False, "apply_seg": True, "apply_bend": False},
        5: {"char_shift": True, "apply_seg": False, "apply_bend": True},
        6: {"char_shift": True, "apply_seg": True, "apply_bend": False},
        7: {"char_shift": False, "apply_seg": True, "apply_bend": True},
        8: {"char_shift": True, "apply_seg": True, "apply_bend": True},
    }

    preset_params = base_parameters.get(preset, base_parameters[1])
    
    # 将预设参数更新到 kwargs 中
    kwargs.update(preset_params)

    # 获取参数，设置默认值
    language = kwargs.get("language", "eng")
    font_size = kwargs.get("font_size", 50)
    bg_color = kwargs.get("bg_color", "white")
    text_color = kwargs.get("text_color", "black")
    line_spacing = kwargs.get("line_spacing", 50)
    char_spacing = kwargs.get("char_spacing", 15)
    angle = kwargs.get("angle", 2)
    margin = kwargs.get("margin", 170)
    char_shift = kwargs.get("char_shift", False)
    apply_seg = kwargs.get("apply_seg", False)
    apply_bend = kwargs.get("apply_bend", False)

    def apply_shifted_segments(image, segment_height=100, max_shift=2, max_num_segments=10):
        """
        随机选择若干个段，在页面随机位置进行平移，模拟页面的错位效果。

        :param image: 输入的PIL图像对象
        :param segment_height: 每个段的最大高度
        :param max_shift: 每个段的最大平移距离
        :param num_segments: 随机选择的段数量
        :return: 应用错位效果后的图像
        """
        img_array = np.array(image)
        h, w, _ = img_array.shape


        num_segments = random.randint(0, max_num_segments)
        for _ in range(num_segments):
            # 随机选择起始位置和段高度
            y = random.randint(0, h - segment_height)
            random_segment_height = random.randint(1, segment_height)
            
            # 随机平移距离
            shift = random.randint(-max_shift, max_shift)
            
            # 对选定的段应用平移
            img_array[y:y + random_segment_height] = np.roll(img_array[y:y + random_segment_height], shift, axis=1)
        
        return Image.fromarray(img_array)

    # --- 模拟页面弯折 ---
    def apply_random_bend(image, bend_intensity_horizontal=30, bend_intensity_vertical=30, num_waves=2):
        """
        对图像应用随机弯曲效果，包括左右和上下弯折，模拟纸张的自然弯曲。

        :param image: 输入的PIL图像对象
        :param bend_intensity_horizontal: 左右弯折的最大强度
        :param bend_intensity_vertical: 上下弯折的最大强度
        :param num_waves: 每个方向的最大波形数量，用于随机弯曲效果
        :return: 经过随机弯曲处理的图像
        """
        # 将图像转换为 NumPy 数组并扩展画布，以避免弯曲后出现黑边
        expanded_image = ImageOps.expand(image, border=50, fill="white")
        img_array = np.array(expanded_image)
        h, w = img_array.shape[:2]

        # --- 随机生成左右弯折波形 ---
        actual_waves_horizontal = random.randint(0, num_waves)
        bend_horizontal = np.zeros(h)
        for _ in range(actual_waves_horizontal):
            phase_shift = random.uniform(0, 2 * np.pi)
            frequency = random.uniform(0.5, 1.5)
            intensity = random.uniform(0, bend_intensity_horizontal)
            bend_horizontal += np.sin(np.linspace(-np.pi, np.pi, h) * frequency + phase_shift) * intensity / actual_waves_horizontal

        bent_img_array = np.zeros_like(img_array)
        for i in range(h):
            shift = int(bend_horizontal[i])
            if shift > 0:
                bent_img_array[i, shift:] = img_array[i, :-shift]
            elif shift < 0:
                bent_img_array[i, :shift] = img_array[i, -shift:]
            else:
                bent_img_array[i] = img_array[i]

        # --- 随机生成上下弯折波形 ---
        actual_waves_vertical = random.randint(0, num_waves)
        bend_vertical = np.zeros(w)
        for _ in range(actual_waves_vertical):
            phase_shift = random.uniform(0, 2 * np.pi)
            frequency = random.uniform(0.5, 1.5)
            intensity = random.uniform(0, bend_intensity_vertical)
            bend_vertical += np.sin(np.linspace(-np.pi, np.pi, w) * frequency + phase_shift) * intensity / actual_waves_vertical

        bent_img_array_vertical = np.zeros_like(bent_img_array)
        for j in range(w):
            shift = int(bend_vertical[j])
            if shift > 0:
                bent_img_array_vertical[shift:, j] = bent_img_array[:-shift, j]
            elif shift < 0:
                bent_img_array_vertical[:shift, j] = bent_img_array[-shift:, j]
            else:
                bent_img_array_vertical[:, j] = bent_img_array[:, j]

        # 将结果转换回 PIL 图像
        bent_image = Image.fromarray(bent_img_array_vertical)
        final_image = bent_image.crop((50, 50, w - 50, h - 50))

        return final_image


    font_dir = Path('./font/latin')
    font_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
    # font_path = random.choice(font_files) if font_files else None
    # font = ImageFont.truetype(str(font_path), font_size)

    if font_files:
        available_fonts = font_files.copy()  # 复制一份列表，避免直接修改原列表
        font = None
        while available_fonts:
            font_path = random.choice(available_fonts)
            try:
                font = ImageFont.truetype(str(font_path), font_size)
                break  # 成功加载后跳出循环
            except Exception as e:
                print(f"加载字体 {font_path} 失败：{e}，尝试其他字体...")
                available_fonts.remove(font_path)  # 移除错误的字体路径以防重复尝试
        else:
            # 如果所有字体都加载失败
            print("未能加载任何字体，可能需要检查字体文件是否正常。")
            font = None





    
    # 使用指数分布生成 line_spacing 和 char_spacing，使得值靠近 0 的概率更高
    line_spacing = min(int(np.random.exponential(scale=line_spacing / 10)), line_spacing)
    char_spacing = min(int(np.random.exponential(scale=char_spacing / 10)), char_spacing)
    
    # 随机生成偏向于0的角度
    angle = min(np.random.exponential(scale=angle / 3), angle)
    angle = angle if random.choice([True, False]) else -angle  # 随机设置正负

    # 将单个整数的边距值扩展为四边边距
    if isinstance(margin, int):
        margin = (margin, margin, margin, margin)
    left_margin, top_margin, right_margin, bottom_margin = margin
    
    # 创建临时图像和绘制对象
    dummy_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_image)
    lines = text.splitlines()
    
    # 计算总宽度和高度
    text_width = max(sum(draw.textbbox((0, 0), char, font=font)[2] + char_spacing for char in line) - char_spacing for line in lines)
    
    # 使用单行高度计算总高度，更准确
    single_line_bbox = draw.textbbox((0, 0), "Ag", font=font)  # 使用包含上下伸部的字符
    line_height = single_line_bbox[3] - single_line_bbox[1]    # 实际行高
    total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)

    # 创建底图，考虑页边距
    base_image_width = text_width + left_margin + right_margin
    base_image_height = total_height + top_margin + bottom_margin
    base_image = Image.new('RGB', (base_image_width, base_image_height), bg_color)
    
    # 创建透明图层用于绘制文字
    text_layer = Image.new('RGBA', (base_image_width, base_image_height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)
    

    # 绘制每一行文本
    y = top_margin
    for line in lines:
        x = left_margin
        for char in line:
            if char_shift:
                # 随机生成每个字符的轻微偏移和旋转角度
                offset_x = random.randint(-1, 1)
                offset_y = random.randint(-1, 1)
                char_angle = random.uniform(-1, 1)  # 控制每个字符的旋转角度
            else:
                # 无偏移或旋转
                offset_x, offset_y, char_angle = 0, 0, 0

            # 创建一个透明图层用于旋转每个字符
            char_layer = Image.new('RGBA', (font_size * 2, font_size * 2), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_layer)
            
            # 在透明图层上绘制字符
            char_draw.text((font_size // 2, font_size // 2), char, font=font, fill=text_color)
            
            # 旋转字符
            rotated_char = char_layer.rotate(char_angle, resample=Image.BICUBIC, expand=True)
            
            # 将旋转的字符粘贴到文本图层
            text_layer.paste(rotated_char, (x + offset_x, y + offset_y), rotated_char)
            
            # 更新 x 坐标，加上字符宽度和字符间距
            char_width = text_draw.textbbox((0, 0), char, font=font)[2]
            x += char_width + char_spacing

        # 更新 y 坐标，加上实际行高和行距
        y += line_height + line_spacing

    # 旋转透明图层
    rotated_text_layer = text_layer.rotate(angle, expand=True)
    
    # 将旋转后的文字图层粘贴到背景图上
    base_image.paste(rotated_text_layer, (0, 0), rotated_text_layer)
    final_image = base_image.convert("RGB")

    if apply_seg:
        final_image = apply_shifted_segments(final_image)


    if apply_bend:
        final_image = apply_random_bend(final_image)

    # Limit image width to 1333 pixels
    # if final_image.width > 1333:


    return final_image

#Noise
def add_noise_and_reduce_resolution(image,preset = "0_level", **kwargs):
    noise_parameters = {
        "default": {
            "max_noise_factor": 40, #噪声越大越垃圾
            "min_scale_factor": 0.8,   #画质降低 越小越垃圾
            "blur_radius": 1,         #高斯模糊
            "background_folder": "./noise_img/background_p",
            "bg_intensity": 0.6,          #背景透明度
            "stain_folder": "./noise_img/stain_p",
            "st_intensity": 0.8,        #污渍透明度
            "max_stains": 7,    #污渍最大个数
            "contrast_factor": 0.7, 
            "black_size": 1,          #黑点尺寸
            "black_number": 1000,       #黑点的个数=总像素/这个数值    数值越大黑点越少
            "white_size": 10, #白点尺寸
            "white_number": 100,     #白点的个数=总像素/这个数值    数值越大黑点越少
            "line_number": 30
        },
        "0_level": {
            "max_noise_factor": 0,
            # "min_scale_factor": 0.3,
            "blur_radius": 1,
            "contrast_factor": 0.6,
            "bg_intensity": 0.0,
            "st_intensity": 0.0,
            "max_stains": 0,
            "black_number": 10000,
            "white_size": 3,
            "white_number": 20000, 
            "line_number": 2
        },
        "1_level": {
            "max_noise_factor": 10,
            # "min_scale_factor": 0.3,
            "blur_radius": 1,
            "contrast_factor": 0.6,
            "bg_intensity": 0.1,
            "st_intensity": 0.3,
            "max_stains": 1,
            "black_number": 3000,
            "white_size": 3,
            "white_number": 500, 
            "line_number": 4
        },
        "2_level": {
            "max_noise_factor": 30,
            # "min_scale_factor": 0.3,
            "blur_radius": 1,
            "contrast_factor": 0.6,
            "bg_intensity": 0.3,
            "st_intensity": 0.6,
            "max_stains": 3,
            "black_number": 2000,
            "white_size": 5,
            "white_number": 300,
            "line_number": 8
        },
        "3_level": {
            "max_noise_factor": 50,
            # "min_scale_factor": 0.3,
            "blur_radius": 2,
            "contrast_factor": 0.6,
            "bg_intensity": 0.6,
            "st_intensity": 0.8,
            "max_stains": 5,
            "black_number": 1000,
            "white_size": 5,
            "white_number": 200,
            "line_number": 10
        },
        "4_level": {
            "max_noise_factor": 50,
            # "min_scale_factor": 0.2,
            "blur_radius": 2,
            "contrast_factor": 0.3,
            "bg_intensity": 0.6,
            "st_intensity": 0.8,
            "max_stains": 5,
            "black_number": 1000,
            "white_size": 5,
            "white_number": 200,
            "line_number": 10
        }
    }
    preset_params = noise_parameters.get(preset, noise_parameters["0_level"])
    kwargs.update(preset_params)
    
    # 从 kwargs 中提取参数，如果未提供则使用默认值
    max_noise_factor = kwargs.get("max_noise_factor", 50)
    min_scale_factor = kwargs.get("min_scale_factor", 0.2)
    blur_radius = kwargs.get("blur_radius", 1)
    background_folder = kwargs.get("background_folder", "./noise_img/background_p")
    bg_intensity = kwargs.get("bg_intensity", 0.2)
    stain_folder = kwargs.get("stain_folder", "./noise_img/stain_p")
    st_intensity = kwargs.get("st_intensity", 0.8)
    max_stains = kwargs.get("max_stains", 5)
    contrast_factor = kwargs.get("contrast_factor", 0.5)
    black_size = kwargs.get("black_size", 1)
    black_number = kwargs.get("black_number", 1000)
    white_size = kwargs.get("white_size", 5)
    white_number = kwargs.get("white_number", 200)
    line_number = kwargs.get("line_number", 10)

    # 首先执行随机插入黑色区域操作
    image = random_erase_black(image, black_size, black_number)
    image = random_erase_white(image, white_size, white_number)
    image = random_erase_white_lines(image, line_width=5, max_line_length=30000, line_number = line_number)
    
    # 定义其他操作及其参数
    operations = [
        # 添加轻微的噪声
        ("add_noise", lambda img: add_noise(img, random.randint(0, max_noise_factor))),
        # 降低分辨率
        ("reduce_resolution", lambda img: reduce_resolution(img, min_scale_factor)),
        # 模拟边缘模糊
        ("blur_edges", lambda img: apply_edge_blur(img, blur_radius)),
        # 模拟纸张纹理
        # ("add_paper_texture", lambda img: add_paper_texture(img)),

        ("add_background", lambda img: add_background(img, background_folder, bg_intensity)),
        # 模拟纸张纹理
        ("add_stain_overlay", lambda img: add_stain_overlay(img, stain_folder, st_intensity, max_stains)),
        # 调整对比度
        ("adjust_contrast", lambda img: adjust_contrast(img, contrast_factor)),
        # 膨胀和腐蚀，模拟文字的不完整性
        ("dilate", lambda img: apply_dilation(np.array(img), np.ones((2, 2), np.uint8), random.randint(0, 1))),
        ("erode", lambda img: apply_erosion(np.array(img), np.ones((3, 3), np.uint8), random.randint(0, 1)))
    ]
    
    random.shuffle(operations)
    # print(f"操作顺序: ['random_erase_black' + {[op[0] for op in operations] + ['random_erase_white']}")

    for op_name, operation in operations:
        if op_name in ["dilate", "erode"]:
            image = Image.fromarray(cv2.cvtColor(operation(image), cv2.COLOR_BGR2RGB))
        else:
            image = operation(image)
    
    # image = add_stain_overlay(image,"../noise_img/stain_p")
    image = image.convert("L")
    scale_factor = 1000 / image.width
    new_width = 1000
    new_height = int(image.height * scale_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def random_erase_black(image, black_size=1, black_number=1000):
    """根据图像尺寸随机插入黑色块，模拟文字磨损，插入次数为0到指定最大值之间的随机数。"""
    img_np = np.array(image)
    h, w, _ = img_np.shape
    num_erases = random.randint(0, (h * w) // black_number)  # 生成0到指定最大插入次数之间的随机数

    for _ in range(num_erases):
        erase_sizex=random.randint(1, black_size)
        erase_sizey=random.randint(1, black_size)
        x = random.randint(0, w - erase_sizex)
        y = random.randint(0, h - erase_sizey)
        img_np[y:y + erase_sizey, x:x + erase_sizex] = 0  # 插入黑色块

    return Image.fromarray(img_np)

def random_erase_white(image, white_size=5, white_number=200):
    """根据图像尺寸随机插入白色块，模拟纸张磨损，插入次数为0到指定最大值之间的随机数。"""
    img_np = np.array(image)
    h, w, _ = img_np.shape
    num_erases = random.randint(0, (h * w) // white_number)  # 生成0到指定最大插入次数之间的随机数

    for _ in range(num_erases):
        erase_sizex=random.randint(1, white_size)
        erase_sizey=random.randint(1, white_size)
        x = random.randint(0, w - erase_sizex)
        y = random.randint(0, h - erase_sizey)
        img_np[y:y + erase_sizey, x:x + erase_sizex] = 255  # 插入白色块

    return Image.fromarray(img_np)

# 辅助函数示例

def apply_edge_blur(image,radius = 1):
    """应用边缘模糊，模拟打印模糊效果。"""
    radius = random.randint(0, radius)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def add_paper_texture(image):
    """叠加纸张纹理，模拟历史书籍纸张效果。"""
    texture = Image.new("RGB", image.size, (255, 225, 200))  # 添加黄褐色的纸张背景
    return Image.blend(image, texture, alpha=0.3)

def adjust_contrast(image, contrast_factor=1):
    """调整图像对比度，模拟褪色效果。"""
    factor = random.uniform(contrast_factor, 1)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def add_noise(image, noise_factor):
    """
    向图片中添加噪声。
    
    :param image: PIL Image对象。
    :param noise_factor: 噪声因子，值越大噪声越多。
    :return: 添加噪声后的图片。
    """
    pixels = list(image.getdata())
    for i in range(len(pixels)):
        r, g, b = pixels[i]
        noise = random.randint(-noise_factor, noise_factor)
        pixels[i] = (max(min(r + noise, 255), 0),
                     max(min(g + noise, 255), 0),
                     max(min(b + noise, 255), 0))
    noisy_image = Image.new('RGB', image.size)
    noisy_image.putdata(pixels)
    return noisy_image

def reduce_resolution(image, scale_factor):
    """
    降低图片的分辨率。
    
    :param image: PIL Image对象。
    :param scale_factor: 缩放因子，值越小分辨率越低。
    :return: 分辨率降低后的图片。
    """
    scale_factor = random.uniform(scale_factor, 1)
    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
    return image.resize(new_size, Image.LANCZOS).resize(image.size, Image.NEAREST)

def apply_dilation(image_np, kernel, dilate_iterations):
    """
    对图片进行膨胀操作。
    
    :param image_np: 图片的NumPy数组。
    :param kernel: 核大小。
    :param dilate_iterations: 膨胀次数。
    :return: 膨胀后的图片NumPy数组。
    """
    return cv2.dilate(image_np, kernel, iterations=dilate_iterations)

def apply_erosion(image_np, kernel, erode_iterations):
    """
    对图片进行腐蚀操作。
    
    :param image_np: 图片的NumPy数组。
    :param kernel: 核大小。
    :param erode_iterations: 腐蚀次数。
    :return: 腐蚀后的图片NumPy数组。
    """
    return cv2.erode(image_np, kernel, iterations=erode_iterations)

def binarize_image(image):
    """
    使用大津算法自动确定阈值，将图像转换为二值图像。
    
    :param image: 输入的PIL图像对象
    :return: 二值化后的图像
    """
    # 转换为灰度图
    grayscale_image = image.convert("L")
    image_np = np.array(grayscale_image)
    
    # 计算大津阈值
    histogram, bin_edges = np.histogram(image_np, bins=256, range=(0, 256))
    total_pixels = image_np.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), histogram)
    sum_foreground, weight_background = 0, 0
    
    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        sum_foreground += i * histogram[i]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    
    # 使用自动确定的阈值进行二值化
    binary_image = grayscale_image.point(lambda x: 255 if x > threshold else 0, mode='1')
    # print(f"自动确定的二值化阈值: {threshold}")
    
    return binary_image


def add_stain_overlay(image, stain_folder, intensity=0.9, max_stains=3):
    """
    将stain文件夹中的随机噪声图像叠加到主图像上，支持随机大小、角度、透明度和叠加次数。
    
    :param image: 主图像 (PIL Image 对象)
    :param stain_folder: 存储噪声图像的文件夹路径
    :param intensity: 最大叠加透明度，范围0到1，1为全强度叠加
    :param max_stains: 最大叠加次数，随机叠加0到max_stains个噪声图像
    :return: 叠加噪声后的图像
    """
    # 从stain_folder中获取噪声图像列表
    stain_images = list(Path(stain_folder).glob("*.png"))
    if not stain_images:
        raise ValueError("指定的噪声文件夹中没有找到图像")
    
    # 随机选择0到max_stains个叠加次数
    num_stains = random.randint(0, max_stains)
    
    # 将主图像转换为RGBA模式以支持透明度
    base_image = image.convert("RGBA")
    
    for _ in range(num_stains):
        stain_image_path = random.choice(stain_images)
        
        with Image.open(stain_image_path) as stain_image:
            # 将噪声图像转换为RGBA模式
            stain_image = stain_image.convert("RGBA")
            
            # 随机调整噪声图像的大小，可以比主图像大
            scale_factor = random.uniform(0.5, 3.0)  # 可根据需求调整范围
            new_size = (int(stain_image.width * scale_factor), int(stain_image.height * scale_factor))
            stain_image = stain_image.resize(new_size, Image.LANCZOS)
            
            # 再次随机调整X和Y轴的缩放
            x_scale = random.uniform(0.5, 1.5)  # X轴随机缩放
            y_scale = random.uniform(0.5, 1.5)  # Y轴随机缩放
            new_size = (int(stain_image.width * x_scale), int(stain_image.height * y_scale))
            stain_image = stain_image.resize(new_size, Image.LANCZOS)

            # 随机旋转噪声图像
            angle = random.uniform(0, 360)
            stain_image = stain_image.rotate(angle, expand=True)
            
            # 随机调整透明度
            alpha = random.uniform(0, intensity)
            stain_data = stain_image.getdata()
            new_data = [(r, g, b, int(a * alpha)) for r, g, b, a in stain_data]
            stain_image.putdata(new_data)
            
            # 随机选择叠加位置，可能会超出主图像边界
            max_x = max(0, base_image.width - stain_image.width)
            max_y = max(0, base_image.height - stain_image.height)
            offset_x = random.randint(-stain_image.width // 2, max_x + stain_image.width // 2)
            offset_y = random.randint(-stain_image.height // 2, max_y + stain_image.height // 2)
            
            # 将噪声图像叠加到主图像上
            temp_image = Image.new("RGBA", base_image.size)
            temp_image.paste(stain_image, (offset_x, offset_y), stain_image)
            base_image = Image.alpha_composite(base_image, temp_image)
    
    return base_image.convert("RGB")  # 返回 RGB 格式的图像



def add_background(image, background_folder, bg_intensity):
    """
    将background文件夹中的随机背景图像叠加到主图像下方，并控制背景透明度。
    
    :param image: 主图像 (PIL Image 对象)
    :param background_folder: 存储背景图像的文件夹路径
    :param intensity: 背景透明度，范围0到1，1为完全不透明
    :return: 带有背景的图像
    """
    # 从background_folder中获取背景图像列表
    background_images = list(Path(background_folder).glob("*.png"))
    if not background_images:
        raise ValueError("指定的背景文件夹中没有找到图像")
    
    # 随机选择一个背景图像
    background_image_path = random.choice(background_images)
    
    with Image.open(background_image_path) as background_image:
        # 调整背景图像大小以匹配主图像大小
        background_image = background_image.resize(image.size, Image.LANCZOS)
        
        # 将背景图像和主图像都转换为 RGBA 模式
        background_image = background_image.convert("RGBA")
        main_image_rgba = image.convert("RGBA")

        alpha = random.uniform(0, bg_intensity)
        
        # 调整背景图像的透明度
        blended_background = Image.blend(main_image_rgba, background_image, alpha=alpha)
        
    return blended_background.convert("RGB")  # 返回 RGB 格式的图像


def random_erase_white_lines(image, line_width=20, max_line_length=3000, line_number=1000):
    """
    根据图像尺寸随机插入白色线条，模拟纸张上的划痕或折痕效果。
    
    :param image: 输入的PIL图像对象
    :param line_width: 线条宽度，默认为2
    :param max_line_length: 线条最大长度，默认为30
    :param line_number: 插入线条数量的控制因子
    :return: 插入白色线条后的图像
    """
    img_np = np.array(image)
    h, w, _ = img_np.shape
    num_lines = random.randint(0, line_number)  # 生成指定数量的线条

    # 将 NumPy 数组转换为 PIL 图像对象，以便绘制
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)

    for _ in range(num_lines):
        # 随机选择线条的起点
        x_start = random.randint(0, w)
        y_start = random.randint(0, h)
        width = random.randint(1, line_width)

        # 随机生成线条的长度和角度
        line_length = random.randint(5, max_line_length)
        angle = random.uniform(0, 360)  # 随机角度

        # 计算线条的终点
        x_end = int(x_start + line_length * math.cos(math.radians(angle)))
        y_end = int(y_start + line_length * math.sin(math.radians(angle)))

        # 绘制白色线条，指定 RGB 颜色为 (255, 255, 255)
        draw.line((x_start, y_start, x_end, y_end), fill=(255, 255, 255), width=width)

    return img_pil