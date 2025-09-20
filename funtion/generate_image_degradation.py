import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from pathlib import Path
import os
import argparse

def add_random_degradation_patches(image, num_patches=None, min_size=15, max_size=80):
    """
    根据图像尺寸随机添加小区域，实现颜色变淡和毛玻璃效果
    """
    # 转换为PIL图像以便处理
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    w, h = pil_image.size
    image_area = w * h
    
    # 计算缩放因子（在函数开始就定义）
    scale_factor = max(0.2, min(w, h) / 1000)
    
    if num_patches is None:
        # 根据图像尺寸动态计算区域数量
        # 基础数量：每10万像素约12-20个区域
        base_patches_per_100k = random.randint(12, 20)
        calculated_patches = int((image_area / 40000) )
        
        # 设置最小和最大值范围（从0开始）
        min_patches = calculated_patches  # 从0开始
        max_patches = calculated_patches + 10  # 最多40个
        
        # 在计算出的范围内随机选择
        num_patches = random.randint(min_patches, max_patches)
        # num_patches = 30
        
    # 根据图像尺寸调整区域大小范围
    adjusted_min_size = max(8, int(min_size * scale_factor))
    calculated_max_size = int(max_size * scale_factor)
    adjusted_max_size = max(adjusted_min_size + 5, calculated_max_size)
    adjusted_min_size = 100
    adjusted_max_size = 150
    
    for _ in range(num_patches):
        # 随机生成区域大小（根据图像尺寸调整）
        patch_w = random.randint(adjusted_min_size, adjusted_max_size)
        patch_h = random.randint(adjusted_min_size, adjusted_max_size)
        
        # 随机生成区域位置
        x = random.randint(0, max(0, w - patch_w))
        y = random.randint(0, max(0, h - patch_h))
        
        # 随机选择效果类型
        effect_type = random.choice(['lighten', 'fade_out', 'frosted_glass'])
        
        # 提取区域
        region = pil_image.crop((x, y, x + patch_w, y + patch_h))
        
        if effect_type == 'lighten':
            # 只对笔画变淡效果 - 背景保持不变
            region_array = np.array(region)
            
            # 创建文字掩码（二值化找笔画）
            if len(region_array.shape) == 3:
                # 彩色图像转灰度
                gray = cv2.cvtColor(region_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = region_array
            
            # 使用OTSU二值化找到文字区域
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_mask = binary == 0  # 黑色区域认为是文字
            
            # 只对文字区域变淡
            lightened_region_array = region_array.copy()
            lighten_factor = random.uniform(1.1, 1.8)  # 随机变淡强度
            
            if len(region_array.shape) == 3:
                # 彩色图像
                for c in range(3):
                    channel = lightened_region_array[:, :, c].astype(float)
                    # 只对文字区域应用变淡
                    channel[text_mask] = np.clip(channel[text_mask] * lighten_factor, 0, 255)
                    lightened_region_array[:, :, c] = channel.astype(np.uint8)
            else:
                # 灰度图像
                channel = lightened_region_array.astype(float)
                channel[text_mask] = np.clip(channel[text_mask] * lighten_factor, 0, 255)
                lightened_region_array = channel.astype(np.uint8)
            
            lightened_region = Image.fromarray(lightened_region_array)
            
        elif effect_type == 'fade_out':
            # 只对笔画褪色效果 - 背景保持不变
            region_array = np.array(region)
            
            # 创建文字掩码
            if len(region_array.shape) == 3:
                gray = cv2.cvtColor(region_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = region_array
            
            # 使用OTSU二值化找到文字区域
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_mask = binary == 0  # 黑色区域认为是文字
            
            # 只对文字区域褪色
            faded_region_array = region_array.copy()
            fade_factor = random.uniform(1.2, 2.0)  # 随机褪色强度
            
            if len(region_array.shape) == 3:
                # 彩色图像 - 降低饱和度并增加亮度
                for c in range(3):
                    channel = faded_region_array[:, :, c].astype(float)
                    # 向255靠近，实现褪色效果
                    channel[text_mask] = channel[text_mask] + (255 - channel[text_mask]) * (fade_factor - 1) / fade_factor
                    channel[text_mask] = np.clip(channel[text_mask], 0, 255)
                    faded_region_array[:, :, c] = channel.astype(np.uint8)
            else:
                # 灰度图像
                channel = faded_region_array.astype(float)
                channel[text_mask] = channel[text_mask] + (255 - channel[text_mask]) * (fade_factor - 1) / fade_factor
                channel[text_mask] = np.clip(channel[text_mask], 0, 255)
                faded_region_array = channel.astype(np.uint8)
            
            lightened_region = Image.fromarray(faded_region_array)
            
        elif effect_type == 'frosted_glass':
            # 毛玻璃效果 - 对整个区域应用模糊和亮度调整
            blur_radius = random.uniform(0.8, 2.0)  # 随机模糊强度
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # 轻微增亮模拟毛玻璃效果
            brightness_factor = random.uniform(1.05, 1.3)  # 随机亮度因子
            brightness_enhancer = ImageEnhance.Brightness(blurred_region)
            lightened_region = brightness_enhancer.enhance(brightness_factor)
        
        # 将处理后的区域粘贴回原图
        pil_image.paste(lightened_region, (x, y))
    
    return pil_image

def apply_random_shadow_effect(image, intensity=None):
    """
    添加随机方向的轻微重影效果，保持原图不变
    只有5%的概率会发生重影
    """
    # 5%的概率发生重影
    if random.random() > 0.05:
        return image
    
    if intensity is None:
        intensity = random.uniform(0.3, 0.8)
    
    # 转换为PIL图像以便处理
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    # 随机选择重影方向 - 8个方向
    directions = [
        (1, 0),   # 右
        (-1, 0),  # 左
        (0, 1),   # 下
        (0, -1),  # 上
        (1, 1),   # 右下
        (-1, 1),  # 左下
        (1, -1),  # 右上
        (-1, -1)  # 左上
    ]
    
    dx, dy = random.choice(directions)
    
    # 随机偏移距离
    offset_x = int(dx * random.uniform(1, 3))
    offset_y = int(dy * random.uniform(1, 3))
    
    # 创建结果图像，先复制原图
    result_image = pil_image.copy()
    
    # 计算粘贴位置
    paste_x = max(0, offset_x)
    paste_y = max(0, offset_y)
    crop_x = max(0, -offset_x)
    crop_y = max(0, -offset_y)
    
    # 计算有效区域大小
    w, h = pil_image.size
    copy_w = min(w - abs(offset_x), w)
    copy_h = min(h - abs(offset_y), h)
    
    if copy_w > 0 and copy_h > 0:
        # 裁剪原图像作为重影
        shadow_crop = pil_image.crop((crop_x, crop_y, crop_x + copy_w, crop_y + copy_h))
        
        # 创建淡化的重影
        shadow_enhancer = ImageEnhance.Brightness(shadow_crop)
        faded_shadow = shadow_enhancer.enhance(1.3)  # 让重影更亮
        
        contrast_enhancer = ImageEnhance.Contrast(faded_shadow)
        faded_shadow = contrast_enhancer.enhance(0.5)  # 降低对比度
        
        # 转换为RGBA模式以支持透明度
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        if faded_shadow.mode != 'RGBA':
            faded_shadow = faded_shadow.convert('RGBA')
        
        # 调整重影的透明度
        shadow_opacity = int(255 * random.uniform(0.2, 0.4))  # 更低的透明度
        shadow_data = []
        for pixel in faded_shadow.getdata():
            # 只对非白色像素添加透明度
            if pixel[:3] != (255, 255, 255):  # 如果不是白色
                shadow_data.append((pixel[0], pixel[1], pixel[2], shadow_opacity))
            else:  # 白色像素完全透明
                shadow_data.append((255, 255, 255, 0))
        faded_shadow.putdata(shadow_data)
        
        # 在重影位置先粘贴重影
        temp_image = Image.new('RGBA', result_image.size, (255, 255, 255, 0))
        temp_image.paste(faded_shadow, (paste_x, paste_y), faded_shadow)
        
        # 将重影合成到背景
        result_image = Image.alpha_composite(result_image, temp_image)
        
        # 再将原图完整地粘贴到最上层，保持原图不变
        if pil_image.mode != 'RGBA':
            original_rgba = pil_image.convert('RGBA')
        else:
            original_rgba = pil_image
        result_image = Image.alpha_composite(result_image, original_rgba)
    
    # 转换回原始模式
    if image.mode == 'RGB':
        return result_image.convert('RGB')
    elif image.mode == 'L':
        return result_image.convert('L')
    else:
        return result_image

def apply_double_print_effect(image, offset_x=None, offset_y=None, opacity=None):
    """
    模拟重印效果 - 创建轻微偏移的重叠图像
    """
    if offset_x is None:
        offset_x = random.randint(1, 2)
    if offset_y is None:
        offset_y = random.randint(1, 2)
    if opacity is None:
        opacity = random.uniform(0.2, 0.5)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # 创建输出图像
    result_array = img_array.copy()
    
    # 计算偏移后的有效区域
    if offset_x > 0 and offset_y > 0:
        # 右下偏移
        overlay_region = img_array[:-offset_y, :-offset_x]
        target_y_start, target_x_start = offset_y, offset_x
    elif offset_x > 0 and offset_y <= 0:
        # 右上偏移
        overlay_region = img_array[-offset_y:, :-offset_x]
        target_y_start, target_x_start = 0, offset_x
    elif offset_x <= 0 and offset_y > 0:
        # 左下偏移
        overlay_region = img_array[:-offset_y, -offset_x:]
        target_y_start, target_x_start = offset_y, 0
    else:
        # 左上偏移
        overlay_region = img_array[-offset_y:, -offset_x:]
        target_y_start, target_x_start = 0, 0
    
    overlay_h, overlay_w = overlay_region.shape[:2]
    
    # 混合图像
    if len(img_array.shape) == 3:
        # 彩色图像
        for c in range(3):
            result_array[target_y_start:target_y_start+overlay_h, 
                        target_x_start:target_x_start+overlay_w, c] = (
                result_array[target_y_start:target_y_start+overlay_h, 
                           target_x_start:target_x_start+overlay_w, c] * (1-opacity) +
                overlay_region[:, :, c] * opacity
            ).astype(np.uint8)
    else:
        # 灰度图像
        result_array[target_y_start:target_y_start+overlay_h, 
                    target_x_start:target_x_start+overlay_w] = (
            result_array[target_y_start:target_y_start+overlay_h, 
                       target_x_start:target_x_start+overlay_w] * (1-opacity) +
            overlay_region * opacity
        ).astype(np.uint8)
    
    return Image.fromarray(result_array)

def add_ink_bleeding(image, intensity=None):
    """
    添加墨水渗透效果 - 保持原图颜色
    """
    if intensity is None:
        intensity = random.uniform(0.3, 1.0)  # 降低强度
    
    # 转换为PIL图像以便处理
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    # 只对文字部分进行轻微的膨胀效果
    img_array = np.array(pil_image)
    
    if len(img_array.shape) == 3:
        # 彩色图像 - 分别处理每个通道
        result = img_array.copy()
        kernel_size = max(1, int(intensity))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for c in range(3):
            # 对每个颜色通道单独应用轻微膨胀
            channel = img_array[:, :, c]
            # 只对暗色区域（文字）进行膨胀
            mask = channel < 200  # 只处理非白色区域
            if mask.any():
                dilated_channel = cv2.dilate(channel, kernel, iterations=1)
                # 只在文字区域应用膨胀效果
                result[:, :, c] = np.where(mask, dilated_channel, channel)
        
        return Image.fromarray(result)
    else:
        # 灰度图像
        kernel_size = max(1, int(intensity))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 只对文字区域进行膨胀
        mask = img_array < 200
        dilated = cv2.dilate(img_array, kernel, iterations=1)
        result = np.where(mask, dilated, img_array)
        
        return Image.fromarray(result)

def process_single_image(input_path, output_path, effects_config=None):
    """
    处理单张图像
    """
    if effects_config is None:
        effects_config = {
            'degradation_patches': True,
            'camera_shake': True,
            'defocus_blur': True,
            'double_print': True,
            'ink_bleeding': True
        }
    
    try:
        # 读取图像
        image = Image.open(input_path)
        
        # 随机应用效果
        available_effects = []
        
        if effects_config.get('degradation_patches', True):
            available_effects.append(('degradation_patches', add_random_degradation_patches))
        
        if effects_config.get('shadow_effect', True):
            available_effects.append(('shadow_effect', apply_random_shadow_effect))
            
            
        
        # 随机选择1-2个效果应用
        num_effects = random.randint(1, min(2, len(available_effects)))
        selected_effects = random.sample(available_effects, num_effects)
        
        # 应用选中的效果
        processed_image = image
        for effect_name, effect_func in selected_effects:
            processed_image = effect_func(processed_image)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 强制转换为RGB格式保存（即使是灰度图像）
        if processed_image.mode == 'L':
            # 灰度图像转换为RGB
            processed_image = processed_image.convert('RGB')
        elif processed_image.mode == 'RGBA':
            # RGBA图像转换为RGB（白色背景）
            rgb_image = Image.new('RGB', processed_image.size, (255, 255, 255))
            rgb_image.paste(processed_image, mask=processed_image.split()[-1] if processed_image.mode == 'RGBA' else None)
            processed_image = rgb_image
        
        # 保存处理后的图像
        processed_image.save(output_path)
        
        return True
        
    except Exception as e:
        import traceback
        print(f"处理图像 {input_path} 时出错: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def batch_process_images(input_folder, output_folder, effects_config=None):
    """
    批量处理文件夹中的所有图像
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在文件夹 {input_folder} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像文件
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(image_files, 1):
        output_file = output_path / input_file.name
        
        print(f"处理第 {i}/{len(image_files)} 个文件: {input_file.name}")
        
        if process_single_image(str(input_file), str(output_file), effects_config):
            successful += 1
        else:
            failed += 1
    
    print(f"处理完成！成功: {successful}, 失败: {failed}")

def main():
    parser = argparse.ArgumentParser(description='图像降质处理工具')
    parser.add_argument('--input', '-i', required=True, help='输入文件夹路径')
    parser.add_argument('--output', '-o', required=True, help='输出文件夹路径')
    parser.add_argument('--no-patches', action='store_true', help='禁用对比度降低效果')
    parser.add_argument('--no-shadow', action='store_true', help='禁用重影效果')
    
    args = parser.parse_args()
    
    # 配置效果
    effects_config = {
        'degradation_patches': not args.no_patches,
        'shadow_effect': not args.no_shadow
    }
    
    # 执行批量处理
    batch_process_images(args.input, args.output, effects_config)

if __name__ == "__main__":
    # 如果直接运行脚本，使用默认参数
    if len(os.sys.argv) == 1:
        # 默认处理示例
        input_folder = "/home/ubuntu/PreP-OCR/data/outputgai3/noisy"
        output_folder = "/home/ubuntu/PreP-OCR/data/outputgai3/noisy_pad2"
        
        print(f"使用默认路径:")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        
        if os.path.exists(input_folder):
            # 使用默认的效果配置
            default_effects_config = {
                'degradation_patches': True,
                'shadow_effect': True
            }
            batch_process_images(input_folder, output_folder, default_effects_config)
        else:
            print(f"输入文件夹不存在: {input_folder}")
            print("请使用命令行参数指定正确的路径:")
            print("python generate_image_degradation.py --input /path/to/input --output /path/to/output")
    else:
        main()