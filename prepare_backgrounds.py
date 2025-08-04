#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
背景图片预处理脚本
================

这个脚本用于预处理背景图片和污渍图片，使其适合作为叠加层使用。
主要功能：
1. 淡化深色墨迹
2. 将白色区域转为透明
"""

from PIL import Image
from pathlib import Path

def process_image_for_overlay(image_path, output_path, white_threshold=240, ink_threshold=80, lighten_factor=1.5):
    """
    处理单张图片，使其适合作为叠加层
    
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :param white_threshold: 白色阈值（大于此值转为透明）
    :param ink_threshold: 墨迹阈值（小于此值进行淡化）
    :param lighten_factor: 淡化系数
    """
    try:
        with Image.open(image_path) as img:
            # 转为灰度图处理墨迹
            gray_img = img.convert("L")
            pixels = gray_img.load()
            
            # 淡化深色墨迹
            for y in range(gray_img.height):
                for x in range(gray_img.width):
                    brightness = pixels[x, y]
                    if brightness < ink_threshold:
                        new_brightness = int(brightness + (255 - brightness) * (lighten_factor - 1))
                        pixels[x, y] = min(new_brightness, 255)

            # 转为RGBA处理透明度
            rgba_img = gray_img.convert("RGBA")
            datas = rgba_img.getdata()

            new_data = []
            for item in datas:
                # 白色区域变透明
                if item[0] >= white_threshold:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)

            rgba_img.putdata(new_data)
            rgba_img.save(output_path, "PNG")
            return True
            
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return False

def process_folder(input_folder, output_folder):
    """处理文件夹中的所有图片"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"输入文件夹不存在：{input_folder}")
        return 0
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(ext))
        image_files.extend(input_path.glob(ext.upper()))
    
    if not image_files:
        print(f"在{input_folder}中没有找到图片文件")
        return 0
    
    print(f"正在处理{len(image_files)}张图片...")
    success_count = 0
    
    for img_file in image_files:
        output_file = output_path / f"{img_file.stem}.png"
        if process_image_for_overlay(img_file, output_file):
            success_count += 1
            print(f"✓ 处理完成：{img_file.name} -> {output_file.name}")
        else:
            print(f"✗ 处理失败：{img_file.name}")
    
    print(f"处理完成：{success_count}/{len(image_files)} 成功")
    return success_count

def main():
    """主函数"""
    print("PreP-OCR 背景图片预处理工具")
    print("=" * 40)
    
    base_folder = Path("./noise_img")
    
    # 处理背景图片
    background_input = base_folder / "background"
    background_output = base_folder / "background_p"
    
    print("\n1. 处理背景图片...")
    if background_input.exists():
        bg_count = process_folder(background_input, background_output)
        print(f"背景图片处理完成：{bg_count} 张图片")
    else:
        print(f"背景文件夹不存在：{background_input}")
        print("请将背景图片放入 noise_img/background/ 文件夹")
    
    # 处理污渍图片
    stain_input = base_folder / "stain"
    stain_output = base_folder / "stain_p"
    
    print("\n2. 处理污渍图片...")
    if stain_input.exists():
        stain_count = process_folder(stain_input, stain_output)
        print(f"污渍图片处理完成：{stain_count} 张图片")
    else:
        print(f"污渍文件夹不存在：{stain_input}")
        print("如需添加污渍效果，请将污渍图片放入 noise_img/stain/ 文件夹")
        # 创建空文件夹备用
        stain_input.mkdir(parents=True, exist_ok=True)
        stain_output.mkdir(parents=True, exist_ok=True)
    
    print("\n🎉 背景预处理完成！")

if __name__ == "__main__":
    main()