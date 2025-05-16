from PIL import Image
from pathlib import Path

def lighten_and_make_transparent(image_path, output_path, white_threshold=240, ink_threshold=80, lighten_factor=1.5):
    """
    先将图像中的浓墨痕迹淡化，然后将白色区域转换为透明。
    
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    :param white_threshold: 白色阈值，定义将哪些像素转换为透明，默认240
    :param ink_threshold: 墨痕浓度阈值，低于此值的像素被视为浓墨痕迹，默认80
    :param lighten_factor: 浓墨痕迹淡化因子，默认1.5
    """
    with Image.open(image_path) as img:
        # 转换为灰度图以淡化浓墨痕迹
        gray_img = img.convert("L")
        pixels = gray_img.load()
        
        for y in range(gray_img.height):
            for x in range(gray_img.width):
                brightness = pixels[x, y]
                if brightness < ink_threshold:  # 将浓墨痕迹变淡
                    new_brightness = int(brightness + (255 - brightness) * (lighten_factor - 1))
                    pixels[x, y] = min(new_brightness, 255)

        # 转换为 RGBA 模式以应用透明化
        rgba_img = gray_img.convert("RGBA")
        datas = rgba_img.getdata()

        new_data = []
        for item in datas:
            # 将接近白色的像素转换为透明
            if item[0] >= white_threshold and item[1] >= white_threshold and item[2] >= white_threshold:
                new_data.append((255, 255, 255, 0))  # 白色变为透明
            else:
                new_data.append(item)

        rgba_img.putdata(new_data)
        rgba_img.save(output_path, "PNG")
        print(f"已保存处理后的图像到: {output_path}")

def process_folder(input_folder, output_folder, white_threshold=240, ink_threshold=80, lighten_factor=1.5):
    """遍历文件夹，淡化浓墨痕迹并将白色背景转换为透明，结果保存到新的文件夹。"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for img_file in Path(input_folder).glob("*.[jp][pn]g"):
        output_file = Path(output_folder) / img_file.stem
        output_file = output_file.with_suffix(".png")  # 将输出文件扩展名设为 .png
        lighten_and_make_transparent(img_file, output_file, white_threshold, ink_threshold, lighten_factor)

# 设置输入和输出文件夹路径
base_folder = "../noise_img"
background_input = f"{base_folder}/background"
stain_input = f"{base_folder}/stain"

# 输出文件夹路径
background_output = f"{base_folder}/background_p"
stain_output = f"{base_folder}/stain_p"

# 批量处理图像，淡化浓墨痕迹并将白色背景转换为透明
process_folder(background_input, background_output, white_threshold=240, ink_threshold=80, lighten_factor=1.5)
process_folder(stain_input, stain_output, white_threshold=240, ink_threshold=80, lighten_factor=1.5)