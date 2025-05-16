import os

# 设置文件路径
input_folder = "/home/shuhaog/PrePOCR/data/Novel_data_UTF8"
output_folder = "/home/shuhaog/PrePOCR/data/Novel_data_UTF8_processed"

# 设置行的最大长度，超过此长度将自动换行
max_line_length = 80  # 设置为印刷书本的平均长度

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            while len(line) > 100:
                # 查找最近的空格，避免分割单词
                split_index = line.rfind(' ', 0, max_line_length)
                if split_index == -1:  # 没有找到空格，直接按最大长度分割
                    split_index = max_line_length
                # 写入当前行，添加换行符
                outfile.write(line[:split_index] + '\n')
                line = line[split_index:].lstrip()  # 去掉前导空格
            # 写入剩余的短行
            outfile.write(line)

# 遍历文件夹中的所有txt文件
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        process_file(input_file, output_file)

print("处理完成，长行已分割并存储在新文件夹中。")