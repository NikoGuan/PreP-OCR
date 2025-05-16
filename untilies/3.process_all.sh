#!/bin/bash

# 定义文件夹路径和 Python 脚本路径
INPUT_FOLDER="/home/shuhaog/PrePOCR/data/Novel_data_UTF8_processed"
PYTHON_SCRIPT="python /home/shuhaog/PrePOCR/untilies/3.gerenate_data.py"

# 设置并发的最大任务数
MAX_JOBS=50

# 逐个处理 txt 文件
for txt_file in "$INPUT_FOLDER"/*.txt; do
    # 启动一个任务并在后台运行
    $PYTHON_SCRIPT "$txt_file" &

    # 检查当前运行的后台任务数
    while (( $(jobs | grep -c 'Running') >= MAX_JOBS )); do
        # 等待任一后台任务完成再继续
        sleep 1
    done
done

# 等待所有后台任务完成
wait

echo "所有任务已完成！"
