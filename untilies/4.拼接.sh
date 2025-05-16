#!/bin/bash

# 循环 50 次，每次后台运行 python 4.拼接.py
for i in {1..60}
do
    python 4.拼接2.py &
done

# 等待所有后台任务结束
wait
echo "所有任务已完成。"