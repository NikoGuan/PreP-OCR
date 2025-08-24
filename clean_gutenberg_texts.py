#!/usr/bin/env python3
"""
Project Gutenberg文本清理脚本
移除*** START OF THE PROJECT GUTENBERG之前的所有内容和之后的空行
"""

import os
import re
from pathlib import Path
from typing import List

def clean_gutenberg_text(file_path: Path) -> bool:
    """
    清理单个Project Gutenberg文本文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否成功处理
    """
    try:
        # 读取原文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找*** START OF THE PROJECT GUTENBERG行
        start_pattern = re.compile(r'^\*\*\* START OF THE PROJECT GUTENBERG', re.IGNORECASE)
        start_index = None
        
        for i, line in enumerate(lines):
            if start_pattern.match(line.strip()):
                start_index = i
                break
        
        if start_index is None:
            print(f"⚠️  未找到START标记: {file_path.name}")
            return False
        
        # 从START行之后开始，跳过空行和重复信息
        content_start = start_index + 1
        while content_start < len(lines):
            line = lines[content_start].strip()
            # 跳过空行和常见的重复信息
            if (line == "" or 
                line.startswith("This eBook was produced") or
                line.startswith("Charles Franks") or
                line.startswith("Online Distributed Proofreading")):
                content_start += 1
            else:
                break
        
        if content_start >= len(lines):
            print(f"⚠️  未找到正文内容: {file_path.name}")
            return False
        
        # 提取清理后的内容
        cleaned_lines = lines[content_start:]
        
        # 移除开头的多余空行
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)
        
        if not cleaned_lines:
            print(f"⚠️  清理后内容为空: {file_path.name}")
            return False
        
        # 写入清理后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        print(f"✅ 已清理: {file_path.name} (移除了{content_start}行前缀)")
        return True
        
    except Exception as e:
        print(f"❌ 处理失败 {file_path.name}: {e}")
        return False

def batch_clean_directory(directory_path: str):
    """
    批量清理目录下所有txt文件
    
    Args:
        directory_path: 目录路径
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ 目录不存在: {directory_path}")
        return
    
    # 获取所有txt文件
    txt_files = list(directory.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ 目录中未找到txt文件: {directory_path}")
        return
    
    print(f"📁 找到 {len(txt_files)} 个txt文件")
    print(f"🔧 开始批量清理...\n")
    
    success_count = 0
    
    for file_path in txt_files:
        if clean_gutenberg_text(file_path):
            success_count += 1
    
    print(f"\n🎉 处理完成!")
    print(f"📊 成功处理: {success_count}/{len(txt_files)} 个文件")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清理Project Gutenberg文本文件")
    parser.add_argument("--dir", "-d", 
                       default="/home/ubuntu/PreP-OCR/data/Novel_data_UTF8_new",
                       help="文本文件目录路径")
    parser.add_argument("--preview", "-p", action="store_true",
                       help="预览模式：只显示将要处理的文件，不实际修改")
    
    args = parser.parse_args()
    
    if args.preview:
        directory = Path(args.dir)
        txt_files = list(directory.glob("*.txt"))
        print(f"📁 将要处理的文件 ({len(txt_files)} 个):")
        for file_path in txt_files:
            print(f"  - {file_path.name}")
        print(f"\n使用 --dir {args.dir} 开始实际处理")
    else:
        batch_clean_directory(args.dir)

if __name__ == "__main__":
    main()