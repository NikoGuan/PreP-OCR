#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PreP-OCR Data Generation Tool (并行版本)
=====================================

此工具通过以下步骤生成合成OCR训练数据：
1. 并行创建干净的基础图像
2. 为每个基础图像并行添加4个级别的噪声/劣化
3. 生成对应的真值文件

特性：
- 多线程并行处理，生成速度更快
- 实时统计的进度条显示
- 可配置工作线程数量
- 线程安全的文件操作

使用方法：
    python generate_ocr_data.py --base 5                    # 生成5个基础图像 + 20个噪声变体
    python generate_ocr_data.py --base 10 --workers 8       # 使用8个工作线程
    python generate_ocr_data.py --base 100 --workers 16     # 大批量生成，使用16个工作线程

性能说明：
- 默认工作线程数: min(32, CPU核心数 + 4)
- I/O密集型操作从更多工作线程中受益更多
- 更多工作线程会因PIL图像对象增加内存使用量
"""

import random
import argparse
from pathlib import Path
from typing import Tuple, List
import sys
import os
from PIL import Image
import concurrent.futures
import threading
import time
from tqdm import tqdm

# Add function directory to path
sys.path.append(str(Path('./funtion')))

from generate_base_add_noise import generate_base_image, add_noise_and_reduce_resolution, binarize_image

class OCRDataGenerator:
    """OCR synthetic data generator"""
    
    def __init__(self, max_workers=None):
        """Initialize the generator"""
        self.text_folder = Path("./data/Novel_data_UTF8_processed")
        self.output_folder = Path("./data/output")
        self.setup_directories()
        self.base_images = []  # Store base images for noise processing
        # 优化worker数量：对于I/O密集型任务，建议使用CPU核心数的1-2倍
        # 但不超过16个，避免过度竞争
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
        self.max_workers = max_workers or min(16, cpu_count)
        self.lock = threading.Lock()
    
    def setup_directories(self):
        """Create output directories"""
        self.clean_dir = self.output_folder / "clean"
        self.noisy_dir = self.output_folder / "noisy" 
        self.gt_dir = self.output_folder / "ground_truth"
        
        for dir_path in [self.clean_dir, self.noisy_dir, self.gt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_text_chunk_that_fits(self, max_attempts: int = 10) -> Tuple[str, str, str]:
        """Get a text chunk that will fit properly in the generated image"""
        text_files = list(self.text_folder.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No text files found in {self.text_folder}")
        
        for attempt in range(max_attempts):
            # Select random text file and chunk
            text_file = random.choice(text_files)
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 90% chance: normal length (15-25 lines)
            # 10% chance: varied length (1 to 45 lines)
            if random.random() < 0.9:
                # Normal length
                lines_per_image = random.randint(15, 45)
            else:
                # Varied length for diversity
                lines_per_image = random.randint(1, 45)
            
            if len(lines) >= lines_per_image:
                start_idx = random.randint(0, len(lines) - lines_per_image)
                selected_lines = lines[start_idx:start_idx + lines_per_image]
            else:
                selected_lines = lines
            
            # Format text with conservative indentation
            formatted_lines = []
            for line in selected_lines:
                line = line.strip()
                if line == "":
                    continue  # Skip empty lines to save space
                # Add modest indentation for new paragraphs
                if len(formatted_lines) > 0 and random.random() < 0.3:
                    formatted_lines.append(" " * random.randint(2, 4) + line)
                else:
                    formatted_lines.append(line)
            
            text_content = "\n".join(formatted_lines).strip()
            
            # Validate text content
            if text_content and len(text_content) > 10:  # Lower threshold for short texts
                return text_content, text_file.stem, f"attempt_{attempt}_lines_{lines_per_image}"
        
        # Fallback: use a simple text chunk
        return "Sample text for OCR training data generation.", "fallback", "simple"
    
    def generate_filename(self) -> str:
        """Generate a sequential filename"""
        return f"{random.randint(10000000000, 99999999999)}"
    
    def _generate_single_base_image(self, index: int) -> dict:
        """Generate a single base image (for parallel execution)"""
        try:
            # Get text content that fits properly
            text_content, source_file, _ = self.get_text_chunk_that_fits()
            
            # Generate base image with conservative settings
            preset = random.randint(1, 8)
            
            # Use smaller font size and better spacing to ensure text fits
            base_image = generate_base_image(
                text_content, 
                preset=preset,
                font_size=random.randint(40, 60),  # Smaller font range
                line_spacing=random.randint(5, 15),  # Tighter line spacing
                char_spacing=random.randint(1, 5),   # Tighter char spacing
                margin=random.randint(80, 120)       # Smaller margins
            )
            
            # Conservative scaling to ensure text visibility
            scale = random.uniform(0.8, 1.1)  # Less aggressive scaling
            new_size = (int(base_image.width * scale), int(base_image.height * scale))
            base_image = base_image.resize(new_size, Image.LANCZOS)
            
            # Generate filename
            filename = self.generate_filename()
            
            # Save base image
            image_path = self.clean_dir / f"{filename}_clean.jpg"
            base_image.save(image_path, format="JPEG", quality=95)
            
            # Save ground truth
            gt_path = self.gt_dir / f"{filename}_clean.txt"
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Return data for noise processing
            return {
                'image': base_image,
                'text': text_content,
                'filename': filename,
                'source_file': source_file,
                'success': True,
                'index': index
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'index': index
            }
    
    def generate_base_images(self, num_images: int):
        """Generate base clean images in parallel and store them for noise processing"""
        print(f"Generating {num_images} base images using {self.max_workers} workers...")
        
        self.base_images = []  # Reset base images list
        success_count = 0
        
        # Use ThreadPoolExecutor for parallel generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._generate_single_base_image, i): i 
                for i in range(num_images)
            }
            
            # Process results as they complete
            with tqdm(total=num_images, desc="Base images", unit="img") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    result = future.result()
                    
                    if result['success']:
                        # Thread-safe append to base_images list
                        with self.lock:
                            self.base_images.append({
                                'image': result['image'],
                                'text': result['text'],
                                'filename': result['filename'],
                                'source_file': result['source_file']
                            })
                        success_count += 1
                        pbar.set_postfix({'successful': success_count})
                    else:
                        pbar.write(f"✗ Failed to generate base image {result['index'] + 1}: {result['error']}")
                    
                    pbar.update(1)
        
        print(f"Base image generation completed: {success_count}/{num_images} successful")
        return success_count
    
    def _generate_single_noisy_variant(self, base_data: dict, level: str) -> dict:
        """Generate a single noisy variant (for parallel execution)"""
        try:
            base_image = base_data['image']
            text_content = base_data['text']
            base_filename = base_data['filename']
            
            # Add noise to the base image
            noisy_image = add_noise_and_reduce_resolution(base_image, preset=level)
            
            # 10% chance of binarization
            if random.random() < 0.1:
                noisy_image = binarize_image(noisy_image)
            
            # Save noisy image
            noisy_filename = f"{base_filename}_{level}"
            image_path = self.noisy_dir / f"{noisy_filename}.jpg"
            noisy_image.save(image_path, format="JPEG", quality=85)
            
            # Save ground truth (same as base image)
            gt_path = self.gt_dir / f"{noisy_filename}.txt"
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            return {
                'success': True,
                'filename': noisy_filename,
                'level': level,
                'base_filename': base_filename
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'level': level,
                'base_filename': base_data.get('filename', 'unknown')
            }
    
    def generate_noisy_variants(self):
        """Generate noisy variants from the base images in parallel"""
        if not self.base_images:
            print("No base images available for noise processing")
            return 0
        
        noise_levels = ["1_level", "2_level", "3_level", "4_level"]
        total_success = 0
        
        # Create all tasks (base_image, level) combinations
        tasks = []
        for base_data in self.base_images:
            for level in noise_levels:
                tasks.append((base_data, level))
        
        total_tasks = len(tasks)
        print(f"\nGenerating {total_tasks} noisy variants from {len(self.base_images)} base images using {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._generate_single_noisy_variant, base_data, level): (base_data, level)
                for base_data, level in tasks
            }
            
            # Process results as they complete
            with tqdm(total=total_tasks, desc="Noisy variants", unit="img") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    result = future.result()
                    
                    if result['success']:
                        total_success += 1
                        pbar.set_postfix({'successful': total_success})
                    else:
                        pbar.write(f"✗ Failed to generate {result['level']} for {result['base_filename']}: {result['error']}")
                    
                    pbar.update(1)
        
        print(f"Noisy variant generation completed: {total_success}/{total_tasks} successful")
        return total_success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PreP-OCR Synthetic Data Generator")
    parser.add_argument("--base", type=int, default=5, 
                       help="Number of base images to generate (each produces 4 noisy variants)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect based on CPU cores)")
    
    args = parser.parse_args()
    
    if args.base <= 0:
        print("Please specify a positive number of base images to generate.")
        print("Example: python generate_ocr_data.py --base 5")
        return 1
    
    try:
        generator = OCRDataGenerator(max_workers=args.workers)
        
        # Step 1: Generate base images
        base_count = generator.generate_base_images(args.base)
        
        if base_count == 0:
            print("❌ No base images were generated successfully.")
            return 1
        
        # Step 2: Generate noisy variants from base images
        noisy_count = generator.generate_noisy_variants()
        
        print(f"\n🎉 Data generation completed!")
        print(f"📊 Results:")
        print(f"  - Base images: {base_count}")
        print(f"  - Noisy variants: {noisy_count}")
        print(f"  - Total images: {base_count + noisy_count}")
        print(f"  - Ground truth files: {base_count + noisy_count}")
        print(f"\n📁 Output directory: {generator.output_folder}")
        print(f"  - Clean images: {generator.clean_dir}")
        print(f"  - Noisy images: {generator.noisy_dir}")
        print(f"  - Ground truth: {generator.gt_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())