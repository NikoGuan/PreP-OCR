#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PreP-OCR Data Generation Tool
=============================

This tool generates synthetic OCR training data by:
1. Creating clean base images from text
2. Adding 4 levels of noise/degradation to each base image
3. Generating corresponding ground truth files

Usage:
    python generate_ocr_data.py --base 5  # Generate 5 base images + 20 noisy variants
"""

import random
import argparse
from pathlib import Path
from typing import Tuple, List
import sys
from PIL import Image

# Add function directory to path
sys.path.append(str(Path('./funtion')))

from generate_base_add_noise import generate_base_image, add_noise_and_reduce_resolution, binarize_image

class OCRDataGenerator:
    """OCR synthetic data generator"""
    
    def __init__(self):
        """Initialize the generator"""
        self.text_folder = Path("./data/Novel_data_UTF8_processed")
        self.output_folder = Path("./data/output")
        self.setup_directories()
        self.base_images = []  # Store base images for noise processing
    
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
    
    def generate_base_images(self, num_images: int):
        """Generate base clean images and store them for noise processing"""
        print(f"Generating {num_images} base images...")
        
        self.base_images = []  # Reset base images list
        success_count = 0
        
        for i in range(num_images):
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
                
                # Store for noise processing
                self.base_images.append({
                    'image': base_image,
                    'text': text_content,
                    'filename': filename,
                    'source_file': source_file
                })
                
                success_count += 1
                print(f"✓ Generated base image {success_count}/{num_images}: {filename}_clean.jpg")
                
            except Exception as e:
                print(f"✗ Failed to generate base image {i+1}: {e}")
                continue
        
        print(f"Base image generation completed: {success_count}/{num_images} successful")
        return success_count
    
    def generate_noisy_variants(self):
        """Generate noisy variants from the base images"""
        if not self.base_images:
            print("No base images available for noise processing")
            return 0
        
        noise_levels = ["1_level", "2_level", "3_level", "4_level"]
        total_success = 0
        
        print(f"\nGenerating noisy variants from {len(self.base_images)} base images...")
        
        for base_data in self.base_images:
            base_image = base_data['image']
            text_content = base_data['text']
            base_filename = base_data['filename']
            
            print(f"\nProcessing base image: {base_filename}")
            
            for level in noise_levels:
                try:
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
                    
                    total_success += 1
                    print(f"  ✓ Generated {level}: {noisy_filename}.jpg")
                    
                except Exception as e:
                    print(f"  ✗ Failed to generate {level} for {base_filename}: {e}")
                    continue
        
        print(f"\nNoisy variant generation completed: {total_success} images")
        return total_success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PreP-OCR Synthetic Data Generator")
    parser.add_argument("--base", type=int, default=5, 
                       help="Number of base images to generate (each produces 4 noisy variants)")
    
    args = parser.parse_args()
    
    if args.base <= 0:
        print("Please specify a positive number of base images to generate.")
        print("Example: python generate_ocr_data.py --base 5")
        return 1
    
    try:
        generator = OCRDataGenerator()
        
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