# PreP-OCR: OCR Synthetic Data Generation Tool

<img width="1687" alt="image" src="https://github.com/user-attachments/assets/f1b099cf-2db8-4720-9146-770932794e58" />

## Overview

PreP-OCR is a specialized tool for generating synthetic OCR training data. It converts text files into high-quality simulated scanned document images suitable for training and testing OCR models.

## Key Features

### 🎯 Core Functionality
1. **Text-to-Image Conversion** - Convert text into clean images with various font styles and layouts
2. **Multi-level Noise Generation** - Generate 4 different degradation levels to simulate real scanning artifacts
3. **Background & Stain Processing** - Add paper textures, stains, and other realistic document effects
4. **Ground Truth Generation** - Automatically generate corresponding annotation files

### 📊 Noise Level Descriptions
- **0_level**: Clean images (no noise)
- **1_level**: Light noise (minimal background texture, slight blur)
- **2_level**: Medium noise (visible background, some stains)
- **3_level**: Heavy noise (strong background, multiple stains, significant blur)
- **4_level**: Extreme noise (severe degradation, heavy artifacts)

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n prep python=3.9 -y
conda activate prep

# Install dependencies
pip install numpy pillow opencv-python matplotlib tqdm
```

### 2. Preprocess Background Images

```bash
# Preprocess background and stain images (required for first use)
python prepare_backgrounds.py
```

### 3. Generate Data

```bash
# Generate 5 base images + 20 noisy variants (4 levels × 5 base images)
python generate_ocr_data.py --base 5

# Generate different numbers of base images
python generate_ocr_data.py --base 10  # 10 base + 40 noisy variants
```

## Project Structure

```
PreP-OCR/
├── data/
│   ├── Novel_data_UTF8_processed/    # Processed text files
│   └── output/                       # Generated images and GT files
│       ├── clean/                    # Clean base images
│       ├── noisy/                    # Noisy variant images
│       └── ground_truth/             # Ground truth text files
├── font/
│   └── latin/                        # Font files
├── noise_img/
│   ├── background/                   # Original background images
│   ├── background_p/                 # Processed background images
│   ├── stain/                       # Original stain images
│   └── stain_p/                     # Processed stain images
├── funtion/
│   └── generate_base_add_noise.py   # Core generation functions
├── generate_ocr_data.py             # Main generation script
├── prepare_backgrounds.py           # Background preprocessing script
├── environment.yml                  # Conda environment config
└── requirements.txt                 # Python dependencies
```

## Detailed Usage

### Text Data Preparation
- Place text files in `data/Novel_data_UTF8_processed/` directory
- Supports UTF-8 encoded .txt files
- Each file will be randomly sampled to generate images

### Font Configuration
- Font files should be placed in `font/latin/` directory
- Supports .ttf and .otf formats
- System randomly selects fonts for image generation

### Background and Stains
- Place original background images in `noise_img/background/`
- Place original stain images in `noise_img/stain/`
- Run `prepare_backgrounds.py` to preprocess before use

### Advanced Configuration

#### Generation Parameter Tuning
You can modify parameters in `generate_ocr_data.py`:
```python
lines_per_image = (15, 25)     # Range of lines per image
font_size = (40, 60)           # Font size range
scale_range = (0.8, 1.1)       # Scaling ratio range
binarize_prob = 0.1            # Binarization probability
```

#### Noise Level Customization
Modify noise parameters in `funtion/generate_base_add_noise.py`:
```python
noise_parameters = {
    "1_level": {
        "max_noise_factor": 10,    # Noise intensity
        "bg_intensity": 0.1,       # Background transparency
        "st_intensity": 0.3,       # Stain transparency
        # ... more parameters
    }
}
```

## Output Description

### File Naming Convention
- Clean images: `{randomID}_clean.jpg`
- Noisy images: `{randomID}_{noise_level}.jpg`
- GT files: `{randomID}_{type}.txt`

### Ground Truth Format
- UTF-8 encoded plain text files
- Contains identical text content as corresponding images
- Preserves original line breaks and indentation formatting

## Technical Features

### Image Generation Techniques
- **Multi-font Support**: Random font selection for diversity
- **Layout Deformation**: Character shifting, page bending, paragraph misalignment
- **Layout Randomization**: Random line spacing, character spacing, margins, rotation angles

### Noise Simulation Techniques
- **Gaussian Noise**: Simulate sensor noise
- **Resolution Reduction**: Simulate scanning quality issues
- **Morphological Operations**: Simulate text wear and blur
- **Background Overlay**: Simulate paper texture
- **Stain Addition**: Simulate coffee stains, ink spots, etc.

### Image Processing Pipeline
1. Text rendering (fonts, layout, deformation)
2. Basic noise addition (Gaussian noise, blur)
3. Background texture overlay
4. Stain and artifact addition
5. Resolution and contrast adjustment
6. Optional binarization processing

## Performance Optimization

- Batch processing to reduce I/O overhead
- Smart font loading to avoid repeated loading
- Memory-optimized image processing pipeline
- Support for parallel processing (extensible)

## Troubleshooting

### Common Issues
1. **Font loading failure**: Check font file format and permissions
2. **Background images not showing**: Run `prepare_backgrounds.py` preprocessing
3. **Slow generation speed**: Reduce noise processing steps or lower image resolution
4. **Memory insufficient**: Reduce batch size or image dimensions

### Debug Mode
Add detailed logging output in scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Generation Logic

The tool follows a two-step process:

1. **Base Image Generation**: Creates clean text images with various fonts and layouts
2. **Noise Variant Generation**: For each base image, generates 4 different noise levels

This ensures that:
- Each noisy image has a corresponding clean version
- Ground truth files perfectly match their respective images
- Consistent text content across all noise levels of the same base

## Contributing

We welcome Issues and Pull Requests!

### Development Environment Setup
```bash
git clone https://github.com/your-repo/PreP-OCR.git
cd PreP-OCR
conda env create -f environment.yml
conda activate prep
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all open source projects contributing fonts and texture resources
- Built with PIL, OpenCV, and NumPy
- Refactored for improved usability and maintainability

---

**Version**: 2.0.0-refactored  
**Last Updated**: January 2025