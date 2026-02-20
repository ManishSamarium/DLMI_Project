# Cell Nuclei Separation using Watershed Segmentation

This project implements cell nuclei separation using the Watershed algorithm in OpenCV. It demonstrates both standard and marker-controlled watershed segmentation techniques for separating overlapping cell nuclei in microscopy images.

## Dataset

The dataset used for this project is from the **Data Science Bowl 2018** competition:

**Dataset Link:** [Data Science Bowl 2018 - Merged Mask](https://www.kaggle.com/datasets/mahmudulhasantasin/data-science-bowl-2018-competition-merged-mask)

## Project Structure

```
├── main.py                 # Main script for nuclei separation
├── README.md               # Project documentation
├── results/                # Output images
│   ├── 01_original.png     # Original input image
│   ├── 02_threshold.png    # Otsu thresholding result
│   ├── 03_without_markers.png  # Watershed without marker control
│   └── 04_with_markers.png     # Marker-controlled watershed
└── dataset/                # Training dataset
```

## Methods Implemented

### 1. Preprocessing
- Grayscale conversion
- Otsu's thresholding for binary segmentation
- Morphological opening for noise removal

### 2. Watershed without Marker Control
- Distance transform on the binary image
- Simple thresholding to find foreground markers
- Standard watershed segmentation

### 3. Marker-Controlled Watershed
- Distance transform with higher threshold (0.6x max)
- Sure foreground and background region identification
- Unknown region calculation
- Controlled watershed with proper markers

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation

```bash
pip install opencv-python numpy
```

## Usage

```bash
python assignemnt_1_main.py
```

The script will process the input image and save results in the `results/` folder.

## Results

The output includes:
- Original image
- Binary threshold image
- Watershed segmentation without markers (may have over-segmentation)
- Marker-controlled watershed segmentation (cleaner separation)

## License

This project is for educational purposes as part of the DLMI (Deep Learning for Medical Imaging) course.
