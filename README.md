# Brain MRI Tumor Segmentation — Otsu vs Sauvola

A comparison of Global Otsu and Sauvola adaptive thresholding methods for brain tumor segmentation in MRI slices.

## Task
Segment tumor regions in brain MRI slices using two thresholding approaches and compare their performance.

## Methods
- **Global Otsu Thresholding**: Automatically determines optimal global threshold
- **Sauvola Adaptive Thresholding**: Local adaptive method based on local mean and standard deviation

## Dataset
Download the [Brain MRI Tumor Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation) from Kaggle and place it in:
```
data/
├── images/
└── masks/
```

## Installation
```bash
pip install numpy opencv-python matplotlib scikit-image tqdm
```

## Usage
```bash
python assignment_1_main.py
```

## Evaluation Metrics
- **Dice Score**: Measures overlap between predicted and ground truth masks
- **Jaccard Index (IoU)**: Intersection over Union of predicted and ground truth

## Results
| Metric | Otsu | Sauvola |
|--------|------|---------|
| Dice Score | 0.0705 | 0.0454 |
| Jaccard Score | 0.0375 | 0.0236 |

## Learning
Global Otsu thresholding outperforms Sauvola adaptive thresholding on this dataset. The scores are relatively low because simple intensity thresholding methods struggle with the complex texture and intensity variations in brain MRI images.
