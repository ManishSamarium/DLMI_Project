# Retinal Vessel Extraction — Sauvola vs Niblack

## Task
Extract thin vessels in fundus images using local thresholding techniques.

## Methods
Comparison between two local adaptive thresholding algorithms:
- **Niblack Thresholding**: Uses local mean and standard deviation to calculate threshold
- **Sauvola Thresholding**: A modification of Niblack that normalizes threshold, better for documents and images with varying illumination

## Dataset
**DRIVE (Digital Retinal Images for Vessel Extraction)** - Kaggle retinal dataset mirrors

The dataset contains:
- Retinal fundus images (`.tif` format)
- Manual segmentation ground truth (1st_manual)
- FOV masks

## Evaluation Metric
**Sensitivity (True Positive Rate)** - Focuses on how well the methods detect thin vessels in the retinal images.

$$\text{Sensitivity} = \frac{TP}{TP + FN}$$

## Results

| Method   | Average Sensitivity |
|----------|---------------------|
| Niblack  | *Run main.py*       |
| Sauvola  | *Run main.py*       |

## Learning Outcomes
- Understanding local threshold behavior on thin structures
- Comparing adaptive thresholding algorithms for vessel segmentation
- Sensitivity analysis for medical image processing

## Usage

### Requirements
```bash
pip install opencv-python numpy scikit-image
```

### Run
```bash
python main.py
```

### Output
- `result_niblack.png` - Sample Niblack thresholding result
- `result_sauvola.png` - Sample Sauvola thresholding result
- `result_gt.png` - Ground truth for comparison
- Console output with average sensitivity scores

## Project Structure
```
├── main.py              # Main comparison script
├── training/
│   ├── images/          # Retinal fundus images
│   ├── 1st_manual/      # Ground truth vessel segmentation
│   └── mask/            # FOV masks
├── .gitignore
└── README.md
```

## References
- DRIVE Dataset: https://drive.grand-challenge.org/
- Niblack, W. (1986). An introduction to digital image processing
- Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization
