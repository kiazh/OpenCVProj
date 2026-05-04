# OpenCV Projects

A collection of computer vision projects built with Python, OpenCV, and TensorFlow. Each project is self-contained with its own environment and documentation.

## Projects

### [Facial Identity Verification](./FacialRecgTensorFlow/README.md)

One-shot facial verification using a Siamese neural network. The model learns a discriminative similarity metric over image pairs — enabling identity verification without retraining on new subjects.

**Key characteristics:**
- Siamese architecture with shared convolutional embedding network (4,096-dim output)
- L1 distance layer as the similarity metric
- Two-stage verification threshold for tunable false-accept / false-reject trade-off
- Trained on anchor / positive / negative image triplets (≤ 3,000 per class)
- Live webcam verification at runtime

**Stack:** TensorFlow · Keras · OpenCV · NumPy

---

### [Document Scanner](./TextScanner/README.md)

Classical computer vision pipeline that rectifies and binarises a photographed document into a flatbed-quality scan.

**Key characteristics:**
- Canny edge detection → quadrilateral contour localisation → perspective transform
- Adaptive Gaussian thresholding for binarisation
- Single-command CLI operation

**Stack:** OpenCV · imutils · scikit-image · NumPy

---

## Repository Structure

```
OpenCVProj/
├── FacialRecgTensorFlow/    # Siamese network facial verification
│   ├── main.ipynb
│   ├── my_model.keras
│   └── README.md
├── TextScanner/             # Document scanning pipeline
│   ├── fileScanner.py
│   └── README.md
└── README.md                # This file
```

## Environment

Both projects share a common virtual environment (`cv_env`):

```bash
source cv_env/bin/activate
```

Python version is pinned in `.python-version` (managed by `pyenv`).
