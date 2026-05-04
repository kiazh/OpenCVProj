# Document Scanner

A command-line document scanning utility that applies classical computer vision techniques to produce a rectified, binarised scan from an arbitrary photograph of a document.

## Overview

The pipeline comprises three sequential stages — edge detection, contour localisation, and perspective correction — replicating the output of a flatbed scanner from a handheld photograph.

## Pipeline

### Stage 1 — Edge Detection

The input image is converted to greyscale, smoothed with a Gaussian kernel (5×5, σ=0) to suppress high-frequency noise, and passed through a Canny edge detector with thresholds `[75, 200]`. The result isolates structural boundaries while discarding texture detail.

### Stage 2 — Document Contour Localisation

Contours are extracted from the edge map and ranked by area (descending). The largest quadrilateral approximation (`approxPolyDP`, ε=2% of arc length) is taken as the document boundary. If no four-sided contour is found, the process exits with a diagnostic message.

### Stage 3 — Perspective Transform and Binarisation

`four_point_transform` warps the detected quadrilateral to an axis-aligned rectangle, correcting projective distortion. The warped region is converted to greyscale and binarised via adaptive Gaussian thresholding (block size 11, offset 10), producing a high-contrast black-and-white scan.

## Usage

```bash
python fileScanner.py -i path/to/document.jpg
```

At each stage a window is displayed. Press any key to advance.

## Dependencies

| Package      | Purpose                          |
|--------------|----------------------------------|
| `opencv-python` | Image I/O, edge detection, display |
| `imutils`    | Contour utilities, perspective transform |
| `scikit-image` | Adaptive local thresholding     |
| `numpy`      | Array operations                 |

```bash
pip install opencv-python imutils scikit-image numpy
```

## Failure Modes

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `ERROR: Could not find a 4-sided contour` | Low contrast between document and background | Use a dark, uniform background; improve lighting |
| Skewed or cropped output | Partial document in frame | Ensure all four corners are visible |
| Noisy binarisation | Glare or shadow on document | Diffuse lighting; avoid direct flash |
