# One-Shot Facial Identity Verification via Siamese Neural Networks

## Abstract

This project presents an implementation of a Siamese neural network architecture for one-shot facial identity verification. The system learns a discriminative similarity metric over image pairs, enabling robust identity verification without the requirement to retrain on newly enrolled subjects. A dual-threshold decision boundary is applied at inference time to govern the trade-off between false acceptance and false rejection rates. The model is trained on a custom-collected dataset of anchor, positive, and negative image pairs using binary cross-entropy loss and the Adam optimiser.

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [System Architecture](#2-system-architecture)
3. [Dataset Construction and Preprocessing](#3-dataset-construction-and-preprocessing)
4. [Network Design](#4-network-design)
5. [Training Procedure](#5-training-procedure)
6. [Evaluation and Results](#6-evaluation-and-results)
7. [Verification Application](#7-verification-application)
8. [Model Persistence and Deployment](#8-model-persistence-and-deployment)
9. [Dependencies and Environment Setup](#9-dependencies-and-environment-setup)
10. [Directory Structure](#10-directory-structure)
11. [Usage](#11-usage)

---

## 1. Theoretical Background

### 1.1 One-Shot Learning

Conventional deep learning classifiers require a substantial volume of labelled samples per class to achieve reliable generalisation. In biometric verification scenarios — where enrolling hundreds of images per individual is impractical — **one-shot learning** offers an alternative paradigm. Rather than learning a class-discriminative mapping, the network learns a **general-purpose similarity function** that transfers to unseen identities.

### 1.2 Siamese Networks

A Siamese network (Koch et al., 2015) consists of two identical sub-networks sharing weights, each processing one image from an input pair. The shared weights enforce a symmetric embedding space: if image A is similar to image B, the distance `d(f(A), f(B))` is small regardless of which image occupies which branch. The architecture is trained on labelled pairs `(x_i, x_j, y_ij)` where `y = 1` indicates same-identity and `y = 0` indicates different-identity.

### 1.3 L1 Similarity Metric

The element-wise **L1 (Manhattan) distance** is used as the similarity metric between embedding vectors:

```
d(u, v) = Σ |u_k - v_k|
```

L1 distance produces sparser gradients than L2, making it particularly effective when the embedding dimensionality is large relative to the training set size. The distance vector is subsequently passed through a sigmoid classifier to produce a scalar match probability `p ∈ (0, 1)`.

### 1.4 Contrastive Learning Objective

Training optimises **binary cross-entropy** between the predicted match probability and the ground-truth label:

```
L(y, ŷ) = -[y · log(ŷ) + (1 − y) · log(1 − ŷ)]
```

This formulation encourages the network to output `ŷ → 1` for same-identity pairs and `ŷ → 0` for different-identity pairs, learning a discriminative metric space without explicit contrastive or triplet margin losses.

---

## 2. System Architecture

```
Input Image (100×100×3)          Validation Image (100×100×3)
       │                                    │
       ▼                                    ▼
┌─────────────────────────────────────────────────────┐
│              Shared Embedding Network               │
│  Conv(64, 10×10, ReLU) → MaxPool(2×2)              │
│  Conv(128, 7×7, ReLU)  → MaxPool(2×2)              │
│  Conv(128, 4×4, ReLU)  → MaxPool(2×2)              │
│  Conv(256, 4×4, ReLU)  → Flatten                   │
│  Dense(4096, Sigmoid)                               │
└─────────────────────────────────────────────────────┘
       │                                    │
   f(input)                           f(validation)
       │                                    │
       └──────────┬─────────────────────────┘
                  ▼
         L1Dist Layer: |f(input) − f(validation)|
                  │
                  ▼
         Dense(1, Sigmoid)
                  │
                  ▼
         Match Probability ŷ ∈ (0, 1)
```

The embedding sub-network is **weight-tied** across both input branches, guaranteeing metric symmetry: `d(A, B) = d(B, A)` by construction.

---

## 3. Dataset Construction and Preprocessing

### 3.1 Triplet Image Sets

Three disjoint image partitions are defined:

| Partition    | Semantics                                    | Label |
|--------------|----------------------------------------------|-------|
| **Anchor**   | Reference face of the enrolled subject        | —     |
| **Positive** | Alternate captures of the same identity       | 1     |
| **Negative** | Faces of distinct, non-enrolled individuals   | 0     |

The Labeled Faces in the Wild (LFW) dataset is recommended as the negative corpus to provide sufficient inter-subject variance.

### 3.2 Image Acquisition

Anchor and positive samples are collected via live webcam capture using OpenCV:

- Press `a` — write current frame to `data/anchor/`
- Press `p` — write current frame to `data/positive/`
- Press `q` — terminate capture session

Each frame is resized to **250×250** prior to disk write; further resizing to **100×100** occurs during the preprocessing pipeline.

### 3.3 Preprocessing Pipeline

Each image undergoes the following transformations before ingestion into the network:

1. JPEG decoding via `tf.io.decode_jpeg`
2. Spatial resizing to **100×100** pixels via bilinear interpolation
3. Pixel normalisation to the range `[0, 1]` by dividing by 255

Normalisation ensures gradient stability during backpropagation and prevents dominance of high-magnitude channels.

### 3.4 Dataset Assembly and Partitioning

Up to **3,000 samples** are drawn from each partition. Anchor-positive pairs are assigned label `1`; anchor-negative pairs are assigned label `0`. The combined dataset of **6,000 pairs** is:

- Cached in memory to eliminate redundant disk reads across epochs
- Shuffled with a buffer of 10,000 elements for stochasticity
- Split **70% training / 30% test**
- Batched at size **16** with prefetching depth **8** to overlap I/O and compute

---

## 4. Network Design

### 4.1 Embedding Network

The convolutional feature extractor (`make_embedding`) maps each `100×100×3` image to a **4,096-dimensional embedding vector** via four successive convolution–pooling blocks:

| Block | Operation              | Output Channels | Kernel Size | Activation |
|-------|------------------------|-----------------|-------------|------------|
| 1     | Conv2D + MaxPool       | 64              | 10×10       | ReLU       |
| 2     | Conv2D + MaxPool       | 128             | 7×7         | ReLU       |
| 3     | Conv2D + MaxPool       | 128             | 4×4         | ReLU       |
| 4     | Conv2D + Flatten       | 256             | 4×4         | ReLU       |
| —     | Dense                  | 4096            | —           | Sigmoid    |

ReLU activations in convolutional layers promote sparse feature representations. The **sigmoid** activation on the final dense layer constrains all embedding dimensions to `[0, 1]`, bounding the L1 distance space.

### 4.2 L1 Distance Layer

`L1Dist` is implemented as a custom `keras.layers.Layer` to enable end-to-end gradient computation through the distance operation:

```python
class L1Dist(Layer):
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
```

Implementing the distance as a trainable layer (rather than a post-hoc computation) ensures correct gradient flow during backpropagation through the Siamese branches.

### 4.3 Classification Head

A single `Dense(1, activation='sigmoid')` neuron maps the 4,096-dimensional L1 distance vector to a scalar match probability. The decision boundary is applied at `p > 0.5` during standard evaluation.

---

## 5. Training Procedure

### 5.1 Optimiser

**Adam** (Adaptive Moment Estimation) with a learning rate of `1 × 10⁻⁴` provides adaptive per-parameter learning rates. Adam is well-suited to this task due to its robustness to sparse gradients, which arise naturally when only a subset of embedding dimensions are discriminative.

### 5.2 Gradient Computation

Each training step is compiled into a TensorFlow graph via `@tf.function`, eliminating Python interpreter overhead and enabling XLA compilation. Gradients are computed with `tf.GradientTape` and applied via `optimizer.apply_gradients`.

### 5.3 Training Loop

The model is trained for **50 epochs** over the training partition. Per-epoch metrics include:

- **Binary Cross-Entropy Loss** — scalar loss over the mini-batch
- **Precision** — proportion of predicted positives that are true positives
- **Recall** — proportion of true positives correctly retrieved

Model weights and optimiser state are serialised via `tf.train.Checkpoint` every **10 epochs**, enabling training resumption without loss of progress.

### 5.4 Hyperparameter Summary

| Hyperparameter          | Value       |
|-------------------------|-------------|
| Learning rate           | `1 × 10⁻⁴` |
| Batch size              | 16          |
| Epochs                  | 50          |
| Embedding dimensionality| 4,096       |
| Dataset size (per split)| ≤ 3,000     |
| Train / test ratio      | 70% / 30%   |
| Checkpoint frequency    | Every 10 epochs |

---

## 6. Evaluation and Results

### 6.1 Metrics

Model performance is assessed using **Precision** and **Recall** over the held-out test partition, aggregated across all test batches:

| Metric        | Value    |
|---------------|----------|
| Recall        | 0.9868   |
| Precision     | 0.4011   |

The high recall indicates the model reliably identifies genuine matches; the comparatively lower precision reflects a tendency toward false acceptance under the default `p > 0.5` threshold. Raising the detection threshold at inference time can be used to improve precision at the cost of recall, depending on the operational security requirement.

### 6.2 Threshold Sensitivity

The verification application exposes two configurable thresholds:

- **Detection threshold** — minimum match probability for a single reference image comparison to be counted as a positive detection
- **Verification threshold** — minimum proportion of reference images that must exceed the detection threshold for the identity to be accepted

This two-stage decision boundary provides finer control over the false acceptance / false rejection trade-off than a single global threshold.

---

## 7. Verification Application

### 7.1 Enrollment

Up to **200 positive samples** from the training corpus are copied into `application_data/verification_images/` as the enrolled reference gallery. A larger gallery increases decision robustness at the cost of per-verification latency.

### 7.2 Runtime Verification

At runtime:

1. A live webcam frame is captured and written to `application_data/input_image/input_image.jpg` on keypress `v`
2. The input image is compared against every image in the verification gallery
3. Match scores exceeding `detection_threshold` are tallied
4. Verification is accepted if the tally proportion exceeds `verification_threshold`

```python
results, verified = verify(siamese_model, detection_threshold=0.5, verification_threshold=0.5)
```

Press `q` to terminate the verification session.

---

## 8. Model Persistence and Deployment

The trained model is serialised in the **Keras native format** (`.keras`), which preserves architecture, weights, and training configuration in a single file:

```python
siamese_model.save('my_model.keras')
```

At load time, the custom `L1Dist` layer must be registered explicitly:

```python
siamese_model = tf.keras.models.load_model(
    'my_model.keras',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
)
```

A legacy HDF5 checkpoint (`siamesemodelv2.h5`) is also retained for compatibility with older Keras versions.

---

## 9. Dependencies and Environment Setup

### 9.1 Python Version

Python `3.10+` is recommended. The project uses a virtual environment managed by `pyenv`.

### 9.2 Core Dependencies

| Package         | Purpose                                  |
|-----------------|------------------------------------------|
| `tensorflow`    | Model definition, training, serialisation|
| `keras`         | High-level neural network API            |
| `opencv-python` | Webcam capture and image I/O             |
| `numpy`         | Numerical array operations               |
| `matplotlib`    | Visualisation utilities                  |
| `uuid`          | Unique filename generation               |

### 9.3 Environment Activation

```bash
# Create and activate the virtual environment
python -m venv cv_env
source cv_env/bin/activate

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib
```

---

## 10. Directory Structure

```
FacialRecgTensorFlow/
├── main.ipynb                          # Primary notebook: training and verification
├── my_model.keras                      # Serialised Keras model (production)
├── siamesemodelv2.h5                   # Legacy HDF5 checkpoint
├── data/
│   ├── anchor/                         # Anchor (reference) face images
│   ├── positive/                       # Positive (same identity) face images
│   └── negative/                       # Negative (different identity) face images
├── application_data/
│   ├── input_image/
│   │   └── input_image.jpg             # Live capture for verification
│   └── verification_images/            # Enrolled reference gallery (≤ 200 images)
└── training_checkpoints/
    ├── checkpoint
    ├── ckpt-{1..5}.data-00000-of-00001
    └── ckpt-{1..5}.index
```

---

## 11. Usage

### 11.1 Data Collection

Open `main.ipynb` and run sections **3** (Image Acquisition) to collect anchor and positive samples. Populate `data/negative/` with an external negative corpus such as LFW.

### 11.2 Training

Run sections **4–14** sequentially to construct the dataset pipeline, define the Siamese network, and execute the training loop. Training for 50 epochs on ~6,000 pairs takes approximately 10–30 minutes depending on hardware.

### 11.3 Verification

1. Run section **16** to enroll reference images into `application_data/verification_images/`
2. Run section **17** to launch the live verification application
3. Press `v` to capture and verify; press `q` to quit

### 11.4 Threshold Tuning

Adjust `detection_threshold` and `verification_threshold` in the `verify()` call to calibrate the system for the target operating point on the precision–recall curve.

---

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-Shot Image Recognition*. ICML Deep Learning Workshop.
- Chopra, S., Hadsell, R., & LeCun, Y. (2005). *Learning a Similarity Metric Discriminatively, with Application to Face Verification*. CVPR.
- Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. UMass Amherst Technical Report 07-49.
