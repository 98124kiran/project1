# UNIT-II: Convolutional Networks and Transfer Learning

## 1) Convolution Operation

Convolution is the core operation in Convolutional Neural Networks (CNNs).  
It applies a small learnable filter (kernel) over local regions of an input (image/feature map) to detect patterns such as edges, corners, and textures.

### How convolution works
- Input tensor shape (image): `H x W x C` (height, width, channels)
- Filter shape: `k x k x C`
- A filter slides spatially over the input and computes dot products.
- Each filter produces one output channel (feature map).
- Multiple filters produce multiple feature maps.

### Key hyperparameters
- **Kernel size (`k`)**: Commonly `3x3`, `5x5`, `7x7`
- **Stride (`s`)**: Step size of filter movement
- **Padding (`p`)**:
  - `valid`: no padding (output shrinks)
  - `same`: padding keeps spatial size similar
- **Dilation**: Spreads kernel elements to enlarge receptive field without extra parameters

### Output size formula
For one spatial dimension:
`Output = floor((Input + 2p - k) / s) + 1`

### Why convolution is effective
- **Local connectivity**: Exploits spatial locality
- **Parameter sharing**: Same filter reused across locations
- **Translation equivariance**: Shift in input causes corresponding shift in output
- **Fewer parameters** than fully connected layers on images

---

## 2) Motivation for Convolutional Networks

Classical fully connected networks are inefficient for image data because:
- They ignore spatial structure.
- Parameter count grows extremely fast with input size.
- They overfit easily with limited data.

CNNs are motivated by:
- Natural image statistics (local patterns repeat globally)
- Need for hierarchical representation learning:
  - Early layers: edges/textures
  - Mid layers: motifs/parts
  - Deep layers: object-level semantics
- Better computational efficiency on large visual inputs

---

## 3) Pooling

Pooling reduces spatial resolution of feature maps to make representations compact and robust.

### Common pooling types
- **Max pooling**: takes maximum value in each window
  - Captures strongest activation (feature presence)
- **Average pooling**: takes average value
  - Smoother aggregation
- **Global average pooling (GAP)**: averages each full feature map to one number
  - Often replaces large fully connected heads

### Benefits
- Reduces computation and memory
- Increases receptive field in deeper layers
- Adds some translation invariance
- Helps control overfitting

### Trade-off
- Too much pooling may lose fine spatial detail (important in dense prediction tasks)

---

## 4) Structured Outputs

Structured outputs are predictions with spatial or relational structure, not just one class label.

### Examples
- Semantic segmentation (class per pixel)
- Instance segmentation (object masks + identities)
- Object detection (boxes + labels)
- Keypoint/pose estimation
- Depth estimation

### CNN design implications for structured outputs
- Need to preserve spatial information (avoid excessive downsampling)
- Use encoder-decoder or multi-scale features
- Upsampling layers (transpose conv, interpolation + conv)
- Skip connections from shallow to deep layers for detail recovery
- Pixel-wise or region-wise loss functions

---

## 5) Data Types in CNN Workflows

CNN pipelines involve several data types:

### By modality
- **Image classification**: single image, single label
- **Detection**: image + bounding boxes + class labels
- **Segmentation**: image + pixel masks
- **Video**: frame sequences (spatiotemporal features)
- **Medical/remote sensing**: multi-channel, often high-resolution imagery

### By tensor representation
- Input images: `uint8` (stored), converted to `float32/float16` for training
- Labels:
  - Integer class IDs (classification)
  - Box coordinates + classes (detection)
  - Dense masks (segmentation)
- Activations/weights: `float32` or mixed precision (`float16` + scaling)

### Data preprocessing
- Normalization/scaling
- Resizing/cropping
- Augmentation (flip, rotate, color jitter, random crop, cutout/mixup variants)
- Class balancing or weighted sampling for imbalanced datasets

---

## 6) Popular CNN Architectures

### 6.1 LeNet (LeNet-5)

### Context
- One of the earliest successful CNNs (digit recognition, e.g., MNIST).

### Core characteristics
- Alternating convolution + pooling layers
- Followed by fully connected layers
- Small input and shallow depth by modern standards

### Significance
- Demonstrated end-to-end learned visual feature extraction
- Established the standard CNN block pattern

---

### 6.2 AlexNet

### Context
- Won ImageNet 2012 and triggered modern deep learning adoption in vision.

### Key innovations
- Deeper and wider than previous models
- ReLU activations (faster training than tanh/sigmoid)
- Dropout in fully connected layers (regularization)
- Data augmentation and GPU training at scale
- Overlapping max pooling

### Significance
- Huge accuracy jump on large-scale image classification
- Proved deep CNNs can generalize with big data + compute

---

### 6.3 VGG (VGG-16 / VGG-19)

### Design philosophy
- Very uniform architecture using stacked `3x3` convolutions and periodic pooling.

### Strengths
- Simple and modular design
- Deep hierarchical features
- Good transfer-learning backbone historically

### Limitations
- Large parameter count (especially FC layers)
- High compute/memory cost

---

## 7) Transfer Learning

Transfer learning reuses knowledge from a model pretrained on a large source dataset (e.g., ImageNet) for a target task.

### Why it helps
- Faster convergence
- Better performance with limited labeled data
- Reduces risk of overfitting on small datasets

### Common strategies
1. **Feature extraction**
   - Freeze pretrained backbone
   - Train only task-specific head
2. **Fine-tuning**
   - Initialize from pretrained weights
   - Unfreeze part/all backbone and continue training with smaller learning rate

### Practical guidelines
- Start with feature extraction when data is small.
- Fine-tune deeper layers when target domain differs significantly.
- Use lower learning rate for pretrained layers than new layers.
- Monitor overfitting carefully during fine-tuning.

---

## 8) DenseNet

DenseNet introduces dense connectivity: each layer receives feature maps from all previous layers in the same dense block.

### Core idea
- For layer `l`, input is concatenation of outputs from layers `0..l-1`.

### Benefits
- Better gradient flow (mitigates vanishing gradients)
- Feature reuse across network depth
- Parameter efficiency compared with similarly accurate very deep nets
- Strong performance on classification and transfer tasks

### Components
- **Dense blocks**: densely connected conv layers
- **Transition layers**: `1x1` conv + pooling to reduce dimensions
- **Growth rate**: number of new feature maps each layer adds

---

## 9) PixelNet

PixelNet-style models target pixel-level prediction tasks by learning representations for each pixel using multi-scale CNN features.

### Motivation
- Pixel-level tasks (segmentation, surface normals, edge detection) need both:
  - Local fine detail
  - Global semantic context

### Key concept
- Build per-pixel descriptors from multiple CNN layers (hypercolumn-like features).
- Use these descriptors for pixel-wise classification/regression.

### Advantages
- Rich multi-scale information per pixel
- Better dense prediction quality than relying on only final deep features
- Flexible framework for different structured output tasks

---

## 10) Summary

UNIT-II centers on how CNNs efficiently learn spatial hierarchies through convolution and pooling, how they are adapted for structured outputs, and how modern architecture evolution (LeNet → AlexNet → VGG → DenseNet) improved depth, trainability, and performance.  
Transfer learning enables practical reuse of pretrained CNNs, while PixelNet-style approaches extend CNNs to strong pixel-level predictions in dense vision tasks.
