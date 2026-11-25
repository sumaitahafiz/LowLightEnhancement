# Zero-Reference Low-Light Image Enhancement Using Adaptive Contrast and Edge Preservation

A Digital Image Processing research project that explores low-light image enhancement using a lightweight Convolutional Neural Network (CNN), adaptive contrast enhancement, and edge-preserving refinement.

## 1. Project Overview

Images captured under poor lighting conditions often suffer from low visibility, poor contrast, color distortion, and noise. These problems can affect image quality and make important visual details difficult to recognize.

This project proposes a **zero-reference low-light image enhancement approach** that enhances low-light images without requiring paired normal-light reference images during training.

The proposed method combines illumination analysis, high-frequency/noise analysis, and edge information to guide adaptive enhancement while attempting to preserve important image details.

## 2. Project Objectives

- Improve the visibility of low-light images.
- Perform illumination-aware adaptive enhancement.
- Analyze high-frequency variations and noise.
- Preserve edges and important image details.
- Reduce unwanted noise and color distortion.
- Develop a lightweight CNN-based enhancement model.
- Evaluate enhanced images using reference-based and no-reference quality metrics.

## 3. Proposed Methodology

The proposed enhancement pipeline consists of the following stages:

1. **Dataset Preparation:** Load and prepare low-light images for processing.
2. **Low-Light Input:** Provide low-light images as input to the enhancement pipeline.
3. **Preprocessing:** Analyze image characteristics before enhancement.
4. **Feature Analysis and Fusion:** Extract and combine three types of information:
   - Illumination map
   - High-frequency variation/noise map
   - Sobel edge/detail map
5. **Lightweight CNN:** Process the six-channel input and predict three-channel enhancement parameters.
6. **Adaptive Enhancement:** Apply illumination-aware adaptive enhancement.
7. **Image Refinement:** Improve the enhanced output through:
   - Noise suppression
   - Color correction
   - Edge preservation
8. **Zero-Reference Loss Calculation:** Calculate six loss components to guide the enhancement process.
9. **Training:** Optimize the CNN using the combined loss.
10. **Evaluation:** Assess the enhancement results using image quality metrics.

## 4. Model Architecture

The proposed lightweight CNN receives a six-channel input consisting of:

| Input Feature | Channels |
|---|---:|
| RGB Image | 3 |
| Illumination Map | 1 |
| High-Frequency/Noise Map | 1 |
| Sobel Edge/Detail Map | 1 |
| **Total** | **6** |

The CNN predicts three-channel enhancement parameters.

- Model type: Lightweight Convolutional Neural Network
- Input channels: 6
- Output channels: 3
- Trainable parameters: 16,067

The predicted parameters are used by the adaptive enhancement stage, followed by image refinement.

## 5. Zero-Reference Loss Functions

The training process uses six loss components:

| Loss Function | Purpose |
|---|---|
| Illumination Loss | Encourages appropriate illumination |
| Exposure Loss | Guides exposure and contrast adjustment |
| Spatial Loss | Preserves spatial relationships |
| Color Loss | Encourages color consistency |
| Noise Loss | Helps control unwanted noise |
| Edge Loss | Encourages preservation of image details |

The total loss combines these six components and guides the optimization of the CNN.

## 6. Datasets

The project uses the following datasets:

| Dataset | Purpose |
|---|---|
| ExDark | Primary low-light training dataset |
| LOL-v1 | Paired reference-based evaluation |
| LOL-v2 Real | Paired reference-based evaluation |

**Dataset usage:**
- ExDark contains 7,363 low-light images.
- LOL-v1 evaluation uses 15 paired low-light and normal-light images.
- LOL-v2 Real Test uses 100 paired images.

The paired LOL datasets are reserved for evaluation rather than being used as normal-light training targets.

## 7. Experimental Results

### 7.1 Reference-Based Evaluation

The model was evaluated using PSNR and SSIM on paired low-light and normal-light images.

| Dataset | Number of Pairs | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| LOL-v1 | 15 | 14.436 | 0.5867 |
| LOL-v2 Real Test | 100 | 15.214 | 0.6027 |

- **PSNR:** Measures pixel-level reconstruction quality. Higher values generally indicate lower reconstruction error.
- **SSIM:** Measures structural similarity between enhanced and reference images. Higher values indicate greater structural similarity.

### 7.2 No-Reference Evaluation on ExDark

The following results were obtained from the ExDark image analysis:

| Metric | Before Enhancement | After Enhancement |
|---|---:|---:|
| NIQE | 7.4368 | 7.0430 |
| BRISQUE | 27.5406 | 28.8336 |
| Brightness | 0.1454 | 0.4838 |
| Contrast | 0.1508 | 0.0799 |
| Noise | 0.010066 | 0.010440 |
| Edge/Detail | 0.151436 | 0.157731 |

**Interpretation:**

- Brightness increased substantially after enhancement.
- NIQE decreased, indicating improvement according to this metric.
- BRISQUE increased, indicating a worse score according to this metric.
- Contrast decreased.
- Noise increased slightly.
- Edge/detail measurements increased slightly.

These results indicate that the proposed approach improves image brightness, but its effects on other image quality characteristics are mixed.

**Evaluation note:** All 7,363 ExDark images were used during training. Therefore, the ExDark analysis is reported as a no-reference analysis, not as a held-out test evaluation.

## 8. Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- scikit-image
- VS Code

## 9. Project Structure

```text
LowLightEnhancement/
│
├── checkpoints/
│   ├── corrected_epoch_1.pth
│   ├── corrected_epoch_2.pth
│   ├── corrected_epoch_3.pth
│   ├── corrected_epoch_4.pth
│   ├── corrected_epoch_5.pth
│   └── training_history_corrected.txt
│
├── data/
│   ├── dataset.py
│   ├── test_dataset.py
│   └── visualize_dataset.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── save_results.py
│   └── diagnostic scripts
│
├── losses/
│   ├── illumination_loss.py
│   ├── exposure_loss.py
│   ├── spatial_loss.py
│   ├── color_loss.py
│   ├── noise_loss.py
│   ├── edge_loss.py
│   └── total_loss.py
│
├── preprocessing/
│   ├── illumination.py
│   ├── noise.py
│   ├── edges.py
│   └── fusion.py
│
├── checkpoints/
├── enhancement.py
├── model.py
├── refinement.py
├── train.py
├── training.py
└── README.md
```

## 10. Future Improvements

- Improve contrast preservation during enhancement.
- Further investigate noise suppression.
- Improve perceptual quality and color consistency.
- Evaluate the approach on additional low-light datasets.
- Compare the proposed approach with established low-light enhancement methods.

## 11. Research Summary

This project investigates a lightweight zero-reference approach to low-light image enhancement by combining illumination-aware processing, feature fusion, adaptive enhancement, and edge-preserving refinement.

The experiments demonstrate substantial brightness enhancement, while the evaluation results also highlight areas where further improvements are needed.

---

**Project:** Zero-Reference Low-Light Image Enhancement Using Adaptive Contrast and Edge Preservation  
**Domain:** Digital Image Processing (DIP)  
**Implementation:** Python and PyTorch