# Image Classifier — Architecture Overview

## 1. Network Architecture

### 1.1 Main Ensemble (3 networks)

- **Member 0:** Base architecture from `NeuralNetwork::determineOptimalArchitecture()`
- **Member 1:** Same + extra layer `max(num_classes*2, 32)`
- **Member 2:** Same base, first hidden layer ×1.5

Input: `image_size × image_size × 3` (default 32×32×3)

### 1.2 Structure Subnets (SubNetworkManager)

Dynamic count: one subnet per structure type encountered during training.

Structure types (ShapeAnalyzer):
- **fuselage** — near center (distance < 0.3)
- **wing** — horizontal (angle ≈ ±45°)
- **tail** — rear (angle ≈ ±135°)
- **additional_element** — other

Up to 4 structure subnets. Topology can be extended by entropy (`morphTopology`).

### 1.3 Problem Class Ensemble

Separate ensemble for classes with low F1/recall/precision. Used during classification with structure analysis.

---

## 2. Structure Analysis and SVM

- **Structure analysis** is applied to **all** images when `use_structure_analysis` is enabled.
- **Problem classes** get extra support: `predictProblemClassSubNetwork` and `problem_ensemble`.
- During **training**, structure and SVM are used only if `use_structure_analysis == true`.
- In **fast mode**, structure analysis is disabled to speed up training.

---

## 3. Segmentation (no manual templates)

- **Segmentation:** Mean Shift (< 1M pixels) or k-means by color (larger images)
- **Shape and mask:** Otsu binarization, morphology, `findContours`
- **Structures:** Sobel gradients, magnitude threshold, `findContours`, area/prominence filters
- **SVM** trains on structure features from the same training dataset

---

## 4. Training Pipeline

1. **Stage 1:** Quality control (sample selection)
2. **Stage 2:** Batch training (main epochs)
3. **Stage 3:** Accumulated training (refinement)
4. **Stage 4:** Problem class training (optional)

---

## 5. Key Files

| Component        | Files                          |
|-----------------|--------------------------------|
| Main classifier | `UniversalImageClassifier.cpp/h` |
| Neural network  | `NeuralNetwork.cpp/h`          |
| Structure       | `ShapeAnalyzer.cpp`            |
| Subnets         | `SubNetworkManager.cpp`        |
| Fusion          | `FuzzyLogicFusion.cpp`        |
| Profiling       | `Profiler.h/cpp`              |
| Config          | `Config.h`                    |
