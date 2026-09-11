# Assignment 2 — Trustworthy AI

This assignment covers **model interpretability and explainability** through two parts: LIME-based local explanations and Grad-CAM/Finer-CAM visual explanations.

> **Note:** This assignment was completed during a period of widespread internet disruption in Iran.

## Assignment 2-1 — LIME

This assignment investigates **LIME (Local Interpretable Model-Agnostic Explanations)** for interpreting predictions of a pretrained **ResNet-18** image classification model.

### Model and Dataset

* **Model:** Pretrained ResNet-18
* **Framework:** PyTorch
* **Images:** `cat.jpg` and `polarBear.png`
* **Target classes:** `282` (cat) and `296` (polar bear)
* Images are resized to `224 × 224` and normalized using ImageNet statistics.

The model is first evaluated on the original cat image, and the probability assigned to class `282` is recorded.

### Image Segmentation

The input image is divided into **18 segments** using a custom K-means implementation.

Each pixel is represented using its RGB values and spatial coordinates. The resulting segments are visualized together with the original image.

### Perturbed Samples

**1000 random binary vectors** of length 18 are generated, where each value determines whether the corresponding image segment is kept or replaced.

Segments with value `0` are replaced with the mean color of the original image. Several perturbed images are visualized.

### LIME Weighting

Each perturbed sample is weighted according to its cosine distance from the original image representation.

The LIME kernel is implemented as:

```text
w(z) = exp(-(dist(z)²) / σ²)
```

with:

```text
σ = 0.25 × √d
```

where `d = 18`.

### Linear Model and Heatmap

The perturbed images are passed through ResNet-18 and the probability of the target class is recorded.

A **Lasso regression** model is then trained using:

* `alpha = 0.1`
* `fit_intercept = False`
* LIME sample weights

The learned coefficients are mapped back to the image segments to produce a **LIME heatmap**, showing the regions that contribute most to the model's local prediction.

### Polar Bear Analysis

The same procedure is applied to `polarBear.png` using class `296` as the target class. The resulting heatmap and model behavior are analyzed and compared with the cat image.

### Notebook

The complete implementation, visualizations, and results are available in:

`notebooks/Assignment2-1.ipynb`

## Assignment 2-2 — Grad-CAM and Finer-CAM

This assignment investigates **Grad-CAM** and **Finer-CAM** for interpreting the predictions of a pretrained **ResNet-18** model on Tiny ImageNet images.

### Model and Dataset

* **Model:** Pretrained ResNet-18
* **Dataset:** Tiny ImageNet
* **Framework:** PyTorch
* **Input size:** `224 × 224`
* **Target layer:** `model.layer4[-1]`

Five images that are confidently classified by the model are selected for the interpretability analysis. The model's predictions and corresponding class indices are used as the target classes for generating explanations.

### Grad-CAM

**Grad-CAM** is implemented from scratch using PyTorch hooks to capture the activations and gradients of the target convolutional layer.

The gradients are spatially averaged to obtain channel importance weights, which are then combined with the feature maps. ReLU is applied to obtain the final class-specific activation map.

The Grad-CAM maps and their overlays on the original images are visualized.

### Guided Grad-CAM

**Guided Backpropagation** is implemented by modifying the gradients through the ReLU layers.

The resulting guided gradients are combined element-wise with the Grad-CAM maps to produce **Guided Grad-CAM** visualizations.

Vanilla Backpropagation, Guided Backpropagation, Grad-CAM, and Guided Grad-CAM results are also compared.

### Finer-CAM

**Finer-CAM** is implemented from scratch using the target class and its top-3 competing classes.

A modified objective is constructed by comparing the target class logit with the logits of the most similar competing classes. The resulting gradients are used to generate the Finer-CAM activation maps.

Different values of `γ` are evaluated:

* `γ = 0.0`
* `γ = 0.3`
* `γ = 0.6`
* `γ = 1.0`

The effect of `γ` on the generated explanations is visualized.

### Comparison

Grad-CAM and Finer-CAM maps are displayed side by side for the five selected images.

The generated maps are also combined with Guided Backpropagation to produce **Guided Finer-CAM**. Difference maps between the two methods are visualized to analyze how their explanations differ and which image regions are emphasized by each method.

### Notebook

The complete implementation, visualizations, and comparisons are available in:

`notebooks/Assignment2-2.ipynb`
