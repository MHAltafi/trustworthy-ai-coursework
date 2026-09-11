# Assignment 1 — Trustworthy AI

This assignment covers spurious correlations and adversarial robustness through two parts.

> **Note:** This assignment was completed during a period of widespread internet disruption in Iran.

## Assignment 1-1 — Spurious Correlation and JTT

This part investigates **spurious correlations** in Colored MNIST and evaluates ERM, JTT-style upsampling, and Generalized Cross-Entropy (GCE).

### Dataset

* MNIST digits are converted into binary classes: `0–4 → class 0` and `5–9 → class 1`.
* Images are converted from grayscale to RGB.
* The training set contains a strong correlation between class and color, while the test set reverses this correlation.

### Model and Experiments

A simple MLP is used with:

* Input: `3 × 28 × 28`
* Hidden layer: `128` units with ReLU
* Output: `2` classes

The following approaches are evaluated:

* **ERM:** Standard training with cross-entropy loss.
* **Error Set:** Training samples misclassified by the ERM model.
* **JTT:** Error Set samples are upsampled by a factor of `50` and the model is retrained.
* **GCE:** Generalized Cross-Entropy loss is used as an alternative training objective.

### Results

The notebook reports training errors, Error Set size, and test accuracy for the different approaches. Sample Colored MNIST images and experimental results are also included.

### Notebook

The complete implementation is available in:

`notebooks/Assignment1-1.ipynb`

## Assignment 1-2 — Adversarial Attacks

This part investigates the robustness of a pretrained **ResNet-20** model on CIFAR-10 using two untargeted white-box adversarial attacks.

### Model and Dataset

* **Model:** Pretrained ResNet-20 for CIFAR-10
* **Dataset:** 25 CIFAR-10 images that are initially classified correctly by the model
* **Framework:** PyTorch
* **Device:** CPU, CUDA, or Apple MPS when available

### FGSM

The **Fast Gradient Sign Method (FGSM)** was implemented as an untargeted white-box attack.

* Epsilon: `0.03`
* The perturbation is computed from the sign of the loss gradient with respect to the input.
* The resulting adversarial images are evaluated using the pretrained model.
* Four successful adversarial examples are visualized with their original image, perturbation, adversarial image, true label, and incorrect prediction.

### PGD

The **Projected Gradient Descent (PGD)** attack was implemented as an iterative extension of FGSM.

* Epsilon: `0.03`
* Step size: `0.005`
* Maximum steps: `20`
* The perturbation is projected back into the allowed epsilon-ball after each step.
* Four successful adversarial examples are visualized.

### Results

The model achieves **100% accuracy on the provided clean images** before applying attacks.

| Evaluation | Accuracy |
|---|---:|
| Clean images | 100.00% |
| FGSM | 28.00% |
| PGD | 4.00% |

### Notebook

The complete implementation and visualizations are available in:

`notebooks/Assignment1-2.ipynb`
